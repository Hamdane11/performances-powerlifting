"""TabM with fit-only numerical embeddings and categorical vocabularies.

Official implementation: https://github.com/yandex-research/tabm (ICLR 2025).
Embedding implementation: https://github.com/yandex-research/rtdl-num-embeddings.
The loss is the mean of individual ensemble-member losses, not the loss of
their mean. Predictions average members. Allometric sample weights preserve
the original kilogram squared-error data-fit term exactly (up to a constant).
Selection uses the supplied selection block only for early stopping; final
fit_candidate uses a frozen number of epochs and no validation targets.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, PeriodicEmbeddings, compute_bins

RAW_NUMERICAL = ['Age', 'BodyweightKg', 'MeetYear']
RAW_CATEGORICAL = ['Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg',
                   'Tested', 'Country', 'State', 'Federation', 'ParentFederation',
                   'MeetCountry', 'MeetState', 'Sanctioned']


def allometric_cat_features(frame):
    """Same deterministic basic features as v2, without importing tree libraries."""
    x = frame[RAW_NUMERICAL + RAW_CATEGORICAL].copy()
    for c in RAW_NUMERICAL:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x.loc[x['Age'] < 5, 'Age'] = np.nan
    x['AgeEstimated'] = np.where(x.Age.notna(), np.isclose(x.Age % 1, .5).astype(float), np.nan)
    for c in RAW_CATEGORICAL:
        x[c] = x[c].fillna('Unknown').astype(str)
    parsed = x.WeightClassKg.str.extract(r'^\s*(\d+(?:\.\d+)?)\s*(\+)?\s*$')
    x['WeightClassBound'] = pd.to_numeric(parsed[0], errors='coerce')
    x['WeightClassOpen'] = np.where(parsed[0].notna(), parsed[1].eq('+').astype(float), np.nan)
    x['WeightClassGapClosed'] = (x.WeightClassBound - x.BodyweightKg).where(x.WeightClassOpen.eq(0))
    for country, state, name in [('Country','State','CountryState'), ('MeetCountry','MeetState','MeetCountryState')]:
        valid = ~x[country].str.strip().str.lower().isin(['unknown','','nan','none'])
        valid &= ~x[state].str.strip().str.lower().isin(['unknown','','nan','none'])
        x[name] = np.where(valid, x[country] + '|' + x[state], 'Unknown')
    return x, RAW_CATEGORICAL + ['CountryState', 'MeetCountryState']


def allometric_scale(frame, exponent):
    weight = frame.BodyweightKg.to_numpy(dtype=float)
    if not np.all(np.isfinite(weight) & (weight > 0)):
        raise ValueError('BodyweightKg must be positive and finite.')
    return (weight / 80.) ** exponent


def make_features(frame, config):
    x, categorical = allometric_cat_features(frame)
    x['EstimatedBirthYear'] = x['MeetYear'] - x['Age']
    x['WeightClassRatio'] = x['BodyweightKg'] / x['WeightClassBound']
    if config.get('interactions', False):
        for a, b in [('Sex', 'Equipment'), ('Sex', 'Tested'), ('Equipment', 'Tested')]:
            c = a + '_' + b
            x[c] = x[a].astype(str) + '|' + x[b].astype(str)
            categorical.append(c)
    if config.get('features', 'engineered') == 'raw':
        keep = ['Age', 'BodyweightKg', 'MeetYear', 'EstimatedBirthYear']
        categorical = [c for c in categorical if c not in ['CountryState', 'MeetCountryState']]
        x = x[keep + categorical]
    numerical = [c for c in x if c not in categorical]
    assert 'id' not in x and 'TotalKg' not in x
    return x, numerical, categorical


def fit_preprocessor(frame, config):
    x, numerical, categorical = make_features(frame, config)
    values = x[numerical].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians = np.nan_to_num(medians)
    missing = ~np.isfinite(values)
    missing_columns = np.flatnonzero(missing.any(axis=0))
    values = np.where(missing, medians[None, :], values)
    values = np.column_stack([values, missing[:, missing_columns].astype(float)])
    active = np.flatnonzero(np.ptp(values, axis=0) > 0)
    values = values[:, active]
    if config.get('normalization', 'standard') == 'quantile':
        transform = QuantileTransformer(n_quantiles=min(1000, len(values)),
            output_distribution='normal', subsample=100000, random_state=int(config['seed']))
    else:
        transform = StandardScaler()
    nums = transform.fit_transform(values).astype('float32')
    vocabularies = {}
    minimum = int(config.get('min_category_count', 10))
    for c in categorical:
        counts = x[c].astype(str).value_counts()
        # Zero is reserved for categories absent or infrequent during fitting.
        levels = sorted(counts[counts >= minimum].index.tolist())
        vocabularies[c] = {v: i + 1 for i, v in enumerate(levels)}
    cats = np.column_stack([x[c].map(vocabularies[c]).fillna(0).to_numpy(dtype='int64')
                           for c in categorical])
    pre = dict(numerical=numerical, categorical=categorical, medians=medians,
               missing_columns=missing_columns, active=active, transform=transform,
               vocabularies=vocabularies, n_fit=len(frame),
               cardinalities=[len(vocabularies[c]) + 1 for c in categorical])
    return pre, torch.from_numpy(nums), torch.from_numpy(cats)


def transform_frame(frame, config, pre):
    x, numerical, categorical = make_features(frame, config)
    assert numerical == pre['numerical'] and categorical == pre['categorical']
    values = x[numerical].to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    values = np.where(missing, pre['medians'][None, :], values)
    values = np.column_stack([values, missing[:, pre['missing_columns']].astype(float)])
    nums = pre['transform'].transform(values[:, pre['active']]).astype('float32')
    cats = np.column_stack([x[c].map(pre['vocabularies'][c]).fillna(0).to_numpy(dtype='int64')
                           for c in categorical])
    assert np.isfinite(nums).all()
    return torch.from_numpy(nums), torch.from_numpy(cats)


def make_model(config, nums, cardinalities, bins=None):
    embedding = config.get('embedding', 'pwl')
    if embedding == 'pwl':
        if bins is None:
            bins = compute_bins(nums, n_bins=int(config.get('n_bins', 32)))
        embeds = PiecewiseLinearEmbeddings(bins, int(config.get('d_embedding', 12)),
                                            activation=False, version='B')
    elif embedding == 'periodic':
        embeds = PeriodicEmbeddings(nums.shape[1], d_embedding=int(config.get('d_embedding', 12)),
                                    n_frequencies=16, frequency_init_scale=float(config.get('frequency_scale', .1)),
                                    lite=True)
    elif embedding == 'none':
        embeds = None
    else:
        raise ValueError(embedding)
    model = TabM.make(n_num_features=nums.shape[1], cat_cardinalities=cardinalities,
                     d_out=1, num_embeddings=embeds, k=int(config.get('k', 8)),
                     n_blocks=int(config.get('n_blocks', 2)), d_block=int(config.get('d_block', 128)),
                     dropout=float(config.get('dropout', .1)), arch_type=config.get('arch_type', 'tabm'))
    return model, bins


@torch.inference_mode()
def predict_encoded(model, nums, cats, batch_size=2048):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    for lo in range(0, len(nums), batch_size):
        values = model(nums[lo:lo+batch_size].to(device), cats[lo:lo+batch_size].to(device))
        predictions.append(values[..., 0].mean(1).cpu().numpy())
    return np.concatenate(predictions)


def _fit(frame, y, config, validation=None, log_path=None):
    config = deepcopy(config)
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('threads', 2)))
    requested = config.get('device', 'auto')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    pre, nums, cats = fit_preprocessor(frame, config)
    model, bins = make_model(config, nums, pre['cardinalities'])
    model.to(device)
    exponent = float(config.get('exponent', 0.))
    scale = allometric_scale(frame, exponent)
    z = np.asarray(y, dtype=float) / scale
    weights = scale ** 2
    target_mean = float(np.average(z, weights=weights))
    target_std = float(np.sqrt(np.average((z-target_mean)**2, weights=weights)))
    target = torch.tensor((z-target_mean)/target_std, dtype=torch.float32)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    if validation is not None:
        vx, vy = validation
        vnums, vcats = transform_frame(vx, config, pre)
        vscale = allometric_scale(vx, exponent)
        vy = np.asarray(vy, dtype=float)
        max_epochs = int(config.get('max_epochs', 120))
    else:
        max_epochs = int(config['epochs'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', .002)),
                                  weight_decay=float(config.get('weight_decay', .0003)))
    batch = int(config.get('batch_size', 512))
    best_score, best_epoch, best_state = np.inf, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs+1):
        model.train()
        order = torch.randperm(len(nums))
        epoch_sse = 0.
        for idx in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            output = model(nums[idx].to(device), cats[idx].to(device))[..., 0]
            # Each member is optimized separately. Averaging precedes no loss.
            loss = ((output - target[idx].to(device)[:, None]).square() * weights[idx].to(device)[:, None]).mean()
            loss.backward()
            optimizer.step()
            epoch_sse += float(loss.detach()) * len(idx)
        row = dict(epoch=epoch, train_objective=epoch_sse/len(nums), seconds=time.perf_counter()-started)
        if validation is not None:
            prediction = np.maximum((predict_encoded(model, vnums, vcats)*target_std+target_mean)*vscale, 0.)
            score = float(np.sqrt(np.mean((prediction-vy)**2)))
            row['selection_rmse'] = score
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row['best_rmse'] = best_score
            if epoch == 1 or epoch % 5 == 0:
                print(json.dumps(dict(name=config['name'], **row)), flush=True)
        elif epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(json.dumps(dict(name=config['name'], fixed_epochs=max_epochs, **row)), flush=True)
        history.append(row)
        if log_path is not None:
            Path(log_path).write_text(json.dumps(history, indent=2))
        if validation is not None and epoch-best_epoch >= int(config.get('patience', 16)):
            break
    if validation is not None:
        model.load_state_dict(best_state)
        config['epochs'] = best_epoch
    else:
        best_epoch = max_epochs
    fitted = dict(config=config, preprocessor=pre, model=model.cpu(), bins=bins,
                  target_mean=target_mean, target_std=target_std, history=history,
                  best_epoch=best_epoch, best_score=None if not np.isfinite(best_score) else best_score,
                  seconds=time.perf_counter()-started)
    return fitted


def fit_candidate(frame, y, config):
    """Frozen-epoch refit: has no access to validation targets."""
    if 'epochs' not in config:
        raise ValueError('Fixed epochs must be selected before fit_candidate.')
    return _fit(frame, y, config)


def fit_selection(frame, y, validation_frame, validation_y, config, log_path=None):
    return _fit(frame, y, config, (validation_frame, validation_y), log_path)


def predict_candidate(fitted, frame):
    torch.set_num_threads(int(fitted['config'].get('threads', 2)))
    nums, cats = transform_frame(frame, fitted['config'], fitted['preprocessor'])
    requested = fitted['config'].get('device', 'auto')
    device = 'cuda' if requested in ['cuda', 'auto'] and torch.cuda.is_available() else 'cpu'
    fitted['model'].to(device)
    z = predict_encoded(fitted['model'], nums, cats)
    fitted['model'].cpu()
    scale = allometric_scale(frame, fitted['config'].get('exponent', 0.))
    prediction = np.maximum((z*fitted['target_std']+fitted['target_mean'])*scale, 0.)
    assert prediction.shape == (len(frame),) and np.isfinite(prediction).all()
    return prediction


def save_candidate(fitted, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {k:v for k,v in fitted.items() if k != 'model'}
    # This preprocessing checkpoint is local trusted data, never an untrusted download.
    joblib.dump(metadata, directory/'preprocessor.joblib')
    torch.save(fitted['model'].state_dict(), directory/'weights.pt')
    (directory/'config.json').write_text(json.dumps(fitted['config'], indent=2))


def load_candidate(directory):
    directory = Path(directory)
    fitted = joblib.load(directory/'preprocessor.joblib')
    pre = fitted['preprocessor']
    dummy = torch.zeros((2, len(pre['active'])), dtype=torch.float32)
    model, _ = make_model(fitted['config'], dummy, pre['cardinalities'], fitted['bins'])
    model.load_state_dict(torch.load(directory/'weights.pt', weights_only=True, map_location='cpu'))
    fitted['model'] = model
    return fitted
