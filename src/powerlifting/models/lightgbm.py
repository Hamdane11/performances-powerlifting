"""Powerlifting v3 boosting recipes; all learned transforms fit on training rows.

Research uses fit/select only. fit_candidate takes a frozen configuration with
fixed tree counts, never evaluates the outer fold. Native categorical vocabularies,
frequency maps and optional cross-fitted target encodings stay inside each fit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from .lightgbm_features import make_features as v2_features, allometric_scale


CONTEXT = [
    ('Sex', 'WeightClassKg'), ('Equipment', 'WeightClassKg'),
    ('Sex', 'Federation'), ('Equipment', 'Federation'),
    ('Sex', 'Equipment', 'Federation'), ('Federation', 'MeetYear'),
    ('Sex', 'AgeClass'), ('Equipment', 'AgeClass'),
    ('Sex', 'Country'), ('Sex', 'Equipment', 'Tested'),
    ('Sex', 'Equipment', 'WeightClassKg'),
]


def target_scale(frame, config):
    """Positive rowwise scale: allometry or a documented IPF-inspired prior.

    Source for the A,B,C coefficients (three-lift total):
    https://www.powerlifting.sport/fileadmin/ipf/data/ipf-formula/IPF_GL_Coefficients-2020.pdf
    This is a modeling transformation, not an official sporting points result:
    Mx uses allometry; below official 40kg M / 35kg F validity thresholds we
    extend continuously by a 2/3 power; non-Raw/Wraps equipment uses the equipped
    curve only in the explicitly named ipf_gl_equipment variant.
    """
    mode = config.get('target_scale', 'allometric')
    if mode == 'allometric':
        return allometric_scale(frame, float(config.get('exponent', 0.)))
    if mode not in ('ipf_gl_classic', 'ipf_gl_equipment'):
        raise ValueError('Unknown target scale.')
    result = allometric_scale(frame, 2/3)
    weight = frame['BodyweightKg'].to_numpy(dtype=float)
    sex = frame['Sex'].to_numpy()
    equipped = ~frame['Equipment'].isin(['Raw', 'Wraps']).to_numpy()
    for category, threshold, classic, equipment in [
        ('M', 40., (1199.72839,1025.18162,.00921), (1236.25115,1449.21864,.01644)),
        ('F', 35., (610.32796,1045.59282,.03048), (758.63878,949.31382,.02435)),
    ]:
        for use_equipment in [False, True]:
            mask = sex == category
            if mode == 'ipf_gl_equipment':
                mask &= equipped == use_equipment
            elif use_equipment:
                continue
            A,B,C = equipment if use_equipment else classic
            w = np.maximum(weight[mask], threshold)
            scale = (A-B*np.exp(-C*w))/600.
            scale *= np.minimum(weight[mask]/threshold,1.) ** (2/3)
            result[mask] = scale
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError('Invalid modeling scale.')
    return result


def _values(frame, columns):
    result = frame[columns[0]].fillna('Unknown').astype(str)
    for column in columns[1:]:
        result = result + '|' + frame[column].fillna('Unknown').astype(str)
    return result


def make_features(frame, config, state=None):
    """Create rowwise features and train-only categorical/frequency mappings."""
    fitting = state is None
    if fitting:
        state = {}
    v2_config = {'features': {'cohort': True, 'interactions': True},
                 'base_feature_options': {'geography': True, 'age_half_year': True}}
    X, base = v2_features(frame, v2_config, None if fitting else state['base'])
    state['base'] = base
    options = config.get('features', {})
    extra = {}
    if options.get('context', False):
        for columns in CONTEXT:
            extra['Cross_' + '_'.join(columns)] = _values(frame, columns)
    if options.get('age_bounds', False):
        for column in ['AgeClass', 'BirthYearClass']:
            bounds = frame[column].astype(str).str.extract(r'^(\d+)-(\d+)$')
            low = pd.to_numeric(bounds[0], errors='coerce')
            high = pd.to_numeric(bounds[1], errors='coerce').mask(lambda v: v >= 999)
            X[column + '_Lower'] = low
            X[column + '_Upper'] = high
            X[column + '_Offset'] = X['Age'] - low
    if options.get('numeric', False):
        X['LogBodyweight'] = np.log(frame['BodyweightKg'])
        X['BodyweightSquared'] = frame['BodyweightKg'] ** 2
        X['BodyweightFraction'] = frame['BodyweightKg'] % 1
        X['CountryMatchesMeet'] = (frame['Country'] == frame['MeetCountry']).astype(int)
        X['StateMatchesMeet'] = (frame['State'] == frame['MeetState']).astype(int)
    if options.get('precision_travel', False):
        weight = frame['BodyweightKg'].to_numpy(float)
        for factor, suffix in [(1, 'Integer'), (2, 'Half'), (10, 'Tenth')]:
            X['Bodyweight' + suffix] = np.isclose(weight * factor, np.round(weight * factor), atol=1e-7, rtol=0).astype(int)
        known_country = frame['Country'].ne('Unknown') & frame['MeetCountry'].ne('Unknown')
        known_state = frame['State'].ne('Unknown') & frame['MeetState'].ne('Unknown')
        same_country = frame['Country'].eq(frame['MeetCountry'])
        X['SameKnownCountry'] = same_country.where(known_country).astype(float)
        X['TravelsState'] = (same_country & frame['State'].ne(frame['MeetState'])).where(known_country & known_state).astype(float)
    if fitting:
        state['extra_categories'] = {}
    for column, values in extra.items():
        if fitting:
            state['extra_categories'][column] = sorted(values.unique().tolist())
        X[column] = pd.Categorical(values, categories=state['extra_categories'][column])
    if options.get('frequency', False):
        if fitting:
            state['frequencies'] = {}
        for columns in [('Federation', 'MeetYear', 'MeetCountry', 'MeetState'),
                        ('Country', 'State'), ('Sex', 'Equipment', 'WeightClassKg')]:
            name = 'Frequency_' + '_'.join(columns)
            values = _values(frame, columns)
            if fitting:
                state['frequencies'][name] = values.value_counts(normalize=True).to_dict()
            X[name] = values.map(state['frequencies'][name]).fillna(0).astype(float)
    if options.get('drop_weight_derived', False):
        X = X.drop(columns=['WeightClassGapClosed', 'WeightClassRatio'])
    if options.get('standardize_numeric', False):
        numeric = X.select_dtypes(include='number').columns.tolist()
        if fitting:
            state['numeric_center'] = X[numeric].mean().to_dict()
            state['numeric_scale'] = X[numeric].std().replace(0, 1).fillna(1).to_dict()
        for column in numeric:
            X[column] = (X[column] - state['numeric_center'][column]) / state['numeric_scale'][column]
    if 'id' in X or 'TotalKg' in X:
        raise ValueError('Target or identifier leaked into features.')
    return X, state


def _transform_target_encoding(X, target=None, config=None, state=None):
    if state is None:
        columns = [c for c in X if isinstance(X[c].dtype, pd.CategoricalDtype)]
        transformer = TargetEncoder(target_type='continuous', smooth=config.get('te_smooth', 30.),
                                    cv=5, shuffle=True, random_state=20261003)
        values = X[columns].astype(str)
        encoded = transformer.fit_transform(values, target)
        state = {'columns': columns, 'transformer': transformer}
    else:
        columns = state['columns']
        encoded = state['transformer'].transform(X[columns].astype(str))
    result = X.copy()
    for i, column in enumerate(columns):
        result['TargetMean_' + column] = encoded[:, i]
    return result, state


def _fit_one(frame, target, config, evaluation=None):
    X, state = make_features(frame, config)
    scale = target_scale(frame, config)
    z = np.asarray(target, dtype=float) / scale
    te = None
    if config.get('target_encoding', False):
        X, te = _transform_target_encoding(X, target=z, config=config)
    params = copy.deepcopy(config['params'])
    params['n_jobs'] = int(config.get('threads', 2))
    params['n_estimators'] = int(config.get('best_iteration', params.get('n_estimators', 4000)))
    if config.get('monotone_bodyweight', False):
        params['monotone_constraints'] = [int(c == 'BodyweightKg') for c in X]
        params['monotone_constraints_method'] = 'advanced'
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {'sample_weight': scale ** 2, 'categorical_feature': 'auto'}
    if evaluation is not None:
        val, yval = evaluation
        XV, _ = make_features(val, config, state)
        if te is not None:
            XV, _ = _transform_target_encoding(XV, state=te)
        sv = target_scale(val, config)
        # Weighted transformed RMSE differs by a constant independent of the
        # iteration; selecting its minimum exactly selects kilogram RMSE.
        fit_kwargs.update(eval_set=[(XV, np.asarray(yval) / sv)],
                          eval_sample_weight=[sv ** 2], eval_metric='rmse',
                          callbacks=[lgb.early_stopping(int(config.get('patience', 150)),
                                                       verbose=False, first_metric_only=True)])
    model.fit(X, z, **fit_kwargs)
    frozen = copy.deepcopy(config)
    frozen['best_iteration'] = int(model.best_iteration_ or model.n_estimators_)
    return {'config': frozen, 'state': state, 'target_encoder': te,
            'feature_names': list(X), 'model': model}


def _predict_one(fitted, frame):
    X, _ = make_features(frame, fitted['config'], fitted['state'])
    if fitted['target_encoder'] is not None:
        X, _ = _transform_target_encoding(X, state=fitted['target_encoder'])
    if list(X) != fitted['feature_names']:
        raise ValueError('Prediction schema differs from training schema.')
    p = fitted['model'].predict(X) * target_scale(frame, fitted['config'])
    return np.maximum(np.asarray(p, dtype=float), 0.)


def fit_candidate(frame, y, config, evaluation=None):
    """Fit fixed recipe; optional evaluation is for selection experiments only."""
    target = np.asarray(y, dtype=float)
    if len(frame) != len(target) or not np.isfinite(target).all():
        raise ValueError('Target must contain one finite value per training row.')
    frozen = copy.deepcopy(config)
    group_columns = config.get('specialist_columns', [])
    if not group_columns:
        fitted = _fit_one(frame, target, config, evaluation)
        return {'config': fitted['config'], 'global': fitted, 'specialists': {}}
    global_config = copy.deepcopy(config)
    global_config.pop('specialist_columns', None)
    global_config['params'].update(config.get('global_params', {}))
    if 'global_iterations' in config:
        global_config['best_iteration'] = config['global_iterations']
    global_fit = _fit_one(frame, target, global_config, evaluation)
    frozen['global_iterations'] = global_fit['config']['best_iteration']
    keys = _values(frame, group_columns)
    valkeys = None if evaluation is None else _values(evaluation[0], group_columns)
    specialists = {}
    frozen['specialist_iterations'] = {}
    for key, count in keys.value_counts().items():
        if count < int(config.get('min_specialist_rows', 1000)):
            continue
        mask = np.asarray(keys == key)
        local_config = copy.deepcopy(config)
        local_config.pop('specialist_columns', None)
        local_config['params'].update(config.get('specialist_params', {}))
        if 'specialist_iterations' in config:
            # Exactly the frozen selected groups; sample-size changes in OOF
            # must not accidentally create a newly selected expert.
            if key not in config['specialist_iterations']:
                continue
            local_config['best_iteration'] = config['specialist_iterations'][key]
        local_eval = None
        if evaluation is not None:
            vmask = np.asarray(valkeys == key)
            if vmask.sum() < 100:
                continue
            local_eval = (evaluation[0].iloc[np.flatnonzero(vmask)], np.asarray(evaluation[1])[vmask])
        specialists[key] = _fit_one(frame.iloc[np.flatnonzero(mask)], target[mask], local_config, local_eval)
        frozen['specialist_iterations'][key] = specialists[key]['config']['best_iteration']
    return {'config': frozen, 'global': global_fit, 'specialists': specialists}


def predict_candidate(fitted, frame):
    prediction = _predict_one(fitted['global'], frame)
    config = fitted['config']
    if fitted['specialists']:
        keys = _values(frame, config['specialist_columns'])
        weight = float(config.get('specialist_weight', 1.))
        for key, expert in fitted['specialists'].items():
            rows = np.flatnonzero(np.asarray(keys == key))
            if len(rows):
                local = _predict_one(expert, frame.iloc[rows])
                prediction[rows] = (1 - weight) * prediction[rows] + weight * local
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid predictions.')
    return prediction


def save_candidate(fitted, directory):
    """Trusted local pickle plus manifest; no external pickle loading."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / 'model.joblib'
    joblib.dump(fitted, file, compress=3)
    metadata = {'format_version': 1, 'config': fitted['config'],
                'model_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'lightgbm_version': lgb.__version__}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return directory


def load_candidate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    file = directory / 'model.joblib'
    if hashlib.sha256(file.read_bytes()).hexdigest() != metadata['model_sha256']:
        raise ValueError('Saved model checksum mismatch.')
    return joblib.load(file)
