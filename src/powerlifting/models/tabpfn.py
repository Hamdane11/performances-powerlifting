"""TabPFN 3.5 : modèle préentraîné, conditionné sur le contexte du challenge.

fit prépare le contexte et les transformations ; il ne met pas à jour les
poids neuronaux. Chaque lot de requêtes utilise le même contexte complet.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from ..features import prepare_tabpfn


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def fit_predict(context, y_context, queries, config, feature_names=None):
    """Prédire avec la recette TabPFN figée ou une configuration d'expérience."""
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    params = config
    expected_version = params.get('package_version')
    if expected_version and importlib.metadata.version('tabpfn') != expected_version:
        raise RuntimeError(f'La configuration exige tabpfn=={expected_version}.')
    if str(params.get('device', 'cuda')).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Un GPU CUDA est nécessaire pour cette configuration TabPFN.')
    clean_age = params.get('age_below_five_as_missing', True)
    x_context = prepare_tabpfn(context, feature_names, clean_age=clean_age)
    x_query = prepare_tabpfn(queries, feature_names, clean_age=clean_age)
    model = TabPFNRegressor.create_default_for_version(
        getattr(ModelVersion, params.get('model_version', 'V3_5')),
        n_estimators=params['n_estimators'],
        device=params['device'],
        fit_mode=params['fit_mode'],
        memory_saving_mode=params['memory_saving_mode'],
        n_preprocessing_jobs=params['n_preprocessing_jobs'],
        random_state=params['random_state'],
        categorical_features_indices=params['categorical_features_indices'],
        inference_config=params['inference_config'],
    )
    if params.get('checkpoint_name') and Path(model.model_path).name != params['checkpoint_name']:
        raise RuntimeError('Le nom du checkpoint TabPFN diffère de la configuration.')

    model.fit(x_context, np.asarray(y_context, dtype=float))
    if params.get('checkpoint_sha256') and _sha256(model.model_path) != params['checkpoint_sha256']:
        raise RuntimeError('Le checkpoint TabPFN diffère de la référence vérifiée.')
    if model.n_train_samples_ != len(context) or model.n_estimators_ != params['n_estimators']:
        raise RuntimeError('La taille du contexte ou le nombre de variantes a changé.')
    expected_subsample = params['inference_config'].get('SUBSAMPLE_SAMPLES')
    if model.get_inference_config().SUBSAMPLE_SAMPLES != expected_subsample:
        raise RuntimeError('Le sous-échantillonnage du contexte diffère de la configuration.')

    predictions = []
    step = int(params['query_chunk_rows'])

    if step < 1:
        raise ValueError('query_chunk_rows doit être strictement positif.')
    for start in range(0, len(x_query), step):
        values = model.predict(
            x_query.iloc[start:start + step], output_type=params.get('output_type', 'mean'),
        )
        predictions.append(np.maximum(np.asarray(values, dtype=float), 0.))
        print(f'TabPFN : {min(start + step, len(x_query))}/{len(x_query)}', flush=True)
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=float)
