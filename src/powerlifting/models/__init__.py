"""Point d'entrée des modèles : les dépendances GPU sont importées à la demande."""
from __future__ import annotations

import importlib

import numpy as np


def predict_model(family, context, y_context, queries, config, feature_names=None):
    """Ajuster un modèle sur le contexte puis prédire les lignes de queries.

    Les paramètres et durées d'entraînement viennent de la configuration.
    Cette fonction ne reçoit pas les cibles de validation : aucun arrêt
    anticipé n'est effectué sur le pli externe.
    """
    if family not in {'lightgbm', 'catboost', 'tabm', 'tabpfn'}:
        raise ValueError(f'Famille de modèle inconnue : {family}')
    module = importlib.import_module(f'.{family}', package=__name__)
    if family == 'tabpfn':
        prediction = module.fit_predict(
            context, y_context, queries, config, feature_names=feature_names,
        )
    else:
        fitted = module.fit_candidate(context, y_context, config)
        prediction = module.predict_candidate(fitted, queries)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(queries),) or not np.isfinite(prediction).all():
        raise ValueError('Le modèle doit fournir une prédiction finie par requête.')
    return prediction
