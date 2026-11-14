"""Colonnes du challenge et préparation déterministe de TabPFN.

Les transformations apprises (vocabulaires, normalisation, etc.) restent
dans chaque modèle et sont ajustées uniquement sur ses lignes d'apprentissage.
"""
from __future__ import annotations

import numpy as np
from pandas.api.types import is_numeric_dtype



RAW_NUMERICAL = ['Age', 'BodyweightKg', 'MeetYear']
RAW_CATEGORICAL = [


    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg',
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation',
    'MeetCountry', 'MeetState', 'Sanctioned',
]
RAW_FEATURES = RAW_NUMERICAL + RAW_CATEGORICAL


def prepare_tabpfn(frame, feature_names=None, *, clean_age=True):
    """Préparer les requêtes ou le contexte sans apprendre de statistiques.

    La recette v5 conserve les seize variables originales. Une expérience
    peut fournir des colonnes supplémentaires ; les colonnes de texte sont
    converties en chaînes et les variables numériques restent numériques.
    """
    names = RAW_FEATURES if feature_names is None else list(feature_names)
    if 'id' in names or 'TotalKg' in names:
        raise ValueError('id et TotalKg ne doivent pas être des variables explicatives.')
    x = frame.loc[:, names].copy()
    if clean_age and 'Age' in x:
        x.loc[x['Age'] < 5, 'Age'] = np.nan
    for column in names:
        if column in RAW_CATEGORICAL or not is_numeric_dtype(x[column].dtype):
            x[column] = x[column].fillna('Unknown').astype(str)
    return x
