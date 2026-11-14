"""Variables CatBoost de la recette historique, calculées ligne par ligne."""
import numpy as np
import pandas as pd
from ..features import RAW_NUMERICAL, RAW_CATEGORICAL

def allometric_cat_features(frame):
    X = frame[RAW_NUMERICAL + RAW_CATEGORICAL].copy()
    for column in RAW_NUMERICAL:
        X[column] = pd.to_numeric(X[column], errors='coerce')
    X.loc[X['Age'] < 5, 'Age'] = np.nan
    X['AgeEstimated'] = np.where(X['Age'].notna(), np.isclose(X['Age'] % 1, 0.5).astype(float), np.nan)
    for column in RAW_CATEGORICAL:
        X[column] = X[column].fillna('Unknown').astype(str)
    parsed = X['WeightClassKg'].str.extract('^\\s*(\\d+(?:\\.\\d+)?)\\s*(\\+)?\\s*$')
    X['WeightClassBound'] = pd.to_numeric(parsed[0], errors='coerce')
    X['WeightClassOpen'] = np.where(parsed[0].notna(), parsed[1].eq('+').astype(float), np.nan)
    X['WeightClassGapClosed'] = (X['WeightClassBound'] - X['BodyweightKg']).where(X['WeightClassOpen'].eq(0))
    for country, state, result in [('Country', 'State', 'CountryState'), ('MeetCountry', 'MeetState', 'MeetCountryState')]:
        known = ~X[country].str.strip().str.lower().isin(['unknown', '', 'nan', 'none'])
        known &= ~X[state].str.strip().str.lower().isin(['unknown', '', 'nan', 'none'])
        X[result] = np.where(known, X[country] + '|' + X[state], 'Unknown')
    categorical = RAW_CATEGORICAL + ['CountryState', 'MeetCountryState']
    return (X, categorical)

def allometric_scale(X, exponent):
    bodyweight = X['BodyweightKg'].to_numpy(dtype=float)
    if not np.all(np.isfinite(bodyweight) & (bodyweight > 0)):
        raise ValueError('Allometric scaling requires positive observed bodyweights.')
    return (bodyweight / 80.0) ** exponent
