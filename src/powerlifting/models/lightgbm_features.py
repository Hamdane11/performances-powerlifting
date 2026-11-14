"""Variables LightGBM de la recette historique : catégories apprises sur le train."""
import numpy as np
import pandas as pd

NUMERIC = ['Age', 'BodyweightKg', 'MeetYear']
INTERACTIONS = [('Sex', 'Equipment', 'SexEquipment'),
                ('Sex', 'Tested', 'SexTested'),
                ('Equipment', 'Tested', 'EquipmentTested')]

def _base_features(frame, engineered=True, vocabulary=None, config_options=None):
    """Fit-only category mapping; unknown prediction categories become missing.

    config_options: age_mask (default True), geography (default False),
    age_half_year (default False). engineered retains original API semantics.
    WeightClassRatio divides bodyweight by the parsed positive boundary for
    all classes: an upper bound for closed classes, lower bound for open +
    classes. WeightClassGapClosed is only defined for closed classes. Raw
    Unknown categories are retained; unparsable weight boundaries are NaN.
    """
    options = {'age_mask': True, 'geography': False, 'age_half_year': False}
    if config_options:
        options.update(config_options)
    result = frame.drop(columns=['id', 'TotalKg'], errors='ignore').copy()
    numeric = set(NUMERIC)
    age = pd.to_numeric(result['Age'], errors='coerce')
    if options['age_half_year']:
        result['AgeIsHalfYear'] = (age % 1 == 0.5).astype(float).where(age.notna())
        numeric.add('AgeIsHalfYear')
    if options['age_mask']:
        result.loc[age < 5, 'Age'] = np.nan
    if engineered:
        parsed = result['WeightClassKg'].astype('string').str.extract('^((?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+))(\\+?)$')
        boundary = pd.to_numeric(parsed[0], errors='coerce')
        boundary = boundary.where(boundary > 0)
        valid = boundary.notna()
        is_open = parsed[1].eq('+').where(valid).astype(float)
        result['WeightClassBoundary'] = boundary
        result['WeightClassIsOpen'] = is_open
        result['WeightClassGapClosed'] = (boundary - result['BodyweightKg']).where(is_open.eq(0))
        result['WeightClassRatio'] = result['BodyweightKg'] / boundary
        numeric.update(['WeightClassBoundary', 'WeightClassIsOpen', 'WeightClassGapClosed', 'WeightClassRatio'])
    if options['geography']:
        for country, region, name in [('Country', 'State', 'CountryState'), ('MeetCountry', 'MeetState', 'MeetCountryState')]:
            country_s = result[country].astype('string')
            region_s = result[region].astype('string')
            valid_country = country_s.notna() & ~country_s.str.lower().isin(['unknown', 'nan', 'none', ''])
            valid_region = region_s.notna() & ~region_s.str.lower().isin(['unknown', 'nan', 'none', ''])
            result[name] = (country_s + '|' + region_s).where(valid_country & valid_region, 'Unknown')
    categorical = [c for c in result.columns if c not in numeric]
    if vocabulary is None:
        vocabulary = {c: sorted(result[c].dropna().astype(str).unique().tolist()) for c in categorical}
    for column in categorical:
        result[column] = pd.Categorical(result[column].astype('string'), categories=vocabulary[column])
    return (result, vocabulary)

def allometric_scale(X, exponent):
    bodyweight = X['BodyweightKg'].to_numpy(dtype=float)
    if not np.all(np.isfinite(bodyweight) & (bodyweight > 0)):
        raise ValueError('Allometric scaling requires positive observed bodyweights.')
    return (bodyweight / 80.0) ** exponent

def make_features(frame, config, vocabulary=None):
    """Return transformed frame and training-only vocabularies.

    Cohort is approximate MeetYear-Age. Native categorical unknowns become
    missing. Raw columns stay present; id and TotalKg are always excluded.
    """
    options=config.get('features', {})
    base_options=config.get('base_feature_options', {'geography':True, 'age_half_year':True})
    base_vocabulary=None if vocabulary is None else vocabulary['base']
    X,base_vocabulary=_base_features(frame, engineered=True,
        vocabulary=base_vocabulary, config_options=base_options)
    if options.get('cohort', False):
        X['EstimatedBirthYear']=X['MeetYear']-X['Age']
    interaction_vocabulary={} if vocabulary is None else vocabulary['interactions']
    if options.get('interactions', False):
        for left,right,name in INTERACTIONS:
            values=frame[left].astype('string')+'|'+frame[right].astype('string')
            if vocabulary is None:
                interaction_vocabulary[name]=sorted(values.dropna().unique().tolist())
            X[name]=pd.Categorical(values,categories=interaction_vocabulary[name])
    return X, {'base':base_vocabulary, 'interactions':interaction_vocabulary}
