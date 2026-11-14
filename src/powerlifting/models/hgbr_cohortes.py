"""Modèles d'arbres à 54 variables avec moyennes de la cible par cohorte (solution finale).

Deux modèles partagent les mêmes 54 variables :
- ``HGBR54`` : ``HistGradientBoostingRegressor`` de scikit-learn ;
- ``LGBM54`` : LightGBM (arbres classiques, 63 feuilles, 800 arbres).

Variables (calculées sur le contexte seul, sans les cibles des requêtes) :
- les 16 variables d'origine et 10 variables dérivées (catégorie de poids, âge estimé,
  année de naissance, croisements, même pays) ;
- des « empreintes » de saisie (poids entier / demi / dixième, poids en livres arrondi,
  âge en ,5, nombre de métadonnées inconnues), le logarithme de l'effectif de 8 cohortes
  et le poids standardisé dans sa cohorte sexe|équipement|fédération ;
- pour ces 8 cohortes, la moyenne de la cible ramenée à 80 kg, codée par ``TargetEncoder``
  (validation croisée interne à 5 plis : une ligne d'apprentissage n'est jamais codée avec
  sa propre cible).
Les modèles apprennent la cible ramenée à 80 kg, y / (poids/80)^(2/3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import TargetEncoder
from threadpoolctl import threadpool_limits

from ..features import RAW_FEATURES, prepare_tabpfn

PARAMS_HGBR = {"loss": "squared_error", "learning_rate": 0.08, "max_iter": 400, "max_leaf_nodes": 31,
               "min_samples_leaf": 60, "l2_regularization": 10.0, "max_bins": 255, "early_stopping": False,
               "random_state": 2026}
PARAMS_LGBM = {"objective": "regression", "metric": "rmse", "n_estimators": 800, "learning_rate": 0.04,
               "n_jobs": 2, "random_state": 44, "verbosity": -1, "deterministic": True, "force_col_wise": True,
               "max_cat_to_onehot": 4, "reg_lambda": 20.0, "reg_alpha": 0.1, "colsample_bytree": 0.9,
               "subsample": 1.0, "subsample_freq": 0, "cat_smooth": 40.0, "cat_l2": 10.0, "num_leaves": 63,
               "min_child_samples": 70, "max_depth": -1, "extra_trees": False}
COHORTES = (
    ("Sex", "Equipment", "Federation"),
    ("Sex", "Equipment", "Federation", "MeetYear"),
    ("Sex", "Equipment", "Federation", "MeetYear", "MeetCountry", "MeetState"),
    ("Sex", "Equipment", "Federation", "Tested"),
    ("Sex", "Equipment", "Federation", "WeightClassKg"),
    ("Sex", "Equipment", "Federation", "AgeClass"),
    ("Sex", "Equipment", "MeetCountry", "MeetState", "MeetYear"),
    ("Sex", "Equipment", "Country", "State"),
)


def echelle(frame):
    poids = frame["BodyweightKg"].to_numpy(dtype=float)
    if not (np.isfinite(poids).all() and (poids > 0).all()):
        raise ValueError("Poids de corps manquant ou nul.")
    return (poids / 80.0) ** (2.0 / 3.0)


def variables_de_base(frame):
    """16 variables d'origine + 10 variables dérivées, sans statistique apprise."""
    x = prepare_tabpfn(frame).reset_index(drop=True)
    parsed = x["WeightClassKg"].str.extract(r"^([0-9]+(?:\.[0-9]+)?)(\+?)$")
    borne = pd.to_numeric(parsed[0], errors="coerce")
    borne = borne.where(borne > 0)
    ouverte = parsed[1].eq("+").astype(float).where(borne.notna())
    x["WeightClassBoundary"] = borne
    x["WeightClassIsOpen"] = ouverte
    x["WeightClassGapClosed"] = (borne - x["BodyweightKg"]).where(ouverte.eq(0))
    x["WeightClassRatio"] = x["BodyweightKg"] / borne
    bornes_age = x["AgeClass"].str.extract(r"^([0-9]+)-([0-9]+)$").apply(pd.to_numeric, errors="coerce")
    milieu = ((bornes_age[0] + bornes_age[1]) / 2).where(bornes_age[1] <= 100)
    x["AgeEstimated"] = x["Age"].fillna(milieu)
    x["EstimatedBirthYear"] = x["MeetYear"] - x["AgeEstimated"]
    for gauche, droite in (("Sex", "Equipment"), ("Sex", "Tested"), ("Equipment", "Tested")):
        x[f"{gauche}_{droite}"] = x[gauche] + "|" + x[droite]
    inconnu = ["unknown", "nan", "none", ""]
    connu = ~x["Country"].str.lower().isin(inconnu) & ~x["MeetCountry"].str.lower().isin(inconnu)
    x["SameCountry"] = x["Country"].eq(x["MeetCountry"]).astype(float).where(connu)
    return x


def cles_cohortes(frame):
    valeurs = {}
    for colonnes in COHORTES:
        serie = frame[colonnes[0]].fillna("Unknown").astype(str)
        for colonne in colonnes[1:]:
            serie = serie + "|" + frame[colonne].fillna("Unknown").astype(str)
        valeurs["__".join(colonnes)] = serie.reset_index(drop=True)
    return pd.DataFrame(valeurs)


def empreintes(contexte, requetes):
    """Variables de base, empreintes de saisie, effectifs et poids standardisé par cohorte."""
    a, b = variables_de_base(contexte), variables_de_base(requetes)
    for x in (a, b):
        poids = x["BodyweightKg"].to_numpy(float)
        livres = poids / 0.45359237
        for multiple, nom in ((1, "Integer"), (2, "Half"), (10, "Tenth")):
            x["BWkg" + nom] = np.isclose(poids * multiple, np.round(poids * multiple), atol=1e-7, rtol=0).astype(float)
        for multiple, nom in ((1, "Integer"), (2, "Half")):
            x["BWpoundDistance" + nom] = np.abs(livres * multiple - np.round(livres * multiple)) / multiple
        x["AgeMissing"] = x["Age"].isna().astype(float)
        x["AgeHalf"] = pd.Series(np.isclose(x["Age"] % 1, .5, atol=1e-7, rtol=0).astype(float), index=x.index).where(x["Age"].notna())
        manquant = x[["Country", "State", "MeetCountry", "MeetState", "ParentFederation", "Tested"]].eq("Unknown")
        x["MetadataMissingCount"] = manquant.sum(axis=1).astype(float)
        x["MetadataMissingSignature"] = manquant.astype(str).agg("|".join, axis=1)
        x["BodyweightFraction"] = poids % 1
        x["LogBodyweight"] = np.log(poids)
    ka, kb = cles_cohortes(contexte), cles_cohortes(requetes)
    for colonne in ka:
        effectifs = ka[colonne].value_counts()
        a["LogCount_" + colonne] = np.log1p(ka[colonne].map(effectifs).to_numpy(float))
        b["LogCount_" + colonne] = np.log1p(kb[colonne].map(effectifs).fillna(0).to_numpy(float))
    cle = ka.columns[0]
    stats = pd.DataFrame({"g": ka[cle], "w": contexte["BodyweightKg"].to_numpy(float)}).groupby("g")["w"].agg(["mean", "std"])
    for x, k in ((a, ka), (b, kb)):
        moyenne = k[cle].map(stats["mean"]).fillna(contexte["BodyweightKg"].mean())
        ecart = k[cle].map(stats["std"]).fillna(20).clip(lower=5)
        x["BodyweightCohortZ"] = (x["BodyweightKg"].to_numpy(float) - moyenne.to_numpy(float)) / ecart.to_numpy(float)
    return a, b, ka, kb


def coder(apprentissage, requetes):
    """Codes entiers appris sur l'apprentissage seul (200 modalités au plus, le reste regroupé)."""
    categorielle = np.asarray([not pd.api.types.is_numeric_dtype(apprentissage[c]) for c in apprentissage])
    xa = np.empty(apprentissage.shape, dtype=np.float64)
    xb = np.empty(requetes.shape, dtype=np.float64)
    for j, colonne in enumerate(apprentissage):
        if not categorielle[j]:
            xa[:, j] = pd.to_numeric(apprentissage[colonne], errors="coerce")
            xb[:, j] = pd.to_numeric(requetes[colonne], errors="coerce")
            continue
        vocabulaire = {v: i for i, v in enumerate(apprentissage[colonne].value_counts().head(200).index)}
        xa[:, j] = apprentissage[colonne].map(vocabulaire).fillna(len(vocabulaire)).to_numpy()
        xb[:, j] = requetes[colonne].map(vocabulaire).fillna(len(vocabulaire)).to_numpy()
    return xa, xb, categorielle


def variables_54(contexte, y_contexte, requetes, graine):
    """Les 54 variables du contexte et des requêtes, et les échelles de la cible."""
    contexte = contexte.loc[:, RAW_FEATURES].reset_index(drop=True)
    requetes = requetes.loc[:, RAW_FEATURES].reset_index(drop=True)
    s_contexte = echelle(contexte)
    a, b, ka, kb = empreintes(contexte, requetes)
    codeur = TargetEncoder(target_type="continuous", smooth=50.0, cv=5, shuffle=True, random_state=graine)
    with threadpool_limits(limits=1):
        za = codeur.fit_transform(ka, np.asarray(y_contexte, float) / s_contexte)
        zb = codeur.transform(kb)
    for j, colonne in enumerate(ka):
        a["CohortTarget_" + colonne] = za[:, j]
        b["CohortTarget_" + colonne] = zb[:, j]
    if a.shape[1] != 54:
        raise AssertionError("54 variables attendues.")
    xa, xb, categorielle = coder(a, b)
    return xa, xb, categorielle, s_contexte, echelle(requetes)


def predire_hgbr(contexte, y_contexte, requetes, graine=2026):
    xa, xb, cat, sa, sb = variables_54(contexte, y_contexte, requetes, graine)
    modele = HistGradientBoostingRegressor(categorical_features=cat, **{**PARAMS_HGBR, "random_state": graine})
    with threadpool_limits(limits=1):
        modele.fit(xa, np.asarray(y_contexte, float) / sa)
        return np.maximum(modele.predict(xb) * sb, 0.0)


def predire_lgbm54(contexte, y_contexte, requetes, graine=2026):
    import lightgbm as lgb

    xa, xb, cat, sa, sb = variables_54(contexte, y_contexte, requetes, graine)
    poids = sa ** 2
    modele = lgb.LGBMRegressor(**PARAMS_LGBM)
    modele.fit(xa.astype(np.float32), np.asarray(y_contexte, float) / sa, sample_weight=poids / poids.mean(),
               categorical_feature=np.flatnonzero(cat).tolist(), feature_name=[f"f{i}" for i in range(xa.shape[1])])
    return np.maximum(modele.predict(xb.astype(np.float32)) * sb, 0.0)
