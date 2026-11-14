"""Section 7 du rapport : quel modèle de second niveau pour la correction, et faut-il plutôt donner
les candidates en entrée d'un modèle individuel ?

1. Base : mélange avec l'ancienne version de TabPFN. Mêmes 43 caractéristiques et même validation
   strictement imbriquée que la correction retenue ; seul le modèle de second niveau change :
   LightGBM (celui du projet), XGBoost, CatBoost, HistGradientBoosting (petits arbres, 200 itérations,
   taux 0,03, moyenne de 5 graines) et régression linéaire (valeurs manquantes imputées + indicateurs).
2. LightGBM (configuration de départ, 32 variables) + caractéristiques d'appariement calculées sur les
   totaux des candidates (sans aucune prédiction), arrêt anticipé sur 10 % de l'apprentissage.

Exécution : python analyses/modeles_second_niveau.py      (environ 30 minutes sur 2 cœurs)
Sortie : resultats/analyses/modeles_second_niveau.json
"""
import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from _commun import CFG, base_melange, bootstrap_ecart, donnees, ecrire, lire_predictions, rmse
from powerlifting import appariement as A
from powerlifting.models import lightgbm as L

train, test, d, dev, hold = donnees()
y, plis = dev.TotalKg.to_numpy(float), d.plis
D = [("tabpfn_ancien" if n == "tabpfn" else n) for n in CFG["solution_finale"]["modeles"]]
p = base_melange(dev, y, plis, modeles=D)
r = y - p
t = A.tableau(dev, np.arange(len(dev)), y=y, p=p)
APP = {(k, m): A.caracteristiques(t[plis == m], t[(plis != m) & (plis != k)]) for k in range(1, 6) for m in range(1, 6)
       if m != k}
EVAL = {k: A.caracteristiques(t[plis == k], t[plis != k]) for k in range(1, 6)}
print("caractéristiques calculées", flush=True)
GRAINES = A.GRAINES


def moyenne_graines(fabrique):
    return lambda X, c, Xn: np.mean([fabrique(s).fit(X, c).predict(Xn) for s in GRAINES], axis=0)


SECOND_NIVEAU = {
    "LightGBM": A.ajuster_predire,
    "XGBoost": moyenne_graines(lambda s: XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, min_child_weight=300,
                                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=10, n_jobs=2,
                                                      random_state=s)),
    "CatBoost": moyenne_graines(lambda s: CatBoostRegressor(iterations=200, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                                                            random_seed=s, thread_count=2, verbose=0,
                                                            allow_writing_files=False)),
    "HistGradientBoosting": moyenne_graines(lambda s: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=10,
        early_stopping=False, random_state=s)),
    "régression linéaire": lambda X, c, Xn: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                          StandardScaler(), LinearRegression()).fit(X, c).predict(Xn),
}
R = {"base": "mélange avec l'ancienne version de TabPFN", "rmse_base": rmse(y, p), "second_niveau": {}}
for nom, ajuster in SECOND_NIVEAU.items():
    c = p.copy()
    for k in range(1, 6):
        X = np.vstack([APP[(k, m)] for m in range(1, 6) if m != k])
        cible = np.concatenate([r[plis == m] for m in range(1, 6) if m != k])
        c[plis == k] = p[plis == k] + ajuster(X, cible, EVAL[k])
    R["second_niveau"][nom] = {"rmse": rmse(y, c), **bootstrap_ecart(y, c, p, tirages=1000)}
    print(nom, round(rmse(y, c) - rmse(y, p), 3), flush=True)


# 2. LightGBM avec les totaux des candidates en entrée.
def totaux_candidates(requetes, reserve):
    """Pour chaque filtre : total moyen des candidates ramené au poids de la requête, log(1 + n),
    écarts moyens de poids et d'année, dispersion. Aucune prédiction n'intervient."""
    q = A.annees_naissance(requetes)[A.CLE + ["ligne", "Equipment", "Federation", "BodyweightKg", "MeetYear"]]
    rv = A.annees_naissance(reserve)[A.CLE + ["ligne", "Equipment", "Federation", "BodyweightKg", "MeetYear", "y"]]
    m = q.merge(rv, on=A.CLE, suffixes=("", "_c"))
    m = m[m.ligne != m.ligne_c].drop_duplicates(["ligne", "ligne_c"])
    m["ty"] = m.y * (m.BodyweightKg / m.BodyweightKg_c) ** (2 / 3)
    m["dbw"] = abs(m.BodyweightKg - m.BodyweightKg_c) / m.BodyweightKg
    m["dan"] = abs(m.MeetYear - m.MeetYear_c)
    position = pd.Series(np.arange(len(requetes)), index=requetes.ligne.to_numpy())
    colonnes = {}
    for nom, filtre in A.FILTRES.items():
        g = m[np.asarray(filtre(m))].groupby("ligne").agg(ty=("ty", "mean"), n=("ty", "size"), dbw=("dbw", "mean"),
                                                            dan=("dan", "mean"), sd=("ty", "std"))
        ix = position[g.index].to_numpy()
        for col in ("ty", "n", "dbw", "dan", "sd"):
            v = np.full(len(requetes), 0.0 if col == "n" else np.nan)
            v[ix] = np.log1p(g[col].to_numpy()) if col == "n" else g[col].to_numpy()
            colonnes[f"app_{nom}_{col}"] = v
    return pd.DataFrame(colonnes)


config = copy.deepcopy(CFG["modeles"]["lightgbm"])
params = {**config["params"], "n_estimators": 8000}
reference = lire_predictions("lightgbm")["hors_pli"]
X_dev = train.drop(columns=["TotalKg"]).iloc[d.dev].reset_index(drop=True)
oof = np.empty(len(y))
for k in range(1, 6):
    blocs = [(np.flatnonzero(plis == m), totaux_candidates(t[plis == m], t[(plis != m) & (plis != k)]))
             for m in range(1, 6) if m != k]
    la = np.concatenate([b[0] for b in blocs])
    XA, etat = L.make_features(X_dev.iloc[la], config)
    XA = pd.concat([XA.reset_index(drop=True), pd.concat([b[1] for b in blocs], ignore_index=True)], axis=1)
    lq = np.flatnonzero(plis == k)
    XQ, _ = L.make_features(X_dev.iloc[lq], config, etat)
    XQ = pd.concat([XQ.reset_index(drop=True), totaux_candidates(t[plis == k], t[plis != k])], axis=1)
    val = np.random.default_rng(k).random(len(la)) < 0.1
    m = lgb.LGBMRegressor(**params).fit(XA[~val], y[la][~val], eval_set=[(XA[val], y[la][val])],
                                        callbacks=[lgb.early_stopping(180, verbose=False)])
    oof[lq] = np.maximum(m.predict(XQ, num_iteration=m.best_iteration_), 0.0)
    print("LightGBM + candidates, pli", k, "fait", flush=True)
R["lightgbm_avec_candidates_en_entree"] = {"rmse": rmse(y, oof), "rmse_lightgbm": rmse(y, reference),
                                           **bootstrap_ecart(y, oof, reference, tirages=1000)}
ecrire("modeles_second_niveau", R)
