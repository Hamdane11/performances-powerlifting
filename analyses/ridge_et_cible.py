"""Section 4.1 du rapport : Ridge, forme de la cible, et même transformation pour LightGBM.

1. Ridge sur les 16 variables, cible en kg : pénalité alpha de 1 à 1 000 ;
2. Ridge (alpha = 100) avec la cible ramenée au poids, z = y / s, s = (poids/80)^p et poids s² ;
3. écart Single-ply - Raw chez les hommes, par tranche de poids de corps (train.csv) ;
4. LightGBM (réglages génériques, 16 variables) : cible en kg contre cible ramenée (p = 2/3).
Validation croisée : 5 plis sur les 84 000 lignes de développement.

Exécution : python analyses/ridge_et_cible.py      (quelques minutes)
Sortie : resultats/analyses/ridge_et_cible.json
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _commun import CAT, NUM, bootstrap_ecart, donnees, ecrire, rmse

train, test, d, dev, hold = donnees()
y, plis = dev.TotalKg.to_numpy(float), d.plis
X = dev[NUM + CAT].copy()
X.loc[X.Age < 5, "Age"] = np.nan
poids = dev.BodyweightKg.to_numpy(float)
R = {}


def ridge_hors_pli(alpha, exposant):
    s = (poids / 80.0) ** exposant
    p = np.empty_like(y)
    for k in range(1, 6):
        a = plis != k
        modele = make_pipeline(ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT)]), Ridge(alpha=alpha))
        modele.fit(X[a], y[a] / s[a], ridge__sample_weight=s[a] ** 2)
        p[~a] = modele.predict(X[~a]) * s[~a]
    return np.maximum(p, 0.0)


R["ridge_alpha_cible_kg"] = {str(a): rmse(y, ridge_hors_pli(a, 0.0)) for a in (1.0, 10.0, 100.0, 1000.0)}
R["ridge_alpha_100_exposant_p"] = {f"{p:.4f}": rmse(y, ridge_hors_pli(100.0, p)) for p in (0.0, 0.5, 2 / 3, 1.0)}
print("Ridge fait", flush=True)

# Écart Single-ply - Raw chez les hommes, par tranche de poids de corps.
h = train[(train.Sex == "M") & train.Equipment.isin(["Raw", "Single-ply"])]
moyennes = h.groupby([pd.cut(h.BodyweightKg, [55, 70, 90, 110, 150]), "Equipment"], observed=True).TotalKg.mean().unstack()
R["single_ply_moins_raw_hommes"] = {str(t): {"raw": float(r["Raw"]), "single_ply": float(r["Single-ply"]),
                                             "ecart_kg": float(r["Single-ply"] - r["Raw"]),
                                             "ecart_relatif": float(r["Single-ply"] / r["Raw"] - 1)}
                                    for t, r in moyennes.iterrows()}

# LightGBM générique : la cible ramenée au poids change-t-elle quelque chose ?
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=50, reg_lambda=10.0,
              cat_smooth=25.0, n_estimators=3000, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
              random_state=0, verbosity=-1, n_jobs=2)
Xl = X.copy()
for c in CAT:
    Xl[c] = Xl[c].astype("category")


def lightgbm_hors_pli(exposant):
    s = (poids / 80.0) ** exposant
    z, w = y / s, s ** 2
    p = np.empty_like(y)
    for k in range(1, 6):
        a, t = np.where(plis != k)[0], np.where(plis == k)[0]
        interne = np.random.default_rng(k).random(len(a)) < 0.1     # arrêt anticipé sur 10 % de l'apprentissage
        fi, vi = a[~interne], a[interne]
        m = lgb.LGBMRegressor(**PARAMS).fit(Xl.iloc[fi], z[fi], sample_weight=w[fi], eval_set=[(Xl.iloc[vi], z[vi])],
                                            eval_sample_weight=[w[vi]], callbacks=[lgb.early_stopping(100, verbose=False)])
        p[t] = m.predict(Xl.iloc[t]) * s[t]
    return np.maximum(p, 0.0)


kg, ramenee = lightgbm_hors_pli(0.0), lightgbm_hors_pli(2 / 3)
R["lightgbm_cible_ramenee_moins_kg"] = {"rmse_kg": rmse(y, kg), "rmse_ramenee": rmse(y, ramenee),
                                        **bootstrap_ecart(y, ramenee, kg, tirages=1000)}
ecrire("ridge_et_cible", R)
