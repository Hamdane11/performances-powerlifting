"""Étape 1 : entraîner les modèles sur processeur et écrire leurs prédictions.

Pour chaque modèle, trois jeux de prédictions sont écrits dans predictions/<modèle>.npz :
- hors_pli     : 84 000 lignes de développement, chaque pli prédit par un modèle entraîné sur les 4 autres ;
- test_interne : 21 000 lignes mises de côté, prédites par un modèle entraîné sur les 84 000 lignes ;
- test         : 45 000 lignes de test.csv, prédites par un modèle entraîné sur les 105 000 lignes.

Si predictions/<modèle>.npz existe déjà (fichiers fournis), le script recalcule le modèle,
compare et n'écrase rien : il affiche l'écart maximal (0 attendu : ces modèles sont déterministes).

Exécution :
  python 1_modeles_cpu.py                       # tous les modèles (plusieurs heures sur 2 cœurs, CatBoost surtout)
  python 1_modeles_cpu.py ridge lightgbm        # seulement certains modèles
  python 1_modeles_cpu.py --ecraser catboost    # remplace le fichier fourni par le nouveau calcul
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from powerlifting.data import PREDICTIONS, charger, configuration, decoupage, ecrire_predictions  # noqa: E402
from powerlifting.features import RAW_CATEGORICAL, RAW_FEATURES, RAW_NUMERICAL  # noqa: E402
from powerlifting.mesures import rmse  # noqa: E402
from powerlifting.models import predict_model  # noqa: E402
from powerlifting.models.hgbr_cohortes import predire_hgbr, predire_lgbm54  # noqa: E402

CFG = configuration()


# ---------------------------------------------------------------- les modèles
# Chaque modèle est une fonction (lignes d'apprentissage, cibles, lignes à prédire) -> prédictions.

def ridge(contexte, y, requetes):
    """Ridge (alpha = 100) sur les 16 variables, cible divisée par s = poids/80 et poids s²."""
    p = CFG["modeles"]["ridge"]
    xa, xq = contexte[RAW_FEATURES].copy(), requetes[RAW_FEATURES].copy()
    for x in (xa, xq):
        x.loc[x.Age < 5, "Age"] = np.nan
    sa = (xa.BodyweightKg.to_numpy(float) / 80.0) ** p["exposant"]
    sq = (xq.BodyweightKg.to_numpy(float) / 80.0) ** p["exposant"]
    modele = make_pipeline(ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), RAW_NUMERICAL),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), RAW_CATEGORICAL)]), Ridge(alpha=p["alpha"]))
    modele.fit(xa, y / sa, ridge__sample_weight=sa ** 2)
    return np.maximum(modele.predict(xq) * sq, 0.0)


def lightgbm(nom, graine=None):
    config = copy.deepcopy(CFG["modeles"][nom])
    if graine is not None:
        config["params"]["random_state"] = graine
    return lambda c, y, q: predict_model("lightgbm", c, y, q, copy.deepcopy(config), CFG["variables"])


def catboost(c, y, q):
    return predict_model("catboost", c, y, q, copy.deepcopy(CFG["modeles"]["catboost"]), CFG["variables"])


def hgbr(graine):
    return lambda c, y, q: predire_hgbr(c[RAW_FEATURES], y, q[RAW_FEATURES], graine)


def lgbm54(c, y, q):
    return predire_lgbm54(c[RAW_FEATURES], y, q[RAW_FEATURES], CFG["modeles"]["lgbm54"]["graine"])


# Un modèle « 5 graines » est la moyenne de 5 entraînements qui ne diffèrent que par la graine.
GRAINES = CFG["graines_lightgbm_5"]
MODELES = {
    "ridge": [ridge],
    "lightgbm": [lightgbm("lightgbm")],
    "lightgbm_p067": [lightgbm("lightgbm_p067")],
    "lightgbm_5graines": [lightgbm("lightgbm", g) for g in GRAINES],
    "lightgbm_p067_5graines": [lightgbm("lightgbm_p067", g) for g in GRAINES],
    "lightgbm_meilleure": [lightgbm("lightgbm_meilleure")],
    "hgbr_5graines": [hgbr(g) for g in CFG["modeles"]["hgbr"]["graines"]],
    "lgbm54": [lgbm54],
    "catboost": [catboost],
}


def predire(fonctions, contexte, y, requetes):
    return np.mean([f(contexte, y, requetes) for f in fonctions], axis=0)


def calculer(nom, train, test, d):
    """Les trois jeux de prédictions d'un modèle."""
    X = train.drop(columns=["TotalKg"])
    y = train.TotalKg.to_numpy(float)
    fonctions = MODELES[nom]
    hors_pli = np.empty(len(d.dev))
    for k in range(1, 6):
        a, q = d.dev[d.plis != k], d.dev[d.plis == k]
        hors_pli[d.plis == k] = predire(fonctions, X.iloc[a], y[a], X.iloc[q])
        print(f"  {nom} : pli {k} fait", flush=True)
    test_interne = predire(fonctions, X.iloc[d.dev], y[d.dev], X.iloc[d.test_interne])
    test_final = predire(fonctions, X, y, test)
    return {"hors_pli": hors_pli, "test_interne": test_interne, "test": test_final}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modeles", nargs="*", help="modèles à calculer, parmi : " + ", ".join(MODELES))
    parser.add_argument("--ecraser", action="store_true", help="remplacer les fichiers existants")
    args = parser.parse_args()
    inconnus = set(args.modeles) - set(MODELES)
    if inconnus:
        parser.error(f"modèle inconnu : {', '.join(sorted(inconnus))}")
    train, test = charger()
    d = decoupage(len(train))
    y = train.TotalKg.to_numpy(float)
    for nom in args.modeles or list(MODELES):
        debut = time.time()
        print(f"{nom} : entraînement", flush=True)
        nouveau = calculer(nom, train, test, d)
        bilan = (f"{nom} : RMSE hors pli {rmse(y[d.dev], nouveau['hors_pli']):.2f} kg, "
                 f"test interne {rmse(y[d.test_interne], nouveau['test_interne']):.2f} kg "
                 f"({(time.time() - debut) / 60:.0f} min)")
        fichier = PREDICTIONS / f"{nom}.npz"
        if fichier.is_file() and not args.ecraser:
            with np.load(fichier) as ancien:
                ecart = max(float(np.abs(ancien[c] - nouveau[c]).max()) for c in nouveau if c in ancien.files)
            print(bilan, f"| écart maximal avec {fichier.name} fourni : {ecart:.1e} kg", flush=True)
        else:
            ecrire_predictions(nom, **nouveau)
            print(bilan, f"| écrit : predictions/{nom}.npz", flush=True)


if __name__ == "__main__":
    main()
