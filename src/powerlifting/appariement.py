"""Correction par appariement d'une prédiction de base.

Aucune variable n'identifie l'athlète. Pour chaque ligne à corriger, on cherche dans un réservoir
de lignes étiquetées des *candidates* : même sexe, pays, région et année de naissance compatible
(déduite de l'âge et de l'année de la compétition). Six filtres de plus en plus stricts
(équipement, poids, année, fédération) donnent des groupes de candidates. Pour chaque groupe :
- moyennes rétrécies de d = total de la candidate ramené au poids de la ligne - prédiction de la ligne ;
- log(1 + nombre de candidates), écarts moyens de poids et d'année, dispersion des d.
Seuls les totaux réels des candidates et la prédiction de la ligne elle-même sont utilisés.
Un petit LightGBM (moyenne de 5 graines) apprend l'erreur de la base à partir de ces 43 caractéristiques.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLE = ["Sex", "Country", "State", "BY"]
COLONNES = ["Sex", "Country", "State", "Equipment", "Federation", "BodyweightKg", "Age", "MeetYear"]
FILTRES = {
    "cohorte": lambda m: np.ones(len(m), bool),
    "equip_bw10": lambda m: (m.Equipment == m.Equipment_c) & (abs(m.BodyweightKg - m.BodyweightKg_c) <= 0.10 * m.BodyweightKg),
    "equip_bw5_an3": lambda m: (m.Equipment == m.Equipment_c) & (abs(m.BodyweightKg - m.BodyweightKg_c) <= 0.05 * m.BodyweightKg)
    & (abs(m.MeetYear - m.MeetYear_c) <= 3),
    "equip_bw3_fed": lambda m: (m.Equipment == m.Equipment_c) & (abs(m.BodyweightKg - m.BodyweightKg_c) <= 0.03 * m.BodyweightKg)
    & (m.Federation == m.Federation_c),
    "equip_bw2_an1": lambda m: (m.Equipment == m.Equipment_c) & (abs(m.BodyweightKg - m.BodyweightKg_c) <= 0.02 * m.BodyweightKg)
    & (abs(m.MeetYear - m.MeetYear_c) <= 1),
    "autre_equip_bw10": lambda m: (m.Equipment != m.Equipment_c) & (abs(m.BodyweightKg - m.BodyweightKg_c) <= 0.10 * m.BodyweightKg),
}
RETRECISSEMENTS = (0.5, 2.0, 8.0)
PARAMS = {"num_leaves": 15, "learning_rate": 0.03, "min_data_in_leaf": 300, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10, "objective": "regression",
          "verbosity": -1, "num_threads": 2}
TOURS, GRAINES = 200, (1, 2, 3, 4, 5)


def noms_caracteristiques():
    noms = []
    for filtre in FILTRES:
        noms += [f"{filtre}:moy_k{k:g}" for k in RETRECISSEMENTS]
        noms += [f"{filtre}:log_n", f"{filtre}:ecart_poids", f"{filtre}:ecart_annee", f"{filtre}:dispersion"]
    return noms + ["prediction"]


def annees_naissance(frame):
    """Une ou deux années de naissance possibles par ligne.

    Un âge en ,5 vient d'une année de naissance connue (OpenPowerlifting) : une seule année.
    Sinon l'anniversaire peut tomber avant ou après la compétition : deux années possibles.
    """
    age, annee = frame.Age.to_numpy(float), frame.MeetYear.to_numpy(float)
    demi = np.isclose(age % 1, 0.5)
    parties = []
    for valeur in (np.where(demi, annee - np.floor(age) - 1, annee - age - 1),
                   np.where(demi, annee - np.floor(age) - 1, annee - age)):
        t = frame.copy()
        t["BY"] = valeur
        parties.append(t)
    return pd.concat(parties).dropna(subset=["BY"]).drop_duplicates(["ligne", "BY"])


def caracteristiques(requetes, reserve):
    """requetes : COLONNES + ligne + p ; reserve : COLONNES + ligne + y. Identifiants « ligne » distincts."""
    q = annees_naissance(requetes)[CLE + ["ligne", "Equipment", "Federation", "BodyweightKg", "MeetYear", "p"]]
    r = annees_naissance(reserve)[CLE + ["ligne", "Equipment", "Federation", "BodyweightKg", "MeetYear", "y"]]
    m = q.merge(r, on=CLE, suffixes=("", "_c"))
    m = m[m.ligne != m.ligne_c].drop_duplicates(["ligne", "ligne_c"])
    m["dy"] = m.y * (m.BodyweightKg / m.BodyweightKg_c) ** (2 / 3) - m.p
    m["dbw"] = abs(m.BodyweightKg - m.BodyweightKg_c) / m.BodyweightKg
    m["dan"] = abs(m.MeetYear - m.MeetYear_c)
    position = pd.Series(np.arange(len(requetes)), index=requetes.ligne.to_numpy())
    colonnes = []
    for filtre in FILTRES.values():
        g = m[np.asarray(filtre(m))].groupby("ligne").agg(md=("dy", "mean"), n=("dy", "size"), dbw=("dbw", "mean"),
                                                            dan=("dan", "mean"), sd=("dy", "std"))
        ix = position[g.index].to_numpy()
        for k in RETRECISSEMENTS:
            v = np.zeros(len(requetes)); v[ix] = g.md.to_numpy() * g.n.to_numpy() / (g.n.to_numpy() + k); colonnes.append(v)
        v = np.zeros(len(requetes)); v[ix] = np.log1p(g.n.to_numpy()); colonnes.append(v)
        for nom in ("dbw", "dan", "sd"):
            v = np.full(len(requetes), np.nan); v[ix] = g[nom].to_numpy(); colonnes.append(v)
    colonnes.append(requetes.p.to_numpy())
    return np.column_stack(colonnes)


def ajuster_predire(X, cible, X_nouveau):
    import lightgbm as lgb

    return np.mean([lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, cible), TOURS).predict(X_nouveau) for s in GRAINES], axis=0)


def tableau(frame, ligne, **valeurs):
    t = frame[COLONNES].copy().reset_index(drop=True)
    t["ligne"] = ligne
    for nom, v in valeurs.items():
        t[nom] = v
    return t


def evaluer_imbrique(dev, y, p, folds):
    """Correction évaluée en validation strictement imbriquée : pour le pli k, le modèle est entraîné
    sur les autres plis avec des caractéristiques calculées sans aucune ligne du pli k."""
    d = tableau(dev, np.arange(len(dev)), y=y, p=p)
    r = y - p
    corrige = p.copy()
    for k in np.unique(folds):
        autres = [m for m in np.unique(folds) if m != k]
        X = np.vstack([caracteristiques(d[folds == m], d[(folds != m) & (folds != k)]) for m in autres])
        cible = np.concatenate([r[folds == m] for m in autres])
        corrige[folds == k] = p[folds == k] + ajuster_predire(X, cible, caracteristiques(d[folds == k], d[folds != k]))
    return corrige


def corriger(dev, y, p, folds, nouvelles, p_nouvelles, reserve_supplementaire=None):
    """Entraînement sur toutes les lignes de développement (réservoir : les autres plis de chaque ligne),
    puis correction de nouvelles lignes avec pour réservoir toutes les lignes de développement.

    reserve_supplementaire : autres lignes étiquetées (COLONNES + TotalKg) ajoutées au réservoir des
    nouvelles lignes seulement, par exemple les lignes mises de côté pour le test interne."""
    d = tableau(dev, np.arange(len(dev)), y=y, p=p)
    X = np.vstack([caracteristiques(d[folds == m], d[folds != m]) for m in np.unique(folds)])
    cible = np.concatenate([(y - p)[folds == m] for m in np.unique(folds)])
    reserve = d
    if reserve_supplementaire is not None:
        plus = tableau(reserve_supplementaire, np.arange(len(reserve_supplementaire)) + 20_000_000,
                       y=reserve_supplementaire.TotalKg.to_numpy(float))
        reserve = pd.concat([d.drop(columns=["p"]), plus], ignore_index=True)
    t = tableau(nouvelles, np.arange(len(nouvelles)) + 10_000_000, p=p_nouvelles)
    return np.maximum(p_nouvelles + ajuster_predire(X, cible, caracteristiques(t, reserve)), 0.0)
