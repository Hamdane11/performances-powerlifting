"""Outils partagés par les scripts d'analyse (chargement, bases du mélange, figures)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from powerlifting.data import charger, configuration, decoupage, lire_predictions  # noqa: E402
from powerlifting.features import RAW_CATEGORICAL as CAT, RAW_NUMERICAL as NUM  # noqa: E402,F401
from powerlifting.melange import (ajuster_poids, appliquer_poids, cle_profil, corriger_profils, groupe,  # noqa: E402
                                  moyennes_profils, profils_hors_pli)
from powerlifting.mesures import bootstrap_ecart, rmse  # noqa: E402,F401

SORTIE = RACINE / "resultats" / "analyses"
GPU = RACINE / "resultats" / "gpu"
FIGURES = RACINE / "rapport" / "figures"
CFG = configuration()


def donnees():
    """train, test, découpage, lignes de développement (84 000) et du test interne (21 000)."""
    train, test = charger()
    d = decoupage(len(train))
    dev = train.iloc[d.dev].reset_index(drop=True)
    hold = train.iloc[d.test_interne].reset_index(drop=True)
    return train, test, d, dev, hold


def base_melange(dev, y, plis, modeles=None, hold=None):
    """Prédictions hors pli du mélange retenu (poids ajustés sur les 84 000 lignes), et sur le test
    interne si hold est donné. C'est la « base » que corrige l'appariement."""
    modeles = modeles or CFG["solution_finale"]["modeles"]
    alpha = CFG["solution_finale"]["alpha_profils_identiques"]
    P = [lire_predictions(n) for n in modeles]
    m, n = profils_hors_pli(dev, y, plis)
    O = corriger_profils(np.column_stack([p["hors_pli"] for p in P]), m, n, alpha)
    cles = cle_profil(dev)
    g = groupe(cles, cles)
    w = ajuster_poids(y, O, g)
    base = appliquer_poids(O, w, g)
    if hold is None:
        return base
    m, n = moyennes_profils(dev, y, hold)
    H = corriger_profils(np.column_stack([p["test_interne"] for p in P]), m, n, alpha)
    return base, np.maximum(appliquer_poids(H, w, groupe(cles, cle_profil(hold))), 0.0)


def ecrire(nom, resultat):
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / f"{nom}.json").write_text(json.dumps(resultat, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(resultat, indent=1, ensure_ascii=False, default=float))
