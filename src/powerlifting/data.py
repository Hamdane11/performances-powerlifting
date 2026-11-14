"""Lecture des données et découpage commun à tous les modèles.

Découpage (identique pour tous les modèles et toutes les analyses) :
- 21 000 des 105 000 lignes étiquetées forment le test interne, mesuré une seule fois à la fin ;
- les 84 000 autres (lignes de développement) sont réparties en 5 plis de 16 800 lignes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import create_partitions

RACINE = Path(__file__).resolve().parents[2]
DONNEES = RACINE / "donnees"
PREDICTIONS = RACINE / "predictions"
RESULTATS = RACINE / "resultats"

# Les 16 variables explicatives du challenge, dans l'ordre des fichiers CSV.
FEATURES = (
    "Age", "BodyweightKg", "MeetYear", "Sex", "Equipment", "AgeClass",
    "BirthYearClass", "WeightClassKg", "Tested", "Country", "State",
    "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned",
)
NUMERIC = ("Age", "BodyweightKg", "MeetYear")

# Découpage historique : développement 80 % (graine 42), 5 plis mélangés (graine 2026).
PROTOCOLE = {"scope": "development", "development_fraction": 0.8, "development_seed": 42,
             "fold_seed": 2026, "fold_count": 5}


def charger():
    """train (105 000 lignes, avec TotalKg) et test (45 000 lignes), dans l'ordre des fichiers."""
    return pd.read_csv(DONNEES / "train.csv"), pd.read_csv(DONNEES / "test.csv")


@dataclass(frozen=True)
class Decoupage:
    dev: np.ndarray            # positions (dans train.csv) des 84 000 lignes de développement
    plis: np.ndarray           # numéro de pli (1 à 5) de chaque ligne de développement
    test_interne: np.ndarray   # positions des 21 000 lignes du test interne


def decoupage(n_lignes: int = 105_000) -> Decoupage:
    p = create_partitions(n_lignes, PROTOCOLE)
    return Decoupage(dev=p.indices, plis=p.fold_ids, test_interne=p.historical_indices)


def configuration() -> dict:
    """Paramètres de tous les modèles et de la solution finale (config/modeles.json)."""
    return json.loads((RACINE / "config" / "modeles.json").read_text())


def lire_predictions(nom: str) -> dict:
    """predictions/<nom>.npz : hors_pli (84 000), test_interne (21 000), test (45 000) selon le modèle."""
    with np.load(PREDICTIONS / f"{nom}.npz", allow_pickle=False) as f:
        return {cle: f[cle].astype(float) for cle in f.files}


def ecrire_predictions(nom: str, dossier: Path | None = None, **tableaux) -> Path:
    dossier = Path(dossier or PREDICTIONS)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom}.npz"
    np.savez_compressed(chemin, **{k: np.asarray(v, float) for k, v in tableaux.items()})
    return chemin
