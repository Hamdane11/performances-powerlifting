"""Mélange des modèles (sections 6 et 8 du rapport).

1. Correction des profils identiques : chaque prédiction est rapprochée du total moyen des lignes
   d'apprentissage identiques sur les 16 variables, quand il en existe : ŷ <- 0,25 ŷ + 0,75 ȳ_profil.
2. Poids positifs de somme 1 (moindres carrés sous contrainte), différents pour les profils rares et
   fréquents. Le profil d'une ligne est le groupe sexe | équipement | catégorie | fédération ; il est
   « rare » s'il compte moins de 20 lignes pour 67 200 lignes d'apprentissage (seuil proportionnel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ensemble import exact_correction, fit_blend_weights, fit_lookup, lookup_diagnostics


def moyennes_profils(apprentissage, y_apprentissage, requetes):
    """Total moyen et nombre des lignes d'apprentissage identiques à chaque requête."""
    diag = lookup_diagnostics(fit_lookup(apprentissage.drop(columns=["TotalKg"], errors="ignore"), y_apprentissage),
                              requetes.drop(columns=["TotalKg"], errors="ignore"))
    return diag.raw_prediction.to_numpy(), diag.n_matches.to_numpy()


def corriger_profils(colonnes, moyennes, nombres, alpha):
    """colonnes : tableau (lignes, modèles). Les lignes sans profil identique ne changent pas."""
    return exact_correction(np.asarray(colonnes, float), np.nan_to_num(moyennes), nombres, alpha)


def profils_hors_pli(dev, y, plis):
    """Moyennes de profils pour les 84 000 lignes, chaque pli n'utilisant que les quatre autres."""
    moyennes, nombres = np.full(len(dev), np.nan), np.zeros(len(dev), int)
    for k in np.unique(plis):
        a = plis != k
        moyennes[~a], nombres[~a] = moyennes_profils(dev[a], y[a], dev[~a])
    return moyennes, nombres


def cle_profil(frame):
    return (frame.Sex + "|" + frame.Equipment + "|" + frame.WeightClassKg.astype(str) + "|" + frame.Federation).to_numpy()


def groupe(cles_apprentissage, cles_cible, seuil=20, lignes_reference=67_200):
    """0 = profil rare, 1 = profil fréquent ; tailles comptées sur les lignes d'apprentissage."""
    s = seuil * len(cles_apprentissage) / lignes_reference
    taille = pd.Series(cles_cible).map(pd.Series(cles_apprentissage).value_counts()).fillna(0).to_numpy()
    return (taille >= s).astype(int)


def ajuster_poids(y, colonnes, g):
    """Un vecteur de poids (positifs, de somme 1) par groupe."""
    return {b: np.asarray(fit_blend_weights(y[g == b], colonnes[g == b]).weights, float) for b in (0, 1)}


def appliquer_poids(colonnes, w, g):
    return np.where(g == 0, colonnes @ w[0], colonnes @ w[1])


def melange_imbrique(y, colonnes, g_cles, plis, rarete=True):
    """Erreur du mélange en validation imbriquée : les poids du pli k sont ajustés sans le pli k."""
    p = np.empty(len(y))
    for k in np.unique(plis):
        a = plis != k
        if rarete:
            g = groupe(g_cles[a], g_cles)
            p[~a] = appliquer_poids(colonnes[~a], ajuster_poids(y[a], colonnes[a], g[a]), g[~a])
        else:
            p[~a] = colonnes[~a] @ np.asarray(fit_blend_weights(y[a], colonnes[a]).weights, float)
    return p
