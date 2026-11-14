"""Étape 3 : solution finale = mélange de 8 modèles + correction par appariement.

Lit predictions/<modèle>.npz (étapes 1 et 2, ou fichiers fournis) et écrit les deux soumissions
retenues pour le classement final :
- resultats/soumissions/solution_finale.csv : réservoir de candidates = 84 000 lignes de développement ;
- resultats/soumissions/solution_finale_reservoir_105000.csv : même solution, réservoir = 105 000 lignes.
Chacune est comparée au fichier déposé sur Kaggle (dossier soumissions/).

Exécution : python 3_solution_finale.py      (environ 2 minutes)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))

from powerlifting.appariement import corriger  # noqa: E402
from powerlifting.data import RESULTATS, charger, configuration, decoupage, lire_predictions  # noqa: E402
from powerlifting.melange import (ajuster_poids, appliquer_poids, cle_profil, corriger_profils, groupe,  # noqa: E402
                                  moyennes_profils, profils_hors_pli)
from powerlifting.mesures import rmse  # noqa: E402


def colonnes(noms, cle):
    return np.column_stack([lire_predictions(n)[cle] for n in noms])


def main():
    cfg = configuration()["solution_finale"]
    train, test = charger()
    d = decoupage(len(train))
    dev = train.iloc[d.dev].reset_index(drop=True)
    y = dev.TotalKg.to_numpy(float)
    alpha = cfg["alpha_profils_identiques"]

    # 1. Correction des profils identiques (hors pli : chaque pli n'utilise que les 4 autres).
    m, n = profils_hors_pli(dev, y, d.plis)
    O = corriger_profils(colonnes(cfg["modeles"], "hors_pli"), m, n, alpha)
    m, n = moyennes_profils(train, train.TotalKg.to_numpy(float), test)
    T = corriger_profils(colonnes(cfg["modeles"], "test"), m, n, alpha)

    # 2. Poids rare / fréquent ajustés sur les 84 000 prédictions hors pli.
    #    Groupes du test comptés sur les 105 000 lignes (seuil 20 x 105 000 / 67 200).
    g_dev = groupe(cle_profil(dev), cle_profil(dev), **cfg["rarete"])
    g_test = groupe(cle_profil(train), cle_profil(test), **cfg["rarete"])
    w = ajuster_poids(y, O, g_dev)
    base, base_test = appliquer_poids(O, w, g_dev), np.maximum(appliquer_poids(T, w, g_test), 0.0)
    print("Poids du mélange (profils rares | fréquents) :")
    for j, nom in enumerate(cfg["modeles"]):
        print(f"  {nom:24s} {w[0][j]:.3f} | {w[1][j]:.3f}")
    print(f"RMSE hors pli du mélange (poids ajustés sur ces mêmes lignes) : {rmse(y, base):.2f} kg")

    # 3. Correction par appariement : modèle de second niveau entraîné sur les 84 000 lignes.
    sortie = RESULTATS / "soumissions"
    sortie.mkdir(parents=True, exist_ok=True)
    variantes = {"solution_finale": None,                                       # réservoir : 84 000 lignes
                 "solution_finale_reservoir_105000": train.drop(index=d.dev)}   # + les 21 000 autres
    for nom, supplement in variantes.items():
        final = corriger(dev, y, base, d.plis, test, base_test, reserve_supplementaire=supplement)
        pd.DataFrame({"id": test.id, "TotalKg": final}).to_csv(sortie / f"{nom}.csv", index=False)
        depose = pd.read_csv(RACINE / "soumissions" / f"{nom}.csv")
        assert depose.id.equals(test.id)
        ecart = float(np.abs(depose.TotalKg.to_numpy() - final).max())
        print(f"{nom}.csv écrit ; écart maximal avec la soumission déposée : {ecart:.1e} kg")


if __name__ == "__main__":
    main()
