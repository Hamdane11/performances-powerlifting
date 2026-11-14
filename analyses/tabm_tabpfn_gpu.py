"""Sections 4.3 et 4.4 du rapport : TabM et TabPFN, à partir des sorties du notebook GPU.

Les calculs GPU sont faits par analyses/analyses_gpu.ipynb (Colab) ; ses sorties brutes sont dans
resultats/gpu/ (une prédiction par fichier .npz). Ce script ne fait que les mesurer :
1. TabPFN sur le pli 1 : nombre de passages (1 à 16) et cible (kg ou ramenée au poids) ;
2. TabM : recherche de 16 configurations + configuration de départ (63 000 / 21 000 lignes) ;
3. TabM réentraîné (38 époques) en validation croisée et sur le test interne, comparé à la colonne
   du mélange (40 époques).

Exécution : python analyses/tabm_tabpfn_gpu.py
Sortie : resultats/analyses/tabm_tabpfn_gpu.json
"""
import numpy as np
import pandas as pd

from _commun import GPU, donnees, ecrire, lire_predictions, rmse

train, test, d, dev, hold = donnees()
y_all = train.TotalKg.to_numpy(float)
with np.load(GPU / "indices.npz") as a:
    assert np.array_equal(a["dev"], d.dev) and np.array_equal(a["hold"], d.test_interne)
pli1 = d.dev[d.plis == 1]
prediction = lambda nom: np.load(GPU / f"{nom}.npz")["prediction"].astype(float)  # noqa: E731
R = {}

# 1. TabPFN : passages et cible, pli 1 (contexte : les 67 200 lignes des plis 2 à 5).
R["tabpfn_pli1"] = {cible: {str(n): rmse(y_all[pli1], prediction(f"B_{cible}_{n}")) for n in passages}
                    for cible, passages in (("allom", (1, 2, 4, 8, 16)), ("kg", (4, 8)))}
a = R["tabpfn_pli1"]["allom"]
R["tabpfn_gains"] = {"de 1 à 4 passages": a["1"] - a["4"], "de 4 à 16 passages": a["4"] - a["16"],
                     "cible ramenée, 4 passages": R["tabpfn_pli1"]["kg"]["4"] - a["4"],
                     "cible ramenée, 8 passages": R["tabpfn_pli1"]["kg"]["8"] - a["8"]}

# 2. Recherche TabM (early stopping sur les 21 000 lignes de validation).
resume = pd.read_csv(GPU / "R_resume.csv")
R["tabm_recherche"] = {"configurations": int(len(resume)), "min": float(resume.rmse_validation.min()),
                       "max": float(resume.rmse_validation.max()),
                       "configuration_de_depart": float(resume.loc[resume.name == "tabm_ref", "rmse_validation"].iloc[0]),
                       "meilleure": resume.iloc[0]["name"]}

# 3. TabM réentraîné : validation croisée et test interne, contre la colonne TabM du mélange.
oof = np.empty(len(d.dev))
for k in range(1, 6):
    oof[d.plis == k] = prediction(f"T_oof_pli{k}")
tabm = lire_predictions("tabm")
R["tabm_reentraine"] = {"validation_croisee": rmse(y_all[d.dev], oof),
                        "validation_croisee_colonne_du_melange": rmse(y_all[d.dev], tabm["hors_pli"]),
                        "test_interne": rmse(y_all[d.test_interne], prediction("T_test_interne")),
                        "test_interne_colonne_du_melange": rmse(y_all[d.test_interne], tabm["test_interne"])}
ecrire("tabm_tabpfn_gpu", R)
