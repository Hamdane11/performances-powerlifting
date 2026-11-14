"""Tableau 3 du rapport : où TabPFN gagne-t-il ? Écart de RMSE avec TabPFN selon la taille du groupe.

Groupe d'une ligne : sexe | équipement | catégorie de poids | fédération, compté sur les 84 000 lignes.
Prédictions hors pli brutes (sans correction des profils identiques) : on juge les modèles eux-mêmes.
TabPFN : 4 passages, cible en kg (version du premier mélange). Aucun entraînement.

Exécution : python analyses/profils_rares.py
Sortie : resultats/analyses/profils_rares.json
"""
from _commun import bootstrap_ecart, cle_profil, donnees, ecrire, lire_predictions, rmse

import pandas as pd

train, test, d, dev, hold = donnees()
y = dev.TotalKg.to_numpy(float)
cles = pd.Series(cle_profil(dev))
taille = cles.map(cles.value_counts()).to_numpy()
groupes = [("1-4", taille < 5), ("5-19", (taille >= 5) & (taille < 20)), ("20-99", (taille >= 20) & (taille < 100)),
           ("100-499", (taille >= 100) & (taille < 500)), ("500+", taille >= 500)]
tabpfn = lire_predictions("tabpfn_ancien")["hors_pli"]
R = {}
for nom in ("lightgbm", "catboost", "tabm"):
    p = lire_predictions(nom)["hors_pli"]
    R[nom] = {g: {"lignes": int(s.sum()), "rmse_modele": rmse(y[s], p[s]), "rmse_tabpfn": rmse(y[s], tabpfn[s]),
                  **bootstrap_ecart(y[s], p[s], tabpfn[s], tirages=1000)} for g, s in groupes}
ecrire("profils_rares", R)
