"""Sections 5 et 6 du rapport : erreurs communes des modèles et limites d'un mélange.

Référence : le premier mélange de six modèles (poids figés de config/modeles.json, « melange_v5 »),
appliqué aux prédictions hors pli corrigées des profils identiques. Aucun entraînement.
1. corrélations des erreurs, désaccord entre modèles, concentration de l'erreur, erreurs de même sens ;
2. exemple de trois athlètes aux variables presque identiques ; calibration de chaque modèle ;
3. décomposition de Krogh-Vedelsby et condition de Bates-Granger (TabPFN avec chaque autre modèle).

Exécution : python analyses/erreurs_et_melange.py      (environ une minute)
Sortie : resultats/analyses/erreurs_et_melange.json
"""
import numpy as np

from _commun import CFG, corriger_profils, donnees, ecrire, lire_predictions, profils_hors_pli, rmse

train, test, d, dev, hold = donnees()
y, plis = dev.TotalKg.to_numpy(float), d.plis
noms = CFG["melange_v5"]["modeles"]
w = np.asarray(CFG["melange_v5"]["poids"], float)
brutes = np.column_stack([lire_predictions(n)["hors_pli"] for n in noms])
m, n = profils_hors_pli(dev, y, plis)
C = corriger_profils(brutes, m, n, CFG["solution_finale"]["alpha_profils_identiques"])
melange = C @ w
e = y - melange                 # > 0 : le mélange sous-estime le total
E = y[:, None] - C              # erreurs des six modèles
R = {"rmse_melange": rmse(y, melange)}

# 1. Les modèles se trompent-ils sur les mêmes lignes, dans le même sens ?
paires = np.triu_indices(len(noms), 1)
corr = np.corrcoef(E.T)
R["correlations_des_erreurs"] = {"min": float(corr[paires].min()), "max": float(corr[paires].max())}
R["ecart_type_moyen_entre_modeles_kg"] = float(C.std(axis=1).mean())
R["erreur_absolue_moyenne_kg"] = float(np.abs(e).mean())
ordre = np.argsort(-e ** 2)
R["part_somme_carres_des_5_pourcent_pires_lignes"] = float(np.sum(e[ordre[:int(0.05 * len(e))]] ** 2) / np.sum(e ** 2))
signe = np.sign(E)
meme_sens = np.all(signe == signe[:, [0]], axis=1)
grande = np.abs(e) > 2 * rmse(y, melange)
R["erreurs_superieures_a_2_rmse"] = {"part_des_lignes": float(grande.mean()),
                                     "part_ou_les_6_modeles_se_trompent_dans_le_meme_sens": float(meme_sens[grande].mean()),
                                     "meme_part_sur_toutes_les_lignes": float(meme_sens.mean()),
                                     # moyenne pondérée de nombres de même signe : au moins la plus petite valeur absolue
                                     "verification_borne": bool(np.all(np.abs(e[grande]) >= np.abs(E[grande]).min(axis=1) - 1e-9))}

# 2. Trois hommes aux variables presque identiques (fédération autrichienne OEVK, Raw, 120 kg, 2019).
lignes = dev[(dev.Federation == "OEVK") & (dev.Equipment == "Raw") & (dev.WeightClassKg.astype(str) == "120")
             & (dev.MeetYear == 2019) & dev.Age.between(27, 31) & dev.BodyweightKg.between(107, 113)].index
R["exemple_trois_athletes"] = [{"age": float(dev.Age[i]), "poids": float(dev.BodyweightKg[i]), "total": float(y[i]),
                                "predictions_des_6_modeles": [round(float(v), 1) for v in brutes[i]]} for i in lignes]
# Calibration : pente de la régression du total réel sur la prédiction (1 = bien calibré).
R["pente_total_sur_prediction"] = {nom: float(np.cov(brutes[:, j], y)[0, 1] / brutes[:, j].var())
                                   for j, nom in enumerate(noms)}

# 3. Krogh-Vedelsby : erreur du mélange = moyenne pondérée des erreurs - dispersion autour du mélange.
moyenne_ponderee = float(np.sum(w * np.mean((C - y[:, None]) ** 2, axis=0)))
ambiguite = float(np.mean(np.sum(w * (C - melange[:, None]) ** 2, axis=1)))
R["krogh_vedelsby"] = {"moyenne_ponderee_des_erreurs_quadratiques": moyenne_ponderee, "dispersion": ambiguite,
                       "erreur_quadratique_du_melange": float(np.mean(e ** 2)),
                       "verification": moyenne_ponderee - ambiguite}
# Bates-Granger : un mélange de deux modèles ne bat le meilleur que si rho < sigma_min / sigma_max,
# avec sigma les RMSE et rho = E[e_i e_j] / (sigma_i sigma_j) (moments non centrés, comme les RMSE).
j = noms.index("tabpfn_ancien")
s = np.sqrt(np.mean(E ** 2, axis=0))
R["bates_granger_avec_tabpfn"] = {nom: {"correlation": float(np.mean(E[:, j] * E[:, k]) / (s[j] * s[k])),
                                        "rapport_des_rmse": float(min(s[j], s[k]) / max(s[j], s[k]))}
                                  for k, nom in enumerate(noms) if k != j}
ecrire("erreurs_et_melange", R)
