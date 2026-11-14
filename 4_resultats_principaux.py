"""Étape 4 : recalculer les tableaux principaux du rapport à partir de predictions/.

- Tableau 1 : prédictions de référence (moyennes par groupe) ;
- Tableau 2 : erreur de chaque modèle, en validation croisée et sur le test interne ;
- Section 6 : mélanges évalués en validation imbriquée ;
- Tableau 4 : gain de la correction par appariement (validation imbriquée, intervalle à 95 %) ;
- Tableau 5 : TabPFN seul, TabPFN corrigé et solution finale (validation croisée et test interne).

Sorties : resultats/resultats_principaux.md (tableaux lisibles) et resultats/resultats_principaux.json.
Exécution : python 4_resultats_principaux.py      (environ 20 minutes sur 2 cœurs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))

from powerlifting.appariement import corriger, evaluer_imbrique  # noqa: E402
from powerlifting.data import RESULTATS, charger, configuration, decoupage, lire_predictions  # noqa: E402
from powerlifting.melange import (ajuster_poids, appliquer_poids, cle_profil, corriger_profils, groupe,  # noqa: E402
                                  melange_imbrique, moyennes_profils, profils_hors_pli)
from powerlifting.mesures import bootstrap_ecart, rmse  # noqa: E402

# Scores du classement public Kaggle (environ 13 500 lignes), relevés sur la page des soumissions :
# ils ne peuvent pas être recalculés, les cibles du test sont cachées.
PUBLIC = {"tabpfn_seul.csv": 76.42075, "tabpfn_appariement.csv": 75.86197, "solution_finale.csv": 75.65106,
          "solution_finale_reservoir_105000.csv": 75.65040}

CFG = configuration()
train, test = charger()
d = decoupage(len(train))
dev = train.iloc[d.dev].reset_index(drop=True)
hold = train.iloc[d.test_interne].reset_index(drop=True)
y, y_hold = dev.TotalKg.to_numpy(float), hold.TotalKg.to_numpy(float)
plis = d.plis
alpha = CFG["solution_finale"]["alpha_profils_identiques"]
R = {}


def par_pli(p, reference):
    """Baisse de RMSE de p par rapport à reference, dans chaque pli (positive = p meilleure)."""
    return [round(rmse(y[plis == k], reference[plis == k]) - rmse(y[plis == k], p[plis == k]), 3) for k in range(1, 6)]


# ------------------------------------------------------------- Tableau 1 : références
def moyenne_par_groupe(colonnes):
    p = np.empty(len(y))
    for k in range(1, 6):
        a, t = plis != k, plis == k
        globale = dev.TotalKg[a].mean()
        if not colonnes:
            p[t] = globale
            continue
        moyennes = dev[a].groupby(colonnes).TotalKg.mean()
        p[t] = moyennes.reindex(pd.MultiIndex.from_frame(dev.loc[t, colonnes])).fillna(globale).to_numpy() \
            if len(colonnes) > 1 else dev.loc[t, colonnes[0]].map(moyennes).fillna(globale).to_numpy()
    return p


variance = float(np.mean((y - y.mean()) ** 2))
R["tableau_1"] = {}
for nom, colonnes in [("Moyenne générale", []), ("Moyenne par sexe", ["Sex"]),
                      ("Moyenne par sexe et catégorie de poids", ["Sex", "WeightClassKg"]),
                      ("Moyenne par sexe, équipement et catégorie de poids", ["Sex", "Equipment", "WeightClassKg"])]:
    e = rmse(y, moyenne_par_groupe(colonnes))
    R["tableau_1"][nom] = {"rmse": e, "r2": 1 - e ** 2 / variance}
print("Tableau 1 fait", flush=True)

# ------------------------------------------------------------- Tableau 2 : modèles individuels
INDIVIDUELS = {"Ridge (p = 1)": "ridge", "CatBoost": "catboost", "LightGBM, configuration de départ": "lightgbm",
               "LightGBM, meilleure configuration": "lightgbm_meilleure",
               "LightGBM, configuration de départ, 5 graines": "lightgbm_5graines",
               "HistGradientBoosting, 5 graines": "hgbr_5graines", "TabM": "tabm",
               "TabPFN (8 passages, cible ramenée)": "tabpfn"}
P = {n: lire_predictions(n) for n in set(INDIVIDUELS.values()) | set(CFG["solution_finale"]["modeles"])
     | set(CFG["melange_v5"]["modeles"])}
R["tableau_2"] = {libelle: {"validation_croisee": rmse(y, P[n]["hors_pli"]),
                            "test_interne": rmse(y_hold, P[n]["test_interne"])} for libelle, n in INDIVIDUELS.items()}
R["tabpfn_4_passages_cible_kg_validation_croisee"] = rmse(y, P["tabpfn_ancien"]["hors_pli"])

# ------------------------------------------------------------- Section 6 : mélanges
m_oof, n_oof = profils_hors_pli(dev, y, plis)
m_hold, n_hold = moyennes_profils(dev, y, hold)
cles = cle_profil(dev)


def colonnes(noms, cle="hors_pli"):
    return np.column_stack([P[n][cle] for n in noms])


V5 = CFG["melange_v5"]["modeles"]
E = CFG["solution_finale"]["modeles"]
D = [("tabpfn_ancien" if n == "tabpfn" else n) for n in E]          # même mélange, ancienne version de TabPFN
O = {nom: corriger_profils(colonnes(noms), m_oof, n_oof, alpha) for nom, noms in (("v5", V5), ("D", D), ("E", E))}
imb = {"v5": melange_imbrique(y, O["v5"], cles, plis, rarete=False), "E": melange_imbrique(y, O["E"], cles, plis)}
g_dev = groupe(cles, cles)
w = {nom: ajuster_poids(y, O[nom], g_dev) for nom in ("D", "E")}
base = {nom: appliquer_poids(O[nom], w[nom], g_dev) for nom in ("D", "E")}
tabpfn = P["tabpfn"]["hors_pli"]
R["melanges"] = {
    "v5, six modèles, poids réajustés, validation imbriquée": rmse(y, imb["v5"]),
    "solution finale, huit modèles, poids rare/fréquent, validation imbriquée": rmse(y, imb["E"]),
    "poids de TabPFN : profils rares / fréquents": [float(w["E"][0][E.index("tabpfn")]), float(w["E"][1][E.index("tabpfn")])],
    "gain du mélange sur TabPFN seul (validation imbriquée)": bootstrap_ecart(y, imb["E"], tabpfn),
    "gain du mélange sur TabPFN seul, par pli": par_pli(imb["E"], tabpfn),
}
print("Mélanges faits", flush=True)

# ------------------------------------------------------------- Tableau 4 : correction par appariement
tabpfn_profils = corriger_profils(tabpfn[:, None], m_oof, n_oof, alpha)[:, 0]
bases = {"TabPFN seul": tabpfn_profils, "Mélange avec l'ancienne version de TabPFN": base["D"],
         "Mélange retenu": base["E"]}
corriges = {}
R["tableau_4"] = {}
for nom, b in bases.items():
    corriges[nom] = evaluer_imbrique(dev, y, b, plis)
    R["tableau_4"][nom] = {"avant": rmse(y, b), "apres": rmse(y, corriges[nom]),
                           "gain": bootstrap_ecart(y, corriges[nom], b), "gain_par_pli": par_pli(corriges[nom], b)}
    print(nom, "corrigé", flush=True)

# ------------------------------------------------------------- Tableau 5 : test interne
H = corriger_profils(colonnes(E, "test_interne"), m_hold, n_hold, alpha)
g_hold = groupe(cles, cle_profil(hold))
base_hold = np.maximum(appliquer_poids(H, w["E"], g_hold), 0.0)
final_hold = corriger(dev, y, base["E"], plis, hold, base_hold)
pfn_hold = P["tabpfn"]["test_interne"]
pfn_profils_hold = corriger_profils(pfn_hold[:, None], m_hold, n_hold, alpha)[:, 0]
pfn_corrige_hold = corriger(dev, y, tabpfn_profils, plis, hold, pfn_profils_hold)
R["tableau_5"] = {
    "TabPFN seul": {"validation_croisee": rmse(y, tabpfn), "test_interne": rmse(y_hold, pfn_hold),
                    "public": PUBLIC["tabpfn_seul.csv"]},
    "TabPFN + correction par appariement": {"validation_croisee": R["tableau_4"]["TabPFN seul"]["apres"],
                                            "test_interne": rmse(y_hold, pfn_corrige_hold),
                                            "public": PUBLIC["tabpfn_appariement.csv"]},
    "Mélange + correction par appariement (solution finale)": {
        "validation_croisee": R["tableau_4"]["Mélange retenu"]["apres"], "test_interne": rmse(y_hold, final_hold),
        "public": PUBLIC["solution_finale.csv"]},
}
R["test_interne"] = {"mélange avant correction": rmse(y_hold, base_hold),
                     "gain du mélange sur TabPFN seul": bootstrap_ecart(y_hold, base_hold, pfn_hold),
                     "gain de la correction": bootstrap_ecart(y_hold, final_hold, base_hold),
                     "solution finale moins TabPFN seul": bootstrap_ecart(y_hold, final_hold, pfn_hold)}
R["tableau_1"]["Notre solution finale"] = {"rmse": R["tableau_4"]["Mélange retenu"]["apres"],
                                           "r2": 1 - R["tableau_4"]["Mélange retenu"]["apres"] ** 2 / variance}
R["public"] = PUBLIC


# ------------------------------------------------------------- écriture
def f(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def ic(g):
    return f"{f(g['ecart'])} [{f(g['ic95'][0])} ; {f(g['ic95'][1])}]"


lignes = ["# Résultats principaux (recalculés par 4_resultats_principaux.py)", "",
          "RMSE en kg. Validation croisée : 5 plis sur les 84 000 lignes de développement. Test interne : "
          "apprentissage sur les 84 000 lignes, mesure unique sur les 21 000 lignes mises de côté.", "",
          "## Tableau 1 : prédictions de référence", "", "| Prédiction | RMSE | R² |", "|---|---:|---:|"]
lignes += [f"| {k} | {f(v['rmse'], 1)} | {f(v['r2'])} |" for k, v in R["tableau_1"].items()]
lignes += ["", "## Tableau 2 : modèles individuels", "", "| Modèle | Validation croisée | Test interne |", "|---|---:|---:|"]
lignes += [f"| {k} | {f(v['validation_croisee'])} | {f(v['test_interne'])} |" for k, v in R["tableau_2"].items()]
lignes += ["", f"TabPFN avec 4 passages et la cible en kg (ancienne version) : "
           f"{f(R['tabpfn_4_passages_cible_kg_validation_croisee'])} en validation croisée.", "",
           "## Section 6 : mélanges (validation imbriquée)", ""]
M = R["melanges"]
lignes += [f"- Premier mélange de six modèles : {f(M['v5, six modèles, poids réajustés, validation imbriquée'])}",
           f"- Mélange retenu (huit modèles) : {f(M['solution finale, huit modèles, poids rare/fréquent, validation imbriquée'])}",
           f"- Poids de TabPFN : {f(100 * M['poids de TabPFN : profils rares / fréquents'][0], 0)} % (profils rares), "
           f"{f(100 * M['poids de TabPFN : profils rares / fréquents'][1], 0)} % (fréquents)",
           f"- Écart avec TabPFN seul : {ic(M['gain du mélange sur TabPFN seul (validation imbriquée)'])} ; "
           f"baisse par pli : {', '.join(f(v) for v in M['gain du mélange sur TabPFN seul, par pli'])}",
           "", "## Tableau 4 : correction par appariement (validation imbriquée)", "",
           "| Prédiction corrigée | Avant | Après | Gain [IC 95 %] | Baisse par pli |", "|---|---:|---:|---:|---|"]
lignes += [f"| {k} | {f(v['avant'])} | {f(v['apres'])} | {ic(v['gain'])} | {', '.join(f(x) for x in v['gain_par_pli'])} |"
           for k, v in R["tableau_4"].items()]
lignes += ["", "## Tableau 5 : soumissions", "", "| Solution | Validation croisée | Test interne | Public Kaggle |",
           "|---|---:|---:|---:|"]
lignes += [f"| {k} | {f(v['validation_croisee'])} | {f(v['test_interne'])} | {f(v['public'])} |"
           for k, v in R["tableau_5"].items()]
T = R["test_interne"]
lignes += ["", "Test interne :",
           f"- mélange avant correction : {f(T['mélange avant correction'])} ; "
           f"écart avec TabPFN seul : {ic(T['gain du mélange sur TabPFN seul'])}",
           f"- gain de la correction : {ic(T['gain de la correction'])}",
           f"- solution finale moins TabPFN seul : {ic(T['solution finale moins TabPFN seul'])}",
           "", "Public : scores relevés sur Kaggle (non recalculables). Variante au réservoir de 105 000 lignes : "
           f"{f(PUBLIC['solution_finale_reservoir_105000.csv'], 5)} contre {f(PUBLIC['solution_finale.csv'], 5)}."]
RESULTATS.mkdir(parents=True, exist_ok=True)
(RESULTATS / "resultats_principaux.md").write_text("\n".join(lignes) + "\n")
(RESULTATS / "resultats_principaux.json").write_text(json.dumps(R, indent=1, ensure_ascii=False) + "\n")
print("\n".join(lignes))
