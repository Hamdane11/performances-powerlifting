"""Section 4.2 et figure 2 du rapport : effet des réglages de LightGBM.

1. Recherche aléatoire de 60 configurations (graine 11) + configuration de départ : apprentissage sur
   63 000 lignes, validation sur 21 000 (découpage des 84 000 lignes, graine 43), taux 0,05,
   nombre d'arbres fixé par arrêt anticipé (150 arbres de patience) ;
2. la meilleure configuration et celle de départ sont réentraînées avec un taux de 0,02 ;
3. validation croisée (5 plis, nombre d'arbres figé) des deux, au taux 0,02.

Exécution : python analyses/lightgbm_recherche.py      (environ 2 heures sur 2 cœurs ; reprend si interrompu)
Sorties : resultats/analyses/lightgbm_recherche.json et lightgbm_recherche_essais.csv,
          rapport/figures/fig_lightgbm_recherche.(pdf|png)
"""
import copy
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _commun import CFG, SORTIE, donnees, ecrire, lire_predictions, rmse
from _style import BLEU, ENCRE, LARGEUR, ORANGE, axe_virgule, enregistrer, plt, virgule
from powerlifting.models import lightgbm as L

train, test, d, dev, hold = donnees()
X, y_all = train.drop(columns=["TotalKg"]), train.TotalKg.to_numpy(float)
fit_idx, select_idx = train_test_split(d.dev, test_size=0.25, random_state=43)

ESPACE = {
    "num_leaves": [15, 31, 63, 127, 255],
    "min_child_samples": [10, 30, 100, 300],
    "reg_lambda": [0.0, 1.0, 10.0, 30.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "subsample": [0.7, 0.9, 1.0],
    "extra_trees": [False, True],
    "cat_smooth": [10.0, 40.0],
    "exponent": [0.0, 2 / 3, 1.0],
}
DEPART = {"num_leaves": 127, "min_child_samples": 30, "reg_lambda": 10.0, "colsample_bytree": 1.0,
          "subsample": 1.0, "extra_trees": True, "cat_smooth": 40.0, "exponent": 0.0}


def configuration(choix, taux):
    c = copy.deepcopy(CFG["modeles"]["lightgbm"])
    c.pop("best_iteration", None)
    c["name"], c["patience"] = "lgb_recherche", 150
    c["params"].update(learning_rate=taux, n_estimators=8000)
    for cle, valeur in choix.items():
        if cle == "exponent":
            c["exponent"] = float(valeur)
        else:
            c["params"][cle] = valeur
    c["params"]["subsample_freq"] = 1 if c["params"]["subsample"] < 1 else 0
    return c


def essayer(choix, taux):
    debut = time.time()
    f = L.fit_candidate(X.iloc[fit_idx], y_all[fit_idx], configuration(choix, taux),
                        evaluation=(X.iloc[select_idx], y_all[select_idx]))
    return {**choix, "taux": taux, "arbres": int(f["config"]["best_iteration"]),
            "rmse_validation": rmse(y_all[select_idx], L.predict_candidate(f, X.iloc[select_idx])),
            "secondes": time.time() - debut}


def hors_pli(choix, arbres):
    c = configuration(choix, 0.02)
    c["best_iteration"] = arbres
    p = np.empty(len(d.dev))
    for k in range(1, 6):
        a, q = d.dev[d.plis != k], d.dev[d.plis == k]
        p[d.plis == k] = L.predict_candidate(L.fit_candidate(X.iloc[a], y_all[a], c), X.iloc[q])
    return p


# Les 60 tirages sont faits d'avance ; chaque essai terminé est enregistré, et une exécution
# interrompue reprend où elle s'était arrêtée.
rng = np.random.default_rng(11)
tirages = [("depart", DEPART)] + [(f"r{i:02d}", {cle: valeurs[int(rng.integers(len(valeurs)))] for cle, valeurs in ESPACE.items()})
                                  for i in range(60)]
SORTIE.mkdir(parents=True, exist_ok=True)
partiel, complet = SORTIE / "lightgbm_recherche_essais_en_cours.csv", SORTIE / "lightgbm_recherche_essais.csv"
faits = pd.read_csv(partiel) if partiel.is_file() else pd.read_csv(complet) if complet.is_file() else pd.DataFrame()
essais = faits.to_dict("records")
for nom, choix in tirages:
    if len(faits) and nom in set(faits.essai):
        continue
    essais.append({**essayer(choix, 0.05), "essai": nom})
    pd.DataFrame(essais).to_csv(partiel, index=False)
    print(nom, round(essais[-1]["rmse_validation"], 3), flush=True)
table = pd.DataFrame(essais).sort_values("rmse_validation")
table.to_csv(complet, index=False)
partiel.unlink(missing_ok=True)

meilleur = table.iloc[0][list(ESPACE)].to_dict()
meilleur = {k: (bool(v) if k == "extra_trees" else int(v) if k in ("num_leaves", "min_child_samples") else float(v))
            for k, v in meilleur.items()}
fin, depart_fin = essayer(meilleur, 0.02), essayer(DEPART, 0.02)
y = y_all[d.dev]
aleatoires = table[table.essai != "depart"]
R = {"essais": int(len(table)), "min": float(table.rmse_validation.min()), "max": float(table.rmse_validation.max()),
     "ecart_type": float(table.rmse_validation.std()),
     "mediane_avec_70_pourcent_des_lignes": float(aleatoires.loc[aleatoires.subsample == 0.7, "rmse_validation"].median()),
     "mediane_sans_sous_echantillonnage": float(aleatoires.loc[aleatoires.subsample == 1.0, "rmse_validation"].median()),
     "meilleure_configuration": meilleur, "meilleure_taux_002": fin, "depart_taux_002": depart_fin,
     "validation_croisee": {"meilleure_taux_002": rmse(y, hors_pli(meilleur, fin["arbres"])),
                            "depart_taux_002": rmse(y, hors_pli(DEPART, depart_fin["arbres"])),
                            "depart_taux_004": rmse(y, lire_predictions("lightgbm")["hors_pli"])}}
ecrire("lightgbm_recherche", R)

# Figure 2 : une case par hyperparamètre, un point par configuration tirée, trait noir = médiane.
NOMS = {"num_leaves": "Feuilles par arbre", "min_child_samples": "Lignes min. par feuille",
        "reg_lambda": "Pénalité L2 (λ)", "colsample_bytree": "Fraction de variables", "subsample": "Fraction de lignes",
        "extra_trees": "Seuils aléatoires", "cat_smooth": "Lissage des catégories", "exponent": "Exposant p de la cible"}


def libelle(cle, v):
    if cle == "extra_trees":
        return "oui" if str(v) in ("True", "1", "1.0") else "non"
    if cle == "exponent":
        return {0.0: "0", 1.0: "1"}.get(round(float(v), 3), "2/3")
    return virgule(float(v), 0) if float(v) == int(float(v)) else virgule(float(v), 1)


reference = table.loc[table.essai == "depart", "rmse_validation"].iloc[0]
fig, axes = plt.subplots(2, 4, figsize=(LARGEUR, 3.2), sharey=True)
alea = np.random.default_rng(0)
for ax, cle in zip(axes.ravel(), NOMS):
    valeurs = sorted(aleatoires[cle].unique(), key=lambda v: float(v) if cle != "extra_trees" else str(v) == "True")
    for i, v in enumerate(valeurs):
        rv = aleatoires.loc[aleatoires[cle] == v, "rmse_validation"].to_numpy()
        ax.scatter(i + alea.uniform(-0.12, 0.12, len(rv)), rv, s=9, color=BLEU, alpha=0.75, linewidths=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(rv)] * 2, color=ENCRE, linewidth=1.6)
    ax.axhline(reference, color=ORANGE, linewidth=1, linestyle=(0, (3, 2)))
    ax.set_xticks(range(len(valeurs)), [libelle(cle, v) for v in valeurs])
    ax.set_xlim(-0.6, len(valeurs) - 0.4)
    ax.set_title(NOMS[cle], fontsize=7.8)
    ax.grid(axis="x", visible=False)
    axe_virgule(ax.yaxis, 1)
axes[0, 0].set_ylabel("RMSE validation (kg)"); axes[1, 0].set_ylabel("RMSE validation (kg)")
bas, haut = aleatoires.rmse_validation.min(), aleatoires.rmse_validation.max()
axes[0, 0].set_ylim(bas - 0.15, min(haut, bas + 2.2) + 0.15)
fig.tight_layout(h_pad=1.2, w_pad=0.6)
enregistrer(fig, "fig_lightgbm_recherche")
