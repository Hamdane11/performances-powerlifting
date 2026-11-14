"""Figure 3 du rapport : erreur sur le pli 1 selon le nombre de lignes d'apprentissage.

On prédit les 16 800 lignes du pli 1 en apprenant sur les n premières lignes d'une permutation
(graine 2026) des 67 200 lignes des plis 2 à 5 (sous-échantillons emboîtés).
- Ridge (cible en kg) et LightGBM (configuration de départ, arrêt anticipé sur 10 % du
  sous-échantillon) : calculés ici, sur processeur ;
- TabPFN et TabM : sorties du notebook GPU (resultats/gpu/C_*.npz) ; à 67 200 lignes, TabPFN = 8 passages
  sur le pli 1 (B_allom_8) et TabM = colonne hors pli du mélange.

Exécution : python analyses/courbes_apprentissage.py      (quelques minutes)
Sorties : resultats/analyses/courbes_apprentissage.json, rapport/figures/fig_courbes_apprentissage.(pdf|png)
"""
import copy

import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _commun import CAT, CFG, GPU, NUM, donnees, ecrire, lire_predictions, rmse
from _style import BLEU, ENCRE, ENCRE2, LARGEUR, ORANGE, VERT, axe_virgule, enregistrer, plt
from powerlifting.models import lightgbm as L

train, test, d, dev, hold = donnees()
y_all = train.TotalKg.to_numpy(float)
contexte, pli1 = d.dev[d.plis != 1], d.dev[d.plis == 1]
y1 = y_all[pli1]
ordre = np.random.default_rng(2026).permutation(len(contexte))
config = copy.deepcopy(CFG["modeles"]["lightgbm"])
config.pop("best_iteration", None)
config["params"]["n_estimators"] = 8000
Xr = train[NUM + CAT].copy()
Xr.loc[Xr.Age < 5, "Age"] = np.nan


def ridge(lignes):
    m = make_pipeline(ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT)]), Ridge(alpha=100.0))
    m.fit(Xr.iloc[lignes], y_all[lignes])
    return np.maximum(m.predict(Xr.iloc[pli1]), 0.0)


def lightgbm(lignes):
    coupe = int(0.9 * len(lignes))
    a, v = lignes[:coupe], lignes[coupe:]
    f = L.fit_candidate(train.iloc[a], y_all[a], config, evaluation=(train.iloc[v], y_all[v]))
    return L.predict_candidate(f, train.iloc[pli1])


tailles = (1000, 3000, 10000, 30000, len(contexte))
R = {"Ridge": {}, "LightGBM": {}, "TabM": {}, "TabPFN": {}}
for n in tailles:
    lignes = contexte[np.sort(ordre[:n])]
    R["Ridge"][str(n)], R["LightGBM"][str(n)] = rmse(y1, ridge(lignes)), rmse(y1, lightgbm(lignes))
    print(n, "fait", flush=True)
gpu = lambda nom: np.load(GPU / f"{nom}.npz")["prediction"].astype(float)  # noqa: E731
for n in tailles[:-1]:
    R["TabPFN"][str(n)], R["TabM"][str(n)] = rmse(y1, gpu(f"C_tabpfn_{n}")), rmse(y1, gpu(f"C_tabm_{n}"))
R["TabPFN"][str(tailles[-1])] = rmse(y1, gpu("B_allom_8"))
R["TabM"][str(tailles[-1])] = rmse(y1, lire_predictions("tabm")["hors_pli"][d.plis == 1])
ecrire("courbes_apprentissage", R)

fig, ax = plt.subplots(figsize=(LARGEUR * 0.62, 2.35))
couleurs = {"Ridge": ENCRE2, "LightGBM": BLEU, "TabM": VERT, "TabPFN": ORANGE}
fins = []
for nom, valeurs in R.items():
    n = [int(k) for k in valeurs]
    ax.plot(n, list(valeurs.values()), marker="o", color=couleurs[nom], label=nom)
    fins.append([list(valeurs.values())[-1], nom, n[-1]])
fins.sort()                                   # étiquettes de fin écartées d'au moins 1,1 kg
for i in range(1, len(fins)):
    fins[i][0] = max(fins[i][0], fins[i - 1][0] + 1.1)
for yv, nom, xv in fins:
    ax.annotate(nom, (xv, yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=ENCRE)
ax.set_xscale("log")
ax.set_xticks(list(tailles), ["1 000", "3 000", "10 000", "30 000", "67 200"])
ax.minorticks_off()
ax.set_xlim(800, 140000)
ax.set_xlabel("Lignes d'apprentissage (échelle log)")
ax.set_ylabel("RMSE sur 16 800 lignes (kg)")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
axe_virgule(ax.yaxis, 0)
ax.legend(loc="upper right")
fig.tight_layout()
enregistrer(fig, "fig_courbes_apprentissage")
