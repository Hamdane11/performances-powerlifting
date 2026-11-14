"""Sections 1 et 3 du rapport : description des données, prétraitements, figure 1.

Exécution : python analyses/donnees.py      (quelques secondes)
Sorties : resultats/analyses/donnees.json, rapport/figures/fig_donnees.(pdf|png)
"""
import numpy as np
import pandas as pd

from _commun import CFG, donnees, ecrire
from _style import BLEU, ENCRE, LARGEUR, ORANGE, enregistrer, plt

train, test, d, dev, hold = donnees()
part = lambda s: round(float(s.mean()), 4)  # noqa: E731

age = train.Age
R = {
    "lignes": {"train": len(train), "test": len(test)},
    "part_hommes": part(train.Sex == "M"),
    "age_median": float(age[age >= 5].median()),
    "poids_median": float(train.BodyweightKg.median()),
    "categories_de_poids": int(train.WeightClassKg.nunique()),
    "equipement": train.Equipment.value_counts(normalize=True).round(4).to_dict(),
    "part_controlee": part(train.Tested == "Yes"),
    "federations": int(train.Federation.nunique()),
    "annees": [int(train.MeetYear.min()), int(train.MeetYear.max())],
    "part_region_inconnue": part(train.State == "Unknown"),
    "valeurs_manquantes": int(train.isna().sum().sum() + test.isna().sum().sum()),
    # Section 3 : prétraitements
    "ages_inferieurs_a_5": int((age < 5).sum()),
    "part_ages_en_demi": part(np.isclose(age % 1, 0.5)),
    "region_inconnue_hors_usa": part(train.State[train.Country != "USA"] == "Unknown"),
    "region_inconnue_usa": part(train.State[train.Country == "USA"] == "Unknown"),
}

# Nombre de colonnes reçues par chaque modèle (variables d'origine + variables ajoutées).
from powerlifting.models import catboost as C, lightgbm as L  # noqa: E402

echantillon = train.drop(columns=["TotalKg"]).head(2000)
R["colonnes_par_modele"] = {"Ridge": 16, "TabPFN": 16,
                            "LightGBM": int(L.make_features(echantillon, CFG["modeles"]["lightgbm"])[0].shape[1]),
                            "CatBoost": int(C.make_features(echantillon, CFG["modeles"]["catboost"])[0].shape[1])}
try:  # TabM demande torch ; facultatif sur une machine sans
    from powerlifting.models import tabm as T  # noqa: E402
    R["colonnes_par_modele"]["TabM"] = int(T.make_features(echantillon, CFG["modeles"]["tabm"])[0].shape[1])
except ImportError:
    pass
ecrire("donnees", R)

# Figure 1 : distribution du total ; total selon le poids de corps (3 000 hommes et 3 000 femmes).
y = dev.TotalKg.to_numpy(float)
fig, (g, dr) = plt.subplots(1, 2, figsize=(LARGEUR, 2.05), gridspec_kw={"width_ratios": [1, 1.25]})
g.hist(y, bins=np.arange(0, 1300, 40), color=BLEU, edgecolor="white", linewidth=0.6)
g.set_title("Distribution du total")
g.set_xlabel("Total soulevé (kg)"); g.set_ylabel("Participations")
g.grid(axis="x", visible=False)
rng = np.random.default_rng(0)
for sexe, couleur, nom in [("M", BLEU, "Hommes"), ("F", ORANGE, "Femmes")]:
    idx = np.where(dev.Sex.to_numpy() == sexe)[0]
    idx = rng.choice(idx, size=min(3000, len(idx)), replace=False)
    dr.scatter(dev.BodyweightKg.to_numpy()[idx], y[idx], s=3, color=couleur, alpha=0.35, linewidths=0, label=nom)
    sous = dev[dev.Sex == sexe]
    tranches = (sous.BodyweightKg // 5) * 5 + 2.5          # médiane du total par tranche de 5 kg
    med = pd.Series(sous.TotalKg.to_numpy(), index=tranches.to_numpy()).groupby(level=0).agg(["median", "size"])
    med = med[med["size"] >= 50]
    dr.plot(med.index, med["median"], color=couleur, linewidth=2)
    dr.annotate(nom, (med.index[-1], med["median"].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=ENCRE)
dr.set_xlim(35, 185); dr.set_ylim(0, 1150)
dr.set_title("Total selon le poids corporel")
dr.set_xlabel("Poids corporel (kg)"); dr.set_ylabel("Total (kg)")
dr.legend(loc="upper left", markerscale=3)
fig.tight_layout(w_pad=2)
enregistrer(fig, "fig_donnees")
