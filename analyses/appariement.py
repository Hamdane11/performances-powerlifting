"""Section 7 du rapport : d'où vient le gain de la correction par appariement ? (et figure 4)

Base corrigée : prédictions hors pli du mélange retenu. Toutes les corrections sont évaluées en
validation strictement imbriquée (comme dans 4_resultats_principaux.py).
1. ablation : candidates de même génération seulement / candidates proches seulement / toutes ;
2. gain selon l'âge (figure 4, à droite) ;
3. pente de l'erreur du mélange sur l'écart moyen des candidates, par filtre et taille de cohorte
   (figure 4, à gauche) ;
4. deux exemples (un succès, un échec) ;
5. test interne : gain selon la taille du réservoir de candidates (section 8).

Exécution : python analyses/appariement.py      (environ 30 minutes sur 2 cœurs)
Sorties : resultats/analyses/appariement.json, rapport/figures/fig_appariement.(pdf|png)
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _commun import base_melange, bootstrap_ecart, donnees, ecrire, rmse
from _style import BLEU, ENCRE, ENCRE2, LARGEUR, axe_virgule, enregistrer, plt, virgule
from powerlifting import appariement as A

train, test, d, dev, hold = donnees()
y, plis, y_hold = dev.TotalKg.to_numpy(float), d.plis, hold.TotalKg.to_numpy(float)
p, base_hold = base_melange(dev, y, plis, hold=hold)
r = y - p
t = A.tableau(dev, np.arange(len(dev)), y=y, p=p)
R = {"rmse_base": rmse(y, p)}

# Caractéristiques de chaque pli : pour l'apprentissage du pli k (calculées sans le pli k)
# et pour le pli k lui-même (réservoir = les quatre autres plis).
APP = {(k, m): A.caracteristiques(t[plis == m], t[(plis != m) & (plis != k)]) for k in range(1, 6) for m in range(1, 6)
       if m != k}
EVAL = {k: A.caracteristiques(t[plis == k], t[plis != k]) for k in range(1, 6)}
noms = A.noms_caracteristiques()
colonnes = lambda prefixes: [i for i, n in enumerate(noms) if n.split(":")[0] in prefixes]  # noqa: E731


def corriger_avec(cols):
    cols = cols + [len(noms) - 1]                       # toujours la prédiction de la base
    c = p.copy()
    for k in range(1, 6):
        X = np.vstack([APP[(k, m)][:, cols] for m in range(1, 6) if m != k])
        cible = np.concatenate([r[plis == m] for m in range(1, 6) if m != k])
        c[plis == k] = p[plis == k] + A.ajuster_predire(X, cible, EVAL[k][:, cols])
    return c


# 1. Ablation.
proches = ["equip_bw10", "equip_bw5_an3", "equip_bw3_fed", "equip_bw2_an1"]
R["ablation"] = {}
for nom, prefixes in [("génération seulement", ["cohorte"]), ("candidates proches seulement", proches),
                      ("toutes", list(A.FILTRES))]:
    c = corriger_avec(colonnes(prefixes))
    R["ablation"][nom] = {"rmse": rmse(y, c), **bootstrap_ecart(y, c, p, tirages=1000)}
    print(nom, "fait", flush=True)
corrige = c                                              # dernière : toutes les caractéristiques

# 2. Gain selon l'âge.
tranches = pd.cut(dev.Age.to_numpy(), [0, 24, 35, 50, 120], labels=["< 24 ans", "24-34 ans", "35-49 ans", "50 ans et plus"])
R["par_age"] = {str(a): {"lignes": int((tranches == a).sum()), "rmse_base": rmse(y[tranches == a], p[tranches == a]),
                         "rmse_corrigee": rmse(y[tranches == a], corrige[tranches == a])} for a in tranches.categories}

# 3. Pente de l'erreur sur l'écart moyen des candidates (candidates prises dans les autres plis).
FIG = {"equip_bw10": "même équip., poids ±10 %", "equip_bw5_an3": "poids ±5 %, ±3 ans",
       "equip_bw3_fed": "poids ±3 %, même féd.", "equip_bw2_an1": "poids ±2 %, ±1 an"}
TAILLES = [(1, 3, "1–3"), (4, 10, "4–10"), (11, 50, "11–50"), (51, 10 ** 9, "> 50")]
parties = []
for k in range(1, 6):
    q, rv = A.annees_naissance(t[plis == k]), A.annees_naissance(t[plis != k])
    m = q.merge(rv[A.CLE + ["ligne", "Equipment", "Federation", "BodyweightKg", "MeetYear", "y"]], on=A.CLE, suffixes=("", "_c"))
    m = m[m.ligne != m.ligne_c].drop_duplicates(["ligne", "ligne_c"])
    m["dy"] = m.y_c * (m.BodyweightKg / m.BodyweightKg_c) ** (2 / 3) - m.p
    s = m.groupby("ligne").size().rename("cohorte").to_frame()
    for f in FIG:
        s = s.join(m[np.asarray(A.FILTRES[f](m))].groupby("ligne").dy.mean().rename(f), how="left")
    parties.append(s)
s = pd.concat(parties)
s["erreur"] = r[s.index.to_numpy()]
pentes = np.full((len(FIG), len(TAILLES)), np.nan)
effectifs = np.zeros_like(pentes, dtype=int)
for i, f in enumerate(FIG):
    sf = s.dropna(subset=[f])
    for j, (bas, haut, _) in enumerate(TAILLES):
        c = sf[(sf.cohorte >= bas) & (sf.cohorte <= haut)]
        pentes[i, j], effectifs[i, j] = np.polyfit(c[f], c.erreur, 1)[0], len(c)
R["pentes"] = {f: {n: {"pente": float(pentes[i, j]), "lignes": int(effectifs[i, j])} for j, (*_, n) in enumerate(TAILLES)}
               for i, f in enumerate(FIG)}

fig, (g, dr) = plt.subplots(1, 2, figsize=(LARGEUR, 2.45), gridspec_kw={"width_ratios": [1.35, 1]})
carte = LinearSegmentedColormap.from_list("bleu", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
g.imshow(pentes, cmap=carte, vmin=0, vmax=0.5, aspect="auto")
for i in range(pentes.shape[0]):
    for j in range(pentes.shape[1]):
        fonce = pentes[i, j] > 0.25
        g.text(j, i - 0.12, virgule(pentes[i, j], 2), ha="center", va="center", fontsize=7.5, fontweight="bold",
               color="white" if fonce else ENCRE)
        g.text(j, i + 0.22, f"{effectifs[i, j]:,}".replace(",", " ") + " l.", ha="center", va="center", fontsize=6,
               color="white" if fonce else ENCRE2)
g.set_xticks(range(len(TAILLES)), [n for *_, n in TAILLES])
g.set_yticks(range(len(FIG)), list(FIG.values()))
g.set_xlabel("Taille de la cohorte (même sexe, pays, région, naissance)")
g.set_title("Pente de l'erreur sur l'écart des candidates")
g.grid(False)
for cote in ("left", "bottom"):
    g.spines[cote].set_visible(False)
g.tick_params(length=0)
age = R["par_age"]
x = np.arange(len(age))
gain = [age[a]["rmse_base"] - age[a]["rmse_corrigee"] for a in age]
dr.bar(x, gain, width=0.55, color=BLEU)
for xi, v in zip(x, gain):
    dr.text(xi, v + 0.03, virgule(v, 2), ha="center", va="bottom", fontsize=7, color=ENCRE)
dr.tick_params(axis="x", labelsize=6.8)
dr.set_xticks(x, [a.replace(" ans", "").replace(" et plus", "+") + "\n" + f"{age[a]['lignes']:,}".replace(",", " ") + " l."
                  for a in age])
dr.set_xlabel("Âge (ans)"); dr.set_ylabel("Baisse de la RMSE (kg)")
dr.set_ylim(0, 1.75); dr.set_yticks([0, 0.5, 1, 1.5]); axe_virgule(dr.yaxis, 1)
dr.set_title("Gain selon l'âge"); dr.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.8)
enregistrer(fig, "fig_appariement")


# 4. Deux exemples : la ligne, sa prédiction avant et après correction, ses candidates proches.
def exemple(masque):
    i = int(np.flatnonzero(masque)[0])
    assert masque.sum() == 1
    rv = t[plis != plis[i]]                                   # réservoir : les quatre autres plis
    q = A.annees_naissance(t.iloc[[i]])
    m = q.merge(A.annees_naissance(rv), on=A.CLE, suffixes=("", "_c"))
    m = m[np.asarray(A.FILTRES["equip_bw10"](m))].drop_duplicates("ligne_c")
    candidates = dev.loc[m.ligne_c.to_numpy(), ["Age", "BodyweightKg", "Equipment", "Federation", "MeetYear", "TotalKg"]]
    return {"ligne": dev.loc[i, ["Age", "BodyweightKg", "Equipment", "Federation", "MeetYear", "TotalKg"]].to_dict(),
            "melange": float(p[i]), "corrigee": float(corrige[i]),
            "candidates_meme_equipement_poids_10_pourcent": candidates.sort_values("MeetYear").to_dict("records")}


R["exemples"] = {
    "succès (68 ans)": exemple((dev.Age == 68) & (dev.BodyweightKg == 92.6) & (dev.TotalKg == 672.5)),
    "échec (54 ans)": exemple((dev.Age == 54) & (dev.BodyweightKg == 122.9) & (dev.TotalKg == 645)),
}

# 5. Test interne : modèle de second niveau entraîné une fois sur les 84 000 lignes, puis réservoir
#    réduit à une fraction tirée au hasard des 84 000 lignes (3 tirages par taille).
X84 = np.vstack([A.caracteristiques(t[plis == m], t[plis != m]) for m in range(1, 6)])
cible = np.concatenate([r[plis == m] for m in range(1, 6)])
h = A.tableau(hold, np.arange(len(hold)) + 10_000_000, p=base_hold)
rng = np.random.default_rng(0)
R["taille_reservoir_test_interne"] = {"rmse_base": rmse(y_hold, base_hold)}
for fraction in (0.4, 0.6, 0.8, 1.0):
    gains = []
    for _ in range(3 if fraction < 1 else 1):
        garde = rng.random(len(t)) < fraction if fraction < 1 else np.ones(len(t), bool)
        c = np.maximum(base_hold + A.ajuster_predire(X84, cible, A.caracteristiques(h, t[garde])), 0.0)
        gains.append(rmse(y_hold, c) - rmse(y_hold, base_hold))
    R["taille_reservoir_test_interne"][str(int(round(fraction * len(t))))] = {"gains": gains, "moyenne": float(np.mean(gains))}
ecrire("appariement", R)
