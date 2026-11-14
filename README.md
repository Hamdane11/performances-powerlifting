# Prédire le total en force athlétique

Challenge Kaggle « Performances en powerlifting » (M2 Mathématiques et IA, Université Paris-Saclay) :
prédire le total soulevé `TotalKg` d'une participation à une compétition, évalué en RMSE.
Hamdane Tchani Yacoubou et Solène Thibault.

**Rapport :** [`rapport/rapport_powerlifting.pdf`](rapport/rapport_powerlifting.pdf)

## Résultat

Notre solution finale mélange huit modèles (dont TabPFN, LightGBM, CatBoost et TabM), puis corrige la
prédiction de chaque ligne à partir des totaux des participations qui viennent probablement du même athlète
(*correction par appariement*, section 7 du rapport).

| Solution (RMSE en kg) | Validation croisée | Test interne | Classement public |
|---|---:|---:|---:|
| TabPFN seul | 76,94 | 76,78 | 76,42 |
| TabPFN + correction par appariement | 76,55 | 76,08 | 75,86 |
| **Mélange + correction par appariement (solution finale)** | **76,27** | **75,83** | **75,65** |

Validation croisée : 5 plis sur 84 000 lignes. Test interne : 21 000 lignes mises de côté dès le départ,
mesurées une seule fois. Les deux soumissions retenues pour le classement final sont dans `soumissions/` :
`solution_finale.csv` et `solution_finale_reservoir_105000.csv` (même solution, les candidates étant
cherchées parmi les 105 000 lignes étiquetées).

## Reproduire les résultats (sans GPU)

Python 3.13, puis :

```bash
pip install -r requirements.txt
python 3_solution_finale.py        # 2 min  : recrée les deux soumissions retenues et les compare aux fichiers déposés
python 4_resultats_principaux.py   # 4 min  : recalcule les tableaux 1, 2, 4 et 5 du rapport
```

Ces deux scripts partent des prédictions de chaque modèle, fournies dans `predictions/`. Résultat attendu :

- `3_solution_finale.py` affiche un écart maximal de l'ordre de 1e-13 kg avec chaque soumission déposée ;
- `4_resultats_principaux.py` écrit [`resultats/resultats_principaux.md`](resultats/resultats_principaux.md),
  qui contient les tableaux du rapport.

## Tout recalculer, modèles compris

| Étape | Commande | Durée | Ce qu'elle produit |
|---|---|---|---|
| 1 | `python 1_modeles_cpu.py` | ~1 h 30 sur 2 cœurs (dont CatBoost 45 min) | `predictions/` des modèles sur processeur : Ridge, LightGBM, CatBoost, HistGradientBoosting |
| 2 | `2_modeles_gpu.ipynb` dans Google Colab (GPU) | ~4 h sur T4 | `predictions/` de TabPFN et TabM |
| 3 | `python 3_solution_finale.py` | 2 min | les deux soumissions |
| 4 | `python 4_resultats_principaux.py` | 4 min | tableaux du rapport |

Chaque fichier `predictions/<modèle>.npz` contient trois jeux de prédictions :
`hors_pli` (84 000 lignes, chaque pli prédit par un modèle entraîné sur les quatre autres),
`test_interne` (21 000 lignes, modèle entraîné sur les 84 000) et `test` (45 000 lignes de `test.csv`,
modèle entraîné sur les 105 000).

Les modèles sur processeur sont déterministes : si `predictions/<modèle>.npz` existe déjà,
`1_modeles_cpu.py` recalcule le modèle, n'écrase rien et affiche l'écart avec le fichier fourni
(0 attendu ; on peut aussi ne recalculer qu'un modèle : `python 1_modeles_cpu.py lightgbm`).
Les calculs sur GPU (TabPFN, TabM) ne sont pas reproductibles au bit près d'une carte à l'autre.

## Organisation

```text
├── 1_modeles_cpu.py              # étape 1 : modèles sur processeur
├── 2_modeles_gpu.ipynb           # étape 2 : TabPFN et TabM (Colab, GPU)
├── 3_solution_finale.py          # étape 3 : mélange + correction -> soumissions
├── 4_resultats_principaux.py     # étape 4 : tableaux principaux du rapport
├── config/modeles.json           # réglages de tous les modèles et de la solution finale
├── src/powerlifting/
│   ├── data.py                   # lecture des données, découpage commun (test interne, 5 plis)
│   ├── models/                   # LightGBM, CatBoost, TabM, TabPFN, HistGradientBoosting (+ leurs variables)
│   ├── ensemble.py, melange.py   # correction des profils identiques, poids du mélange
│   └── appariement.py            # correction par appariement
├── donnees/                      # train.csv, test.csv, sample_submission.csv (Kaggle)
├── predictions/                  # prédictions de chaque modèle (sorties des étapes 1 et 2)
├── soumissions/                  # fichiers déposés sur Kaggle
├── analyses/                     # autres chiffres et figures du rapport (voir ci-dessous)
├── resultats/                    # sorties des scripts (tableaux, JSON, sorties GPU brutes)
└── rapport/                      # rapport (PDF et LaTeX) et ses figures
```

## Comment fonctionne la solution finale

1. **Modèles individuels** (`src/powerlifting/models/`, réglages dans `config/modeles.json`) : chacun
   prédit les 84 000 lignes de développement hors pli, le test interne et le test.
2. **Correction des profils identiques** (`melange.py`) : une prédiction est rapprochée du total moyen
   des lignes d'apprentissage identiques sur les 16 variables, quand il en existe.
3. **Mélange** (`melange.py`) : poids positifs de somme 1, ajustés sur les prédictions hors pli,
   différents pour les profils rares et fréquents.
4. **Correction par appariement** (`appariement.py`) : pour chaque ligne, on cherche les lignes
   étiquetées de même sexe, pays, région et année de naissance compatible ; un petit LightGBM apprend
   l'erreur du mélange à partir de leurs totaux (43 caractéristiques).

## D'où vient chaque chiffre du rapport

| Rapport | Script | Sortie |
|---|---|---|
| Tableaux 1, 2, 4, 5 ; section 6 (mélanges) ; section 8 (test interne) | `4_resultats_principaux.py` | `resultats/resultats_principaux.md` |
| Sections 1 et 3 (description des données), figure 1 | `analyses/donnees.py` | `resultats/analyses/donnees.json` |
| Section 4.1 (Ridge, cible ramenée au poids) | `analyses/ridge_et_cible.py` | `resultats/analyses/ridge_et_cible.json` |
| Section 4.2 (recherche LightGBM), figure 2 | `analyses/lightgbm_recherche.py` (~2 h) | `resultats/analyses/lightgbm_recherche.json` |
| Section 4.2 (CatBoost, variable de contexte) | `analyses/catboost_contexte.py` (~40 min) | `resultats/analyses/catboost_contexte.json` |
| Sections 4.3 et 4.4 (TabM, passages de TabPFN) | `analyses/tabm_tabpfn_gpu.py` | `resultats/analyses/tabm_tabpfn_gpu.json` |
| Figure 3 (courbes d'apprentissage) | `analyses/courbes_apprentissage.py` | `resultats/analyses/courbes_apprentissage.json` |
| Tableau 3 (profils rares) | `analyses/profils_rares.py` | `resultats/analyses/profils_rares.json` |
| Sections 5 et 6 (erreurs communes, Krogh-Vedelsby, Bates-Granger) | `analyses/erreurs_et_melange.py` | `resultats/analyses/erreurs_et_melange.json` |
| Section 7 (ablation, âge, exemples), figure 4 ; section 8 (taille du réservoir) | `analyses/appariement.py` (~30 min) | `resultats/analyses/appariement.json` |
| Section 7 (modèle de second niveau, candidates en entrée de LightGBM) | `analyses/modeles_second_niveau.py` (~30 min) | `resultats/analyses/modeles_second_niveau.json` |

Les scripts d'analyse se lancent depuis la racine du dépôt (`python analyses/donnees.py`) et ne demandent
pas de GPU : les calculs GPU des sections 4.3, 4.4 et de la figure 3 (`analyses/analyses_gpu.ipynb`)
ont été faits sur Colab et leurs sorties brutes sont fournies dans `resultats/gpu/`.

## Remarques

- **Classement public.** Les scores publics sont ceux affichés par Kaggle après dépôt ; ils ne peuvent pas
  être recalculés, les cibles du test étant cachées.
- **Données.** Les fichiers de `donnees/` sont ceux du challenge Kaggle (licence MIT indiquée sur la page
  des données ; échantillon de la base publique OpenPowerlifting).
- **TabPFN.** Les poids du modèle préentraîné sont téléchargés par la bibliothèque `tabpfn` (version 9.1.0)
  avec une clé d'API Prior Labs (secret Colab `TABPFN_TOKEN`).
