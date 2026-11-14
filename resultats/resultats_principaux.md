# Résultats principaux (recalculés par 4_resultats_principaux.py)

RMSE en kg. Validation croisée : 5 plis sur les 84 000 lignes de développement. Test interne : apprentissage sur les 84 000 lignes, mesure unique sur les 21 000 lignes mises de côté.

## Tableau 1 : prédictions de référence

| Prédiction | RMSE | R² |
|---|---:|---:|
| Moyenne générale | 171,3 | -0,00 |
| Moyenne par sexe | 128,3 | 0,44 |
| Moyenne par sexe et catégorie de poids | 105,5 | 0,62 |
| Moyenne par sexe, équipement et catégorie de poids | 100,3 | 0,66 |
| Notre solution finale | 76,3 | 0,80 |

## Tableau 2 : modèles individuels

| Modèle | Validation croisée | Test interne |
|---|---:|---:|
| Ridge (p = 1) | 83,93 | 83,81 |
| CatBoost | 78,40 | 78,32 |
| LightGBM, configuration de départ | 78,23 | 78,10 |
| LightGBM, meilleure configuration | 78,10 | 77,98 |
| LightGBM, configuration de départ, 5 graines | 78,07 | 77,89 |
| HistGradientBoosting, 5 graines | 78,26 | 78,01 |
| TabM | 78,12 | 78,16 |
| TabPFN (8 passages, cible ramenée) | 76,94 | 76,78 |

TabPFN avec 4 passages et la cible en kg (ancienne version) : 77,08 en validation croisée.

## Section 6 : mélanges (validation imbriquée)

- Premier mélange de six modèles : 76,76
- Mélange retenu (huit modèles) : 76,65
- Poids de TabPFN : 68 % (profils rares), 51 % (fréquents)
- Écart avec TabPFN seul : -0,28 [-0,33 ; -0,23] ; baisse par pli : 0,26, 0,32, 0,40, 0,20, 0,23

## Tableau 4 : correction par appariement (validation imbriquée)

| Prédiction corrigée | Avant | Après | Gain [IC 95 %] | Baisse par pli |
|---|---:|---:|---:|---|
| TabPFN seul | 76,91 | 76,55 | -0,36 [-0,42 ; -0,31] | 0,40, 0,41, 0,34, 0,27, 0,40 |
| Mélange avec l'ancienne version de TabPFN | 76,69 | 76,29 | -0,39 [-0,45 ; -0,34] | 0,45, 0,45, 0,34, 0,30, 0,42 |
| Mélange retenu | 76,64 | 76,27 | -0,37 [-0,43 ; -0,32] | 0,42, 0,42, 0,33, 0,30, 0,39 |

## Tableau 5 : soumissions

| Solution | Validation croisée | Test interne | Public Kaggle |
|---|---:|---:|---:|
| TabPFN seul | 76,94 | 76,78 | 76,42 |
| TabPFN + correction par appariement | 76,55 | 76,08 | 75,86 |
| Mélange + correction par appariement (solution finale) | 76,27 | 75,83 | 75,65 |

Test interne :
- mélange avant correction : 76,50 ; écart avec TabPFN seul : -0,27 [-0,38 ; -0,17]
- gain de la correction : -0,68 [-0,80 ; -0,55]
- solution finale moins TabPFN seul : -0,95 [-1,11 ; -0,79]

Public : scores relevés sur Kaggle (non recalculables). Variante au réservoir de 105 000 lignes : 75,65040 contre 75,65106.
