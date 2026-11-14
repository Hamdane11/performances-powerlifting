"""Section 4.2 du rapport : CatBoost, que rapporte la variable de contexte fédération|pays|région|année ?

Configuration retenue (profondeur 7, taux 0,06, lambda 8, p = 1), avec et sans cette seule variable.
Apprentissage sur 63 000 lignes, validation sur 21 000 (découpage des 84 000 lignes, graine 43),
nombre d'arbres fixé par arrêt anticipé (250 arbres de patience, 6 500 au plus) : même protocole que la
recherche CatBoost d'octobre, dont la configuration retenue avait obtenu 78,18 kg.

Exécution : python analyses/catboost_contexte.py      (environ 40 minutes sur 2 cœurs ; reprend si interrompu)
Sortie : resultats/analyses/catboost_contexte.json
"""
import copy
import json
import time

from sklearn.model_selection import train_test_split

from _commun import CFG, SORTIE, donnees, ecrire, rmse
from powerlifting.models import catboost as C

train, test, d, dev, hold = donnees()
X, y_all = train.drop(columns=["TotalKg"]), train.TotalKg.to_numpy(float)
fit_idx, select_idx = train_test_split(d.dev, test_size=0.25, random_state=43)
fichier = SORTIE / "catboost_contexte.json"
R = json.loads(fichier.read_text()) if fichier.is_file() else {}
for nom, avec in (("avec la variable de contexte", True), ("sans la variable de contexte", False)):
    if nom in R:
        continue
    config = copy.deepcopy(CFG["modeles"]["catboost"])
    config.pop("best_iteration", None)                  # nombre d'arbres fixé par l'arrêt anticipé
    config["competition_context"] = avec
    debut = time.time()
    f = C.fit_candidate(X.iloc[fit_idx], y_all[fit_idx], config, validation=(X.iloc[select_idx], y_all[select_idx]))
    R[nom] = {"rmse_validation": rmse(y_all[select_idx], C.predict_candidate(f, X.iloc[select_idx])),
              "arbres": int(f["config"]["best_iteration"]), "minutes": (time.time() - debut) / 60}
    SORTIE.mkdir(parents=True, exist_ok=True)
    fichier.write_text(json.dumps(R, indent=1, ensure_ascii=False) + "\n")
    print(nom, R[nom], flush=True)
R["gain_de_la_variable"] = R["sans la variable de contexte"]["rmse_validation"] - R["avec la variable de contexte"]["rmse_validation"]
ecrire("catboost_contexte", R)
