"""Mesures d'erreur : RMSE et intervalle de confiance d'un écart de RMSE."""
from __future__ import annotations

import numpy as np


def rmse(y, prediction) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(prediction, float)) ** 2)))


def bootstrap_ecart(y, prediction, reference, tirages=2000, graine=1):
    """Écart de RMSE (prediction - reference) et intervalle percentile à 95 %.

    Bootstrap apparié sur les lignes : les prédictions sont fixes, on ne mesure
    que l'incertitude due à l'échantillon évalué.
    """
    y, prediction, reference = (np.asarray(v, float) for v in (y, prediction, reference))
    rng = np.random.default_rng(graine)
    e1, e0 = (y - prediction) ** 2, (y - reference) ** 2
    n = len(y)
    ecarts = np.empty(tirages)
    for b in range(tirages):
        i = rng.integers(0, n, n)
        ecarts[b] = np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())
    return {"ecart": rmse(y, prediction) - rmse(y, reference),
            "ic95": [float(v) for v in np.percentile(ecarts, [2.5, 97.5])]}
