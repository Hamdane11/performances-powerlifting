




















"""Exact-profile correction and mixtures of aligned OOF predictions.

An exact profile is a tuple of the 16 raw predictors, not an identified athlete.
Lookup fitting must use the learning portion of a fold, never its query labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import FEATURES, NUMERIC


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing lookup features: {sorted(missing)}")
    result = frame.loc[:, FEATURES].copy().reset_index(drop=True)
    if result.isna().any().any():
        raise ValueError("Exact lookup requires nonmissing raw feature values.")
    for column in NUMERIC:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise ValueError(f"Numeric lookup feature has nonnumeric dtype: {column}")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite lookup feature: {column}")
    for column in set(FEATURES).difference(NUMERIC):
        if not result[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Categorical lookup values must be strings: {column}")
        result[column] = result[column].astype(object)
    return result


@dataclass(frozen=True)
class ExactProfileLookup:
    statistics: pd.DataFrame
    training_rows: int


def fit_lookup(frame: pd.DataFrame, target: Sequence[float]) -> ExactProfileLookup:
    """Learn target mean, count and population variance for each raw profile."""
    keys = _keys(frame)
    y = np.asarray(target, dtype=float)
    if y.shape != (len(keys),) or not np.isfinite(y).all() or len(y) == 0:
        raise ValueError("Need a nonempty finite one-dimensional training target.")
    keys["_lookup_target"] = y
    grouped = keys.groupby(list(FEATURES), sort=False, dropna=False, observed=True)["_lookup_target"]
    stats = grouped.agg(raw_prediction="mean", n_matches="size", target_variance="var").reset_index()
    stats["target_variance"] = stats["target_variance"].fillna(0) * (stats.n_matches - 1) / stats.n_matches
    if stats.duplicated(list(FEATURES)).any():
        raise AssertionError("Lookup statistics must have one row per exact key.")
    return ExactProfileLookup(statistics=stats, training_rows=len(keys))


def lookup_diagnostics(fitted: ExactProfileLookup, frame: pd.DataFrame) -> pd.DataFrame:
    """Return raw_prediction (NaN if unseen), n_matches and target_variance."""
    keys = _keys(frame)
    keys["_lookup_position"] = np.arange(len(keys))
    merged = keys.merge(fitted.statistics, on=list(FEATURES), how="left", sort=False,
                        validate="many_to_one")
    merged = merged.sort_values("_lookup_position", kind="stable")
    if not np.array_equal(merged["_lookup_position"], np.arange(len(keys))):
        raise AssertionError("Lookup merge changed row positions.")
    result = merged[["raw_prediction", "n_matches", "target_variance"]].reset_index(drop=True)
    result["n_matches"] = result["n_matches"].fillna(0).astype(np.int64)
    return result


def exact_correction(
    matrix: np.ndarray, means: np.ndarray, counts: np.ndarray, alpha: float = 0.75,
) -> np.ndarray:
    """Blend only matched rows, preserving the historical v5 arithmetic order."""
    raw = np.asarray(matrix, dtype=float)
    means, counts = np.asarray(means, dtype=float), np.asarray(counts)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Predictions must be a finite two-dimensional matrix.")
    if means.shape != (len(raw),) or counts.shape != (len(raw),):
        raise ValueError("Lookup means/counts must have one value per prediction row.")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Lookup counts must be finite and nonnegative.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Lookup alpha must lie in [0, 1].")
    matched = counts > 0
    if not np.isfinite(means[matched]).all():
        raise ValueError("Matched profiles must have a finite target mean.")
    corrected = raw.copy()
    corrected[matched] = (1 - alpha) * raw[matched] + alpha * means[matched, None]
    return corrected


def _weights(weights: Sequence[float], n_columns: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (n_columns,) or not np.isfinite(values).all():
        raise ValueError("Need one finite mixture weight per model column.")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Mixture weights must be nonnegative and sum to one.")
    return values


def weighted_predictions(
    matrix: np.ndarray, weights: Sequence[float],
    original_indices: Sequence[int] | None = None,
    original_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Mix models, optionally retaining v5's exact original 18-column dot product.

    The reference contains six active columns among 18 original candidates.
    Rebuilding the zero columns preserves the CSV down to its last decimal;
    simply multiplying six columns can change floating-point rounding.
    """
    predictions = np.asarray(matrix, dtype=float)
    if predictions.ndim != 2 or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be a finite two-dimensional matrix.")
    active_weights = _weights(weights, predictions.shape[1])
    if original_indices is None and original_weights is None:
        return predictions @ active_weights
    if original_indices is None or original_weights is None:
        raise ValueError("Reference blending requires both original indices and weights.")
    indices = np.asarray(original_indices)
    full_weights = _weights(original_weights, len(original_weights))
    if indices.shape != (predictions.shape[1],) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Need one original integer column index per active model.")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(full_weights)):
        raise ValueError("Original model column indices must be distinct and in range.")
    if not np.array_equal(full_weights[indices], active_weights):
        raise ValueError("Active and original mixture weights differ.")
    omitted = np.ones(len(full_weights), dtype=bool)
    omitted[indices] = False
    if np.any(full_weights[omitted] != 0):
        raise ValueError("A nonzero original weight has no prediction column.")
    complete = np.zeros((len(predictions), len(full_weights)))
    complete[:, indices] = predictions
    return complete @ full_weights


def rmse(target: Sequence[float], prediction: Sequence[float]) -> float:
    """Root mean squared error in kilograms."""
    y, pred = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or len(y) == 0 or pred.shape != y.shape:
        raise ValueError("Targets and predictions must be equal-length nonempty vectors.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets and predictions must be finite.")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


@dataclass(frozen=True)
class BlendFit:
    weights: np.ndarray
    rmse: float
    success: bool
    message: str
    iterations: int


def fit_blend_weights(target: Sequence[float], predictions: np.ndarray) -> BlendFit:
    """Choose convex weights by minimizing MSE on an aligned OOF matrix.

    This is development-set selection: its optimized OOF error is not an
    independent evaluation. Freeze weights before evaluating another split.
    """
    from scipy.optimize import minimize

    y, matrix = np.asarray(target, dtype=float), np.asarray(predictions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(y) or matrix.shape[1] == 0:
        raise ValueError("Need an aligned OOF matrix with at least one model.")
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(matrix).all():
        raise ValueError("OOF predictions and targets must all be finite.")

    def objective(weight):
        return float(np.mean((matrix @ weight - y) ** 2))

    def gradient(weight):
        return (2.0 / len(y)) * (matrix.T @ (matrix @ weight - y))

    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    fitted = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not fitted.success:
        raise RuntimeError(f"Blend weight optimization failed: {fitted.message}")
    weight = np.clip(np.asarray(fitted.x, dtype=float), 0.0, 1.0)
    weight /= weight.sum()
    _weights(weight, matrix.shape[1])
    return BlendFit(weight, rmse(y, matrix @ weight), bool(fitted.success),
                    str(fitted.message), int(fitted.nit))
