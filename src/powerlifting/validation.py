"""Reproducible partitions and checks for aligned OOF prediction banks.

The v5 protocol uses 84,000 development rows, in the exact order returned by
train_test_split. A separate protocol can use all 105,000 rows. Those protocols
must never share a prediction bank or blend weights fitted on mismatched OOFs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.model_selection import KFold, train_test_split


@dataclass(frozen=True)
class Partition:
    """All fold positions are relative to ``indices``, not the original CSV."""

    indices: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    fold_ids: np.ndarray
    historical_indices: np.ndarray
    protocol: dict

    @property
    def dev_idx(self) -> np.ndarray:
        """Compatibility name for historical development prediction caches."""
        return self.indices


def create_partitions(n_rows: int, protocol: Mapping) -> Partition:
    """Build deterministic development or full-data cross-validation folds.

    Protocol keys: scope (development/full), development_seed,
    development_fraction, fold_seed, fold_count. The historical v5 defaults
    are 80% development, seed 42, and five shuffled folds with seed 2026.
    """
    if not isinstance(n_rows, (int, np.integer)) or n_rows < 2:
        raise ValueError("Need at least two training rows.")
    scope = protocol.get("scope", "development")
    scope = {"development84k": "development", "full105k": "full"}.get(scope, scope)
    fraction = float(protocol.get("development_fraction", 0.8))
    development_seed = int(protocol.get("development_seed", 42))
    fold_seed = int(protocol.get("fold_seed", 2026))
    fold_count = int(protocol.get("fold_count", 5))
    if scope not in {"development", "full"}:
        raise ValueError("Validation scope must be development or full.")
    if not 0 < fraction < 1:
        raise ValueError("Development fraction must be strictly between zero and one.")
    rows = np.arange(n_rows, dtype=np.int64)
    if scope == "development":
        # The literal 0.2 matters: 1 - 0.8 can alter test_size rounding for some
        # row counts. This spelling exactly matches the historical split.
        test_size = 0.2 if fraction == 0.8 else 1.0 - fraction
        indices, historical = train_test_split(
            rows, test_size=test_size, random_state=development_seed,
        )
    else:
        indices, historical = rows, np.empty(0, dtype=np.int64)
    if not 2 <= fold_count <= len(indices):
        raise ValueError("Fold count must lie between two and the number of OOF rows.")
    folds = tuple(KFold(fold_count, shuffle=True, random_state=fold_seed).split(indices))
    fold_ids = np.zeros(len(indices), dtype=np.int64)
    for fold_id, (learning, query) in enumerate(folds, 1):
        if np.intersect1d(learning, query).size or np.any(fold_ids[query]):
            raise AssertionError("Validation fold membership overlaps.")
        fold_ids[query] = fold_id
    if np.any(fold_ids == 0) or np.intersect1d(indices, historical).size:
        raise AssertionError("Each OOF row needs exactly one validation fold.")
    if not np.array_equal(np.sort(np.concatenate((indices, historical))), rows):
        raise AssertionError("Development and historical rows must partition train.csv.")
    normalized = {
        "scope": scope, "development_fraction": fraction,
        "development_seed": development_seed, "fold_seed": fold_seed,
        "fold_count": fold_count,
    }
    return Partition(indices, folds, fold_ids, historical, normalized)


def validate_prediction_alignment(
    cached: Mapping, partition: Partition, test_ids: np.ndarray, model_count: int,
) -> None:
    """Reject prediction banks from another row order, fold protocol or test set."""
    index_key = "indices" if "indices" in cached else "dev_idx"
    checks = (
        (cached[index_key], partition.indices, "OOF CSV row positions"),
        (cached["fold_ids"], partition.fold_ids, "OOF folds"),
        (cached["test_ids"], np.asarray(test_ids), "test id order"),
    )
    for actual, expected, label in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"Prediction cache does not match the current {label}.")
    for key, expected_rows in (("raw_oof", len(partition.indices)), ("raw_test", len(test_ids))):
        values = np.asarray(cached[key])
        if values.shape != (expected_rows, model_count) or not np.isfinite(values).all():
            raise ValueError(f"{key} must have finite predictions with aligned rows/models.")
