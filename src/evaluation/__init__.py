"""Evaluation utilities for time-series models.

Exposes cross-validation splitters that respect the temporal structure of
overlapping-window financial targets.
"""

from src.evaluation.purged_cv import (
    PurgedWalkForward,
    embargo_mask,
    purge_train_indices,
)

__all__ = [
    "PurgedWalkForward",
    "embargo_mask",
    "purge_train_indices",
]
