"""Evaluation utilities for time-series models.

Exposes cross-validation splitters that respect the temporal structure of
overlapping-window financial targets, and discrimination-vs-horizon
curves with block-bootstrap confidence intervals.
"""

from src.evaluation.horizon_curve import (
    HorizonCurve,
    HorizonPoint,
    bootstrap_auc,
    horizon_auc_curve,
)
from src.evaluation.purged_cv import (
    PurgedWalkForward,
    embargo_mask,
    purge_train_indices,
)

__all__ = [
    "HorizonCurve",
    "HorizonPoint",
    "PurgedWalkForward",
    "bootstrap_auc",
    "embargo_mask",
    "horizon_auc_curve",
    "purge_train_indices",
]
