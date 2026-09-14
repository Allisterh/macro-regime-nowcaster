"""Purged and embargoed cross-validation for overlapping financial targets.

Standard k-fold assumes samples are independent.  Financial targets break
that in two ways at once:

1. **Overlapping labels.** A target like "return over the next 3 months"
   observed at *t* is computed from prices in ``[t, t+3m]``.  A training
   sample at ``t-1m`` has a label window that overlaps it, so the two
   share outcome data.  Put one in train and the other in test and the
   model has partly seen its own answer.

2. **Serial correlation.** Macro state variables move slowly.  A training
   sample taken immediately after a test fold is nearly a copy of the
   last test sample, which inflates scores even without overlapping
   labels.

López de Prado (*Advances in Financial Machine Learning*, ch. 7) handles
these with **purging** — dropping training samples whose label window
intersects the test window — and an **embargo** — additionally dropping
training samples for a short period after the test fold.

This module implements the walk-forward variant, which is the right shape
here: folds are contiguous and always trained on the past, matching how
the nowcaster itself is refit.

Usage
-----
>>> cv = PurgedWalkForward(n_splits=5, label_horizon=3, embargo=3)
>>> for train_idx, test_idx in cv.split(feature_index):
...     ...
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


def purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_horizon: int,
) -> np.ndarray:
    """Drop training samples whose label window overlaps the test window.

    A sample at position *i* carries a label computed over
    ``[i, i + label_horizon]``.  It is purged when that interval
    intersects the closed interval spanned by the test fold's own label
    windows.

    Parameters
    ----------
    train_idx, test_idx : np.ndarray
        Positional indices into the sample sequence.
    label_horizon : int
        Number of periods the label looks forward.  ``0`` means the label
        is known at the sample's own timestamp and nothing is purged.

    Returns
    -------
    np.ndarray
        The surviving training indices, in order.
    """
    if len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx
    if label_horizon <= 0:
        return train_idx

    test_start = int(np.min(test_idx))
    test_end = int(np.max(test_idx)) + label_horizon

    # A training sample survives only if its label window [i, i+h] lies
    # entirely before the test window starts or entirely after it ends.
    starts = train_idx
    ends = train_idx + label_horizon
    overlaps = (starts <= test_end) & (ends >= test_start)
    return train_idx[~overlaps]


def embargo_mask(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Drop training samples falling within *embargo* periods after the test fold.

    Purging removes label overlap; the embargo removes the residual
    advantage that serial correlation gives a training sample sitting
    immediately after the test window.

    Returns
    -------
    np.ndarray
        The surviving training indices, in order.
    """
    if embargo <= 0 or len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx
    test_end = int(np.max(test_idx))
    banned = (train_idx > test_end) & (train_idx <= test_end + embargo)
    return train_idx[~banned]


@dataclass
class PurgedWalkForward:
    """Expanding-window splitter with purging and an embargo.

    Fold *k* trains on everything before the fold's test block (minus
    purged and embargoed samples) and tests on that block.  Training
    never extends past the test block, so this is strictly causal —
    unlike purged k-fold, which trains on future folds too.

    Parameters
    ----------
    n_splits : int
        Number of test blocks.
    label_horizon : int
        Periods the target looks forward.  Samples whose label window
        overlaps the test window are purged from training.
    embargo : int
        Periods after each test block to exclude from training.
    min_train : int
        Minimum training samples required for a fold to be yielded.
        Folds with fewer are skipped rather than fitted on too little.
    """

    n_splits: int = 5
    label_horizon: int = 1
    embargo: int = 0
    min_train: int = 60

    def split(
        self, X: pd.DataFrame | pd.Series | pd.Index | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` positional index pairs."""
        n = len(X)
        if n == 0:
            return

        # Contiguous, near-equal test blocks over the tail of the sample.
        # The first block starts after min_train so that fold 1 has
        # something to learn from.
        first = max(self.min_train, n // (self.n_splits + 1))
        if first >= n:
            return
        edges = np.linspace(first, n, self.n_splits + 1, dtype=int)

        for k in range(self.n_splits):
            lo, hi = int(edges[k]), int(edges[k + 1])
            if hi <= lo:
                continue
            test_idx = np.arange(lo, hi)
            # Expanding window: everything strictly before the test block.
            train_idx = np.arange(0, lo)
            train_idx = purge_train_indices(
                train_idx, test_idx, self.label_horizon
            )
            train_idx = embargo_mask(train_idx, test_idx, self.embargo)
            if len(train_idx) < self.min_train:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds actually yielded for *X*."""
        if X is None:
            return self.n_splits
        return sum(1 for _ in self.split(X))
