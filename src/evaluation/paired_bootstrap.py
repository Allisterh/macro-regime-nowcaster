"""Paired block bootstrap for comparing two signals on the same labels.

"Does A beat B?" on eight recessions is a question about eight episodes,
not about 700 months: adjacent months share a state, so resampling months
independently manufactures confidence. This resamples whole contiguous
runs of the label — each recession and each expansion is one block — and
recomputes both signals' scores on every draw, so the interval on their
*difference* reflects how few independent episodes there are.

Used by ``scripts/measure_recession.py`` and ``scripts/measure_spf.py``;
one implementation, so the two scripts cannot disagree about method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedDifference:
    """``metric(a) - metric(b)`` with a block-bootstrap interval."""

    point: float
    lo: float
    hi: float
    p_positive: float   # share of draws with a > b on the metric
    p_negative: float
    draws: int

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def paired_block_bootstrap(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    blocks: list[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedDifference:
    """Difference in ``metric`` between signals ``a`` and ``b``.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    a, b : np.ndarray
        Scores for the same observations as ``y``. Rows where either is
        non-finite are dropped from every draw, so both are always scored
        on identical observations.
    blocks : list[np.ndarray]
        Positional index arrays, one per contiguous label run
        (see :func:`src.evaluation.horizon_curve._label_blocks`).
    metric : callable
        ``metric(labels, scores) -> float``, e.g. ``roc_auc`` or
        ``brier_score``.
    """
    y = np.asarray(y)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)

    point = metric(y[ok], a[ok]) - metric(y[ok], b[ok])

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        idx = idx[ok[idx]]
        yb = y[idx]
        if yb.size == 0 or yb.min() == yb.max():
            continue
        d = metric(yb, a[idx]) - metric(yb, b[idx])
        if np.isfinite(d):
            diffs.append(d)

    diffs = np.asarray(diffs)
    if diffs.size < 20:
        nan = float("nan")
        return PairedDifference(point, nan, nan, nan, nan, int(diffs.size))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return PairedDifference(
        point=float(point),
        lo=float(lo),
        hi=float(hi),
        p_positive=float((diffs > 0).mean()),
        p_negative=float((diffs < 0).mean()),
        draws=int(diffs.size),
    )
