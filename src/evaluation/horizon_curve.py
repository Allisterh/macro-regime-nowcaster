"""Discrimination as a function of forecast horizon, with uncertainty.

A recession signal that scores well at horizon 0 has told you the economy
is weak *now*.  Whether it tells you anything about six or twelve months
ahead is a separate question, and the two can point in opposite
directions: a coincident index like CFNAI is close to unbeatable at h=0
and falls below chance by h=18, while a yield-curve probit is mediocre at
h=0 and still informative at h=18.

This module computes AUC against a recession *h* periods ahead, for a set
of candidate signals, with block-bootstrap confidence intervals.

The intervals matter more than the point estimates.  With four recessions
in the post-1990 sample, differences of a few AUC points are noise, and a
plot without error bars invites exactly the over-reading this project has
already been guilty of once.  Resampling is over contiguous runs of the
label rather than over individual months.  The reason is not clustering
alone but that performance *varies by episode*: on this project's panel
the model calls 2001 perfectly and misses 2020 entirely, so which
recessions a resample draws matters enormously.  Measured there, the
block interval is 0.232 wide against 0.069 for an i.i.d. bootstrap —
the naive interval is roughly three times too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.regime_backtest import roc_auc


@dataclass
class HorizonPoint:
    """AUC for one signal at one horizon, with a bootstrap interval."""

    signal: str
    horizon: int
    auc: float
    lo: float
    hi: float
    n_obs: int
    n_positive: int

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "horizon": self.horizon,
            "auc": self.auc,
            "lo": self.lo,
            "hi": self.hi,
            "n_obs": self.n_obs,
            "n_positive": self.n_positive,
        }


@dataclass
class HorizonCurve:
    """Discrimination-vs-horizon for several signals."""

    points: list[HorizonPoint] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Long-format table: one row per (signal, horizon)."""
        return pd.DataFrame([p.as_dict() for p in self.points])

    def pivot(self, value: str = "auc") -> pd.DataFrame:
        """Wide table indexed by horizon, one column per signal."""
        frame = self.to_frame()
        if frame.empty:
            return frame
        return frame.pivot(index="horizon", columns="signal", values=value)


def _label_blocks(y: pd.Series) -> list[np.ndarray]:
    """Contiguous runs of identical label, as positional index arrays.

    Resampling these rather than individual observations preserves the
    clustering of recession months, which is what makes the naive
    interval far too narrow.
    """
    runs = (y.diff() != 0).cumsum()
    return [np.where(runs == r)[0] for r in runs.unique()]


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    blocks: list[np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and block-bootstrap interval for AUC.

    Returns
    -------
    (auc, lo, hi)
        ``nan`` bounds when the resample degenerates (for example a draw
        containing only one class).
    """
    point = roc_auc(y_true, scores)
    if not np.isfinite(point) or not blocks:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = []
    n_blocks = len(blocks)
    for _ in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([blocks[i] for i in pick])
        yb = y_true[idx]
        if yb.min() == yb.max():  # single-class resample
            continue
        val = roc_auc(yb, scores[idx])
        if np.isfinite(val):
            draws.append(val)

    if len(draws) < 20:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def horizon_auc_curve(
    features: pd.DataFrame,
    labels: pd.Series,
    signals: dict[str, str] | list[str],
    horizons: tuple[int, ...] = (0, 3, 6, 9, 12, 18),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> HorizonCurve:
    """AUC against a recession *h* periods ahead, per signal, with intervals.

    Parameters
    ----------
    features : pd.DataFrame
        Point-in-time panel; its index must align with *labels*.
    labels : pd.Series
        Binary recession indicator on the same index.
    signals : dict[str, str] | list[str]
        Either ``{display_name: column}`` or a list of column names.
    horizons : tuple[int, ...]
        Forecast horizons in periods (months, for a monthly panel).

    Returns
    -------
    HorizonCurve
    """
    if not isinstance(signals, dict):
        signals = {name: name for name in signals}

    missing = [c for c in signals.values() if c not in features.columns]
    if missing:
        raise KeyError(f"signals not present in the panel: {missing}")

    y_full = labels.reindex(features.index)
    curve = HorizonCurve()

    for horizon in horizons:
        # Target: was there a recession h periods after this observation?
        shifted = y_full.shift(-horizon).dropna()
        if shifted.nunique() < 2:
            continue
        blocks = _label_blocks(shifted)
        y_arr = shifted.to_numpy(dtype=float)

        for display, column in signals.items():
            scores = features[column].reindex(shifted.index)
            ok = scores.notna().to_numpy()
            if ok.sum() < 20 or len(set(y_arr[ok])) < 2:
                continue
            # Blocks index into the full shifted series; restrict them to
            # the rows this particular signal actually has.
            keep = np.where(ok)[0]
            remap = {orig: i for i, orig in enumerate(keep)}
            sig_blocks = [
                np.array([remap[o] for o in b if o in remap], dtype=int)
                for b in blocks
            ]
            sig_blocks = [b for b in sig_blocks if len(b)]

            auc, lo, hi = bootstrap_auc(
                y_arr[ok], scores.to_numpy(dtype=float)[ok],
                sig_blocks, n_boot=n_boot, alpha=alpha, seed=seed,
            )
            curve.points.append(
                HorizonPoint(
                    signal=display,
                    horizon=horizon,
                    auc=float(auc),
                    lo=float(lo),
                    hi=float(hi),
                    n_obs=int(ok.sum()),
                    n_positive=int(y_arr[ok].sum()),
                )
            )

    return curve
