"""Tests for discrimination-vs-horizon curves and their intervals.

The point of this module is to stop horizon comparisons being read too
confidently, so the tests check that the intervals behave: that they
widen as evidence thins, that a block bootstrap is wider than an i.i.d.
one on clustered labels, and that a signal with a known lead is actually
ranked ahead of a coincident one at the horizon where it should be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.horizon_curve import (
    HorizonCurve,
    bootstrap_auc,
    horizon_auc_curve,
)


@pytest.fixture()
def leading_and_coincident() -> tuple[pd.DataFrame, pd.Series]:
    """A coincident signal and one that leads by six months.

    ``coincident`` sees the recession only while it is happening;
    ``leading`` sees it six months early and is blind to the present.
    """
    rng = np.random.default_rng(3)
    n = 480
    idx = pd.date_range("1980-01-31", periods=n, freq="ME")

    labels = np.zeros(n, dtype=int)
    for start in (90, 200, 310, 420):
        labels[start : start + 10] = 1
    y = pd.Series(labels, index=idx)

    coincident = y.astype(float) + rng.standard_normal(n) * 0.35
    leading = y.shift(-6).fillna(0).astype(float) + rng.standard_normal(n) * 0.35

    features = pd.DataFrame(
        {"coincident": coincident, "leading": leading}, index=idx
    )
    return features, y


def test_coincident_signal_peaks_at_horizon_zero(leading_and_coincident):
    features, y = leading_and_coincident
    curve = horizon_auc_curve(
        features, y, {"coincident": "coincident"},
        horizons=(0, 6, 12), n_boot=300,
    )
    aucs = curve.pivot("auc")["coincident"]
    assert aucs.loc[0] > aucs.loc[6] > aucs.loc[12], (
        f"a coincident signal should decay with horizon, got\n{aucs}"
    )


def test_leading_signal_peaks_at_its_lead(leading_and_coincident):
    """The signal built to lead by six months must win at six months."""
    features, y = leading_and_coincident
    curve = horizon_auc_curve(
        features, y, {"leading": "leading", "coincident": "coincident"},
        horizons=(0, 6), n_boot=300,
    )
    wide = curve.pivot("auc")
    assert wide.loc[0, "coincident"] > wide.loc[0, "leading"]
    assert wide.loc[6, "leading"] > wide.loc[6, "coincident"], (
        f"the six-month-lead signal did not win at h=6:\n{wide}"
    )


def test_intervals_bracket_the_point_estimate(leading_and_coincident):
    features, y = leading_and_coincident
    curve = horizon_auc_curve(
        features, y, ["coincident"], horizons=(0, 6), n_boot=400,
    )
    for point in curve.points:
        assert point.lo <= point.auc <= point.hi, (
            f"{point.signal} at h={point.horizon}: {point.auc:.3f} outside "
            f"[{point.lo:.3f}, {point.hi:.3f}]"
        )


def test_block_bootstrap_is_wider_when_episodes_differ():
    """An i.i.d. bootstrap understates uncertainty across episodes.

    What makes the naive interval too narrow is not clustering alone: it
    is that performance *varies by episode*. On the real panel the model
    calls 2001 perfectly and misses 2020 entirely, so which recessions a
    resample happens to draw matters enormously — measured there, the
    block interval is 0.232 wide against 0.069 for i.i.d.

    Simulating episodes of *equal* difficulty does not reproduce this,
    which is worth knowing: the fixture has to vary the signal quality
    between recessions, as reality does.
    """
    from src.evaluation.horizon_curve import _label_blocks

    rng = np.random.default_rng(11)
    n = 400
    labels = np.zeros(n, dtype=int)
    episodes = [(80, 20), (190, 20), (300, 20)]
    for start, length in episodes:
        labels[start : start + length] = 1
    y = pd.Series(labels)

    # Baseline noise, then per-episode signal quality: the first is seen
    # clearly, the second barely, the third not at all.
    scores = rng.standard_normal(n) * 0.5
    for (start, length), strength in zip(episodes, [2.5, 0.6, -0.4]):
        scores[start : start + length] += strength

    blocks = _label_blocks(y)
    _, blo, bhi = bootstrap_auc(
        y.to_numpy(float), scores, blocks, n_boot=1500, seed=1
    )
    iid_blocks = [np.array([i]) for i in range(n)]
    _, ilo, ihi = bootstrap_auc(
        y.to_numpy(float), scores, iid_blocks, n_boot=1500, seed=1
    )

    assert (bhi - blo) > (ihi - ilo), (
        f"block interval {bhi - blo:.3f} is not wider than the i.i.d. "
        f"interval {ihi - ilo:.3f} despite episodes of differing difficulty"
    )


def test_intervals_widen_as_the_horizon_lengthens(leading_and_coincident):
    """Less signal further out should show up as a wider band."""
    features, y = leading_and_coincident
    curve = horizon_auc_curve(
        features, y, ["coincident"], horizons=(0, 12), n_boot=600,
    )
    frame = curve.to_frame().set_index("horizon")
    width = frame["hi"] - frame["lo"]
    assert width.loc[12] > width.loc[0]


def test_missing_signal_column_raises(leading_and_coincident):
    features, y = leading_and_coincident
    with pytest.raises(KeyError, match="not present"):
        horizon_auc_curve(features, y, ["does_not_exist"], n_boot=50)


def test_pivot_and_frame_shapes(leading_and_coincident):
    features, y = leading_and_coincident
    horizons = (0, 6, 12)
    curve = horizon_auc_curve(
        features, y, ["coincident", "leading"], horizons=horizons, n_boot=100,
    )
    frame = curve.to_frame()
    assert set(frame.columns) >= {"signal", "horizon", "auc", "lo", "hi"}
    assert len(frame) == len(horizons) * 2
    assert curve.pivot("auc").shape == (len(horizons), 2)


def test_empty_curve_pivots_without_error():
    assert HorizonCurve().to_frame().empty
