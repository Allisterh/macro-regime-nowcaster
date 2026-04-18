"""Regression tests for expanding-window standardization.

Verifies that the standardised value at time *t* does not change when
additional observations are appended after *t*.  This is the core
look-ahead-freeness property that the data pipeline must satisfy after
the audit fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.transformations import (
    standardize,
    standardize_expanding,
)


def _series(values):
    idx = pd.date_range("2010-01-31", periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


def test_expanding_z_does_not_change_when_future_appended():
    """Extending the series must not alter past z-scores."""
    vals_short = np.linspace(0.0, 10.0, 60)
    short = _series(vals_short)

    # Append more data AFTER index 60.
    vals_long = np.concatenate([vals_short, np.linspace(10.1, 20.0, 60)])
    long = _series(vals_long)

    z_short = standardize_expanding(short, min_periods=24)
    z_long = standardize_expanding(long, min_periods=24).iloc[: len(short)]

    # Every past z-score must be identical whether or not future data exists.
    np.testing.assert_allclose(
        z_short.dropna().values,
        z_long.dropna().values,
        rtol=0, atol=0,
    )


def test_full_sample_standardize_LEAKS_future():
    """Sanity check: the old full-sample function DOES change past values."""
    vals_short = np.linspace(0.0, 10.0, 60)
    short = _series(vals_short)
    vals_long = np.concatenate([vals_short, np.linspace(10.1, 100.0, 60)])
    long = _series(vals_long)

    z_short = standardize(short).dropna()
    z_long = standardize(long).iloc[: len(short)].dropna()

    # They must differ — otherwise our expanding-vs-full distinction
    # is vacuous and this regression suite is not measuring anything.
    assert not np.allclose(z_short.values, z_long.values)


def test_expanding_z_returns_nan_before_min_periods():
    s = _series(np.arange(50.0))
    z = standardize_expanding(s, min_periods=24)
    # First 23 observations must be NaN — the expanding std is undefined
    # (or defined off too few points to be trustworthy) below min_periods.
    assert z.iloc[:23].isna().all()
    assert z.iloc[23:].notna().all()


def test_expanding_z_has_roughly_zero_mean_on_large_sample():
    """Over a long stationary series, the expanding z-score should
    approximate the running z-score and therefore average near zero."""
    rng = np.random.default_rng(0)
    s = _series(rng.standard_normal(500))
    z = standardize_expanding(s, min_periods=24).dropna()
    # Not exactly zero because the expanding mean shifts each step,
    # but the full-series average should be small.
    assert abs(z.mean()) < 0.3


def test_expanding_z_handles_nan_gaps():
    """NaN observations must not shift the scale or corrupt later values."""
    vals = np.arange(60.0)
    s = _series(vals)
    s.iloc[[10, 20, 30]] = np.nan

    z = standardize_expanding(s, min_periods=24)
    # Non-NaN entries remain non-NaN (once past min_periods) and the
    # explicitly-NaN positions stay NaN.
    assert z.iloc[[10, 20, 30]].isna().all()
    # With NaN at indices 10 and 20, the 24th valid observation arrives at
    # index 25 (26 total minus 2 NaN = 24 valid), so index 25 onward is valid.
    assert z.iloc[25:].drop(index=s.index[[30]]).notna().all()
