"""Tests for the Sahm rule recession indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.sahm_rule import (
    sahm_indicator,
    sahm_recession_probability,
    sahm_recession_signal,
)


def _monthly_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def test_sahm_indicator_basic():
    """Indicator should be zero when UNRATE is constant."""
    idx = _monthly_index("2015-01-31", 36)
    unrate = pd.Series(5.0, index=idx)
    indicator = sahm_indicator(unrate)
    assert (indicator.dropna() == 0.0).all()


def test_sahm_triggers_on_half_point_rise():
    """A 0.5pp rise from the 12-month min must trigger the signal."""
    idx = _monthly_index("2019-01-31", 24)
    vals = np.full(24, 3.5)
    # Sudden rise in the last 3 months
    vals[-3:] = 4.1
    unrate = pd.Series(vals, index=idx)
    signal = sahm_recession_signal(unrate, threshold=0.50)
    assert signal.iloc[-1] == 1


def test_sahm_does_not_trigger_small_rise():
    """A rise of less than 0.5pp should not trigger."""
    idx = _monthly_index("2019-01-31", 24)
    vals = np.full(24, 3.5)
    vals[-3:] = 3.8  # only +0.3pp
    unrate = pd.Series(vals, index=idx)
    signal = sahm_recession_signal(unrate, threshold=0.50)
    assert signal.iloc[-1] == 0


def test_sahm_probability_range():
    """Probability must always be in [0, 1]."""
    rng = np.random.default_rng(42)
    idx = _monthly_index("2000-01-31", 120)
    unrate = pd.Series(5.0 + rng.standard_normal(120).cumsum() * 0.1, index=idx)
    prob = sahm_recession_probability(unrate)
    valid = prob.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_sahm_fires_on_synthetic_recession():
    """Simulate a recession-like UNRATE path and verify the signal fires."""
    idx = _monthly_index("2018-01-31", 36)
    vals = np.full(36, 3.5)
    # Gradual rise starting month 24 (mimicking a real recession)
    vals[24:] = 3.5 + np.linspace(0.0, 1.5, 12)
    unrate = pd.Series(vals, index=idx)
    signal = sahm_recession_signal(unrate)
    # Signal should fire by the end
    assert signal.iloc[-6:].sum() >= 3


def test_sahm_indicator_matches_known_values():
    """The indicator should equal MA3(UNRATE) - min12(UNRATE)."""
    idx = _monthly_index("2010-01-31", 60)
    rng = np.random.default_rng(123)
    unrate = pd.Series(5.0 + rng.standard_normal(60).cumsum() * 0.05, index=idx)
    indicator = sahm_indicator(unrate)
    ma3 = unrate.rolling(3, min_periods=2).mean()
    min12 = unrate.rolling(12, min_periods=6).min()
    expected = ma3 - min12
    pd.testing.assert_series_equal(indicator, expected)
