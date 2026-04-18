"""Tests for the Estrella-Mishkin yield-curve probit baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recession_probit import RecessionProbit


def _monthly_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


@pytest.fixture
def synthetic_yield_data():
    """Create synthetic yield spread + NBER labels.

    Spread is negative during recessions (inversion) and positive
    during expansions, with noise.
    """
    rng = np.random.default_rng(42)
    idx = _monthly_index("1990-01-31", 360)
    nber = pd.Series(0, index=idx)
    # Synthetic recession windows
    nber.iloc[24:36] = 1    # 1992
    nber.iloc[132:144] = 1  # 2001
    nber.iloc[216:234] = 1  # 2008-2009
    nber.iloc[360 - 6:360 - 4] = 1  # 2020 (short)

    # Spread: negative before/during recessions, positive otherwise
    spread = pd.Series(1.5 + rng.standard_normal(360) * 0.5, index=idx)
    for start, end in [(12, 36), (120, 144), (204, 234), (354, 360)]:
        spread.iloc[start:end] = -0.5 + rng.standard_normal(end - start) * 0.3

    return spread, nber


def test_estrella_mishkin_fits_and_predicts(synthetic_yield_data):
    """The classmethod should return a fitted model that can predict."""
    spread, nber = synthetic_yield_data
    model = RecessionProbit.estrella_mishkin(
        spread, nber, horizon=12, regularization=0.01,
    )
    assert model._is_fitted
    proba = model.predict_proba(spread.to_frame("spread"))
    assert len(proba) == len(spread)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_estrella_mishkin_negative_spread_higher_prob(synthetic_yield_data):
    """Inverted yield curve should predict higher recession probability."""
    spread, nber = synthetic_yield_data
    model = RecessionProbit.estrella_mishkin(spread, nber, horizon=12)
    inverted = pd.DataFrame({"spread": [-2.0]})
    normal = pd.DataFrame({"spread": [2.0]})
    p_inverted = model.predict_proba(inverted)[0]
    p_normal = model.predict_proba(normal)[0]
    assert p_inverted > p_normal


def test_estrella_mishkin_horizon_respected(synthetic_yield_data):
    """Different horizons should produce different models."""
    spread, nber = synthetic_yield_data
    model_6m = RecessionProbit.estrella_mishkin(spread, nber, horizon=6)
    model_12m = RecessionProbit.estrella_mishkin(spread, nber, horizon=12)
    coef_6 = model_6m.get_coefficients()
    coef_12 = model_12m.get_coefficients()
    # The coefficients should differ
    assert coef_6 != coef_12


def test_estrella_mishkin_class_balanced(synthetic_yield_data):
    """Model should use class balancing by default."""
    spread, nber = synthetic_yield_data
    model = RecessionProbit.estrella_mishkin(spread, nber, horizon=12)
    assert model.class_balanced is True


def test_estrella_mishkin_too_few_obs_raises():
    """Should raise if not enough valid training observations."""
    idx = _monthly_index("2020-01-31", 10)
    spread = pd.Series(1.0, index=idx)
    nber = pd.Series(0, index=idx)
    with pytest.raises(ValueError, match="Too few"):
        RecessionProbit.estrella_mishkin(spread, nber, horizon=12)
