"""Shared pytest fixtures for all test modules.

Provides:
- Synthetic panel data with a known 2-factor structure
- Pre-built Kalman filter with known analytic solution
- Mock regime probability series
- Sample NowcastResult
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from src.models.kalman_filter import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from src.models.nowcaster import NowcastResult
except ImportError:
    NowcastResult = None


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
# Every fixture draws from its OWN generator, seeded per fixture.
#
# There used to be one shared ``RNG = default_rng(42)`` and every
# session-scoped fixture pulled from it. Session fixtures are built
# lazily, in the order tests first ask for them, so a fixture's contents
# depended on which tests were selected and how many draws they had
# already consumed. ``synthetic_panel`` was one panel under
# `pytest tests/` and a different one under
# `pytest tests/test_dynamic_factor_model.py`, and
# test_dfm_rotated_factors_are_distinct passed in the first case and
# failed at |r| = 0.785 in the second. A green full-suite run was
# therefore partly a statement about collection order.
def _rng(seed: int) -> np.random.Generator:
    """A generator private to one fixture, so order cannot reach it."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Return a (120 × 10) panel with a known 2-factor structure.

    Y_t = C @ F_t + noise,  F_t = 0.8 * F_{t-1} + η_t
    """
    rng = _rng(42)
    T, N, K = 120, 10, 2
    dates = pd.date_range("2010-01-31", periods=T, freq="ME")

    # True loading matrix
    C_true = rng.standard_normal((N, K))

    # True factor path
    factors = np.zeros((T, K))
    for t in range(1, T):
        factors[t] = 0.8 * factors[t - 1] + rng.standard_normal(K) * 0.3

    # Observations with noise
    noise = rng.standard_normal((T, N)) * 0.5
    Y = factors @ C_true.T + noise

    df = pd.DataFrame(Y, index=dates, columns=[f"series_{i}" for i in range(N)])
    return df


@pytest.fixture(scope="session")
def synthetic_panel_with_nans(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Return the synthetic panel with ~10% random NaN values (ragged edge)."""
    rng = _rng(43)
    df = synthetic_panel.copy()
    mask = rng.random(df.shape) < 0.10
    df[mask] = np.nan
    return df


@pytest.fixture(scope="session")
def local_level_kf():
    """Return a Kalman filter configured as a local level model (scalar).

    State:  μ_t = μ_{t-1} + η_t,  η_t ~ N(0, σ²_η)
    Obs:    Y_t = μ_t + ε_t,       ε_t ~ N(0, σ²_ε)
    """
    if KalmanFilter is None:
        pytest.skip("src.models.kalman_filter not available")
    sigma_eta = 1.0
    sigma_eps = 2.0
    A = np.array([[1.0]])
    C = np.array([[1.0]])
    Q = np.array([[sigma_eta**2]])
    R = np.array([[sigma_eps**2]])
    return KalmanFilter(
        A=A, C=C, Q=Q, R=R,
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[10.0]]),
    )


@pytest.fixture(scope="session")
def local_level_observations() -> np.ndarray:
    """Return 50 observations from the local level model."""
    rng = _rng(44)
    T = 50
    mu = np.zeros(T)
    for t in range(1, T):
        mu[t] = mu[t - 1] + rng.normal(0, 1.0)
    y = mu + rng.normal(0, 2.0, T)
    return y.reshape(-1, 1)


@pytest.fixture(scope="session")
def synthetic_regime_series() -> pd.Series:
    """Return a scalar time series with two clear regime changes."""
    rng = _rng(45)
    T = 100
    dates = pd.date_range("2010-01-31", periods=T, freq="ME")
    y = np.concatenate([
        rng.normal(1.5, 0.3, 30),   # regime 0: expansion (high mean)
        rng.normal(-0.5, 0.3, 20),  # regime 1: recession (low mean)
        rng.normal(1.0, 0.3, 30),   # regime 0: recovery
        rng.normal(-0.3, 0.3, 20),  # regime 1: slowdown
    ])
    return pd.Series(y, index=dates, name="composite")


@pytest.fixture(scope="session")
def sample_nowcast_result():
    """Return a fixed NowcastResult for testing downstream consumers."""
    if NowcastResult is None:
        pytest.skip("src.models.nowcaster not available")
    return NowcastResult(
        gdp_nowcast=2.4,
        gdp_ci_lower=0.9,
        gdp_ci_upper=3.9,
        regime_probabilities={
            "expansion": 0.85,
            "recession": 0.15,
        },
        current_regime="expansion",
        factor_values={
            "real_activity": 0.8,
            "labor_market": 0.5,
            "inflation": -0.2,
            "financial_conditions": 0.3,
        },
        recession_probability=0.15,
        ensemble_detail={
            "rsm": 0.20,
            "probit": 0.10,
            "cfnai": 0.18,
            "ensemble": 0.15,
        },
    )


@pytest.fixture(scope="session")
def regime_probs_df() -> pd.DataFrame:
    """Return a (60 × 4) DataFrame of synthetic regime probabilities."""
    rng = _rng(46)
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    raw = rng.dirichlet(alpha=[4, 1], size=60)
    return pd.DataFrame(raw, index=dates, columns=["expansion", "recession"])


@pytest.fixture(scope="session")
def asset_returns_df(regime_probs_df: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic monthly asset returns aligned to regime_probs_df."""
    rng = _rng(47)
    T = len(regime_probs_df)
    data = {
        "equities":    rng.normal(0.008, 0.04, T),
        "bonds":       rng.normal(0.003, 0.015, T),
        "commodities": rng.normal(0.005, 0.05, T),
        "cash":        rng.normal(0.001, 0.001, T),
    }
    return pd.DataFrame(data, index=regime_probs_df.index)
