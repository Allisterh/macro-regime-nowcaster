"""Extreme observations must degrade the fit, not abort it.

Panels spanning 2020 carry 20-sigma readings in the claims and payrolls
series. That drove the DFM's EM to diverge, producing NaN loadings, and
the failure only surfaced later inside varimax's SVD as "SVD did not
converge" — 40 of 710 windows in an extended walk-forward, every one of
them in 2020 or later. The failures therefore clustered on exactly the
dates a live dashboard would be asking about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dynamic_factor_model import (
    DynamicFactorModel,
    _params_are_sane,
    _varimax,
)
from src.models.kalman_filter import _robust_inverse

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_varimax_refuses_non_finite_loadings():
    """Non-finite input returns unrotated rather than failing inside an SVD."""
    loadings = np.array([[1.0, 2.0], [np.nan, 0.5], [3.0, np.inf]])
    rotated, rotation = _varimax(loadings)
    np.testing.assert_array_equal(rotated, loadings)
    np.testing.assert_allclose(rotation, np.eye(2))


def test_params_are_sane_rejects_overflowed_but_finite_values():
    """Finiteness alone does not detect a diverged EM.

    This is the check the original guard was missing. Overflow climbs to
    ~1e300 while ``np.isfinite`` stays True, so the "last finite
    estimate" the guard fell back to was already garbage: R so large the
    Kalman gain underflowed, the state pinned at its zero
    initialisation, and 808 months of factors all exactly 0.0.
    """
    runaway = np.array([[1e300, 0.0], [0.0, 1e300]])
    assert np.isfinite(runaway).all(), "precondition: the old guard passed this"
    assert not _params_are_sane(runaway)


def test_params_are_sane_accepts_ordinary_parameters():
    """Parameters on a standardised panel are O(1) and must not trip it."""
    A = np.array([[0.85, 0.0], [0.0, 0.79]])
    C = np.random.default_rng(0).standard_normal((10, 2))
    Q = np.eye(2) * 0.3
    R = np.eye(10) * 0.4
    assert _params_are_sane(A, C, Q, R)


def test_params_are_sane_rejects_non_finite():
    assert not _params_are_sane(np.array([[1.0, np.nan]]))
    assert not _params_are_sane(np.array([[1.0, np.inf]]))


def test_robust_inverse_handles_a_singular_matrix():
    """A singular innovation covariance must still yield a usable inverse."""
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    inv = _robust_inverse(singular)
    assert np.isfinite(inv).all()
    assert inv.shape == (2, 2)


def test_robust_inverse_matches_pinv_when_well_conditioned():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    well = a @ a.T + np.eye(5) * 2.0
    np.testing.assert_allclose(
        _robust_inverse(well), np.linalg.pinv(well), rtol=1e-8, atol=1e-10
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _panel_with_outlier_shock(n: int = 240, shock: float = 25.0) -> pd.DataFrame:
    """Factor panel with a COVID-scale spike in a few series."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    factor = np.zeros(n)
    for t in range(1, n):
        factor[t] = 0.9 * factor[t - 1] + rng.standard_normal() * 0.3
    loadings = rng.standard_normal((10, 1)) * 0.2 + 1.1
    obs = factor[:, None] @ loadings.T + rng.standard_normal((n, 10)) * 0.4

    # Two months where several series move by twenty-plus sigma.
    obs[n - 40 : n - 38, :4] += shock
    obs[n - 38 : n - 36, :4] -= shock * 0.8
    return pd.DataFrame(obs, index=idx, columns=[f"X{i}" for i in range(10)])


@pytest.mark.slow
def test_dfm_survives_an_extreme_shock():
    """The fit must complete and return finite factors."""
    panel = _panel_with_outlier_shock()
    dfm = DynamicFactorModel(n_factors=2, max_iter=100).fit(panel)

    assert dfm.factors_ is not None
    assert np.isfinite(dfm.factors_.values).all(), "factors contain NaN/inf"
    assert np.isfinite(dfm.get_loadings()).all(), "loadings contain NaN/inf"


@pytest.mark.slow
def test_diverging_em_keeps_the_last_finite_parameters():
    """A shock large enough to diverge must not yield NaN parameters."""
    panel = _panel_with_outlier_shock(shock=5_000.0)
    dfm = DynamicFactorModel(n_factors=2, max_iter=100).fit(panel)

    for name in ("_A", "_Q", "_R"):
        matrix = getattr(dfm, name)
        assert matrix is not None and np.isfinite(matrix).all(), (
            f"{name} is non-finite after an extreme shock"
        )
    assert np.isfinite(dfm.get_loadings()).all()

    # Finite is not enough: the fit must also have produced factors that
    # vary. A diverged EM used to return every factor identically zero,
    # which is finite, correctly shaped and completely uninformative.
    sd = dfm.factors_.std().to_numpy()
    assert float(np.max(sd)) > 1e-12, (
        f"factors are constant after an extreme shock (sd = {sd}); the fit "
        f"degenerated but reported success"
    )
