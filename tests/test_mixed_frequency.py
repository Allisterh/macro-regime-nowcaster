"""Tests for mixed-frequency panel construction and DFM cumulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.mixed_frequency import (
    align_mixed_frequency,
    augment_observation_matrix,
    build_cumulator_matrices,
)

# ---------------------------------------------------------------------------
# align_mixed_frequency
# ---------------------------------------------------------------------------


def test_align_daily_to_monthly_last():
    """Daily series should be collapsed to month-end via last obs."""
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    raw = {"SP500": pd.Series(np.arange(90, dtype=float), index=idx)}
    cfg = [{"code": "SP500", "frequency": "daily"}]
    panel = align_mixed_frequency(raw, cfg, method="last")
    assert panel.index.freq == "ME" or all(
        d.is_month_end for d in panel.index
    )
    assert len(panel) >= 2


def test_align_weekly_to_monthly_mean():
    """Weekly series with method='mean' should average within each month."""
    idx = pd.date_range("2020-01-01", periods=52, freq="W")
    vals = np.ones(52) * 10.0
    raw = {"ICSA": pd.Series(vals, index=idx)}
    cfg = [{"code": "ICSA", "frequency": "weekly"}]
    panel = align_mixed_frequency(raw, cfg, method="mean")
    assert np.allclose(panel["ICSA"].dropna().values, 10.0)


def test_align_quarterly_produces_nan_padding():
    """Quarterly series should have NaN for non-quarter-end months."""
    idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    raw = {"GDPC1": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)}
    cfg = [{"code": "GDPC1", "frequency": "quarterly"}]
    panel = align_mixed_frequency(raw, cfg, method="last")
    # Only quarter-end months should have values
    non_nan = panel["GDPC1"].dropna()
    assert len(non_nan) == 4
    for dt in non_nan.index:
        assert dt.month in (3, 6, 9, 12)


def test_align_mixed_frequencies_join():
    """Monthly + daily + quarterly should all join on monthly index."""
    monthly_idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    daily_idx = pd.date_range("2020-01-01", periods=365, freq="D")
    quarterly_idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    raw = {
        "PAYEMS": pd.Series(np.arange(12, dtype=float), index=monthly_idx),
        "SP500": pd.Series(np.arange(365, dtype=float), index=daily_idx),
        "GDPC1": pd.Series([1.0, 2.0, 3.0, 4.0], index=quarterly_idx),
    }
    cfg = [
        {"code": "PAYEMS", "frequency": "monthly"},
        {"code": "SP500", "frequency": "daily"},
        {"code": "GDPC1", "frequency": "quarterly"},
    ]
    panel = align_mixed_frequency(raw, cfg)
    assert "PAYEMS" in panel.columns
    assert "SP500" in panel.columns
    assert "GDPC1" in panel.columns
    # GDPC1 should be NaN for non-quarter months
    jan_val = panel.loc["2020-01-31", "GDPC1"] if "2020-01-31" in panel.index else np.nan
    assert np.isnan(jan_val) or jan_val is np.nan


# ---------------------------------------------------------------------------
# Cumulator matrices
# ---------------------------------------------------------------------------


def test_cumulator_augments_state_dim():
    """With quarterly columns present, state dim should be K + 2K."""
    result = build_cumulator_matrices(
        n_factors=3, quarterly_columns=[0, 2], n_series=5,
    )
    assert result["n_states_aug"] == 9  # 3 + 2*3


def test_cumulator_no_quarterly():
    """Without quarterly columns, state dim should equal K."""
    result = build_cumulator_matrices(
        n_factors=3, quarterly_columns=[], n_series=5,
    )
    assert result["n_states_aug"] == 3


def test_augmented_C_quarterly_loads_all_lags():
    """Quarterly series should load on f_t + f_{t-1} + f_{t-2}."""
    K = 2
    N = 4
    C = np.eye(N, K)
    quarterly_cols = [2]
    n_aug = K + 2 * K  # 6
    C_aug = augment_observation_matrix(C, quarterly_cols, n_aug, K)
    assert C_aug.shape == (N, n_aug)
    # Quarterly column (2) should have equal loadings on all 3 blocks
    np.testing.assert_array_equal(C_aug[2, :K], C[2, :K])
    np.testing.assert_array_equal(C_aug[2, K:2*K], C[2, :K])
    np.testing.assert_array_equal(C_aug[2, 2*K:3*K], C[2, :K])
    # Monthly column (0) should only load on first block
    np.testing.assert_array_equal(C_aug[0, K:], np.zeros(2 * K))


# ---------------------------------------------------------------------------
# DFM integration with quarterly columns
# ---------------------------------------------------------------------------


def test_dfm_fit_with_quarterly_column():
    """DFM should fit without error when quarterly columns are specified."""
    from src.models.dynamic_factor_model import DynamicFactorModel

    rng = np.random.default_rng(42)
    T = 120
    idx = pd.date_range("2010-01-31", periods=T, freq="ME")
    # 3 monthly series + 1 quarterly (NaN except quarter-ends)
    panel = pd.DataFrame({
        "A": rng.standard_normal(T),
        "B": rng.standard_normal(T),
        "C": rng.standard_normal(T),
        "Q": np.nan,
    }, index=idx)
    # Fill quarterly values
    for i in range(T):
        if idx[i].month in (3, 6, 9, 12):
            panel.iloc[i, 3] = rng.standard_normal()

    dfm = DynamicFactorModel(
        n_factors=2,
        max_iter=5,
        quarterly_columns=["Q"],
    )
    dfm.fit(panel)
    assert dfm._is_fitted
    assert dfm.factors_.shape == (T, 2)


def test_dfm_fit_without_quarterly_unchanged():
    """DFM without quarterly columns should work exactly as before."""
    from src.models.dynamic_factor_model import DynamicFactorModel

    rng = np.random.default_rng(42)
    T = 60
    idx = pd.date_range("2015-01-31", periods=T, freq="ME")
    panel = pd.DataFrame(
        rng.standard_normal((T, 4)),
        index=idx,
        columns=["A", "B", "C", "D"],
    )

    dfm = DynamicFactorModel(n_factors=2, max_iter=5)
    dfm.fit(panel)
    assert dfm._is_fitted
    assert dfm.factors_.shape == (T, 2)


# ---------------------------------------------------------------------------
# Empty series
# ---------------------------------------------------------------------------


def test_to_monthly_handles_an_empty_series():
    """An empty input is a normal condition, not an error.

    A backtest as of 1985 asks for series that do not start until 1990,
    and FRED correctly returns nothing. Raising here took down the entire
    as-of date, which silently truncated an extended walk-forward from
    1967 to 1996 — losing six of the ten recessions it was run to cover.
    """
    from src.data.mixed_frequency import _to_monthly

    for freq in ("daily", "weekly", "monthly", "quarterly"):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]), name="X")
        out = _to_monthly(empty, freq)
        assert out.empty, f"{freq}: expected an empty result"
        assert isinstance(out.index, pd.DatetimeIndex)


def test_align_mixed_frequency_tolerates_a_series_with_no_observations():
    """One not-yet-existing series must not sink the whole panel."""
    idx = pd.date_range("1985-01-31", periods=24, freq="ME")
    raw = {
        "PRESENT": pd.Series(range(24), index=idx, dtype=float),
        "NOT_YET": pd.Series(dtype=float, index=pd.DatetimeIndex([])),
    }
    cfg = [
        {"code": "PRESENT", "frequency": "monthly"},
        {"code": "NOT_YET", "frequency": "quarterly"},
    ]
    panel = align_mixed_frequency(raw, series_config=cfg)
    assert "PRESENT" in panel.columns
    assert panel["PRESENT"].notna().sum() == 24
