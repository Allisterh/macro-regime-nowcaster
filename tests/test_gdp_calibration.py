"""The GDP nowcast must actually be calibrated, not fall through to a trend.

FRED stamps ``GDPC1`` at the *start* of its quarter (2002-01-01) while
resampling the monthly factors produces quarter *ends* (2002-03-31). The
two indexes never intersected, so ``_calibrate_gdp`` raised "Only 0
common quarters", the exception was caught, and every nowcast this
project ever produced used a hard-coded 2.5% trend with a fabricated
confidence band — while the README advertised an OLS-calibrated figure.

Another silent fallback: it produced a plausible number, logged at
warning level, and nothing downstream could tell the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.nowcaster import Nowcaster


class _StubFred:
    """Serves GDPC1 stamped at quarter starts, exactly as FRED does."""

    def __init__(self, n_quarters: int = 60) -> None:
        idx = pd.date_range("2005-01-01", periods=n_quarters, freq="QS")
        rng = np.random.default_rng(0)
        level = 10_000 * np.cumprod(1 + rng.normal(0.005, 0.01, n_quarters))
        self.series = pd.Series(level, index=idx, name="GDPC1")

    def get_series(self, code, start_date=None, end_date=None, **kwargs):
        if code != "GDPC1":
            raise KeyError(code)
        return self.series


class _StubPipeline:
    RAW_LEVEL_SERIES = ("UNRATE", "CFNAI")

    def __init__(self, n_months: int = 240) -> None:
        rng = np.random.default_rng(1)
        idx = pd.date_range("2005-01-31", periods=n_months, freq="ME")
        factor = np.zeros(n_months)
        for t in range(1, n_months):
            factor[t] = 0.9 * factor[t - 1] + rng.standard_normal() * 0.3
        loadings = rng.standard_normal((6, 1)) * 0.2 + 1.1
        self._panel = pd.DataFrame(
            factor[:, None] @ loadings.T + rng.standard_normal((n_months, 6)) * 0.4,
            index=idx, columns=[f"X{i}" for i in range(6)],
        )
        self.raw_levels_ = pd.DataFrame(
            {"UNRATE": 5.0 + np.abs(factor), "CFNAI": factor * 0.5}, index=idx
        )
        self.fred_client = _StubFred()
        self._series_cfg = [
            {"code": c, "frequency": "monthly"} for c in self._panel.columns
        ]

    def run(self, end_date=None):  # noqa: ARG002
        return self._panel


def test_quarter_start_gdp_aligns_with_quarter_end_factors():
    """The index mismatch must be reconciled, not silently swallowed."""
    pipeline = _StubPipeline()
    nowcaster = Nowcaster(
        pipeline=pipeline, n_factors=2, n_regimes=2, use_ensemble=False,
        factor_names=["real_activity", "labor_market"],
    )
    nowcaster.run()

    factors = nowcaster._last_factors
    intercept, betas, se = nowcaster._calibrate_gdp(factors, nowcaster._last_panel)

    # The fallback returns exactly these; a real fit will not.
    assert (intercept, float(se)) != (
        Nowcaster._GDP_TREND_DEFAULT, 1.0
    ), "calibration fell through to the hard-coded trend"
    assert len(betas) == factors.shape[1]
    assert np.isfinite(betas).all()
    assert se > 0


def test_fallback_still_applies_when_gdp_is_genuinely_unavailable():
    """No GDP series means the documented fallback, not a crash."""
    pipeline = _StubPipeline()
    pipeline.fred_client = None
    nowcaster = Nowcaster(
        pipeline=pipeline, n_factors=2, n_regimes=2, use_ensemble=False,
        factor_names=["real_activity", "labor_market"],
    )
    nowcaster.run()
    intercept, betas, se = nowcaster._calibrate_gdp(
        nowcaster._last_factors, nowcaster._last_panel
    )
    assert intercept == Nowcaster._GDP_TREND_DEFAULT
    assert len(betas) == nowcaster._last_factors.shape[1]


def test_confidence_interval_reflects_the_fitted_residual():
    """The band must come from the fit, not from a constant."""
    pipeline = _StubPipeline()
    nowcaster = Nowcaster(
        pipeline=pipeline, n_factors=2, n_regimes=2, use_ensemble=False,
        factor_names=["real_activity", "labor_market"],
    )
    result = nowcaster.run()
    width = result.gdp_ci_upper - result.gdp_ci_lower
    assert width > 0
    # The fallback produced exactly 2 * 1.645 * max(1.0, 0.5) = 3.29.
    assert abs(width - 3.29) > 1e-6, (
        "interval width matches the hard-coded fallback exactly"
    )
