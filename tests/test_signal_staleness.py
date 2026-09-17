"""The ragged panel edge must not silently become a 50% reading.

Publication lags mask the last month or two of every series — that is
what makes the backtest honest. But the ensemble read ``.iloc[-1]`` of
those masked series, saw NaN, and fell back to 0.5. On the dashboard that
rendered as "Probit 50.0%", indistinguishable from a genuine coin-flip
reading, while the headline probability silently averaged fill values
with real ones.

A nowcast uses the latest *published* observation and says how old it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nowcaster import _latest_published


def test_latest_published_skips_the_masked_tail():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    series = pd.Series([0.1, 0.2, 0.3, 0.4, np.nan, np.nan], index=idx)

    value, as_of = _latest_published(series)
    assert value == pytest.approx(0.4)
    assert as_of == idx[3], "should report the date the reading refers to"


def test_latest_published_falls_back_when_nothing_is_observed():
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    value, as_of = _latest_published(pd.Series([np.nan] * 4, index=idx))
    assert value == 0.5
    assert as_of is None, "a fallback must not claim a reference date"


def test_latest_published_handles_empty_and_none():
    assert _latest_published(pd.Series(dtype=float)) == (0.5, None)
    assert _latest_published(None) == (0.5, None)


def test_latest_published_respects_a_custom_fallback():
    value, _ = _latest_published(pd.Series(dtype=float), fallback=0.25)
    assert value == 0.25


# ---------------------------------------------------------------------------
# End-to-end: a ragged edge must not produce a 0.5 headline
# ---------------------------------------------------------------------------


class _RaggedPipeline:
    """Pipeline whose last two months are masked, as publication lags do."""

    RAW_LEVEL_SERIES = ("UNRATE", "CFNAI")

    def __init__(self, n: int = 240) -> None:
        rng = np.random.default_rng(5)
        idx = pd.date_range("2005-01-31", periods=n, freq="ME")
        factor = np.zeros(n)
        for t in range(1, n):
            factor[t] = 0.9 * factor[t - 1] + rng.standard_normal() * 0.3

        loadings = rng.standard_normal((6, 1)) * 0.2 + 1.1
        obs = factor[:, None] @ loadings.T + rng.standard_normal((n, 6)) * 0.4
        self._panel = pd.DataFrame(
            obs, index=idx, columns=[f"X{i}" for i in range(6)]
        )
        # The edge is ragged: the last two months are unpublished.
        self._panel.iloc[-2:, :] = np.nan

        self.raw_levels_ = pd.DataFrame(
            {
                "UNRATE": 5.0 + np.abs(factor) * 1.2,
                "CFNAI": factor * 0.5,
            },
            index=idx,
        )
        self.raw_levels_.iloc[-2:, :] = np.nan
        self._series_cfg = [
            {"code": c, "frequency": "monthly"} for c in self._panel.columns
        ]

    def run(self, end_date=None):  # noqa: ARG002
        return self._panel


@pytest.mark.slow
def test_ensemble_does_not_report_a_fallback_headline_on_a_ragged_edge():
    """CFNAI and Sahm must read their last published value, not 0.5."""
    from src.models.nowcaster import Nowcaster

    pipeline = _RaggedPipeline()
    nowcaster = Nowcaster(
        pipeline=pipeline, n_factors=2, n_regimes=2, use_ensemble=True,
        factor_names=["real_activity", "labor_market"],
    )
    result = nowcaster.run()

    detail = result.ensemble_detail
    for key in ("cfnai", "sahm"):
        assert detail[key] != 0.5, (
            f"{key} fell back to 0.5 despite a published observation two "
            f"months before the panel edge"
        )
        assert nowcaster._signal_as_of[key] is not None

    # The headline must equal the weighted point estimates, so the number
    # and the breakdown beside it cannot disagree.
    w = nowcaster.ensemble_weights
    expected = sum(w.get(k, 0.0) * detail[k] for k in ("rsm", "probit", "cfnai", "sahm"))
    assert result.recession_probability == pytest.approx(expected, abs=1e-9)
