"""Property tests for look-ahead-freeness.

Asserts that the data pipeline and expanding standardisation produce
identical values for time *t* regardless of whether observations after *t*
are present.  This is the central correctness invariant of the Phase 1
audit fixes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import DataPipeline
from src.data.transformations import standardize_expanding
from src.utils.date_utils import ragged_edge_mask


# ---------------------------------------------------------------------------
# Expanding standardisation: no look-ahead
# ---------------------------------------------------------------------------


def test_expanding_standardize_prefix_invariant():
    """Z-score at time t must be the same whether we feed [0..t] or [0..t+k]."""
    rng = np.random.default_rng(99)
    idx_long = pd.date_range("2000-01-31", periods=120, freq="ME")
    vals = rng.standard_normal(120).cumsum()
    long = pd.Series(vals, index=idx_long)

    z_long = standardize_expanding(long, min_periods=24)

    for cut in [60, 80, 100]:
        short = long.iloc[:cut]
        z_short = standardize_expanding(short, min_periods=24)
        shared = z_short.dropna()
        z_long_prefix = z_long.reindex(shared.index)
        np.testing.assert_allclose(
            shared.values,
            z_long_prefix.values,
            atol=0, rtol=0,
            err_msg=f"Z-scores diverged when truncating at {cut}",
        )


# ---------------------------------------------------------------------------
# Ragged-edge mask: no future data leaks through
# ---------------------------------------------------------------------------


def test_ragged_edge_truncation_invariant():
    """Masking at as_of=T on a long panel must equal masking on a panel
    that only extends to T in the first place (for shared rows)."""
    idx = pd.date_range("2019-01-31", periods=24, freq="ME")
    df = pd.DataFrame(
        {
            "PAYEMS": np.arange(24, dtype=float),
            "T10Y2Y": np.arange(24, dtype=float) * 0.1,
        },
        index=idx,
    )
    lags = {"PAYEMS": 35, "T10Y2Y": 1}
    as_of = pd.Timestamp("2020-03-31")

    masked_full = ragged_edge_mask(df, series_lags=lags, as_of_date=as_of)
    masked_short = ragged_edge_mask(
        df.loc[:as_of], series_lags=lags, as_of_date=as_of,
    )

    shared_idx = masked_short.index
    pd.testing.assert_frame_equal(
        masked_full.loc[shared_idx],
        masked_short,
    )


# ---------------------------------------------------------------------------
# Pipeline end-to-end: output at T is stable regardless of later data
# ---------------------------------------------------------------------------


def _build_mock_pipeline(raw: dict, series_cfg: list, *, apply_lags: bool = True):
    """Construct a DataPipeline with mocked FRED I/O."""
    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.fred_client = MagicMock()
    pipeline.start_date = "2015-01-01"
    pipeline.series_config_path = "config/fred_series.yaml"
    pipeline.storage = None
    pipeline.apply_publication_lags = apply_lags
    pipeline.standardize_min_periods = 3
    pipeline._series_cfg = series_cfg
    pipeline._fetch_all = lambda end: raw
    pipeline._align_monthly = lambda r: pd.concat(r, axis=1)
    return pipeline


def test_pipeline_output_stable_across_end_dates():
    """Values at T must not change when we extend the panel past T."""
    dates_long = pd.date_range("2015-01-31", periods=72, freq="ME")
    vals = np.arange(72, dtype=float) + 100.0
    raw_long = {
        "PAYEMS": pd.Series(vals, index=dates_long),
        "T10Y2Y": pd.Series(vals * 0.5, index=dates_long),
    }
    cfg = [
        {"code": "PAYEMS", "transform": "none", "publication_lag_days": 35},
        {"code": "T10Y2Y", "transform": "none", "publication_lag_days": 1},
    ]

    # Run with full panel
    pipe_long = _build_mock_pipeline(raw_long, cfg, apply_lags=False)
    result_long = pipe_long.run(end_date="2020-12-31")

    # Run with panel truncated to mid-2019
    cut = pd.Timestamp("2019-06-30")
    raw_short = {k: v.loc[:cut] for k, v in raw_long.items()}
    pipe_short = _build_mock_pipeline(raw_short, cfg, apply_lags=False)
    result_short = pipe_short.run(end_date="2019-06-30")

    # Shared dates must have identical values (expanding standardisation)
    shared = result_short.index.intersection(result_long.index)
    assert len(shared) > 10
    for col in result_short.columns:
        short_vals = result_short.loc[shared, col].dropna()
        long_vals = result_long.loc[shared, col].reindex(short_vals.index)
        np.testing.assert_allclose(
            short_vals.values,
            long_vals.values,
            atol=1e-12, rtol=0,
            err_msg=f"{col} values changed when extending the panel",
        )
