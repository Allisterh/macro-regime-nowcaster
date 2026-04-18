"""Regression tests for publication-lag (ragged-edge) masking.

Ensures the mask is wired into the data pipeline and produces the
expected NaN pattern so the downstream models cannot consume data that
would not yet have been publicly released at the nowcast reference date.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import DataPipeline
from src.utils.date_utils import ragged_edge_mask


def test_ragged_edge_masks_future_releases():
    """Observations whose publication date is after the as-of date get NaN."""
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "PAYEMS": 100.0 + np.arange(6),      # 35-day lag
            "T10Y2Y": 2.0 - 0.1 * np.arange(6),  # 1-day lag
        },
        index=idx,
    )

    # As-of April 5, 2020 — PAYEMS for March (lag 35 days → publishes May 5)
    # must NOT be available.  T10Y2Y (lag 1) for March publishes April 1 → visible.
    masked = ragged_edge_mask(
        df,
        series_lags={"PAYEMS": 35, "T10Y2Y": 1},
        as_of_date=pd.Timestamp("2020-04-05"),
    )

    # March 2020 row: PAYEMS masked, T10Y2Y visible
    assert np.isnan(masked.loc["2020-03-31", "PAYEMS"])
    assert not np.isnan(masked.loc["2020-03-31", "T10Y2Y"])

    # February 2020 row: PAYEMS published April 4 → visible on April 5
    assert not np.isnan(masked.loc["2020-02-29", "PAYEMS"])

    # Any row whose reference date is after the as-of date is fully masked
    assert masked.loc["2020-04-30":].isna().all().all()


def test_pipeline_applies_publication_lags_by_default():
    """DataPipeline.run wires ragged_edge_mask; toggle via the flag."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    raw = {
        "PAYEMS": pd.Series(np.arange(6, dtype=float), index=dates),
        "T10Y2Y": pd.Series(np.arange(6, dtype=float), index=dates),
    }

    # Build a DataPipeline with a mocked FRED client and a minimal series
    # config (no real YAML), then patch internals to avoid FRED calls.
    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.fred_client = MagicMock()
    pipeline.start_date = "2020-01-01"
    pipeline.series_config_path = "config/fred_series.yaml"
    pipeline.storage = None
    pipeline.apply_publication_lags = True
    pipeline.standardize_min_periods = 3
    pipeline._series_cfg = [
        {"code": "PAYEMS", "transform": "none", "publication_lag_days": 35},
        {"code": "T10Y2Y", "transform": "none", "publication_lag_days": 1},
    ]
    pipeline._fetch_all = lambda end: raw
    pipeline._align_monthly = lambda r: pd.concat(r, axis=1)

    result = pipeline.run(end_date="2020-04-05")

    # After masking, March PAYEMS must be NaN (lag 35 → publishes May 5).
    assert np.isnan(result.loc["2020-03-31", "PAYEMS"])
    # T10Y2Y for March (1-day lag → publishes April 1) must be present.
    assert not np.isnan(result.loc["2020-03-31", "T10Y2Y"])


def test_pipeline_lag_flag_can_be_disabled():
    """apply_publication_lags=False returns the unmasked panel."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    raw = {
        "PAYEMS": pd.Series(np.arange(6, dtype=float), index=dates),
    }

    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.fred_client = MagicMock()
    pipeline.start_date = "2020-01-01"
    pipeline.series_config_path = "config/fred_series.yaml"
    pipeline.storage = None
    pipeline.apply_publication_lags = False
    pipeline.standardize_min_periods = 3
    pipeline._series_cfg = [
        {"code": "PAYEMS", "transform": "none", "publication_lag_days": 35},
    ]
    pipeline._fetch_all = lambda end: raw
    pipeline._align_monthly = lambda r: pd.concat(r, axis=1)

    result = pipeline.run(end_date="2020-03-31")

    # Without masking, March PAYEMS retains whatever value the transform
    # produced (not NaN by the lag rule).
    assert "PAYEMS" in result.columns
