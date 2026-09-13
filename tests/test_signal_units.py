"""Unit contracts for threshold-based signals.

The Sahm rule (0.50pp) and the CFNAI convention (-0.7) are defined on
*published units*.  The modelling panel transforms and expanding-z-scores
every column, so reading these signals from it compares a threshold
against a different quantity — numerically plausible, economically
meaningless.

These tests pin the contract at both ends: the signal functions are
anchored to published history, and the pipeline is required to hand the
Nowcaster raw levels rather than the standardised panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import DataPipeline
from src.data.transformations import apply_transform, standardize_expanding
from src.models.nowcaster import Nowcaster
from src.models.sahm_rule import sahm_indicator, sahm_recession_probability

# ---------------------------------------------------------------------------
# Anchored to published history
# ---------------------------------------------------------------------------

# Unemployment rate (FRED UNRATE), percentage points.  Enough history for
# the 12-month rolling minimum around each episode of interest.
_UNRATE_2007_2010 = {
    "2007-01-31": 4.6, "2007-02-28": 4.5, "2007-03-31": 4.4,
    "2007-04-30": 4.5, "2007-05-31": 4.4, "2007-06-30": 4.6,
    "2007-07-31": 4.7, "2007-08-31": 4.6, "2007-09-30": 4.7,
    "2007-10-31": 4.7, "2007-11-30": 4.7, "2007-12-31": 5.0,
    "2008-01-31": 5.0, "2008-02-29": 4.9, "2008-03-31": 5.1,
    "2008-04-30": 5.0, "2008-05-31": 5.4, "2008-06-30": 5.6,
    "2008-07-31": 5.8, "2008-08-31": 6.1, "2008-09-30": 6.1,
    "2008-10-31": 6.5, "2008-11-30": 6.8, "2008-12-31": 7.3,
    "2009-01-31": 7.8, "2009-02-28": 8.3, "2009-03-31": 8.7,
    "2009-04-30": 9.0, "2009-05-31": 9.4, "2009-06-30": 9.5,
}


@pytest.fixture()
def unrate_2007_2010() -> pd.Series:
    idx = pd.to_datetime(list(_UNRATE_2007_2010.keys()))
    return pd.Series(list(_UNRATE_2007_2010.values()), index=idx, dtype=float)


def test_sahm_fires_in_great_recession(unrate_2007_2010):
    """The Sahm gap must exceed 0.50pp well inside the 2008-09 recession."""
    gap = sahm_indicator(unrate_2007_2010)
    assert gap.loc["2008-09-30"] >= 0.50, (
        f"Sahm gap was {gap.loc['2008-09-30']:.2f}pp in Sep-2008; the rule "
        f"is documented to have triggered during this recession"
    )
    assert gap.loc["2009-06-30"] >= 2.0


def test_sahm_quiet_in_calm_expansion(unrate_2007_2010):
    """It must stay well below the threshold through 2007."""
    gap = sahm_indicator(unrate_2007_2010)
    calm = gap.loc["2007-07-31":"2007-11-30"]
    assert (calm < 0.50).all(), f"Sahm falsely triggered in 2007:\n{calm}"


def test_sahm_rejects_standardised_input(unrate_2007_2010):
    """A z-scored first difference must not masquerade as a level.

    This is the exact substitution that made the deployed Sahm signal fire
    in 87% of months and correlate -0.07 with the real rule.
    """
    piped = standardize_expanding(
        apply_transform(unrate_2007_2010.copy(), "diff"), min_periods=3
    )
    correct = sahm_recession_probability(unrate_2007_2010)
    wrong = sahm_recession_probability(piped)

    # The two must not be interchangeable.
    assert not np.allclose(
        correct.dropna().values,
        wrong.reindex(correct.dropna().index).values,
        atol=0.05,
    ), "z-scored input produced the same signal as the raw level"


# ---------------------------------------------------------------------------
# Pipeline contract: raw levels are preserved
# ---------------------------------------------------------------------------


def _mock_pipeline(raw: dict, cfg: list) -> DataPipeline:
    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.fred_client = None
    pipeline.start_date = "2000-01-01"
    pipeline.series_config_path = "config/fred_series.yaml"
    pipeline.storage = None
    pipeline.apply_publication_lags = False
    pipeline.standardize_min_periods = 6
    pipeline.use_vintages = False
    pipeline._series_cfg = cfg
    pipeline.raw_levels_ = None
    pipeline._fetch_all = lambda end: raw
    pipeline._align_monthly = lambda r: pd.concat(r, axis=1)
    return pipeline


@pytest.fixture()
def pipeline_with_levels() -> DataPipeline:
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    unrate = pd.Series(5.0 + rng.standard_normal(60).cumsum() * 0.1, index=idx)
    cfnai = pd.Series(rng.standard_normal(60) * 0.4, index=idx)
    cfg = [
        {"code": "UNRATE", "transform": "diff", "publication_lag_days": 35},
        {"code": "CFNAI", "transform": "none", "publication_lag_days": 24},
    ]
    return _mock_pipeline({"UNRATE": unrate, "CFNAI": cfnai}, cfg)


def test_pipeline_exposes_raw_levels(pipeline_with_levels):
    """run() must populate raw_levels_ with untransformed values."""
    panel = pipeline_with_levels.run(end_date="2004-12-31")
    raw = pipeline_with_levels.raw_levels_

    assert raw is not None, "DataPipeline.run() did not populate raw_levels_"
    assert "UNRATE" in raw.columns and "CFNAI" in raw.columns

    # The panel is z-scored; the raw frame must not be.
    assert raw["UNRATE"].dropna().min() > 1.0, (
        "raw_levels_['UNRATE'] looks standardised, not a percentage-point level"
    )
    assert abs(panel["UNRATE"].dropna().mean()) < 1.0, (
        "panel['UNRATE'] is expected to be standardised"
    )


def test_nowcaster_rejects_standardised_raw_levels(pipeline_with_levels):
    """_raw_level must refuse a frame whose values are not in published units."""
    pipeline_with_levels.run(end_date="2004-12-31")
    nowcaster = Nowcaster(pipeline=pipeline_with_levels)

    # Substitute z-scores for the level, as the pre-fix code effectively did.
    idx = pipeline_with_levels.raw_levels_.index
    pipeline_with_levels.raw_levels_ = pd.DataFrame(
        {"UNRATE": np.linspace(-3, 3, len(idx))}, index=idx
    )
    with pytest.raises(ValueError, match="does not look like a raw level"):
        nowcaster._raw_level("UNRATE", idx)


def test_nowcaster_accepts_genuine_levels(pipeline_with_levels):
    pipeline_with_levels.run(end_date="2004-12-31")
    nowcaster = Nowcaster(pipeline=pipeline_with_levels)
    idx = pipeline_with_levels.raw_levels_.index
    series = nowcaster._raw_level("UNRATE", idx)
    assert series is not None
    assert series.dropna().between(0, 30).all()


def test_missing_raw_level_returns_none(pipeline_with_levels):
    """An absent series degrades to None rather than raising."""
    pipeline_with_levels.run(end_date="2004-12-31")
    nowcaster = Nowcaster(pipeline=pipeline_with_levels)
    idx = pipeline_with_levels.raw_levels_.index
    assert nowcaster._raw_level("NOT_A_SERIES", idx) is None
