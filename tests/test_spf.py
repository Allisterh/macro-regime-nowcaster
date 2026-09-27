"""SPF alignment must be point-in-time, or the benchmark flatters it.

The SPF is compared against a model whose every input is aligned to the
date it was published. If a survey were aligned to the quarter it
describes instead, it would see data that had not been released yet, and
the comparison would be rigged in its favour. These tests pin the
alignment rules on synthetic inputs, since ``data/`` is not in the repo.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.spf import as_of, knowable_dates, load_release_dates

# Mirrors the real file: year printed once per block, footnote markers
# after some dates, and a retroactive 1990Q2 released in late August.
_RELEASE_TEXT = """Deadline and Release Dates for the Survey of Professional Forecasters
True deadline and news release dates for surveys prior to 1990:Q2 are not known.


Survey         True Deadline Date       News Release Date

1990 Q2             8/23/90*            8/31/90*
     Q3             8/23/90             8/31/90
     Q4             11/22/90            11/28/90

1991 Q1             2/16/91             2/21/91
     Q2             5/18/91             5/24/91

2019 Q1             3/12/19**           3/22/19**

*The 1990Q2 survey was not taken in real time.
"""


@pytest.fixture
def release_dates():
    d = Path(tempfile.mkdtemp())
    try:
        f = d / "spf-release-dates.txt"
        f.write_text(_RELEASE_TEXT, encoding="utf-8")
        yield load_release_dates(f)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_release_dates_carry_the_year_forward(release_dates):
    """Q3 and Q4 1990 have no year on their line; they must not be lost."""
    assert release_dates.loc[pd.Timestamp("1990-07-01")] == pd.Timestamp("1990-08-31")
    assert release_dates.loc[pd.Timestamp("1990-10-01")] == pd.Timestamp("1990-11-28")
    assert release_dates.loc[pd.Timestamp("1991-04-01")] == pd.Timestamp("1991-05-24")


def test_footnote_markers_are_stripped(release_dates):
    assert release_dates.loc[pd.Timestamp("2019-01-01")] == pd.Timestamp("2019-03-22")


def test_the_retroactive_1990q2_survey_is_not_known_in_may(release_dates):
    """Forecasters gave May 1990 forecasts in August, released 31 August.

    Aligning it to its survey quarter would give it three months of
    hindsight, including the start of the July 1990 recession.
    """
    quarters = pd.DatetimeIndex([pd.Timestamp("1990-04-01")])
    known = knowable_dates(quarters, release_dates, unknown_lag_days=45)
    assert known.iloc[0] == pd.Timestamp("1990-08-31")

    values = pd.Series([0.9], index=quarters, name="RECESS1")
    seen = as_of(values, known, pd.DatetimeIndex(["1990-05-31", "1990-08-31"]))
    assert np.isnan(seen.iloc[0]), "the May reading was visible before release"
    assert seen.iloc[1] == pytest.approx(0.9)


def test_the_assumed_lag_applies_only_where_the_date_is_unknown(release_dates):
    quarters = pd.DatetimeIndex(["1980-01-01", "1991-01-01"])
    known = knowable_dates(quarters, release_dates, unknown_lag_days=120)
    assert known.loc["1980-01-01"] == pd.Timestamp("1980-01-01") + pd.Timedelta(days=120)
    assert known.loc["1991-01-01"] == pd.Timestamp("1991-02-21")


def test_as_of_takes_the_latest_published_survey_not_the_latest_quarter():
    """A later survey quarter published later must not appear early."""
    quarters = pd.DatetimeIndex(["2000-01-01", "2000-04-01"])
    values = pd.Series([0.1, 0.5], index=quarters, name="RECESS1")
    known = pd.Series(
        [pd.Timestamp("2000-02-15"), pd.Timestamp("2000-05-15")], index=quarters
    )
    dates = pd.DatetimeIndex(["2000-02-14", "2000-03-31", "2000-05-14", "2000-05-15"])
    seen = as_of(values, known, dates)

    assert np.isnan(seen.iloc[0]), "a value appeared before any survey was out"
    assert seen.iloc[1] == pytest.approx(0.1)
    assert seen.iloc[2] == pytest.approx(0.1), "Q2 was visible before its release"
    assert seen.iloc[3] == pytest.approx(0.5)


def test_as_of_never_back_fills():
    quarters = pd.DatetimeIndex(["2000-01-01"])
    values = pd.Series([0.3], index=quarters, name="RECESS1")
    known = pd.Series([pd.Timestamp("2000-03-01")], index=quarters)
    seen = as_of(values, known, pd.date_range("1999-10-31", periods=6, freq="ME"))
    assert seen.loc[:"2000-02-29"].isna().all()
    assert (seen.loc["2000-03-31":] == 0.3).all()
