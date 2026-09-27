"""Philadelphia Fed Survey of Professional Forecasters: recession probabilities.

The SPF asks forecasters each quarter for the probability that real GDP
(real GNP before 1992) declines quarter-on-quarter in the survey quarter
(``RECESS1``) and each of the next four (``RECESS2``-``RECESS5``).
``RECESS2`` is what the Philadelphia Fed calls the Anxious Index.

It is published continuously since 1968Q4, so unlike prediction markets it
spans all eight NBER recessions this project is evaluated on.

Point-in-time alignment
-----------------------
A survey is aligned to the date its results were *published*, never to
the quarter it describes. The Philadelphia Fed publishes true release
dates from 1990Q2 onward, and they matter:

- The 1990Q2 survey was not taken in real time. Forecasters supplied
  "dated forecasts from May 1990" retroactively in August, and it was
  released on 31 August 1990. Treating it as known in May would hand the
  survey three months of hindsight.
- Several surveys were delayed by federal shutdowns (1996Q1, 2013Q4,
  2019Q1, 2026Q1).

For surveys before 1990Q2 the release dates "are not known". Those are
aligned with an assumed lag from the start of the survey quarter, which
the caller must choose; ``measure_spf.py`` reports results under several,
so a conclusion that only survives an optimistic assumption is visible.

Data files are downloaded from the Philadelphia Fed into ``data/spf/``
(gitignored); see :func:`download`.
"""

from __future__ import annotations

import re
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

SPF_BASE = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters"
)
FILES = {
    "mean": "data-files/files/Mean_RECESS_Level.xlsx",
    "median": "data-files/files/Median_RECESS_Level.xlsx",
    "release_dates": "spf-release-dates.txt",
}
DEFAULT_DIR = Path("data/spf")

# First survey whose true release date is published.
FIRST_KNOWN_RELEASE = pd.Timestamp("1990-04-01")

_DATE_LINE = re.compile(
    r"^\s*(?P<year>\d{4})?\s*Q(?P<q>[1-4])\s+"
    r"(?P<deadline>\d{1,2}/\d{1,2}/\d{2})\**\s+"
    r"(?P<release>\d{1,2}/\d{1,2}/\d{2})\**"
)


def download(directory: str | Path = DEFAULT_DIR) -> dict[str, Path]:
    """Fetch the RECESS files and the release-date table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, rel in FILES.items():
        dest = directory / Path(rel).name
        req = urllib.request.Request(
            f"{SPF_BASE}/{rel}", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        paths[key] = dest
    return paths


def load_recess(path: str | Path) -> pd.DataFrame:
    """RECESS1-5 as probabilities in [0, 1], indexed by survey quarter start."""
    with warnings.catch_warnings():
        # The workbook's print header is malformed; the data is not.
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(path)
    quarters = pd.PeriodIndex.from_fields(
        year=raw["YEAR"].astype(int), quarter=raw["QUARTER"].astype(int),
        freq="Q",
    )
    cols = [c for c in raw.columns if str(c).startswith("RECESS")]
    out = raw[cols].astype(float) / 100.0
    out.index = quarters.to_timestamp(how="start")
    out.index.name = "survey_quarter"
    return out


def load_release_dates(path: str | Path) -> pd.Series:
    """News release date for each survey quarter, where it is known.

    The file lists each year once, on that year's first line, so the
    year is carried forward. Footnote markers (``*``) are stripped.
    """
    year = None
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _DATE_LINE.match(line)
        if not m:
            continue
        if m["year"]:
            year = int(m["year"])
        if year is None:
            continue
        quarter = pd.Period(year=year, quarter=int(m["q"]), freq="Q")
        rows[quarter.to_timestamp(how="start")] = pd.to_datetime(
            m["release"], format="%m/%d/%y"
        )
    series = pd.Series(rows, name="release_date").sort_index()
    series.index.name = "survey_quarter"
    return series


def knowable_dates(
    survey_quarters: pd.DatetimeIndex,
    release_dates: pd.Series,
    unknown_lag_days: int,
) -> pd.Series:
    """Date each survey could first have been used.

    The published release date where one exists; otherwise the survey
    quarter's start plus ``unknown_lag_days``.
    """
    known = release_dates.reindex(survey_quarters)
    assumed = pd.Series(
        survey_quarters + pd.Timedelta(days=unknown_lag_days),
        index=survey_quarters,
    )
    out = known.fillna(assumed)
    out.name = "knowable_at"
    return out


def as_of(
    values: pd.Series, knowable: pd.Series, dates: pd.DatetimeIndex
) -> pd.Series:
    """The latest survey value knowable on or before each date.

    ``values`` and ``knowable`` are indexed by survey quarter. Dates
    before the first survey is knowable are NaN — never back-filled.
    """
    frame = pd.DataFrame({"value": values, "knowable_at": knowable}).dropna()
    frame = frame.sort_values("knowable_at")
    pos = frame["knowable_at"].searchsorted(dates, side="right") - 1
    result = pd.Series(float("nan"), index=dates, name=values.name)
    ok = pos >= 0
    result[ok] = frame["value"].to_numpy()[pos[ok]]
    return result
