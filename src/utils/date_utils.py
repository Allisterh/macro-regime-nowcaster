"""Date and frequency alignment utilities for economic data.

Provides helpers for business-day handling, publication lag mapping,
and frequency alignment needed when working with mixed-frequency
FRED data.
"""

from __future__ import annotations

import pandas as pd


def to_business_day_end(dt: pd.Timestamp) -> pd.Timestamp:
    """Shift a timestamp to the nearest preceding business day.

    Args:
        dt: Input timestamp.

    Returns:
        Timestamp adjusted to the last business day on or before *dt*.
    """
    if dt.weekday() < 5:
        return dt
    # Saturday → Friday, Sunday → Friday
    offset = dt.weekday() - 4
    return dt - pd.Timedelta(days=offset)


def align_to_monthly(series: pd.Series, method: str = "last") -> pd.Series:
    """Resample a series to monthly frequency.

    Args:
        series: Input time series with DatetimeIndex.
        method: Aggregation method — ``"last"``, ``"mean"``, or ``"sum"``.

    Returns:
        Monthly series with period-end dates.

    Raises:
        ValueError: If *method* is not recognised.
    """
    methods = {
        "last": lambda s: s.resample("ME").last(),
        "mean": lambda s: s.resample("ME").mean(),
        "sum":  lambda s: s.resample("ME").sum(),
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)}")
    return methods[method](series)


def get_publication_date(
    obs_date: pd.Timestamp,
    publication_lag_days: int,
) -> pd.Timestamp:
    """Compute the earliest date on which an observation would be public.

    Args:
        obs_date: The reference observation period end date.
        publication_lag_days: Number of calendar days after *obs_date*
            until the data is typically published.

    Returns:
        Estimated publication date.
    """
    return obs_date + pd.Timedelta(days=publication_lag_days)


def business_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count the number of business days between two dates (exclusive of start).

    Args:
        start: Start date (exclusive).
        end: End date (inclusive).

    Returns:
        Number of business days.
    """
    dates = pd.bdate_range(start=start + pd.Timedelta(days=1), end=end)
    return len(dates)


def ragged_edge_mask(
    df: pd.DataFrame,
    series_lags: dict[str, int],
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply publication lags to create a ragged-edge mask.

    Sets values to NaN for any (date, series) pair where the observation
    would not yet be publicly available as of *as_of_date*.

    Args:
        df: Wide DataFrame (T × N) with DatetimeIndex.
        series_lags: Mapping series_id → publication_lag_days.
        as_of_date: Reference date.  Defaults to today.

    Returns:
        DataFrame with NaN where data is not yet released.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()

    result = df.copy()
    if result.empty:
        return result

    # Vectorised over the index: publication date is a pure function of
    # (observation date, lag), so one timedelta shift per column replaces
    # the former T x N scalar .loc assignments (~38k per run on the full
    # panel, which dominated pipeline time in the expanding backtests).
    index = pd.DatetimeIndex(df.index)
    for col in df.columns:
        lag = int(series_lags.get(col, 0))
        pub_dates = index + pd.Timedelta(days=lag)
        unavailable = pub_dates > as_of_date
        if unavailable.any():
            result.loc[unavailable, col] = float("nan")

    return result
