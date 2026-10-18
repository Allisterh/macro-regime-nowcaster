"""Wrapper around the FRED API (``fredapi`` library).

Provides caching, rate-limiting, and transparent retry logic so that
the rest of the pipeline never needs to talk to the network directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # type: ignore[misc]


# How long a cached copy of a series' *latest* values is trusted.
#
# The cache used to never expire. It was filled on 2026-03-03 and served
# for the next seven months, so the CLI nowcast, the walk-forward panel,
# the benchmarks and the backtests all saw an economy that ended in
# January 2026 — while FRED had data through September — and every run
# logged an unremarkable "Cache hit". A nowcaster whose inputs stop
# updating is not a nowcaster. Vintage (ALFRED) caches are exempt: a
# series as it stood on a past date never changes.
DEFAULT_MAX_AGE = pd.Timedelta(days=1)


class FREDClient:
    """Thin wrapper around ``fredapi.Fred`` with local CSV caching.

    Parameters
    ----------
    api_key : str
        FRED API key.  Falls back to ``FRED_API_KEY`` env-var.
    cache_dir : str | Path | None
        Directory for cached CSV files.  ``None`` disables caching.
    request_delay : float
        Minimum seconds between API calls (rate limiting).
    max_age : pd.Timedelta | None
        How old a cached copy of a series' latest values may be before it
        is re-fetched. ``None`` pins the cache — it never expires, which
        is what a walk-forward rebuild needs so every window sees the same
        data. ``pd.Timedelta(0)`` re-fetches everything. Vintage caches
        (``as_of=...``) never expire whatever this is set to.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        request_delay: float = 0.25,
        max_age: pd.Timedelta | None = DEFAULT_MAX_AGE,
    ) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.request_delay = request_delay
        self.max_age = max_age

        if cache_dir is not None:
            self.cache_dir: Path | None = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        if Fred is None:
            raise ImportError(
                "fredapi is required: pip install fredapi"
            )
        self._fred = Fred(api_key=self.api_key)
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        as_of: str | pd.Timestamp | None = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch a single FRED series, caching to disk when possible.

        Parameters
        ----------
        as_of : str | pd.Timestamp | None
            When set, fetch the **vintage** of the series as it was known
            on this date, using the ALFRED ``realtime_end`` parameter.
            This is the only way to genuinely backtest a real-time
            nowcaster — it returns data values that were publicly
            available on *as_of*, not the latest-revised values.
            Caching is keyed by ``(series_id, as_of)`` so that different
            vintages coexist.
        """
        cache_suffix = ""
        if as_of is not None:
            vintage_str = pd.Timestamp(as_of).strftime("%Y%m%d")
            cache_suffix = f"_v{vintage_str}"

        cache_path = (
            self._cache_path(series_id, suffix=cache_suffix)
            if self.cache_dir else None
        )

        cached_exists = bool(use_cache and cache_path and cache_path.exists())
        expired = cached_exists and as_of is None and self._is_expired(cache_path)

        if cached_exists and not expired:
            logger.debug(f"Cache hit for {series_id}{cache_suffix}")
            return self._read_cache(cache_path, start_date, end_date)

        try:
            return self._fetch(
                series_id, cache_path, start_date, end_date, as_of
            )
        except Exception as exc:
            if not expired:
                raise
            # Offline, or FRED is down. Serving the stale copy keeps the
            # model usable, but it must not pass silently — that is how
            # seven months of frozen data went unnoticed.
            stale = self._read_cache(cache_path, start_date, end_date)
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            last = stale.dropna().index.max() if len(stale) else None
            logger.warning(
                f"Could not refresh {series_id} ({exc}); using a cached copy "
                f"{age_days:.0f} days old whose latest observation is "
                f"{last.date() if last is not None else 'unknown'}."
            )
            return stale

    def _is_expired(self, cache_path: Path) -> bool:
        if self.max_age is None:
            return False
        # Both sides in epoch seconds. A first version compared
        # pd.Timestamp.now() (local time) with a timestamp built from
        # st_mtime (UTC), which on a machine behind UTC made fresh files
        # look hours in the future and expire that much late.
        age = time.time() - cache_path.stat().st_mtime
        return age >= self.max_age.total_seconds()

    @staticmethod
    def _slice(
        series: pd.Series, start_date: str | None, end_date: str | None
    ) -> pd.Series:
        if start_date:
            series = series.loc[start_date:]
        if end_date:
            series = series.loc[:end_date]
        return series

    @classmethod
    def _read_cache(
        cls, cache_path: Path, start_date: str | None, end_date: str | None
    ) -> pd.Series:
        cached = pd.read_csv(
            cache_path, index_col=0, parse_dates=True
        ).squeeze("columns")
        return cls._slice(cached, start_date, end_date)

    def _fetch(
        self,
        series_id: str,
        cache_path: Path | None,
        start_date: str | None,
        end_date: str | None,
        as_of: str | pd.Timestamp | None,
    ) -> pd.Series:
        self._rate_limit()

        # Always fetch — and so always cache — the full history, then
        # slice locally. The cache is keyed by series id alone, so a file
        # must not depend on the date range of whoever last fetched it.
        # It used to: requests passed observation_start/end straight to
        # FRED and cached the result. That was masked while the cache
        # never expired, since each file was fetched once. With expiry it
        # would bite: a backtest refitting as of 1995 would overwrite the
        # file with data ending in 1995, and a run starting in 1980 would
        # strip the pre-1980 history the walk-forward needs — each reader
        # afterwards silently getting a truncated series.
        requested = (start_date, end_date)
        start_date = end_date = None

        if as_of is not None:
            vintage = pd.Timestamp(as_of).strftime("%Y-%m-%d")
            logger.debug(
                f"Fetching {series_id} vintage as-of {vintage} "
                f"from FRED/ALFRED API"
            )
            # Both realtime bounds must be the vintage date.  Passing
            # realtime_start=start_date asks ALFRED for every vintage
            # published between start_date and as_of, which returns
            # duplicated observation dates rather than the single
            # point-in-time snapshot this function promises.
            data: pd.Series = self._fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
                realtime_start=vintage,
                realtime_end=vintage,
            )
            # Belt and braces: if a provider still returns overlapping
            # vintages, keep the last value per observation date.
            if data.index.has_duplicates:
                data = data[~data.index.duplicated(keep="last")]
        else:
            logger.debug(f"Fetching {series_id} from FRED API")
            data = self._fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )

        data.name = series_id

        if cache_path is not None:
            data.to_csv(cache_path)
            logger.debug(f"Cached {series_id} → {cache_path}")

        return self._slice(data, *requested)

    def get_series_info(self, series_id: str) -> dict:
        """Return metadata for a FRED series."""
        self._rate_limit()
        return self._fred.get_series_info(series_id).to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_call = time.time()

    def _cache_path(self, series_id: str, suffix: str = "") -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{series_id}{suffix}.csv"
