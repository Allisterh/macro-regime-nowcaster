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
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        request_delay: float = 0.25,
    ) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.request_delay = request_delay

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

        if use_cache and cache_path and cache_path.exists():
            logger.debug(f"Cache hit for {series_id}{cache_suffix}")
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            if start_date:
                cached = cached.loc[start_date:]
            if end_date:
                cached = cached.loc[:end_date]
            return cached

        self._rate_limit()

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

        return data

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
