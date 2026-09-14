"""End-to-end data pipeline: fetch → align → transform → panel.

Orchestrates :class:`FREDClient` and the transformation functions to
produce a clean, aligned ``pd.DataFrame`` ready for modelling.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.data.fred_client import FREDClient
from src.data.mixed_frequency import align_mixed_frequency
from src.data.transformations import (
    apply_transform,
    standardize_expanding,
)
from src.utils.date_utils import ragged_edge_mask


class DataPipeline:
    """Build a transformed panel of macro indicators from FRED.

    Parameters
    ----------
    fred_client : FREDClient
        Authenticated FRED client.
    start_date : str
        Earliest observation date (ISO-8601).
    series_config_path : str | Path
        Path to ``fred_series.yaml``.
    storage : object | None
        Optional persistence backend (``DataStorage``).
    """

    # Series whose downstream signals are defined on *published units*
    # rather than on z-scores: the Sahm rule needs the unemployment rate
    # in percentage points, and the CFNAI convention thresholds the
    # published index at -0.7.  The modelling panel standardises every
    # column, so these are kept aside untransformed in ``raw_levels_``.
    RAW_LEVEL_SERIES = ("UNRATE", "CFNAI")

    def __init__(
        self,
        fred_client: FREDClient | None = None,
        start_date: str = "1980-01-01",
        series_config_path: str | Path = "config/fred_series.yaml",
        storage: object | None = None,
        *,
        apply_publication_lags: bool = True,
        standardize_min_periods: int = 24,
        use_vintages: bool = False,
    ) -> None:
        self.fred_client = fred_client
        self.start_date = start_date
        self.series_config_path = str(series_config_path)
        self.storage = storage
        self.apply_publication_lags = apply_publication_lags
        self.standardize_min_periods = standardize_min_periods
        # When True, fetch each series as it stood on the run's end_date
        # (ALFRED point-in-time) rather than the latest revised values.
        # Publication-lag masking models *when* a number appeared; this
        # models *what* the number said at the time.  Backtests need both.
        self.use_vintages = use_vintages
        self._series_cfg = self._load_series_config(self.series_config_path)
        # Monthly-aligned, *untransformed* levels for the series whose
        # signals are defined on published units (see RAW_LEVEL_SERIES).
        # Populated by run(); None until then.
        self.raw_levels_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        end_date: str | None = None,
        save_vintage: bool = False,
    ) -> pd.DataFrame:
        """Fetch, align, and transform all series into a clean panel.

        Returns
        -------
        pd.DataFrame
            Rows = monthly end-dates, columns = series codes.
        """
        if end_date is None:
            end_date = str(dt.date.today())

        raw = self._fetch_all(end_date)
        aligned = self._align_monthly(raw)

        # Keep untransformed levels for threshold-based signals *before*
        # the transform/standardise step destroys their units.
        self.raw_levels_ = aligned[
            [c for c in self.RAW_LEVEL_SERIES if c in aligned.columns]
        ].copy()

        transformed = self._transform(aligned)

        # Apply publication-lag mask so each observation is only present on
        # dates after it would have been publicly released (ragged edge).
        # Using the as-of date supplied by the caller (or "today" by default).
        if self.apply_publication_lags:
            series_lags = {
                entry["code"]: int(entry.get("publication_lag_days", 0))
                for entry in self._series_cfg
            }
            transformed = ragged_edge_mask(
                transformed,
                series_lags=series_lags,
                as_of_date=pd.Timestamp(end_date),
            )
            # The raw levels feed real-time signals too, so they get the
            # same ragged edge — otherwise the Sahm/CFNAI signals would
            # read observations that had not been published at end_date.
            self.raw_levels_ = ragged_edge_mask(
                self.raw_levels_,
                series_lags=series_lags,
                as_of_date=pd.Timestamp(end_date),
            )

        if save_vintage and self.storage is not None:
            try:
                self.storage.save_vintage(transformed, as_of=end_date)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning(f"Failed to save vintage: {exc}")

        return transformed

    def get_series_list(self) -> list[dict]:
        """Return the parsed series configuration."""
        return self._series_cfg

    def get_recession_sensitive_codes(self) -> list[str]:
        """Return the list of recession-sensitive series codes.

        These are defined in the ``recession_sensitive_series`` key of
        ``fred_series.yaml`` and represent indicators particularly
        informative for business-cycle turning-point detection.
        """
        config_path = Path(self.series_config_path)
        if not config_path.exists():
            return []
        with config_path.open() as fh:
            data = yaml.safe_load(fh)
        return data.get("recession_sensitive_series", [])

    # ------------------------------------------------------------------
    # Config loading (also used by tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_series_config(path: str | Path) -> list[dict]:
        """Read the YAML series catalogue and return a list of dicts."""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Series config not found at {path}, using empty list")
            return []
        with config_path.open() as fh:
            data = yaml.safe_load(fh)
        return data.get("series", [])

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _fetch_all(self, end_date: str) -> dict[str, pd.Series]:
        """Fetch every series from FRED."""
        if self.fred_client is None:
            raise RuntimeError("No FREDClient configured — cannot fetch data.")

        raw: dict[str, pd.Series] = {}
        # Point-in-time mode asks for the vintage that existed on
        # end_date, so revisions published later never reach the model.
        as_of = end_date if self.use_vintages else None
        for entry in self._series_cfg:
            code = entry["code"]
            try:
                s = self.fred_client.get_series(
                    code,
                    start_date=self.start_date,
                    end_date=end_date,
                    as_of=as_of,
                )
                raw[code] = s
                logger.debug(f"Fetched {code}: {len(s)} observations")
            except Exception as exc:
                logger.warning(f"Could not fetch {code}: {exc}")
        return raw

    def _align_monthly(self, raw: dict[str, pd.Series]) -> pd.DataFrame:
        """Resample all series to month-end frequency and join.

        Uses the mixed-frequency alignment module which is aware of each
        series' native frequency.  Quarterly series are expanded to monthly
        with NaN padding for non-quarter-end months (Mariano–Murasawa
        cumulator preparation).  Weekly/daily series are collapsed to
        monthly via last-observation-in-month.
        """
        return align_mixed_frequency(
            raw,
            series_config=self._series_cfg,
            method="last",
        )

    def _transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply per-series transforms as specified in config, then standardise.

        Standardisation uses an **expanding window** so the z-score at
        time *t* depends only on observations up to and including *t*.
        This removes the look-ahead bias of a full-sample mean/std.

        Drops columns that have fewer than ``standardize_min_periods``
        valid observations after transformation (too sparse to be useful).
        """
        code_to_transform = {
            entry["code"]: entry.get("transform", "none")
            for entry in self._series_cfg
        }

        min_periods = self.standardize_min_periods
        result = pd.DataFrame(index=panel.index)
        for col in panel.columns:
            t_name = code_to_transform.get(col, "none")
            transformed = apply_transform(panel[col].copy(), t_name)
            n_valid = transformed.notna().sum()
            if n_valid < min_periods:
                logger.debug(
                    f"Dropping {col}: only {n_valid} valid obs after transform"
                )
                continue
            result[col] = standardize_expanding(
                transformed, min_periods=min_periods,
            )

        # Drop any rows that are entirely NaN (before first obs window)
        result = result.dropna(how="all")

        logger.info(
            f"Pipeline: {result.shape[1]} series × {result.shape[0]} months "
            f"after filtering"
        )
        return result
