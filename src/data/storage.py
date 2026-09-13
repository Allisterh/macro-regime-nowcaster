"""DuckDB-backed persistence layer for nowcaster data.

Stores raw series, transformed panels, and model vintages so that
historical nowcasts can be replayed without re-fetching from FRED.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DataStorage:
    """Thin wrapper around a DuckDB database file.

    Parameters
    ----------
    db_path : str | Path
        Filesystem path for the DuckDB file. Created if absent.
    """

    def __init__(self, db_path: str | Path = "data/nowcaster.duckdb") -> None:
        if duckdb is None:
            raise ImportError("duckdb is required: pip install duckdb")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path))
        self._ensure_tables()
        logger.debug(f"DataStorage connected to {self.db_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_vintage(
        self,
        panel: pd.DataFrame,
        as_of: str | None = None,
    ) -> None:
        """Persist a vintage of the transformed panel."""
        as_of = as_of or str(dt.date.today())
        # Store as a long-format table
        long = panel.reset_index().melt(
            id_vars=panel.index.name or "index",
            var_name="series_code",
            value_name="value",
        )
        long.rename(columns={panel.index.name or "index": "date"}, inplace=True)
        long["vintage_date"] = as_of
        self._con.execute(
            "INSERT INTO vintages SELECT * FROM long",
        )
        logger.debug(f"Saved vintage as_of={as_of}, {len(long)} rows")

    def load_vintage(self, as_of: str) -> pd.DataFrame:
        """Load a vintage panel by its as-of date."""
        df = self._con.execute(
            "SELECT date, series_code, value FROM vintages WHERE vintage_date = ?",
            [as_of],
        ).fetchdf()
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="date", columns="series_code", values="value")

    def list_vintages(self) -> list[str]:
        """Return all available vintage dates."""
        result = self._con.execute(
            "SELECT DISTINCT vintage_date FROM vintages ORDER BY vintage_date"
        ).fetchdf()
        return result["vintage_date"].tolist()

    def close(self) -> None:
        """Close the underlying connection."""
        self._con.close()
        logger.debug("DataStorage connection closed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS vintages (
                date          DATE,
                series_code   VARCHAR,
                value         DOUBLE,
                vintage_date  VARCHAR
            )
        """)
