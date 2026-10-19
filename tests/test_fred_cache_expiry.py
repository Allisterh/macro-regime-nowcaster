"""The FRED cache must expire, or the nowcaster stops nowcasting.

It never used to. The cache was filled on 2026-03-03 and served for
seven months: the CLI nowcast, the walk-forward panel, the benchmarks and
the backtests all saw macro data ending in January 2026 while FRED had it
through September, and every run logged an unremarkable "Cache hit".

The rules pinned here: latest-value caches expire after ``max_age``;
vintage caches never do, because a series as it stood on a past date
cannot change; ``max_age=None`` pins the cache for reproducible
rebuilds; and a failed refresh serves the stale copy *with a warning*,
never silently.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from loguru import logger

from src.data.fred_client import DEFAULT_MAX_AGE, FREDClient

_OLD = pd.Series(
    [1.0, 2.0], index=pd.date_range("2026-01-01", periods=2, freq="MS")
)
_NEW = pd.Series(
    [1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3, freq="MS")
)


@pytest.fixture
def cache_dir():
    d = Path(tempfile.mkdtemp(prefix="fred_cache_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fred():
    with patch("src.data.fred_client.Fred") as MockFred:
        api = MagicMock()
        api.get_series.return_value = _NEW.copy()
        MockFred.return_value = api
        yield api


def _seed(cache_dir: Path, name: str, age_days: float) -> Path:
    """Write a cached series and backdate its modification time."""
    path = cache_dir / f"{name}.csv"
    _OLD.rename(name).to_csv(path)
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_the_default_expiry_is_a_day():
    assert DEFAULT_MAX_AGE == pd.Timedelta(days=1)


def test_a_fresh_cache_is_served_without_a_call(cache_dir, fred):
    _seed(cache_dir, "PAYEMS", age_days=0.1)
    out = FREDClient(api_key="k", cache_dir=cache_dir).get_series("PAYEMS")
    assert fred.get_series.call_count == 0
    assert len(out) == 2


def test_an_expired_cache_is_refetched_and_rewritten(cache_dir, fred):
    """The case that went wrong: a months-old file must not be trusted."""
    path = _seed(cache_dir, "PAYEMS", age_days=200)
    out = FREDClient(api_key="k", cache_dir=cache_dir).get_series("PAYEMS")

    assert fred.get_series.call_count == 1
    assert len(out) == 3, "the stale copy was served instead of FRED's"
    assert len(pd.read_csv(path, index_col=0)) == 3, "the cache was not updated"


def test_vintage_caches_never_expire(cache_dir, fred):
    """A series as published on a past date cannot change."""
    _seed(cache_dir, "PAYEMS_v20200615", age_days=2000)
    out = FREDClient(api_key="k", cache_dir=cache_dir).get_series(
        "PAYEMS", as_of="2020-06-15"
    )
    assert fred.get_series.call_count == 0
    assert len(out) == 2


def test_a_pinned_cache_never_expires(cache_dir, fred):
    """max_age=None is what a reproducible walk-forward rebuild needs."""
    _seed(cache_dir, "PAYEMS", age_days=2000)
    out = FREDClient(
        api_key="k", cache_dir=cache_dir, max_age=None
    ).get_series("PAYEMS")
    assert fred.get_series.call_count == 0
    assert len(out) == 2


def test_zero_max_age_refetches_everything(cache_dir, fred):
    _seed(cache_dir, "PAYEMS", age_days=0.0)
    FREDClient(
        api_key="k", cache_dir=cache_dir, max_age=pd.Timedelta(0)
    ).get_series("PAYEMS")
    assert fred.get_series.call_count == 1


def test_a_failed_refresh_serves_the_stale_copy_loudly(cache_dir, fred):
    """Offline must still work — but never silently."""
    _seed(cache_dir, "PAYEMS", age_days=200)
    fred.get_series.side_effect = ConnectionError("no network")

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        out = FREDClient(api_key="k", cache_dir=cache_dir).get_series("PAYEMS")
    finally:
        logger.remove(sink)

    assert len(out) == 2
    warning = " ".join(messages)
    assert "Could not refresh PAYEMS" in warning
    assert "2026-02-01" in warning, "the warning must name how stale it is"


def test_a_narrow_request_does_not_truncate_the_cache(cache_dir, fred):
    """The cache is keyed by series id, so it must hold the whole series.

    Requests used to pass their date range straight to FRED and cache the
    result. A backtest refitting as of an early date would then leave a
    file ending on that date, and every later reader — wanting recent
    data — would silently get the truncated copy. Expiry makes refetches
    routine, so this had to be fixed alongside it.
    """
    full = pd.Series(
        range(120), index=pd.date_range("2000-01-01", periods=120, freq="MS"),
        dtype=float,
    )

    def like_fred(series_id, observation_start=None, observation_end=None, **_):
        # Honour the bounds the way the real API does. A mock that always
        # returned the full series let an earlier version of this test pass
        # against the very bug it describes.
        out = full
        if observation_start:
            out = out.loc[observation_start:]
        if observation_end:
            out = out.loc[:observation_end]
        return out.copy()

    fred.get_series.side_effect = like_fred

    client = FREDClient(api_key="k", cache_dir=cache_dir)
    early = client.get_series("PAYEMS", start_date="2001-01-01", end_date="2002-12-31")
    assert early.index.min() >= pd.Timestamp("2001-01-01")
    assert early.index.max() <= pd.Timestamp("2002-12-31")

    later = client.get_series("PAYEMS", start_date="2000-01-01")
    assert fred.get_series.call_count == 1, "the second read should hit the cache"
    assert len(later) == 120, (
        f"the cache held only the first caller's window ({len(later)} rows "
        f"of 120)"
    )


def test_with_no_cache_a_failed_fetch_still_raises(cache_dir, fred):
    """The fallback exists only when there is something stale to fall back to."""
    fred.get_series.side_effect = ConnectionError("no network")
    with pytest.raises(ConnectionError):
        FREDClient(api_key="k", cache_dir=cache_dir).get_series("PAYEMS")
