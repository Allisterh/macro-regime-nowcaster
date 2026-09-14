"""Tests for ALFRED vintage data support in FREDClient."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


def _mock_fred_context():
    """Return a patched Fred + mock instance + sample series."""
    ctx = patch("src.data.fred_client.Fred")
    MockFred = ctx.start()
    mock_inst = MagicMock()
    MockFred.return_value = mock_inst
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    mock_inst.get_series.return_value = pd.Series(
        range(12), index=idx, name="PAYEMS",
    )
    return ctx, mock_inst


def test_get_series_as_of_requests_a_single_vintage():
    """Both realtime bounds must equal as_of.

    ALFRED treats ``realtime_start``/``realtime_end`` as a *range* of
    vintages.  Passing ``realtime_start=start_date`` (as this code
    originally did) asks for every revision published between the start
    of the sample and ``as_of``, which returns duplicated observation
    dates rather than the single point-in-time snapshot ``as_of``
    promises.  A true vintage pins both bounds to the same date.
    """
    ctx, mock_inst = _mock_fred_context()
    try:
        from src.data.fred_client import FREDClient
        client = FREDClient(api_key="test_key", cache_dir=None)
        client.get_series(
            "PAYEMS",
            start_date="2020-01-01",
            end_date="2020-12-31",
            as_of="2020-06-15",
            use_cache=False,
        )
        call_kwargs = mock_inst.get_series.call_args
        assert call_kwargs.kwargs.get("realtime_end") == "2020-06-15"
        assert call_kwargs.kwargs.get("realtime_start") == "2020-06-15"
        # Observation bounds are unaffected by the vintage request.
        assert call_kwargs.kwargs.get("observation_start") == "2020-01-01"
        assert call_kwargs.kwargs.get("observation_end") == "2020-12-31"
    finally:
        ctx.stop()


def test_get_series_as_of_deduplicates_overlapping_vintages():
    """Duplicate observation dates collapse to the latest value."""
    ctx = patch("src.data.fred_client.Fred")
    MockFred = ctx.start()
    mock_inst = MagicMock()
    MockFred.return_value = mock_inst
    # Simulate a provider returning two vintages of the same month.
    idx = pd.to_datetime(
        ["2020-01-31", "2020-02-29", "2020-02-29", "2020-03-31"]
    )
    mock_inst.get_series.return_value = pd.Series(
        [1.0, 2.0, 2.5, 3.0], index=idx, name="PAYEMS",
    )
    try:
        from src.data.fred_client import FREDClient
        client = FREDClient(api_key="test_key", cache_dir=None)
        out = client.get_series("PAYEMS", as_of="2020-06-15", use_cache=False)
        assert not out.index.has_duplicates
        assert out.loc["2020-02-29"] == 2.5
    finally:
        ctx.stop()


def test_get_series_without_as_of_no_realtime():
    """Without as_of, realtime_end should not be passed."""
    ctx, mock_inst = _mock_fred_context()
    try:
        from src.data.fred_client import FREDClient
        client = FREDClient(api_key="test_key", cache_dir=None)
        client.get_series(
            "PAYEMS",
            start_date="2020-01-01",
            use_cache=False,
        )
        call_kwargs = mock_inst.get_series.call_args
        assert "realtime_end" not in (call_kwargs.kwargs or {})
    finally:
        ctx.stop()


def test_vintage_cache_keyed_by_date():
    """Different as_of dates should produce separate cache files."""
    ctx, mock_inst = _mock_fred_context()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            from src.data.fred_client import FREDClient
            client = FREDClient(api_key="test_key", cache_dir=cache_dir)
            client.get_series("PAYEMS", as_of="2020-06-15", use_cache=True)
            client.get_series("PAYEMS", as_of="2020-09-15", use_cache=True)
            cache_files = list(cache_dir.glob("PAYEMS*.csv"))
            assert len(cache_files) == 2
            names = {f.stem for f in cache_files}
            assert "PAYEMS_v20200615" in names
            assert "PAYEMS_v20200915" in names
    finally:
        ctx.stop()


def test_vintage_cache_separate_from_latest():
    """Latest (no as_of) and vintage should be separate cache entries."""
    ctx, mock_inst = _mock_fred_context()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            from src.data.fred_client import FREDClient
            client = FREDClient(api_key="test_key", cache_dir=cache_dir)
            client.get_series("PAYEMS", use_cache=True)
            client.get_series("PAYEMS", as_of="2020-06-15", use_cache=True)
            cache_files = list(cache_dir.glob("PAYEMS*.csv"))
            assert len(cache_files) == 2
    finally:
        ctx.stop()
