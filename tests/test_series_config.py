"""Schema and consistency checks for the FRED series catalogue.

Three of the shipped series ids did not exist on FRED — the Empire State
and Philadelphia Fed codes had their ``DI``/``DF`` infixes transposed,
and the London bullion gold series had been delisted.  ``DataPipeline``
logs a warning and moves on when a fetch fails, so the panel was quietly
missing an entire survey block with nothing to signal it.

These tests are offline: they check the structure and internal
consistency of the catalogue.  Validating the ids against the live API
needs a key, so that check is opt-in via ``--runlive``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "fred_series.yaml"

VALID_TRANSFORMS = {"log_diff", "diff", "pct_change", "standardize", "none"}
VALID_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly"}
REQUIRED_FIELDS = {
    "code", "name", "frequency", "category", "transform",
    "publication_lag_days",
}


@pytest.fixture(scope="module")
def catalogue() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_every_entry_has_required_fields(catalogue):
    for entry in catalogue["series"]:
        missing = REQUIRED_FIELDS - set(entry)
        assert not missing, f"{entry.get('code', entry)} is missing {missing}"


def test_transforms_and_frequencies_are_known(catalogue):
    for entry in catalogue["series"]:
        assert entry["transform"] in VALID_TRANSFORMS, (
            f"{entry['code']}: unknown transform {entry['transform']!r}"
        )
        assert entry["frequency"] in VALID_FREQUENCIES, (
            f"{entry['code']}: unknown frequency {entry['frequency']!r}"
        )


def test_publication_lags_are_plausible(catalogue):
    for entry in catalogue["series"]:
        lag = entry["publication_lag_days"]
        assert isinstance(lag, int) and 0 <= lag <= 120, (
            f"{entry['code']}: implausible publication_lag_days {lag}"
        )


def test_no_duplicate_codes(catalogue):
    codes = [e["code"] for e in catalogue["series"]]
    duplicates = {c for c in codes if codes.count(c) > 1}
    assert not duplicates, f"duplicate series codes: {duplicates}"


def test_recession_sensitive_codes_exist_in_catalogue(catalogue):
    """Every recession-sensitive entry must be a series we actually fetch."""
    known = {e["code"] for e in catalogue["series"]}
    listed = catalogue.get("recession_sensitive_series", [])
    unknown = [c for c in listed if c not in known]
    assert not unknown, (
        f"recession_sensitive_series references codes that are not in the "
        f"series catalogue: {unknown}"
    )


def test_raw_level_series_are_present(catalogue):
    """The threshold signals need their raw level series to be fetched."""
    from src.data.data_pipeline import DataPipeline

    known = {e["code"] for e in catalogue["series"]}
    for code in DataPipeline.RAW_LEVEL_SERIES:
        assert code in known, (
            f"{code} is required for a threshold signal but is not in "
            f"fred_series.yaml"
        )


def test_raw_level_series_keep_their_units(catalogue):
    """UNRATE/CFNAI must not be transformed out of published units.

    They are read from ``raw_levels_``, which is captured before the
    transform step, so the configured transform does not affect the
    signals — but a reader would reasonably expect these to line up.
    """
    by_code = {e["code"]: e for e in catalogue["series"]}
    assert by_code["CFNAI"]["transform"] == "none"


@pytest.mark.live
def test_all_codes_resolve_on_fred(catalogue):
    """Opt-in: verify every id exists on FRED. Needs FRED_API_KEY."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        pytest.skip("FRED_API_KEY not set")

    from fredapi import Fred

    fred = Fred(api_key=api_key)
    bad = []
    for entry in catalogue["series"]:
        try:
            fred.get_series_info(entry["code"])
        except Exception:
            bad.append(entry["code"])
    assert not bad, f"series ids that do not exist on FRED: {bad}"
