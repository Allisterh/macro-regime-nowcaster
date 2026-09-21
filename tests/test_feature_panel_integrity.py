"""The downstream feature panel must not contain fabricated values.

Three defects found in a shipped ``data/features.csv`` of 690 rows, none
of which raised anything:

1. 25 rows had every factor exactly ``0.0``, ``signal_rsm`` exactly 0.5
   and ``p_stay_recession`` exactly 0.0 — the DFM had returned zeros
   rather than failing, and the row was written with a full set of
   plausible numbers. They clustered in 2020 and later, 11 of them in
   the final 25 months.
2. ``expected_recession_duration`` reached 7.5e11 months, because
   ``1 / (1 - p)`` was guarded only against ``p == 1.0`` exactly and an
   estimated ``p`` of 0.9999999999946 clears that test.
3. ``generate_features_asof`` defaulted to ``n_factors=4`` after the
   validated default became 5, so the documented way to rebuild the
   panel produced an unmeasured configuration — and with five names
   competing for four factors, ``long_rates`` was dropped entirely.

These tests use a stubbed Nowcaster rather than a real fit, so they run
in milliseconds and stay in the fast suite; the class of bug they guard
is precisely the kind that a slow, rarely-run test would not catch.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.nowcaster import Nowcaster
from src.models.walk_forward import (
    generate_feature_panel,
    generate_features_asof,
)


@pytest.fixture
def scratch():
    """A throwaway directory.

    Uses ``tempfile`` rather than pytest's ``tmp_path`` because this
    environment's pytest basetemp is not writable; ``test_walk_forward``
    does the same.
    """
    path = Path(tempfile.mkdtemp())
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


_IDX = pd.date_range("2000-01-31", periods=60, freq="ME")


class _StubResult:
    recession_probability = 0.30
    ensemble_detail = {"rsm": 0.5, "probit": 0.2, "cfnai": 0.1, "sahm": 0.1}
    gdp_nowcast = 2.0
    gdp_ci_lower = 0.0
    gdp_ci_upper = 4.0


class _StubRSM:
    def __init__(self, stay_recession: float) -> None:
        self._stay = stay_recession

    def get_transition_matrix(self) -> np.ndarray:
        return np.array([[self._stay, 1.0 - self._stay], [0.05, 0.95]])


def _stub_nowcaster_factory(factor_values: float, stay_recession: float):
    """A Nowcaster replacement that fits nothing and returns fixed state."""

    class _StubNowcaster:
        DEFAULT_FACTOR_NAMES = Nowcaster.DEFAULT_FACTOR_NAMES

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self._last_factors = pd.DataFrame(
                factor_values, index=_IDX, columns=["a", "b"],
            )
            self._ensemble_recession_ts = pd.Series(0.3, index=_IDX)
            self._rsm = _StubRSM(stay_recession)

        def run(self, end_date=None):
            return _StubResult()

    return _StubNowcaster


def _patch(monkeypatch, factor_values: float, stay_recession: float) -> None:
    monkeypatch.setattr(
        "src.models.walk_forward.Nowcaster",
        _stub_nowcaster_factory(factor_values, stay_recession),
    )


# ---------------------------------------------------------------------------
# 1. A failed fit must not be written as a row of zeros
# ---------------------------------------------------------------------------


def test_all_zero_factors_raise_instead_of_being_written(monkeypatch):
    """The panel skips a failed window; it must never fabricate one."""
    _patch(monkeypatch, factor_values=0.0, stay_recession=0.8)

    with pytest.raises(RuntimeError, match="all-zero factors"):
        generate_features_asof(
            object(), "2015-12-31", respect_nber_announcement_lag=False,
        )


def test_ordinary_factors_are_accepted(monkeypatch):
    """The guard must not reject a healthy fit."""
    _patch(monkeypatch, factor_values=0.4, stay_recession=0.8)

    row = generate_features_asof(
        object(), "2015-12-31", respect_nber_announcement_lag=False,
    )
    assert row["factor_a"] == pytest.approx(0.4)


def test_a_single_zero_factor_is_not_rejected(monkeypatch):
    """Only an entirely zero row indicates failure, not one zero column."""
    factory = _stub_nowcaster_factory(0.0, 0.8)

    class _OneNonZero(factory):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._last_factors["b"] = 0.7

    monkeypatch.setattr("src.models.walk_forward.Nowcaster", _OneNonZero)

    row = generate_features_asof(
        object(), "2015-12-31", respect_nber_announcement_lag=False,
    )
    assert row["factor_a"] == 0.0
    assert row["factor_b"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. Expected duration must stay on a usable scale
# ---------------------------------------------------------------------------


def test_expected_duration_is_bounded_by_the_sample(monkeypatch):
    """A near-unit persistence must saturate, not return 1e13."""
    _patch(monkeypatch, factor_values=0.4, stay_recession=1.0 - 1e-13)

    row = generate_features_asof(
        object(), "2015-12-31", respect_nber_announcement_lag=False,
    )
    duration = row["expected_recession_duration"]
    assert np.isfinite(duration)
    assert duration <= len(_IDX), (
        f"expected_recession_duration is {duration:.3g}, beyond what a "
        f"{len(_IDX)}-observation sample can identify"
    )


def test_exact_unit_persistence_saturates_rather_than_returning_nan(monkeypatch):
    _patch(monkeypatch, factor_values=0.4, stay_recession=1.0)

    row = generate_features_asof(
        object(), "2015-12-31", respect_nber_announcement_lag=False,
    )
    assert row["expected_recession_duration"] == pytest.approx(len(_IDX))


def test_ordinary_persistence_is_left_alone(monkeypatch):
    """The cap must not distort values the sample can actually resolve."""
    _patch(monkeypatch, factor_values=0.4, stay_recession=0.9)

    row = generate_features_asof(
        object(), "2015-12-31", respect_nber_announcement_lag=False,
    )
    assert row["expected_recession_duration"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. The factor count belongs to the model
# ---------------------------------------------------------------------------


def test_default_factor_count_comes_from_the_model():
    """A literal here silently builds the panel at the wrong K."""
    default = inspect.signature(
        generate_features_asof
    ).parameters["n_factors"].default

    assert default == len(Nowcaster.DEFAULT_FACTOR_NAMES), (
        f"generate_features_asof defaults to n_factors={default} while the "
        f"model defines {len(Nowcaster.DEFAULT_FACTOR_NAMES)} factors; the "
        f"downstream panel would be built at an unvalidated K"
    )


# ---------------------------------------------------------------------------
# 4. A run that skips most of its history must not report success
# ---------------------------------------------------------------------------


def _failing_panel(monkeypatch, fail_before: str):
    """Patch feature generation to fail for every date before *fail_before*."""
    cutoff = pd.Timestamp(fail_before)

    def _fake(pipeline, as_of, **kwargs):
        as_of = pd.Timestamp(as_of)
        if as_of < cutoff:
            raise RuntimeError("Data pipeline returned an empty panel")
        return {"p_recession": 0.2, "knowable_at": as_of}

    monkeypatch.setattr(
        "src.models.walk_forward.generate_features_asof", _fake,
    )


def test_a_long_run_of_failed_windows_raises(monkeypatch):
    """Skipping 13 years must not be reported as a successful build.

    Pointing the pipeline at 1980 while asking for as-of dates from 1967
    failed 120 consecutive windows — the 1969-70 and 1973-75 recessions —
    each a logged warning, and the run still finished and wrote a panel.
    """
    _failing_panel(monkeypatch, fail_before="1980-01-01")

    with pytest.raises(RuntimeError, match="windows failed"):
        generate_feature_panel(
            object(), start="1967-01-31", end="2000-01-31", step_months=1,
        )


def test_a_few_failed_windows_are_tolerated(monkeypatch):
    """Isolated failures are normal and must not abort a long run."""
    _failing_panel(monkeypatch, fail_before="1967-04-30")

    panel = generate_feature_panel(
        object(), start="1967-01-31", end="2000-01-31", step_months=1,
    )
    assert len(panel) > 380
    assert panel.index.is_monotonic_increasing


def test_the_failure_threshold_can_be_relaxed(monkeypatch):
    """A caller who knows the history is short can opt out."""
    _failing_panel(monkeypatch, fail_before="1980-01-01")

    panel = generate_feature_panel(
        object(), start="1967-01-31", end="2000-01-31", step_months=1,
        max_failure_fraction=1.0,
    )
    assert panel.index[0] >= pd.Timestamp("1980-01-01")


# ---------------------------------------------------------------------------
# 5. A cache from superseded code must not be resumed from
# ---------------------------------------------------------------------------


def test_cache_without_knowable_at_is_ignored(monkeypatch, scratch):
    """A stale cache in an old schema must not be merged into a fresh run.

    ``data/oos_validation_interim.csv`` was sitting in the repository in
    the superseded ``p_rsm, p_probit, p_cfnai, p_ensemble`` format, with
    an inverted RSM at 0.98 and a probit pinned to its since-removed 0.05
    clip floor. ``run_validation.py`` passes that path as ``cache_path``,
    so a run today would have resumed from 28 rows of known-bad numbers
    under column names nothing downstream reads — and the metrics
    computed over the result would have looked entirely ordinary.
    """
    stale = scratch / "old_format.csv"
    pd.DataFrame(
        {"p_rsm": [0.98, 0.98], "p_probit": [0.05, 0.05]},
        index=pd.to_datetime(["1990-01-31", "1990-02-28"]),
    ).to_csv(stale)

    _failing_panel(monkeypatch, fail_before="1900-01-01")  # nothing fails

    panel = generate_feature_panel(
        object(), start="1990-01-31", end="1992-01-31", step_months=1,
        cache_path=stale,
    )

    assert "p_rsm" not in panel.columns, (
        "columns from a superseded cache schema leaked into the panel"
    )
    assert panel["p_recession"].notna().all(), (
        "rows resumed from the stale cache left holes in the output"
    )


def test_a_matching_cache_is_still_resumed(monkeypatch, scratch):
    """The guard must not disable resuming, which is why the cache exists."""
    good = scratch / "partial.csv"
    pd.DataFrame(
        {
            "p_recession": [0.11, 0.22],
            "knowable_at": pd.to_datetime(["1990-04-01", "1990-05-01"]),
        },
        index=pd.to_datetime(["1990-01-31", "1990-02-28"]),
    ).to_csv(good)

    _failing_panel(monkeypatch, fail_before="1900-01-01")

    panel = generate_feature_panel(
        object(), start="1990-01-31", end="1992-01-31", step_months=1,
        cache_path=good,
    )

    # The cached values must survive rather than being recomputed.
    assert panel.loc[pd.Timestamp("1990-01-31"), "p_recession"] == 0.11
    assert panel.loc[pd.Timestamp("1990-02-28"), "p_recession"] == 0.22
