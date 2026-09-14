"""Point-in-time guarantees for the walk-forward feature generator.

``tests/test_no_leakage.py`` establishes prefix-invariance for the data
pipeline.  These tests carry the same property up to the model layer,
which is where the leaks actually were: a single full-sample fit
estimates its EM parameters, rotation, normalisation and probit
coefficients from the whole sample, so slicing a history out of it gives
values no observer could have computed at the time.

The generator avoids this by fitting once per as-of date and keeping only
the final row.  These tests hold it to that.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.data_pipeline import DataPipeline
from src.models.walk_forward import (
    MAX_PUBLICATION_LAG,
    assert_point_in_time,
    generate_feature_panel,
    generate_features_asof,
)

# Every test here refits the DFM/RSM, which takes minutes.
# Run the fast suite with `pytest -m "not slow"`.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# A small, fast synthetic pipeline
# ---------------------------------------------------------------------------


class _SyntheticPipeline(DataPipeline):
    """Pipeline over deterministic synthetic series, no network or cache.

    Deliberately small (8 series, 2 factors) so the DFM refits quickly
    enough to run a walk-forward inside a unit test.
    """

    def __init__(self, n_months: int = 300, seed: int = 4) -> None:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("1995-01-31", periods=n_months, freq="ME")

        f = np.zeros((n_months, 2))
        for t in range(1, n_months):
            f[t] = 0.9 * f[t - 1] + rng.standard_normal(2) * 0.3

        loadings = rng.standard_normal((6, 2)) * 0.2
        loadings[:3, 0] += 1.3
        loadings[3:, 1] += 1.3
        obs = f @ loadings.T + rng.standard_normal((n_months, 6)) * 0.4

        self._series = {
            f"X{i}": pd.Series(obs[:, i] + 100.0, index=idx)
            for i in range(6)
        }
        # Levels consumed by the threshold signals, in published units.
        self._series["UNRATE"] = pd.Series(
            5.0 + np.abs(f[:, 0]) * 1.5, index=idx,
        )
        self._series["CFNAI"] = pd.Series(f[:, 0] * 0.5, index=idx)

        self.fred_client = None
        self.start_date = "1995-01-01"
        self.series_config_path = "config/fred_series.yaml"
        self.storage = None
        self.apply_publication_lags = False
        self.standardize_min_periods = 24
        self.use_vintages = False
        self.raw_levels_ = None
        self._series_cfg = (
            [
                {"code": f"X{i}", "transform": "none",
                 "frequency": "monthly", "publication_lag_days": 0}
                for i in range(6)
            ]
            + [
                {"code": "UNRATE", "transform": "diff",
                 "frequency": "monthly", "publication_lag_days": 0},
                {"code": "CFNAI", "transform": "none",
                 "frequency": "monthly", "publication_lag_days": 0},
            ]
        )

    def _fetch_all(self, end_date: str) -> dict[str, pd.Series]:
        return {k: v.loc[:end_date] for k, v in self._series.items()}

    def _align_monthly(self, raw: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.concat(raw, axis=1)


@pytest.fixture(scope="module")
def pipeline() -> _SyntheticPipeline:
    return _SyntheticPipeline()


_FAST = {"n_factors": 2, "n_regimes": 2, "factor_names": ["f0", "f1"]}


# ---------------------------------------------------------------------------
# Single-row generation
# ---------------------------------------------------------------------------


def test_features_asof_returns_expected_keys(pipeline):
    row = generate_features_asof(pipeline, "2015-12-31", **_FAST)

    for key in (
        "p_recession", "signal_rsm", "signal_probit", "signal_cfnai",
        "signal_sahm", "signal_dispersion", "regime_age_months",
        "gdp_nowcast", "knowable_at",
    ):
        assert key in row, f"missing feature: {key}"

    assert 0.0 <= row["p_recession"] <= 1.0


def test_components_are_emitted_separately(pipeline):
    """The blend must not be the only thing exposed.

    A downstream model should be able to learn its own weighting — and to
    discover that one component carries no information.
    """
    row = generate_features_asof(pipeline, "2015-12-31", **_FAST)
    components = [row[f"signal_{s}"] for s in ("rsm", "probit", "cfnai", "sahm")]
    assert sum(not pd.isna(c) for c in components) >= 3


def test_knowable_at_trails_the_reference_month(pipeline):
    """Timing metadata must reflect publication lag, not the reference date."""
    as_of = pd.Timestamp("2015-12-31")
    row = generate_features_asof(pipeline, as_of, **_FAST)
    assert row["knowable_at"] == as_of + MAX_PUBLICATION_LAG
    assert row["knowable_at"] > as_of


# ---------------------------------------------------------------------------
# The central property
# ---------------------------------------------------------------------------


def test_features_are_point_in_time(pipeline):
    """History must not move when later data arrives.

    This is the gate a feature table has to pass before a downstream
    model consumes it.  Run against a single full-sample fit instead, the
    same comparison moved 21% of months by more than 0.10.
    """
    diffs = assert_point_in_time(
        pipeline,
        early_cutoff="2013-12-31",
        late_cutoff="2016-12-31",
        start="2010-01-31",
        step_months=6,
        **_FAST,
    )
    assert float(diffs["max_abs_diff"].max()) < 1e-8


def test_panel_has_no_duplicate_or_unsorted_dates(pipeline):
    panel = generate_feature_panel(
        pipeline, start="2012-01-31", end="2014-12-31",
        step_months=6, **_FAST,
    )
    assert panel.index.is_monotonic_increasing
    assert not panel.index.has_duplicates
    assert len(panel) >= 4


def test_panel_resumes_from_cache(pipeline):
    """A cached run must not recompute rows it already has."""
    # tempfile rather than pytest's tmp_path: the shared
    # ``pytest-of-<user>`` base directory is not always writable on
    # Windows, and that has nothing to do with what this test checks.
    tmp_dir = tempfile.mkdtemp(prefix="walkfwd-")
    cache = Path(tmp_dir) / "features.csv"
    first = generate_feature_panel(
        pipeline, start="2012-01-31", end="2013-12-31",
        step_months=6, cache_path=cache, **_FAST,
    )
    assert cache.exists()

    # Corrupt a cached value; if the row were recomputed it would change.
    cached = pd.read_csv(cache, index_col=0, parse_dates=True)
    cached.loc[cached.index[0], "p_recession"] = -99.0
    cached.to_csv(cache)

    second = generate_feature_panel(
        pipeline, start="2012-01-31", end="2013-12-31",
        step_months=6, cache_path=cache, **_FAST,
    )
    assert second.loc[second.index[0], "p_recession"] == -99.0, (
        "cached rows were recomputed instead of reused"
    )
    assert len(second) == len(first)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_assert_point_in_time_detects_a_leak(pipeline, monkeypatch):
    """The guard must actually fail when a feature depends on the future."""
    import src.models.walk_forward as wf

    real = wf.generate_features_asof
    calls = {"n": 0}

    def leaky(pipe, as_of, **kwargs):
        row = real(pipe, as_of, **kwargs)
        # Simulate a value that shifts once more data exists.
        calls["n"] += 1
        row["p_recession"] = row["p_recession"] + 0.01 * calls["n"]
        return row

    monkeypatch.setattr(wf, "generate_features_asof", leaky)

    with pytest.raises(AssertionError, match="not point-in-time"):
        assert_point_in_time(
            pipeline,
            early_cutoff="2013-12-31",
            late_cutoff="2015-12-31",
            start="2012-01-31",
            step_months=6,
            **_FAST,
        )
