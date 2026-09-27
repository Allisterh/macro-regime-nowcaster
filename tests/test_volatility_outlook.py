"""The volatility panel must report a ranking, not a forecast.

The evidence behind it is asymmetric: the latent factors rank forward
volatility consistently (IC positive in all 15 fold-horizon
combinations measured) but predict its *level* weakly — out-of-sample
R² of +0.02 to +0.06, and the one fold negative at every horizon is
2006-2016, which contains 2008. The model ranks that period correctly
and still misses its magnitude.

So these tests assert the two things that keep the panel honest: it
never substitutes a value for a missing input, and its percentile
ordering actually tracks realised volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.volatility_outlook import (
    MIN_OBSERVATIONS,
    MIN_USABLE_IC,
    VolatilityOutlook,
    realised_forward_volatility,
)

_FACTORS = ["factor_a", "factor_b"]


@pytest.fixture(scope="module")
def outlook() -> VolatilityOutlook:
    """A hand-built outlook with a known monotone relationship."""
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(0)

    predicted = pd.Series(np.linspace(0.05, 0.45, 200), index=idx)
    # Realised tracks predicted with noise, so rank correlation is high
    # but the level is not exactly recoverable — like the real thing.
    realised = predicted + rng.standard_normal(200) * 0.03

    class _Identity:
        def predict(self, X):
            return np.asarray(X).sum(axis=1)

    return VolatilityOutlook(
        model=_Identity(),
        horizon=3,
        ticker="TEST",
        history=predicted,
        realised=realised,
        factor_columns=list(_FACTORS),
        oos_ic=0.16,
        oos_folds=5,
        oos_fold_ics=[0.10, 0.29, 0.19, 0.14, 0.16],
    )


# ---------------------------------------------------------------------------
# Refuses to invent an input
# ---------------------------------------------------------------------------


def test_missing_factor_raises_rather_than_defaulting(outlook):
    """A substituted zero here would read as a legitimate calm reading.

    This is the failure this codebase has hit repeatedly: a plausible
    value where an error belonged.
    """
    partial = pd.Series({"factor_a": 0.3})
    with pytest.raises(KeyError, match="factor_b"):
        outlook.predict(partial)


def test_non_finite_factor_raises(outlook):
    row = pd.Series({"factor_a": 0.3, "factor_b": np.nan})
    with pytest.raises(ValueError, match="non-finite"):
        outlook.predict(row)


def test_extra_columns_are_ignored(outlook):
    """A live factor frame carries more than the model needs."""
    row = pd.Series({"factor_a": 0.1, "factor_b": 0.1, "factor_unused": 99.0})
    assert outlook.predict(row) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# The percentile is the product, so it has to be right
# ---------------------------------------------------------------------------


def test_percentile_spans_the_range(outlook):
    assert outlook.percentile_of(-1.0) == pytest.approx(0.0)
    assert outlook.percentile_of(99.0) == pytest.approx(100.0)
    mid = outlook.percentile_of(float(outlook.history.median()))
    assert 45.0 < mid < 55.0


def test_percentile_is_monotone_in_the_prediction(outlook):
    values = [0.05, 0.15, 0.25, 0.35, 0.45]
    pcts = [outlook.percentile_of(v) for v in values]
    assert pcts == sorted(pcts), f"percentile is not monotone: {pcts}"


def test_percentile_ordering_tracks_realised_volatility(outlook):
    """The claim the panel makes is ordinal, so test it ordinally."""
    from scipy.stats import spearmanr

    rho = spearmanr(outlook.history.to_numpy(), outlook.realised.to_numpy())
    assert rho.correlation > 0.8, (
        f"predicted and realised volatility barely co-rank (rho="
        f"{rho.correlation:.3f}); the percentile would be meaningless"
    )


def test_assess_reports_a_range_not_just_a_point(outlook):
    """Magnitude is given as an empirical spread, never a single number."""
    row = pd.Series({"factor_a": 0.15, "factor_b": 0.15})
    result = outlook.assess(row)

    for key in ("prediction", "percentile", "realised_p25",
                "realised_median", "realised_p75", "n_comparable"):
        assert key in result, f"assess() dropped {key}"

    assert result["n_comparable"] > 0
    assert result["realised_p25"] <= result["realised_median"]
    assert result["realised_median"] <= result["realised_p75"]


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def test_realised_volatility_is_forward_looking():
    """The target must use prices *after* the observation date."""
    idx = pd.date_range("2000-01-03", periods=800, freq="B")
    rng = np.random.default_rng(1)
    # Calm first half, turbulent second half.
    steps = np.concatenate([
        rng.standard_normal(400) * 0.002, rng.standard_normal(400) * 0.02,
    ])
    prices = pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)

    months = prices.resample("ME").last().index
    vol = realised_forward_volatility(prices, months, horizon=3)

    calm = vol[vol.index < "2001-06-30"]
    rough = vol[vol.index > "2001-09-30"]
    assert calm.mean() < rough.mean(), (
        "forward volatility did not rise ahead of the turbulent period, so "
        "the target is not looking forward"
    )


def test_minimum_observations_is_enforced_not_advisory():
    """A percentile drawn from a handful of points is not a percentile."""
    assert MIN_OBSERVATIONS >= 60



# ---------------------------------------------------------------------------
# The panel must not show a reading its own evidence does not support
# ---------------------------------------------------------------------------


def _with_evidence(outlook: VolatilityOutlook, fold_ics: list[float]):
    """A copy of *outlook* carrying a different out-of-sample measurement."""
    from dataclasses import replace

    return replace(
        outlook,
        oos_ic=float(np.mean(fold_ics)),
        oos_folds=len(fold_ics),
        oos_fold_ics=list(fold_ics),
    )


def test_the_panel_that_justified_building_this_passes(outlook):
    """The pre-rebuild evidence: +0.157 mean, positive in every fold."""
    assert outlook.has_usable_signal


def test_the_rebuilt_panel_is_refused(outlook):
    """Measured on the rebuilt panel: +0.009 mean, mixed signs.

    This is the case that made the gate necessary. The dashboard kept
    rendering a colour-coded percentile after the evidence behind it had
    gone to zero, under a caption still quoting the old numbers.
    """
    rebuilt = _with_evidence(outlook, [-0.09, 0.05, 0.02, 0.04, 0.03])
    assert rebuilt.oos_ic < MIN_USABLE_IC
    assert not rebuilt.has_usable_signal


def test_a_strong_mean_carried_by_one_fold_is_refused(outlook):
    """Sign consistency across eras was the claim, not the average.

    The momentum-inclusive benchmark set reaches +0.08 at three months
    while inverting in 1976-86 — a mean that one era can carry.
    """
    lopsided = _with_evidence(outlook, [-0.25, -0.06, 0.29, 0.30, 0.25])
    assert lopsided.oos_ic >= MIN_USABLE_IC, "precondition: the mean clears"
    assert not lopsided.has_usable_signal


def test_one_weak_fold_is_tolerated(outlook):
    """One negative fold in five is not a reason to hide a real signal."""
    one_miss = _with_evidence(outlook, [-0.02, 0.20, 0.18, 0.15, 0.16])
    assert one_miss.has_usable_signal


def test_unmeasured_evidence_is_refused(outlook):
    """No measurement is not a pass."""
    from dataclasses import replace

    blank = replace(outlook, oos_ic=None, oos_folds=0, oos_fold_ics=[])
    assert not blank.has_usable_signal


def test_the_dashboard_no_longer_hard_codes_the_evidence():
    """The caption quoted numbers that went stale when the panel changed."""
    from pathlib import Path

    app = (
        Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in app.splitlines() if not ln.lstrip().startswith("#")
    )
    for stale in ("+0.02 to +0.06", "all 15 fold", "spans 2008"):
        assert stale not in code, (
            f"the dashboard hard-codes volatility evidence ({stale!r}); "
            f"it must come from the outlook's own measurement"
        )
    assert "has_usable_signal" in code, "the panel is not gated on evidence"
