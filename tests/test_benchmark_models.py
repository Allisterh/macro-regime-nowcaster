"""The downstream benchmark must measure the features, not their units.

``benchmark_features.py`` compared feature sets with
``Ridge(alpha=1.0)`` fitted on raw columns.  The L2 penalty is not
scale-invariant: a column's coefficient scales as 1/sd, so the penalty
``alpha * b**2`` falls hardest on the *narrowest* columns.  The panel
mixes probabilities (sd ~ 0.1) with ``expected_recession_duration``
(sd ~ 3e10), so how much each feature was shrunk was decided by the
units it happened to be recorded in.

Measured on the panel as published, standardising alone moved the
full-panel R² on returns from -0.299 to -3.355 — the raw fit had been
over-shrinking the probability columns, which is regularisation by
accident and would move again the moment a column changed units. The
verdict did not change (every R² is negative either way), but the
numbers were an artifact of scale.

These tests assert invariance out-of-sample, through the benchmark's own
``evaluate`` and CV splitter, because that is where the property lives:
in-sample predictions barely move (~1e-7 relative) even for the broken
model, so an in-sample check passes happily against the bug.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_features import build_models, evaluate
from src.evaluation.purged_cv import PurgedWalkForward

# Invariance should hold to floating-point precision, not approximately.
_TOL = 1e-9


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    """A panel with real signal, wide enough that the penalty binds."""
    rng = np.random.default_rng(0)
    n, p = 300, 12
    X = pd.DataFrame(
        rng.standard_normal((n, p)), columns=[f"c{i}" for i in range(p)]
    )
    beta = rng.standard_normal(p) * 0.3
    y = pd.Series(X.values @ beta + rng.standard_normal(n) * 1.5)
    return X, y


@pytest.fixture(scope="module")
def cv() -> PurgedWalkForward:
    return PurgedWalkForward(
        n_splits=5, label_horizon=3, embargo=3, min_train=80
    )


def _oos_r2(X: pd.DataFrame, y: pd.Series, model_fn, cv) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, r2, _ = evaluate(X, y, model_fn, cv)
    return r2


@pytest.mark.parametrize("factor", [1e-4, 1e4])
def test_ridge_is_invariant_to_a_change_of_units(sample, cv, factor):
    """Recording one column in different units must not change the score.

    Both directions are checked. Widening a column is the case the
    duration bug presented as, but *narrowing* is where a fixed-alpha
    ridge moves most, because the penalty bites hardest on narrow
    columns — that is the direction an in-sample check misses.
    """
    X, y = sample
    ridge = build_models()["ridge"]

    rescaled = X.copy()
    rescaled["c0"] = rescaled["c0"] * factor

    base = _oos_r2(X, y, ridge, cv)
    moved = _oos_r2(rescaled, y, ridge, cv)

    assert abs(moved - base) < _TOL, (
        f"scaling one column by {factor:g} moved out-of-sample R² from "
        f"{base:+.4f} to {moved:+.4f}. The benchmark is measuring the "
        f"units the features are recorded in, not the features."
    )


def test_ridge_penalty_is_selected_not_fixed(sample):
    """A single hard-coded alpha cannot suit every feature-set width.

    The benchmark compares a one-column set against a 35-column set; one
    constant cannot be right for both, and fixing it makes the
    comparison partly a statement about that choice.
    """
    X, y = sample
    fitted = build_models()["ridge"]().fit(X, y)

    selected = getattr(fitted, "best_params_", None) or getattr(
        fitted, "alpha_", None
    )
    assert selected is not None, (
        "the benchmark's ridge does not select its penalty; a fixed alpha "
        "makes the feature-set comparison depend on that constant"
    )


def test_penalty_search_is_purged_not_leave_one_out(sample):
    """The inner search must respect the same label overlap as the outer.

    ``RidgeCV``'s leave-one-out GCV is wrong for an overlapping forward
    target: the held-out point's label is computed from prices it shares
    with its immediate neighbours, and those neighbours stay in the
    training set. The held-out error is therefore optimistic and the
    search is pulled toward too small a penalty — under-regularising the
    exact comparison this script exists to make.
    """
    X, y = sample
    fitted = build_models(label_horizon=3, embargo=3)["ridge"]()

    inner = getattr(fitted, "cv", None)
    found = type(inner).__name__ if inner is not None else "a default (LOO) splitter"
    assert isinstance(inner, PurgedWalkForward), (
        f"the penalty search uses {found}; with overlapping forward "
        f"windows it must purge and embargo like the outer loop"
    )
    assert inner.label_horizon == 3, (
        "the inner splitter's label horizon does not match the target's, "
        "so it purges the wrong samples"
    )
    assert inner.embargo == 3


def test_purged_splitter_accepts_the_sklearn_signature(sample):
    """It has to be usable as a `cv=` argument for the above to hold."""
    X, y = sample
    cv = PurgedWalkForward(n_splits=3, label_horizon=3, embargo=3, min_train=30)

    folds = list(cv.split(X, y))
    assert folds, "no folds yielded"
    assert cv.get_n_splits(X) == len(folds)
    # Splits must depend on position only, never on the target.
    assert [
        (a.tolist(), b.tolist()) for a, b in cv.split(X, y * 100.0)
    ] == [(a.tolist(), b.tolist()) for a, b in folds]


@pytest.mark.parametrize("factor", [1e-4, 1e4])
def test_gbm_is_scale_free(sample, cv, factor):
    """Trees split on order, so the GBM was never affected — keep it so."""
    X, y = sample
    gbm = build_models()["gbm"]

    rescaled = X.copy()
    rescaled["c0"] = rescaled["c0"] * factor

    base = _oos_r2(X, y, gbm, cv)
    moved = _oos_r2(rescaled, y, gbm, cv)

    assert abs(moved - base) < _TOL, (
        f"the GBM moved from {base:+.4f} to {moved:+.4f} under a change of "
        f"units, which trees should be immune to"
    )
