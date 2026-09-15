"""Tests for the probit recession probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recession_probit import RecessionProbit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probit_data():
    """Synthetic labelled data: recession when composite < -0.5."""
    rng = np.random.default_rng(123)
    T = 300
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")

    # Two features that dip during "recessions"
    cycle = np.sin(np.linspace(0, 6 * np.pi, T))  # 3 full cycles
    x1 = cycle + rng.normal(0, 0.3, T)
    x2 = 0.6 * cycle + rng.normal(0, 0.3, T)
    X = pd.DataFrame({"activity": x1, "credit": x2}, index=dates)

    # Label: recession when cycle < -0.5
    y = pd.Series((cycle < -0.5).astype(int), index=dates, name="recession")
    return X, y


# ---------------------------------------------------------------------------
# Construction / API
# ---------------------------------------------------------------------------


def test_probit_init():
    m = RecessionProbit(add_lags=2, regularization=0.05)
    assert m.add_lags == 2
    assert m.regularization == 0.05
    assert not m._is_fitted


def test_probit_fit_returns_self(probit_data):
    X, y = probit_data
    m = RecessionProbit()
    result = m.fit(X, y)
    assert result is m
    assert m._is_fitted


def test_probit_predict_proba_shape(probit_data):
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_probit_predict_binary(probit_data):
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    pred = m.predict(X, threshold=0.5)
    assert set(np.unique(pred)).issubset({0, 1})


def test_probit_accuracy_above_chance(probit_data):
    """With a clean cyclic signal, accuracy should be well above 50 %."""
    X, y = probit_data
    m = RecessionProbit().fit(X, y)
    pred = m.predict(X, threshold=0.5)
    acc = float((pred == y.values).mean())
    assert acc > 0.70


def test_probit_coefficients_dict(probit_data):
    X, y = probit_data
    m = RecessionProbit(add_lags=1).fit(X, y)
    coefs = m.get_coefficients()
    assert isinstance(coefs, dict)
    # 2 features + 2 lagged features + 1 intercept = 5
    assert len(coefs) == 5
    assert "intercept" in coefs


def test_probit_before_fit_raises():
    m = RecessionProbit()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict_proba(np.zeros((5, 2)))


def test_probit_handles_nan_in_predict(probit_data):
    """A partially-missing row is imputed, not discarded.

    This test previously asserted the opposite — that a single NaN sent
    the whole row to 0.5. That behaviour is what silenced the probit in
    64% of months once features with differing start dates were added,
    so the contract deliberately changed: below
    ``max_missing_fraction`` the missing entries are filled with training
    means, and only a mostly-empty row falls back.
    """
    X, y = probit_data
    m = RecessionProbit(add_lags=0).fit(X, y)
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan  # one of two features missing

    proba = m.predict_proba(X_nan)
    assert proba[0] != 0.5, "partially-missing row should still be scored"
    assert 0 < proba[0] < 1
    assert 0 < proba[1] < 1  # other rows unaffected

    # A row with nothing in it remains uninformative.
    X_blank = X.copy()
    X_blank.iloc[0, :] = np.nan
    assert m.predict_proba(X_blank)[0] == 0.5


def test_probit_no_lags_mode(probit_data):
    X, y = probit_data
    m = RecessionProbit(add_lags=0).fit(X, y)
    coefs = m.get_coefficients()
    # 2 features + 1 intercept = 3
    assert len(coefs) == 3


def test_probit_numpy_input():
    """Probit should accept plain numpy arrays."""
    rng = np.random.default_rng(42)
    T = 100
    X = rng.standard_normal((T, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    m = RecessionProbit(add_lags=0).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (T,)


# ---------------------------------------------------------------------------
# Partial feature availability
# ---------------------------------------------------------------------------


def test_partial_missing_features_are_imputed_not_discarded():
    """One short-history feature must not silence the whole model.

    predict_proba used to return 0.5 whenever *any* feature was NaN. With
    features of differing history a row is usable only where every one
    exists, so adding leading indicators starting in 1990 and 2023 turned
    the probit off in 64% of months across 1967-2025 — it was not
    underperforming, it was not running. The failure is invisible because
    0.5 looks like a legitimate probability.
    """
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
    lin = -1.2 + 1.4 * X["a"] - 0.8 * X["b"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float))

    model = RecessionProbit(add_lags=1, regularization=1.0).fit(X, y)

    # Column 'd' only starts halfway through, as a real leading series would.
    partial = X.copy()
    partial.loc[partial.index[:200], "d"] = np.nan
    proba = model.predict_proba(partial)

    stuck = int((proba[:200] == 0.5).sum())
    assert stuck < 10, (
        f"{stuck} of 200 rows fell back to 0.5 despite only one of four "
        f"features being missing"
    )
    assert len(np.unique(np.round(proba[:200], 6))) > 50, (
        "imputed predictions are nearly constant; imputation is not working"
    )


def test_rows_missing_everything_still_fall_back():
    """Imputation must not invent a prediction from nothing."""
    rng = np.random.default_rng(2)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=list("abc"))
    y = pd.Series((rng.random(n) < 0.3).astype(float))
    model = RecessionProbit(add_lags=1, regularization=1.0).fit(X, y)

    blank = X.copy()
    blank.loc[blank.index[:40], :] = np.nan
    proba = model.predict_proba(blank)
    assert (proba[:40] == 0.5).all(), (
        "rows with no features at all should stay uninformative"
    )


def test_max_missing_fraction_is_respected():
    """The fallback threshold is configurable and actually applied."""
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=list("abcd"))
    y = pd.Series((rng.random(n) < 0.3).astype(float))

    strict = RecessionProbit(
        add_lags=0, regularization=1.0, max_missing_fraction=0.0
    ).fit(X, y)
    partial = X.copy()
    partial.loc[partial.index[:50], "d"] = np.nan
    assert (strict.predict_proba(partial)[:50] == 0.5).all(), (
        "max_missing_fraction=0 should refuse to impute anything"
    )


def test_a_feature_with_no_training_data_does_not_empty_the_fit():
    """One not-yet-existing feature must not abort training.

    fit() dropped rows containing any NaN, so a single column with no
    data in the training window removed *every* row, raised, and left the
    caller's signal pinned at its 0.5 fallback. Combined with the
    catch-and-warn in Nowcaster, this switched the probit off in 64% of
    months across 1967-2025 with no visible error.

    Columns are now screened before rows: a feature that is absent for
    this window is excluded from the fit and from prediction.
    """
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    X["not_yet"] = np.nan  # e.g. an index that starts in 2023
    lin = -1.0 + 1.4 * X["a"] - 0.8 * X["b"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float))

    model = RecessionProbit(add_lags=1, regularization=1.0).fit(X, y)
    proba = model.predict_proba(X)

    assert int((proba == 0.5).sum()) == 0, (
        "predictions are pinned at the 0.5 fallback despite three usable "
        "features being present"
    )
    assert 0.0 < proba.mean() < 1.0


def test_predictions_survive_a_feature_appearing_midway():
    """A feature that starts partway through must not break prediction."""
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    X["late"] = np.nan
    y = pd.Series((rng.random(n) < 0.3).astype(float))
    model = RecessionProbit(add_lags=1, regularization=1.0).fit(X, y)

    arrived = X.copy()
    arrived.loc[arrived.index[200:], "late"] = rng.standard_normal(100)
    proba = model.predict_proba(arrived)
    assert int((proba == 0.5).sum()) == 0
