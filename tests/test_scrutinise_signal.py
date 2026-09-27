"""The circular-shift null must separate a real signal from none.

``scripts/scrutinise_signal.py`` is what decides whether a downstream
lead is written up. Its verdict is only as good as the null it compares
against, so the null is tested on data where the answer is known: a
planted relationship must land far outside it, and an absent one must
land inside it.

The synthetic series are deliberately autocorrelated, with overlapping
forward targets — the conditions under which a naive test reports large
spurious ICs, and the reason the null is built by shifting rather than
shuffling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.scrutinise_signal import fold_ics, gbm_factory, null_ics
from src.evaluation.purged_cv import PurgedWalkForward

pytestmark = pytest.mark.slow

_N = 480
_H = 3


def _ar1(rng, n: int, phi: float = 0.95) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def _forward_mean(x: np.ndarray, h: int) -> np.ndarray:
    """Overlapping h-period forward average, as the real targets are."""
    out = np.full(len(x), np.nan)
    for t in range(len(x) - h):
        out[t] = x[t + 1 : t + 1 + h].mean()
    return out


# Weight on the planted driver. Measured across seeds, the null test
# detects weight 1.0 and above at p = 0.00 every time (IC ~0.4-0.6), but
# weight 0.6 (IC ~0.2) is borderline — p = 0.07, 0.00, 0.05 on three
# seeds. The first version of this test planted 0.6 and failed on the
# seed where the effect happened to fall inside the null.
#
# That borderline zone is itself the finding: five purged folds of a few
# hundred autocorrelated months cannot distinguish a true IC of ~0.2 from
# chance. The GBM regime-signal volatility lead sits exactly there.
_PLANTED_WEIGHT = 1.5


def _panel(signal: bool, seed: int):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-31", periods=_N, freq="ME")
    X = pd.DataFrame(
        {f"s{i}": _ar1(rng, _N) for i in range(4)}, index=idx
    )
    shock = _ar1(rng, _N, phi=0.9)
    driver = X["s0"].to_numpy() if signal else _ar1(rng, _N)
    y = pd.Series(
        _forward_mean(driver * _PLANTED_WEIGHT + shock, _H), index=idx
    )
    ok = y.notna()
    return X[ok], y[ok]


@pytest.fixture(scope="module")
def cv() -> PurgedWalkForward:
    return PurgedWalkForward(n_splits=5, label_horizon=_H, embargo=3,
                             min_train=80)


def _observed_and_null(signal: bool, cv, seed: int):
    X, y = _panel(signal, seed)
    model = gbm_factory(100, 2)
    obs = float(np.nanmean([ic for *_, ic in fold_ics(X, y, model, cv)]))
    null = null_ics(X, y, model, cv, n=40, rng=np.random.default_rng(1))
    return obs, null


def test_a_planted_signal_lands_outside_the_null(cv):
    obs, null = _observed_and_null(signal=True, cv=cv, seed=3)
    assert obs > np.quantile(null, 0.95), (
        f"a real relationship (IC {obs:+.3f}) did not clear the null's 95th "
        f"percentile ({np.quantile(null, 0.95):+.3f}); the test has no power"
    )


def test_no_signal_lands_inside_the_null(cv):
    obs, null = _observed_and_null(signal=False, cv=cv, seed=4)
    assert obs < np.quantile(null, 0.95), (
        f"pure noise (IC {obs:+.3f}) cleared the null's 95th percentile "
        f"({np.quantile(null, 0.95):+.3f}); the test would pass anything"
    )


def test_the_null_is_wide_on_autocorrelated_data(cv):
    """The reason for the whole exercise.

    On persistent features and overlapping targets, a model finds
    sizeable ICs in noise. If this null were tight around zero, a mean
    IC of +0.15 would look significant when it is not.
    """
    _, null = _observed_and_null(signal=False, cv=cv, seed=5)
    assert np.quantile(null, 0.95) > 0.05, (
        f"the null's 95th percentile is only {np.quantile(null, 0.95):+.3f}; "
        f"on autocorrelated data it should be wide enough to matter"
    )
