"""The paired bootstrap must find differences that exist and only those.

It decides the README's "does A beat B?" claims — the ensemble against
CFNAI, and the SPF blend against the ensemble — so it is tested where the
answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.horizon_curve import _label_blocks
from src.evaluation.paired_bootstrap import paired_block_bootstrap
from src.models.regime_backtest import brier_score, roc_auc


@pytest.fixture(scope="module")
def episodes():
    """Eight recessions of 6-18 months in 60 years of monthly labels."""
    rng = np.random.default_rng(0)
    y = np.zeros(720, dtype=int)
    for start in np.linspace(40, 660, 8).astype(int):
        y[start : start + rng.integers(6, 18)] = 1
    blocks = _label_blocks(pd.Series(y))
    return y, blocks, rng


def test_a_signal_compared_with_itself_differs_by_nothing(episodes):
    y, blocks, rng = episodes
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, len(y)), 0, 1)
    d = paired_block_bootstrap(y, s, s.copy(), blocks, roc_auc, n_boot=300)
    assert d.point == pytest.approx(0.0)
    assert d.lo <= 0 <= d.hi
    assert not d.excludes_zero()


def test_an_informative_signal_beats_noise(episodes):
    y, blocks, rng = episodes
    good = np.clip(y * 0.7 + rng.normal(0, 0.15, len(y)), 0, 1)
    noise = rng.uniform(0, 1, len(y))
    d = paired_block_bootstrap(y, good, noise, blocks, roc_auc, n_boot=300)
    assert d.point > 0.3
    assert d.excludes_zero() and d.lo > 0
    assert d.p_positive > 0.99


def test_brier_direction_is_reported_the_right_way_round(episodes):
    """Lower Brier is better, so a better signal gives a negative point."""
    y, blocks, rng = episodes
    good = np.clip(y * 0.8 + 0.1, 0, 1)
    bad = np.full(len(y), 0.5)
    d = paired_block_bootstrap(y, good, bad, blocks, brier_score, n_boot=300)
    assert d.point < 0
    assert d.p_negative > 0.99


def test_both_signals_are_scored_on_the_same_rows(episodes):
    """A NaN in either signal drops the row from both, never just one.

    Otherwise a signal with missing early history would be scored on an
    easier sample than the one it is compared against.
    """
    y, blocks, rng = episodes
    s = np.clip(y * 0.6 + rng.normal(0, 0.2, len(y)), 0, 1)
    gappy = s.copy()
    gappy[:200] = np.nan

    d = paired_block_bootstrap(y, gappy, s, blocks, roc_auc, n_boot=200)
    assert d.point == pytest.approx(0.0), (
        "identical values were scored differently because one signal's "
        "missing rows were dropped only from its own side"
    )
