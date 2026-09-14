"""Leakage guarantees for the purged / embargoed splitter.

The whole point of this splitter is what it *excludes*, so these tests
assert the exclusions directly rather than checking that it produces
plausible-looking folds.

The final test is the one that matters: on data where the target is pure
noise, a naive split reports skill that is not there, and this splitter
does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.purged_cv import (
    PurgedWalkForward,
    embargo_mask,
    purge_train_indices,
)

# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def test_purge_removes_overlapping_label_windows():
    train = np.arange(0, 50)
    test = np.arange(50, 60)
    kept = purge_train_indices(train, test, label_horizon=3)

    # A sample at 47 carries a label over [47, 50], which touches the
    # test block starting at 50, so it must go.
    assert 47 not in kept
    assert 48 not in kept and 49 not in kept
    # A sample at 46 ends at 49, strictly before the test block.
    assert 46 in kept


def test_purge_is_a_no_op_for_contemporaneous_labels():
    train, test = np.arange(0, 50), np.arange(50, 60)
    kept = purge_train_indices(train, test, label_horizon=0)
    np.testing.assert_array_equal(kept, train)


def test_purge_scales_with_horizon():
    train, test = np.arange(0, 50), np.arange(50, 60)
    short = purge_train_indices(train, test, label_horizon=1)
    long = purge_train_indices(train, test, label_horizon=10)
    assert len(long) < len(short) <= len(train)


def test_purge_handles_empty_inputs():
    empty = np.array([], dtype=int)
    np.testing.assert_array_equal(
        purge_train_indices(empty, np.arange(3), 2), empty
    )
    train = np.arange(5)
    np.testing.assert_array_equal(
        purge_train_indices(train, empty, 2), train
    )


# ---------------------------------------------------------------------------
# Embargo
# ---------------------------------------------------------------------------


def test_embargo_removes_samples_just_after_the_test_block():
    train = np.concatenate([np.arange(0, 50), np.arange(60, 80)])
    test = np.arange(50, 60)
    kept = embargo_mask(train, test, embargo=5)

    # 60..64 sit inside the embargo window after the test block ends at 59.
    for i in range(60, 65):
        assert i not in kept
    assert 65 in kept
    # Samples before the test block are untouched by the embargo.
    assert 49 in kept


def test_embargo_zero_is_a_no_op():
    train = np.arange(0, 80)
    test = np.arange(50, 60)
    np.testing.assert_array_equal(embargo_mask(train, test, 0), train)


# ---------------------------------------------------------------------------
# Splitter structure
# ---------------------------------------------------------------------------


@pytest.fixture()
def index() -> pd.DatetimeIndex:
    return pd.date_range("1990-01-31", periods=300, freq="ME")


def test_folds_are_causal_and_disjoint(index):
    cv = PurgedWalkForward(n_splits=5, label_horizon=3, embargo=3, min_train=60)
    folds = list(cv.split(index))
    assert len(folds) >= 3

    seen_test: list[np.ndarray] = []
    for train_idx, test_idx in folds:
        # Training must lie strictly before the test block: no future data.
        assert train_idx.max() < test_idx.min(), (
            "training data extends into or past the test block"
        )
        # Folds must not overlap each other.
        for prior in seen_test:
            assert not set(prior) & set(test_idx)
        seen_test.append(test_idx)


def test_purge_gap_is_present_in_every_fold(index):
    horizon = 4
    cv = PurgedWalkForward(
        n_splits=5, label_horizon=horizon, embargo=0, min_train=60
    )
    for train_idx, test_idx in cv.split(index):
        gap = test_idx.min() - train_idx.max()
        assert gap > horizon, (
            f"only {gap} periods between the last training sample and the "
            f"test block, with a {horizon}-period label window"
        )


def test_training_grows_across_folds(index):
    cv = PurgedWalkForward(n_splits=5, label_horizon=1, embargo=0, min_train=60)
    sizes = [len(tr) for tr, _ in cv.split(index)]
    assert sizes == sorted(sizes), f"training set is not expanding: {sizes}"


def test_short_series_yields_nothing_rather_than_tiny_folds():
    cv = PurgedWalkForward(n_splits=5, label_horizon=3, min_train=100)
    assert list(cv.split(pd.RangeIndex(50))) == []


def test_get_n_splits_matches_actual(index):
    cv = PurgedWalkForward(n_splits=5, label_horizon=3, embargo=2, min_train=60)
    assert cv.get_n_splits(index) == len(list(cv.split(index)))


# ---------------------------------------------------------------------------
# The property that justifies the whole module
# ---------------------------------------------------------------------------


def test_shuffled_kfold_finds_skill_in_noise_and_purged_split_does_not():
    """Shuffled k-fold reports large skill on an unpredictable target.

    This is the realistic failure mode: reaching for ``KFold(shuffle=True)``
    on a financial panel. The features here are slow and autocorrelated
    and the target is an *overlapping* forward sum of pure noise, so there
    is nothing to predict. Shuffling scatters near-duplicate neighbours
    across folds and puts future rows in training, and the model recovers
    its own labels — measured at roughly +0.7 correlation against a true
    value of zero.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    n, horizon = 400, 12

    # Slow features, so adjacent rows are near-duplicates.
    feats = np.cumsum(rng.standard_normal((n, 3)) * 0.1, axis=0)
    noise = rng.standard_normal(n + horizon)
    # Overlapping target: consecutive labels share horizon-1 noise terms.
    target = np.array([noise[i : i + horizon].sum() for i in range(n)])

    X = pd.DataFrame(feats, columns=["a", "b", "c"])
    y = pd.Series(target)

    def run(train_idx, test_idx) -> float:
        model = RandomForestRegressor(
            n_estimators=80, random_state=0, min_samples_leaf=5, n_jobs=1
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx].values
        if np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
            return 0.0
        return float(np.corrcoef(pred, actual)[0, 1])

    shuffled = [
        run(tr, te)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X)
    ]
    purged = [
        run(tr, te)
        for tr, te in PurgedWalkForward(
            n_splits=4, label_horizon=horizon, embargo=horizon, min_train=100
        ).split(X)
    ]

    assert purged, "splitter produced no folds"
    shuffled_mean, purged_mean = float(np.mean(shuffled)), float(np.mean(purged))

    # Shuffled k-fold should look impressively wrong here.
    assert shuffled_mean > 0.4, (
        f"expected shuffled k-fold to report large spurious skill, got "
        f"{shuffled_mean:+.3f} — the fixture may no longer be leaky"
    )
    # The purged estimate must be far closer to the truth, which is zero.
    assert purged_mean < shuffled_mean - 0.25, (
        f"purging barely reduced the apparent skill: shuffled "
        f"{shuffled_mean:+.3f} vs purged {purged_mean:+.3f}"
    )
    assert abs(purged_mean) < 0.35, (
        f"purged folds still report correlation {purged_mean:+.3f} on a "
        f"target that is noise by construction"
    )
