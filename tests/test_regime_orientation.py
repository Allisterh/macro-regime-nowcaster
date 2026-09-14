"""Orientation and recovery tests for the regime-switching model.

Two properties that nothing previously checked:

1. **Orientation** — ``get_recession_probability()`` must rise in the
   low-mean state regardless of how the caller names the regimes.  The
   previous implementation resolved the column by label text, so the
   ordering ``["expansion", "recession"]`` in ``settings.yaml`` returned
   the *expansion* column and inverted the signal end to end.

2. **Recovery** — the estimator must actually recover regimes it was
   given.  The suite had shape assertions but nothing that would fail if
   the model returned noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.regime_switching import RegimeSwitchingModel

# Every test here refits the DFM/RSM, which takes minutes.
# Run the fast suite with `pytest -m "not slow"`.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------


def _simulate_markov_chain(
    seed: int = 3,
    T: int = 400,
    D: int = 3,
    separation: float = 0.9,
    noise: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Two-regime Gaussian HMM with a known state path.

    State 0 has the *lower* mean and stands in for recession.  Default
    separation is deliberately modest so the test exercises a realistic
    signal-to-noise ratio rather than a trivially separable one.
    """
    rng = np.random.default_rng(seed)
    P = np.array([[0.90, 0.10], [0.04, 0.96]])
    means = np.array([[-separation * 0.67] * D, [separation * 0.33] * D])

    states = np.zeros(T, dtype=int)
    states[0] = 1
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t - 1]])

    obs = means[states] + rng.standard_normal((T, D)) * noise
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    frame = pd.DataFrame(obs, index=idx, columns=[f"f{i}" for i in range(D)])
    truth = pd.Series((states == 0).astype(float), index=idx, name="recession")
    return frame, truth


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multivariate", [True, False])
def test_recession_probability_tracks_low_mean_state(multivariate):
    """P(recession) must correlate positively with the true low-mean state."""
    data, truth = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2,
        regime_labels=["recession", "expansion"],
        multivariate=multivariate,
        n_restarts=1,
    ).fit(data)

    corr = model.get_recession_probability().corr(truth)
    assert corr > 0.5, (
        f"P(recession) correlates {corr:+.3f} with the true recession state "
        f"(multivariate={multivariate}); a negative or near-zero value means "
        f"the regime columns are inverted"
    )


def test_reversed_regime_labels_are_rejected():
    """The ordering that silently inverted the signal must now raise.

    ``["expansion", "recession"]`` is what config/settings.yaml carried,
    and it put the name "recession" on the high-mean column.
    """
    with pytest.raises(ValueError, match="lowest-mean to highest-mean"):
        RegimeSwitchingModel(
            n_regimes=2, regime_labels=["expansion", "recession"],
        )


def test_valid_orderings_are_accepted():
    RegimeSwitchingModel(n_regimes=2, regime_labels=["recession", "expansion"])
    RegimeSwitchingModel(
        n_regimes=3, regime_labels=["recession", "slowdown", "expansion"],
    )
    RegimeSwitchingModel(n_regimes=2)  # defaults


def test_recession_column_is_positional_not_textual():
    """Column 0 is the recession state even under neutral labels."""
    data, truth = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["state_a", "state_b"], n_restarts=1,
    ).fit(data)

    probs = model.get_regime_probabilities()
    assert model.get_recession_probability().equals(probs.iloc[:, 0])
    # And column 0 really is the low-mean state.
    assert model._means[0] < model._means[-1]


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_recovers_known_regimes():
    """Classification accuracy against the true state path."""
    data, truth = _simulate_markov_chain(separation=1.4)
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=2,
    ).fit(data)

    pred = (model.get_recession_probability() > 0.5).astype(float)
    accuracy = float((pred == truth).mean())
    assert accuracy > 0.8, f"Recovered only {accuracy:.1%} of states"


def test_transition_matrix_is_sticky():
    """Recovered persistence should resemble the simulated chain."""
    data, _ = _simulate_markov_chain(separation=1.4)
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=2,
    ).fit(data)

    P = model.get_transition_matrix()
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(np.diag(P) > 0.5), (
        f"Transition matrix is not persistent:\n{P}"
    )


# ---------------------------------------------------------------------------
# Filtered vs smoothed
# ---------------------------------------------------------------------------


def test_filtered_and_smoothed_are_both_available():
    data, _ = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=1,
    ).fit(data)

    assert model.filtered_probs_ is not None
    assert model.smoothed_probs_ is not None
    assert model.filtered_probs_.shape == model.smoothed_probs_.shape
    # They must genuinely differ in the interior, or the smoother is a no-op.
    assert not np.allclose(model.filtered_probs_, model.smoothed_probs_)


def test_default_is_filtered():
    """The default must be the real-time-safe estimate."""
    data, _ = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=1,
    ).fit(data)
    np.testing.assert_allclose(
        model.get_regime_probabilities().values, model.filtered_probs_,
    )


def test_final_row_matches_between_estimates():
    """At t=T the Kim smoother coincides with the Hamilton filter.

    The walk-forward feature generator depends on this: it keeps only the
    last row of each fit precisely because no future observation can
    influence it.
    """
    data, _ = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=1,
    ).fit(data)
    np.testing.assert_allclose(
        model.filtered_probs_[-1], model.smoothed_probs_[-1], atol=1e-10,
    )


def test_probabilities_are_not_saturated():
    """Output must retain gradation, not latch onto the clip bounds.

    The committed backtest had 421 of 434 months sitting at a bound,
    which left the probability with no usable information content.
    """
    data, _ = _simulate_markov_chain(separation=0.9)
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=1,
    ).fit(data)

    p = model.get_recession_probability()
    interior = float(((p > 0.05) & (p < 0.95)).mean())
    assert interior > 0.10, (
        f"Only {interior:.1%} of probabilities lie strictly inside "
        f"[0.05, 0.95]; the signal has collapsed to a step function"
    )


# ---------------------------------------------------------------------------
# Multivariate ordering across disagreeing dimensions
# ---------------------------------------------------------------------------


def test_ordering_uses_all_dimensions_not_just_the_first():
    """Regime ranking must reflect the whole state vector.

    Ordering on ``means[:, 0]`` alone identified the recession regime by
    the first factor only.  When the fitted factors disagree about which
    regime is weaker, dim-0 ordering can invert the labelling.

    Fitting the real panel as of July 2019 produced exactly that:

        regime A: real_activity -0.23, labor_market +1.05   (composite +0.41)
        regime B: real_activity +0.14, labor_market -0.66   (composite -0.26)

    dim-0 ranked A first and so called it "recession", even though A is
    the *stronger* state overall.  With labour running hot the filter sat
    in A and the model reported P(recession) = 0.999 through a late-cycle
    expansion.

    This reproduces that disagreement: the truly weak state is strong on
    dim 0 and weak on dim 1, so a dim-0 rule inverts the labels while a
    composite rule gets them right.
    """
    rng = np.random.default_rng(17)
    T, D = 500, 2
    P = np.array([[0.93, 0.07], [0.04, 0.96]])

    # State 0 is the genuine recession: weaker overall (composite -0.25),
    # but *higher* than state 1 on dim 0.  A dim-0 sort therefore ranks
    # state 1 first and mislabels it.
    means = np.array([
        [0.15, -0.65],   # state 0 — truly weak  (composite -0.250)
        [-0.25, 1.05],   # state 1 — truly strong (composite +0.400)
    ])
    assert means[0].mean() < means[1].mean()      # composite: 0 is weaker
    assert means[0, 0] > means[1, 0]              # dim 0 disagrees

    states = np.zeros(T, dtype=int)
    states[0] = 1
    for t in range(1, T):
        states[t] = rng.choice(2, p=P[states[t - 1]])

    obs = means[states] + rng.standard_normal((T, D)) * 0.45
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    data = pd.DataFrame(obs, index=idx, columns=["real_activity", "labor_market"])
    truth = pd.Series((states == 0).astype(float), index=idx)

    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"],
        multivariate=True, n_restarts=3,
    ).fit(data)

    # The regime ranked first must be the one that is weaker overall.
    assert model._means_mv[0].mean() < model._means_mv[1].mean(), (
        "regime 0 is not the weaker state overall: "
        f"{np.round(model._means_mv, 3).tolist()}"
    )

    corr = model.get_recession_probability().corr(truth)
    assert corr > 0.5, (
        f"P(recession) correlates {corr:+.3f} with the truly weak state; "
        f"regime means are {np.round(model._means_mv, 3).tolist()}"
    )


def test_means_attribute_is_the_ranking_key():
    """``_means`` must stay ascending — it is what the sort is based on."""
    data, _ = _simulate_markov_chain()
    model = RegimeSwitchingModel(
        n_regimes=2, regime_labels=["recession", "expansion"], n_restarts=1,
    ).fit(data)
    assert model._means[0] < model._means[-1]
