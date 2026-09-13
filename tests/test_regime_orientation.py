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
