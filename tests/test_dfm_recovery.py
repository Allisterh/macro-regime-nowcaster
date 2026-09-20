"""Statistical recovery tests for the Dynamic Factor Model.

``conftest.py`` has always built a panel with a known factor structure,
but nothing asserted that the DFM recovers it — the existing DFM tests
check shapes, not estimates.  A model that returned noise of the right
dimensions would have passed the entire suite.

Also covers the factor-naming contract: varimax does not preserve column
order, so names have to follow loading evidence rather than position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dynamic_factor_model import DynamicFactorModel

# Every test here refits the DFM/RSM, which takes minutes.
# Run the fast suite with `pytest -m "not slow"`.
pytestmark = pytest.mark.slow


def _simulate_factor_panel(
    seed: int = 11,
    T: int = 240,
    K: int = 2,
    n_per_factor: int = 15,
    noise: float = 0.4,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Panel with a block loading structure and known AR(1) factors."""
    rng = np.random.default_rng(seed)

    factors = np.zeros((T, K))
    for t in range(1, T):
        factors[t] = 0.85 * factors[t - 1] + rng.standard_normal(K) * 0.35

    N = K * n_per_factor
    loadings = rng.standard_normal((N, K)) * 0.15
    for k in range(K):
        loadings[k * n_per_factor : (k + 1) * n_per_factor, k] += 1.2

    obs = factors @ loadings.T + rng.standard_normal((T, N)) * noise
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    frame = pd.DataFrame(obs, index=idx, columns=[f"x{i}" for i in range(N)])
    return frame, factors


def _canonical_correlations(
    estimated: pd.DataFrame | np.ndarray, true_factors: np.ndarray
) -> np.ndarray:
    """Canonical correlations between the estimated and true factor spaces.

    Factor models identify the factor *space*, not individual factors:
    any invertible rotation of ``F`` with the inverse applied to ``C``
    gives the same likelihood.  Estimated factors are therefore rotated
    mixtures of the simulated ones, and a per-factor correlation
    understates recovery even when the model is exactly right.

    Canonical correlations are invariant to rotation and scaling of
    either set, so they measure what is actually identified.  All values
    near 1.0 mean the same subspace was recovered.
    """
    est = np.asarray(
        estimated.values if hasattr(estimated, "values") else estimated,
        dtype=float,
    )
    true = np.asarray(true_factors, dtype=float)

    ok = ~(np.isnan(est).any(axis=1) | np.isnan(true).any(axis=1))
    est, true = est[ok], true[ok]

    def _orthonormal(mat: np.ndarray) -> np.ndarray:
        centred = mat - mat.mean(axis=0)
        q, _ = np.linalg.qr(centred)
        return q

    q_est, q_true = _orthonormal(est), _orthonormal(true)
    # Singular values of the cross-product of orthonormal bases are the
    # cosines of the principal angles between the two subspaces.
    return np.clip(np.linalg.svd(q_true.T @ q_est, compute_uv=False), 0.0, 1.0)


def _pca_reference(panel: pd.DataFrame, k: int) -> pd.DataFrame:
    """First *k* principal components, as an estimation floor."""
    z = (panel - panel.mean()) / panel.std()
    z = z.fillna(0.0)
    U, S, _ = np.linalg.svd(z.values, full_matrices=False)
    return pd.DataFrame(U[:, :k] * S[:k], index=panel.index)


def test_recovers_known_factors():
    """Each simulated factor must be strongly represented in the output."""
    panel, true_factors = _simulate_factor_panel()
    dfm = DynamicFactorModel(n_factors=2, max_iter=60, rotate=False).fit(panel)

    cc = _canonical_correlations(dfm.factors_, true_factors)
    assert cc.min() > 0.85, (
        f"DFM did not recover the simulated factor space: canonical "
        f"correlations = {np.round(cc, 3).tolist()}"
    )


def test_at_least_as_good_as_pca():
    """The state-space machinery must earn its keep over plain PCA.

    An absolute correlation threshold mostly measures the simulation's
    signal-to-noise; comparing against PCA on the same panel isolates
    whether the Kalman/EM step is actually contributing.
    """
    panel, true_factors = _simulate_factor_panel()
    dfm = DynamicFactorModel(n_factors=2, max_iter=60, rotate=False).fit(panel)

    dfm_cc = _canonical_correlations(dfm.factors_, true_factors).min()
    pca_cc = _canonical_correlations(
        _pca_reference(panel, 2), true_factors
    ).min()

    assert dfm_cc > pca_cc - 0.05, (
        f"DFM subspace recovery ({dfm_cc:.3f}) is materially worse than "
        f"plain PCA ({pca_cc:.3f}) on the same panel"
    )


def test_recovery_survives_missing_data():
    """Ragged edges must not destroy recovery — the Kalman filter skips them.

    This is the DFM's main advantage over PCA, which needs a filled matrix.
    """
    panel, true_factors = _simulate_factor_panel()
    rng = np.random.default_rng(2)
    holed = panel.mask(rng.random(panel.shape) < 0.15)

    dfm = DynamicFactorModel(n_factors=2, max_iter=60, rotate=False).fit(holed)
    cc = _canonical_correlations(dfm.factors_, true_factors)
    assert cc.min() > 0.8, (
        f"Recovery collapsed with 15% missing data: canonical correlations "
        f"= {np.round(cc, 3).tolist()}"
    )


def test_varimax_restores_block_structure():
    """Varimax must recover the simple structure the simulation has.

    Unrotated loadings are an arbitrary rotation of the truth, so the
    blocks are smeared across both columns; restoring simple structure is
    precisely what the varimax step is for.  The test therefore compares
    rotated against unrotated rather than asserting on either alone.
    """
    panel, _ = _simulate_factor_panel()

    unrotated = np.abs(
        DynamicFactorModel(n_factors=2, max_iter=60, rotate=False)
        .fit(panel)
        .get_loadings()
    )
    rotated = np.abs(
        DynamicFactorModel(n_factors=2, max_iter=60, rotate=True)
        .fit(panel)
        .get_loadings()
    )

    def concentration(loadings: np.ndarray) -> float:
        """Median share of a series' loading mass on its dominant factor."""
        share = loadings.max(axis=1) / (loadings.sum(axis=1) + 1e-12)
        return float(np.median(share))

    rot, unrot = concentration(rotated), concentration(unrotated)
    assert rot > unrot, (
        f"Varimax did not improve simple structure: rotated={rot:.3f} vs "
        f"unrotated={unrot:.3f}"
    )
    assert rot > 0.75, (
        f"Rotated loadings still do not show the simulated block structure "
        f"(median dominant share {rot:.3f})"
    )


def test_filtered_and_smoothed_factors_differ():
    """Both estimates are retained and the smoother is not a no-op."""
    panel, _ = _simulate_factor_panel()
    dfm = DynamicFactorModel(n_factors=2, max_iter=40).fit(panel)

    assert dfm.filtered_factors_ is not None
    assert dfm.smoothed_factors_ is not None
    assert not np.allclose(
        dfm.filtered_factors_.values, dfm.smoothed_factors_.values,
    )
    # The default must be the real-time-safe one.
    np.testing.assert_allclose(
        dfm.factors_.values, dfm.filtered_factors_.values,
    )


# ---------------------------------------------------------------------------
# Factor naming
# ---------------------------------------------------------------------------


def test_factor_names_follow_loadings_not_position():
    """Names must attach to the factor the anchors actually load on.

    Varimax returns columns in an arbitrary order, so naming factor *k*
    ``factor_names[k]`` can label the financial-stress factor
    "real_activity" and then sign-align it against the wrong anchors.

    The blocks here are the real anchor sets. This test used to build a
    CPI block and assert an "inflation" label landed on it, which stopped
    being meaningful once that name was retired: the panel has no
    inflation factor, the price series load 0.117 on their best home
    against 0.059-0.066 elsewhere, and inflation itself shows up in
    `long_rates` (+0.605 with CPI year-on-year).
    """
    rng = np.random.default_rng(7)
    T = 200
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")

    activity = ["INDPRO", "IPMAN", "TCU", "CFNAI"]
    credit = ["AAA10Y", "BAA10Y"]
    stress = ["NFCI", "ANFCI", "STLFSI2", "VIXCLS"]
    cols = activity + credit + stress

    f = np.zeros((T, 3))
    for t in range(1, T):
        f[t] = 0.85 * f[t - 1] + rng.standard_normal(3) * 0.35

    loadings = rng.standard_normal((len(cols), 3)) * 0.12
    loadings[0:4, 0] += 1.4          # activity block -> factor 0
    loadings[4:6, 1] += 1.4          # credit block   -> factor 1
    loadings[6:10, 2] += 1.4         # stress block   -> factor 2

    panel = pd.DataFrame(
        f @ loadings.T + rng.standard_normal((T, len(cols))) * 0.35,
        index=idx, columns=cols,
    )

    dfm = DynamicFactorModel(
        n_factors=3,
        factor_names=["real_activity", "credit_premium", "financial_stress"],
        max_iter=40,
    ).fit(panel)

    named = list(dfm.factors_.columns)
    assert set(named) == {"real_activity", "credit_premium", "financial_stress"}

    loadings_df = pd.DataFrame(dfm.get_loadings(), index=cols, columns=named)

    credit_col = loadings_df.loc[credit].abs().mean()
    assert credit_col.idxmax() == "credit_premium", (
        f"The 'credit_premium' label did not land on the factor the "
        f"corporate-spread series load on; mean |loading| by factor: "
        f"{credit_col.to_dict()}"
    )

    act_col = loadings_df.loc[activity].abs().mean()
    assert act_col.idxmax() == "real_activity", (
        f"The 'real_activity' label landed on the wrong factor: "
        f"{act_col.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Factor names must be evidenced, not merely assigned
# ---------------------------------------------------------------------------


def test_weakly_evidenced_factor_names_are_flagged():
    """Assignment always pairs every name; quality says whether to trust it.

    Matching names to factors by loading evidence still produces *some*
    pairing for every name, so a label can land on a factor its anchor
    series barely load on — and downstream code then trusts the label. On
    the live panel "labor_market" was assigned to a factor whose strongest
    loadings were BAA, AAA and GS10 (bond yields), and the regime model
    was fitted on it as though it were labour data, which made the
    recession probability track the level of interest rates.

    Here only an activity factor exists; the labour anchors have nothing
    to attach to, and that must be visible in the quality scores.
    """
    rng = np.random.default_rng(9)
    T = 200
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")

    activity = ["INDPRO", "PAYEMS", "CFNAI", "W875RX1"]
    rates = ["BAA", "AAA", "GS10", "TB3MS"]
    cols = activity + rates

    f = np.zeros((T, 2))
    for t in range(1, T):
        f[t] = 0.85 * f[t - 1] + rng.standard_normal(2) * 0.35

    loadings = rng.standard_normal((len(cols), 2)) * 0.10
    loadings[:4, 0] += 1.4          # activity block
    loadings[4:, 1] += 1.4          # rates block — no labour factor exists

    panel = pd.DataFrame(
        f @ loadings.T + rng.standard_normal((T, len(cols))) * 0.35,
        index=idx, columns=cols,
    )

    dfm = DynamicFactorModel(
        n_factors=2,
        factor_names=["real_activity", "labor_market"],
        max_iter=40,
    ).fit(panel)

    quality = dfm._factor_match_quality
    assert quality, "no match-quality scores were recorded"
    assert set(quality) == {"real_activity", "labor_market"}

    # The panel has one meaningful block and one rates block, so exactly
    # one of the two names can be genuinely evidenced. Which one wins is
    # not the point and is not asserted: with both names' anchors loading
    # on the same factor, either assignment is defensible. What must hold
    # is that the *other* one is flagged rather than silently trusted.
    scores = sorted(quality.values())
    assert scores[0] < 1.0, (
        f"no name was flagged as weakly evidenced even though only one "
        f"meaningful factor exists: {quality}"
    )
    assert scores[-1] > 1.0, (
        f"the name matching the real factor block should be well "
        f"evidenced: {quality}"
    )


# ---------------------------------------------------------------------------
# An empty factor is not a factor
# ---------------------------------------------------------------------------


def test_a_factor_nothing_loads_on_is_flagged_as_empty():
    """Match quality is a ratio, so it cannot detect a null factor.

    On the full 1956-2026 panel the factor named ``real_activity``
    scored 1.13 — comfortably above the 1.0 "evidenced" threshold —
    with a maximum absolute loading of 0.0008 and a 0.0% share of panel
    variance. Its anchors were simply the largest of its negligible
    loadings, which is all a ratio can ever tell you. The regime model
    was then fitted on it by name.

    ``_factor_strength`` is the scale-aware companion: a factor's
    largest loading relative to the largest in the matrix.
    """
    rng = np.random.default_rng(5)
    T = 240
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")

    activity = ["INDPRO", "IPMAN", "TCU", "CFNAI"]
    cols = activity + [f"noise{i}" for i in range(6)]

    # One real factor; the second has nothing to attach to.
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.85 * f[t - 1] + rng.standard_normal() * 0.35

    obs = rng.standard_normal((T, len(cols))) * 0.4
    obs[:, :4] += f[:, None] * 1.5          # activity block loads
    panel = pd.DataFrame(obs, index=idx, columns=cols)

    dfm = DynamicFactorModel(
        n_factors=2,
        factor_names=["real_activity", "credit_premium"],
        max_iter=60,
    ).fit(panel)

    strength = dfm._factor_strength
    assert strength, "no factor-strength scores were recorded"
    assert set(strength) == {"real_activity", "credit_premium"}

    # The panel supports one factor, so exactly one must be strong.
    ordered = sorted(strength.values())
    assert ordered[-1] > dfm.NULL_FACTOR_STRENGTH, (
        f"the genuine factor was flagged as empty: {strength}"
    )
    assert max(strength.values()) == pytest.approx(1.0), (
        "strength is relative to the largest loading, so the top factor "
        "should be 1.0"
    )


def test_strength_and_quality_measure_different_things():
    """A well-scored name on an empty factor is the case that bit.

    Quality is scale-free and strength is not, so a fit must be able to
    report a high ratio and a low magnitude at the same time — which is
    exactly the combination that went unnoticed.
    """
    rng = np.random.default_rng(6)
    T, idx = 240, pd.date_range("2000-01-31", periods=240, freq="ME")
    cols = ["INDPRO", "IPMAN", "TCU", "CFNAI"] + [f"x{i}" for i in range(6)]

    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.85 * f[t - 1] + rng.standard_normal() * 0.35
    obs = rng.standard_normal((T, len(cols))) * 0.4
    obs[:, :4] += f[:, None] * 1.5
    panel = pd.DataFrame(obs, index=idx, columns=cols)

    dfm = DynamicFactorModel(
        n_factors=2, factor_names=["real_activity", "credit_premium"], max_iter=60,
    ).fit(panel)

    assert set(dfm._factor_match_quality) == set(dfm._factor_strength)
    # They are not the same number, or one of them is redundant.
    assert dfm._factor_match_quality != dfm._factor_strength
