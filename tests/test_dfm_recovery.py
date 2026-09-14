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
    """
    rng = np.random.default_rng(7)
    T = 200
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")

    activity = ["INDPRO", "PAYEMS", "CFNAI", "W875RX1"]
    inflation = ["CPIAUCSL", "CPILFESL", "PCEPI", "PPIFIS"]
    stress = ["BAA10Y", "TEDRATE", "VIXCLS"]
    cols = activity + inflation + stress

    f = np.zeros((T, 3))
    for t in range(1, T):
        f[t] = 0.85 * f[t - 1] + rng.standard_normal(3) * 0.35

    loadings = rng.standard_normal((len(cols), 3)) * 0.12
    loadings[0:4, 0] += 1.4          # activity block -> factor 0
    loadings[4:8, 1] += 1.4          # inflation block -> factor 1
    loadings[8:11, 2] += 1.4         # stress block -> factor 2

    panel = pd.DataFrame(
        f @ loadings.T + rng.standard_normal((T, len(cols))) * 0.35,
        index=idx, columns=cols,
    )

    dfm = DynamicFactorModel(
        n_factors=3,
        factor_names=["real_activity", "inflation", "financial_stress"],
        max_iter=40,
    ).fit(panel)

    named = list(dfm.factors_.columns)
    assert set(named) == {"real_activity", "inflation", "financial_stress"}

    # The factor named "inflation" must be the one the CPI series load on.
    loadings_df = pd.DataFrame(dfm.get_loadings(), index=cols, columns=named)
    infl_col = loadings_df.loc[inflation].abs().mean()
    assert infl_col.idxmax() == "inflation", (
        f"The 'inflation' label did not land on the factor the CPI series "
        f"load on; mean |loading| by factor:\n{infl_col}"
    )

    act_col = loadings_df.loc[activity].abs().mean()
    assert act_col.idxmax() == "real_activity", (
        f"The 'real_activity' label landed on the wrong factor:\n{act_col}"
    )
