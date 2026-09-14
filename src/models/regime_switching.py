"""Markov-switching / Hamilton-filter regime classification.

Implements a Gaussian mixture Hidden-Markov Model with EM estimation via
the Hamilton (1989) filter.  Suitable for classifying the regime state
(e.g., expansion / slowdown / recession / recovery) from factor estimates.

When ``use_statsmodels=True`` the estimation is delegated to
``statsmodels.tsa.regime_switching.markov_regression`` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_density(
    y: float | np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Univariate Gaussian density for each regime.

    Parameters
    ----------
    y : scalar or array
        Observation value(s).
    means : np.ndarray
        Per-regime means, shape ``(K,)``.
    variances : np.ndarray
        Per-regime variances, shape ``(K,)`` (must be > 0).

    Returns
    -------
    np.ndarray
        Density values, shape ``(K,)``.
    """
    y = np.asarray(y, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi * variances)) * np.exp(
        -0.5 * (y - means) ** 2 / variances
    )


def _multivariate_gaussian_density(
    y: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> np.ndarray:
    """Multivariate Gaussian density for each regime.

    Parameters
    ----------
    y : np.ndarray, shape ``(D,)``
        Single observation vector.
    means : np.ndarray, shape ``(K, D)``
        Per-regime mean vectors.
    covs : np.ndarray, shape ``(K, D, D)``
        Per-regime covariance matrices.

    Returns
    -------
    np.ndarray, shape ``(K,)``
        Density values for each regime.
    """
    K, D = means.shape
    densities = np.zeros(K)
    for k in range(K):
        diff = y - means[k]
        cov_k = covs[k]
        try:
            L = np.linalg.cholesky(cov_k)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff)
            mahal = np.dot(solved, solved)
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(cov_k)
            if sign <= 0:
                log_det = D * np.log(1e-6)
            inv_cov = np.linalg.pinv(cov_k)
            mahal = diff @ inv_cov @ diff
        log_p = -0.5 * (D * np.log(2.0 * np.pi) + log_det + mahal)
        densities[k] = np.exp(np.clip(log_p, -500, 500))
    return densities


_PROB_FLOOR = 0.001  # Numerical guard only — see _clip_probs


def _hamilton_forward(
    densities: np.ndarray, P: np.ndarray, pi0: np.ndarray
) -> tuple[np.ndarray, float]:
    """Hamilton (1989) forward recursion.

    Parameters
    ----------
    densities : np.ndarray, shape ``(T, K)``
        Emission density of each observation under each regime.
    P : np.ndarray, shape ``(K, K)``
        Transition matrix.
    pi0 : np.ndarray, shape ``(K,)``
        Initial state distribution.

    Returns
    -------
    (alpha, log_likelihood)
        ``alpha[t]`` conditions only on observations ``0..t``.
    """
    T, K = densities.shape
    alpha = np.zeros((T, K))
    ll = 0.0
    for t in range(T):
        predicted = pi0.copy() if t == 0 else alpha[t - 1] @ P
        joint = predicted * densities[t]
        total = joint.sum()
        if total < 1e-300:
            alpha[t] = predicted
        else:
            alpha[t] = joint / total
            ll += np.log(total)
    return alpha, ll


def _kim_smoother(alpha: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Kim (1994) backward smoother.

    ``beta[t]`` conditions on the whole sample ``0..T-1``.  By
    construction ``beta[-1] == alpha[-1]``, which is why the final row of
    a fit is identical under either estimate — the property the
    walk-forward feature generator relies on.
    """
    T, K = alpha.shape
    beta = np.zeros((T, K))
    beta[-1] = alpha[-1]
    for t in range(T - 2, -1, -1):
        predicted_next = np.maximum(alpha[t] @ P, 1e-300)
        ratio = beta[t + 1] / predicted_next
        beta[t] = alpha[t] * (P @ ratio)
        beta_sum = beta[t].sum()
        if beta_sum > 0:
            beta[t] /= beta_sum
    return beta


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    """Clip regime probabilities away from exactly 0/1 and renormalise.

    This is a numerical guard so downstream log/odds transforms stay
    finite, not a confidence adjustment.  The floor is deliberately tiny
    (0.1%): a wider one (this module previously used 2%) latches a
    well-separated model's output onto the bounds and destroys the
    gradation that makes the probability useful as a feature.
    """
    clipped = np.clip(probs, _PROB_FLOOR, 1.0 - _PROB_FLOOR)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _validate_regime_label_order(labels: list[str]) -> None:
    """Reject ``regime_labels`` that contradict the ascending-mean sort.

    Regimes are sorted by ascending mean, so index 0 is always the
    lowest-mean (recession) state and the last index the highest-mean
    (expansion) state.  Labels are zipped onto that order positionally,
    so supplying them the other way round mislabels every column and
    inverts the recession probability.  Catching it here turns a silent
    sign error into an immediate, explicit failure.
    """
    if not labels or len(labels) < 2:
        return
    lowered = [str(x).lower() for x in labels]
    if "recession" in lowered[-1] or "expansion" in lowered[0]:
        raise ValueError(
            f"regime_labels must run lowest-mean to highest-mean, i.e. "
            f"recession first and expansion last; got {list(labels)}. "
            f"Regimes are sorted by ascending mean after fitting, so this "
            f"ordering would label the expansion state 'recession' and "
            f"invert get_recession_probability()."
        )


# ---------------------------------------------------------------------------
# RegimeSwitchingModel
# ---------------------------------------------------------------------------


class RegimeSwitchingModel:
    """Markov-switching model with Gaussian emissions.

    Parameters
    ----------
    n_regimes : int
        Number of hidden regimes ``K``.
    regime_labels : list[str] | None
        Human-readable names for the regimes.
    use_statsmodels : bool
        If ``True``, use ``statsmodels`` for estimation; otherwise use
        the built-in Hamilton filter EM.
    max_iter : int
        Maximum EM iterations.
    tol : float
        EM convergence tolerance.
    use_filtered : bool
        When ``True`` (default), ``get_regime_probabilities()`` returns
        Hamilton-**filtered** probabilities, which condition only on data
        through *t* and are therefore safe to use as a historical series.
        When ``False`` it returns Kim-**smoothed** probabilities, which
        condition on *t+1..T* — appropriate for retrospective dating of
        turning points, but they leak future information into any
        supervised model or feature built from the history.  Both are
        always available as ``filtered_probs_`` / ``smoothed_probs_``.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        regime_labels: list[str] | None = None,
        use_statsmodels: bool = False,
        multivariate: bool = True,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_restarts: int = 3,
        use_filtered: bool = True,
    ) -> None:
        self.n_regimes = n_regimes
        self.regime_labels = regime_labels or [
            f"regime_{i}" for i in range(n_regimes)
        ]
        _validate_regime_label_order(self.regime_labels)
        self.use_statsmodels = use_statsmodels
        self.multivariate = multivariate
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.use_filtered = use_filtered

        # Fitted attributes
        self._is_fitted: bool = False
        self._transition_matrix: np.ndarray | None = None
        self._regime_probs: np.ndarray | None = None  # (T, K)
        # Hamilton-filtered (uses data through t only) and Kim-smoothed
        # (uses t+1..T) probabilities are both retained; ``_regime_probs``
        # aliases whichever ``use_filtered`` selects.
        self.filtered_probs_: np.ndarray | None = None
        self.smoothed_probs_: np.ndarray | None = None
        self._means: np.ndarray | None = None  # (K,)
        self._variances: np.ndarray | None = None  # (K,)
        self._means_mv: np.ndarray | None = None  # (K, D)
        self._covs_mv: np.ndarray | None = None   # (K, D, D)
        self._index: pd.DatetimeIndex | None = None

    def _set_probs(self, filtered: np.ndarray, smoothed: np.ndarray) -> None:
        """Store both probability estimates and alias the selected one."""
        self.filtered_probs_ = filtered
        self.smoothed_probs_ = smoothed
        self._regime_probs = filtered if self.use_filtered else smoothed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame | np.ndarray) -> RegimeSwitchingModel:
        """Estimate parameters from a time series of factor values.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            If DataFrame with multiple columns and ``multivariate=True``,
            fits a multivariate Gaussian per regime.  Otherwise collapses
            to 1-D via first principal component.

        Returns
        -------
        self
        """
        if isinstance(data, pd.DataFrame):
            self._index = data.index
            vals = data.values.astype(float)
        else:
            vals = np.asarray(data, dtype=float)
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)

        T_full, D = vals.shape
        K = self.n_regimes

        if D > 1 and self.multivariate:
            # --- Multivariate path: use all columns directly ---
            # Only standardise if the data isn't already roughly unit-scale.
            # DFM factors are z-scored → re-standardising squashes separation.
            col_means = np.nanmean(vals, axis=0)
            col_stds = np.nanstd(vals, axis=0)
            col_stds[col_stds == 0] = 1.0
            already_scaled = np.allclose(col_stds, 1.0, atol=0.3)
            if already_scaled:
                vals_std = vals.copy()
            else:
                vals_std = (vals - col_means) / col_stds

            valid_mask = ~np.isnan(vals_std).any(axis=1)
            Y_clean = vals_std[valid_mask]
            T_clean = len(Y_clean)

            if self.use_statsmodels:
                self._fit_statsmodels(Y_clean[:, 0])
            else:
                self._fit_hamilton_em_multivariate(Y_clean, T_clean, K, D)

            mode_str = "multivariate"
        else:
            # --- Univariate path (original behaviour) ---
            if D > 1:
                col_means = np.nanmean(vals, axis=0)
                col_stds = np.nanstd(vals, axis=0)
                col_stds[col_stds == 0] = 1.0
                vals_std = (vals - col_means) / col_stds
                nan_mask = np.isnan(vals_std)
                vals_std[nan_mask] = 0.0
                U, S, Vt = np.linalg.svd(vals_std, full_matrices=False)
                y = U[:, 0] * S[0]
                # The sign of a singular vector is arbitrary, so the
                # collapsed series can come out inverted — which then
                # inverts the ascending-mean regime sort and hence
                # get_recession_probability().  Orient it to agree with
                # the average input series, so "low" still means "weak".
                panel_mean = np.nanmean(vals_std, axis=1)
                if np.corrcoef(y, panel_mean)[0, 1] < 0:
                    y = -y
                y = (y - y.mean()) / (y.std() + 1e-10)
            else:
                y = vals.ravel()

            valid_mask = ~np.isnan(y)
            y = y[valid_mask]
            T_clean = len(y)

            if self.use_statsmodels:
                self._fit_statsmodels(y)
            else:
                self._fit_hamilton_em(y, T_clean, K)

            mode_str = "univariate"

        # Expand probs back if NaN rows were dropped.  Both estimates are
        # expanded so filtered_probs_ / smoothed_probs_ stay aligned with
        # the caller's index.
        if not valid_mask.all():
            def _expand(probs: np.ndarray) -> np.ndarray:
                full = np.full((T_full, K), np.nan)
                full[valid_mask] = probs
                for t in range(1, len(full)):
                    if np.isnan(full[t, 0]):
                        full[t] = full[t - 1]
                return full

            self.filtered_probs_ = _expand(self.filtered_probs_)
            self.smoothed_probs_ = _expand(self.smoothed_probs_)
            self._regime_probs = (
                self.filtered_probs_ if self.use_filtered
                else self.smoothed_probs_
            )

        self._is_fitted = True
        logger.debug(f"RSM fitted: T={T_clean}, K={K} [{mode_str}, D={D}]")
        return self

    def get_regime_probabilities(self) -> np.ndarray | pd.DataFrame:
        """Return the filtered regime probabilities ``(T × K)``."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before accessing regime probabilities.")
        if self._index is not None:
            return pd.DataFrame(
                self._regime_probs,
                index=self._index,
                columns=self.regime_labels[: self._regime_probs.shape[1]],
            )
        return self._regime_probs

    def get_current_regime(self) -> dict:
        """Return the most-probable regime at the last time step.

        Returns
        -------
        dict
            Keys: ``regime`` (str), ``probabilities`` (dict[str, float]).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit first.")
        last_probs = self._regime_probs[-1]
        best_idx = int(np.argmax(last_probs))
        prob_dict = {
            self.regime_labels[i]: float(last_probs[i])
            for i in range(len(last_probs))
        }
        return {
            "regime": self.regime_labels[best_idx],
            "probabilities": prob_dict,
        }

    def get_transition_matrix(self) -> np.ndarray:
        """Return the ``(K × K)`` transition matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit first.")
        return self._transition_matrix

    def get_recession_probability(self) -> pd.Series | np.ndarray:
        """Return the model's recession probability as a time series.

        The regime with the **lowest** mean on the first observable
        dimension is treated as "recession".  Regimes are sorted by
        ascending mean after fitting, so this is always column 0.

        Resolution is by position, never by label text: the sort order is
        the real invariant, and matching on the name "recession" silently
        returns the *expansion* column whenever ``regime_labels`` is
        supplied in the opposite order.  ``__init__`` rejects that
        ordering outright, so the two can no longer disagree.

        Returns
        -------
        pd.Series or np.ndarray, shape ``(T,)``
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit first.")
        probs = self.get_regime_probabilities()
        if isinstance(probs, pd.DataFrame):
            return probs.iloc[:, 0]
        return probs[:, 0]

    # ------------------------------------------------------------------
    # Hamilton filter EM
    # ------------------------------------------------------------------

    def _fit_hamilton_em(self, y: np.ndarray, T: int, K: int) -> None:
        """Run EM via the Hamilton filter (built-in, no external deps)."""

        # --- Initialise with k-means for better starting points ---
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=10, random_state=42)
            labels = km.fit_predict(y.reshape(-1, 1))
            means = km.cluster_centers_.ravel().copy()
            variances = np.array([
                np.var(y[labels == k]) + 1e-6 for k in range(K)
            ])
        except ImportError:
            # Fallback: quantile-based initialisation
            sorted_y = np.sort(y)
            quantile_idx = np.linspace(0, T - 1, K + 2, dtype=int)[1:-1]
            means = sorted_y[quantile_idx].copy()
            variances = np.full(K, np.var(y) * 0.5) + 1e-6

        # Sticky transition matrix
        P = np.full((K, K), 0.05 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(P, 0.95)
        P = P / P.sum(axis=1, keepdims=True)
        # Uniform initial distribution
        pi0 = np.ones(K) / K

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # --- E-step: Hamilton filter forward pass ---
            alpha = np.zeros((T, K))  # filtered probs (scaled)
            ll = 0.0

            for t in range(T):
                if t == 0:
                    predicted = pi0.copy()
                else:
                    predicted = alpha[t - 1] @ P

                # Emission likelihood for each regime
                eta = _gaussian_density(y[t], means, variances)
                joint = predicted * eta
                total = joint.sum()

                if total < 1e-300:
                    alpha[t] = predicted
                else:
                    alpha[t] = joint / total
                    ll += np.log(total)

            # Check convergence
            if abs(ll - prev_ll) < self.tol and iteration > 0:
                logger.debug(f"Hamilton EM converged at iteration {iteration}")
                break
            prev_ll = ll

            # --- E-step: backward (Kim smoother) ---
            beta = np.zeros((T, K))
            beta[-1] = alpha[-1]  # smoothed probs at T

            for t in range(T - 2, -1, -1):
                predicted_next = alpha[t] @ P
                predicted_next = np.maximum(predicted_next, 1e-300)
                ratio = beta[t + 1] / predicted_next
                beta[t] = alpha[t] * (P @ ratio)
                # Normalise
                beta_sum = beta[t].sum()
                if beta_sum > 0:
                    beta[t] /= beta_sum

            smoothed = beta  # (T, K)

            # --- M-step ---
            gamma_sum = smoothed.sum(axis=0) + 1e-10

            # Update means
            for k in range(K):
                means[k] = (smoothed[:, k] * y).sum() / gamma_sum[k]

            # Update variances
            for k in range(K):
                diff2 = (y - means[k]) ** 2
                variances[k] = (smoothed[:, k] * diff2).sum() / gamma_sum[k]
                variances[k] = max(variances[k], 1e-6)

            # Update transition matrix
            for i in range(K):
                for j in range(K):
                    num = 0.0
                    for t in range(1, T):
                        predicted_j = alpha[t - 1, i] * P[i, j]
                        predicted_all = (alpha[t - 1] @ P)[j]
                        if predicted_all > 1e-300:
                            xi_ij = predicted_j * smoothed[t, j] / predicted_all
                        else:
                            xi_ij = 0.0
                        num += xi_ij
                    P[i, j] = num
                row_sum = P[i].sum()
                if row_sum > 0:
                    P[i] /= row_sum

        # Final E-step with the converged parameters (see the multivariate
        # path for why this is needed).
        final_densities = np.array(
            [_gaussian_density(y[t], means, variances) for t in range(T)]
        )
        alpha, _ = _hamilton_forward(final_densities, P, pi0)
        smoothed = _kim_smoother(alpha, P)

        # --- Sort regimes by ascending mean for consistency ---
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        P = P[np.ix_(order, order)]
        smoothed = smoothed[:, order]
        filtered = alpha[:, order]

        self._means = means
        self._variances = variances
        self._transition_matrix = P
        self._set_probs(
            filtered=_clip_probs(filtered), smoothed=_clip_probs(smoothed),
        )

    def _fit_statsmodels(self, y: np.ndarray) -> None:
        """Delegate estimation to ``statsmodels``."""
        try:
            from statsmodels.tsa.regime_switching.markov_regression import (
                MarkovRegression,
            )
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for use_statsmodels=True"
            ) from exc

        model = MarkovRegression(y, k_regimes=self.n_regimes, trend="c")
        result = model.fit(disp=False)

        smoothed = np.asarray(result.smoothed_marginal_probabilities)
        filtered = np.asarray(result.filtered_marginal_probabilities)
        # statsmodels returns filtered probabilities with one fewer row in
        # some versions; align to the smoothed length conservatively.
        if filtered.shape[0] != smoothed.shape[0]:
            filtered = smoothed.copy()
        self._set_probs(
            filtered=_clip_probs(filtered), smoothed=_clip_probs(smoothed),
        )
        self._transition_matrix = result.regime_transition.reshape(
            self.n_regimes, self.n_regimes
        )
        self._means = np.array([result.params[f"const[{i}]"] for i in range(self.n_regimes)])
        self._variances = np.array([result.params.get(f"sigma2[{i}]", 1.0) for i in range(self.n_regimes)])

    # ------------------------------------------------------------------
    # Multivariate Hamilton filter EM
    # ------------------------------------------------------------------

    def _fit_hamilton_em_multivariate(
        self, Y: np.ndarray, T: int, K: int, D: int
    ) -> None:
        """Run EM via the Hamilton filter with multivariate Gaussian emissions.

        Runs ``n_restarts`` independent initialisations and keeps the
        result with the highest log-likelihood.  Final smoothed
        probabilities are clipped to ``[_PROB_FLOOR, 1 - _PROB_FLOOR]``
        to prevent degenerate 0 / 1 extremes.

        Parameters
        ----------
        Y : np.ndarray, shape ``(T, D)``
            Cleaned observation matrix (no NaN).
        T : int
            Number of time steps.
        K : int
            Number of regimes.
        D : int
            Observation dimensionality.
        """
        best_ll = -np.inf
        best_state: tuple | None = None

        for restart in range(self.n_restarts):
            seed = 42 + restart * 17

            # --- Initialise with k-means ---
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=K, n_init=10, random_state=seed)
                labels = km.fit_predict(Y)
                means = km.cluster_centers_.copy()  # (K, D)
                covs = np.zeros((K, D, D))
                for k in range(K):
                    members = Y[labels == k]
                    if len(members) >= 2:
                        covs[k] = np.cov(members.T) + np.eye(D) * 1e-4
                    else:
                        covs[k] = np.eye(D) * np.var(Y) * 0.5
            except ImportError:
                # Quantile-based fallback
                sorted_idx = np.argsort(Y[:, 0])
                chunk = max(T // K, 1)
                means = np.array(
                    [Y[sorted_idx[k * chunk : (k + 1) * chunk]].mean(axis=0) for k in range(K)]
                )
                covs = np.array([np.eye(D) * np.var(Y) * 0.5 for _ in range(K)])

            # Sticky transition matrix
            P = np.full((K, K), 0.05 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
            np.fill_diagonal(P, 0.95)
            P = P / P.sum(axis=1, keepdims=True)
            pi0 = np.ones(K) / K

            prev_ll = -np.inf
            smoothed = np.full((T, K), 1.0 / K)

            for iteration in range(self.max_iter):
                # --- E-step: Hamilton filter forward pass ---
                alpha = np.zeros((T, K))
                ll = 0.0

                for t in range(T):
                    predicted = pi0.copy() if t == 0 else alpha[t - 1] @ P
                    eta = _multivariate_gaussian_density(Y[t], means, covs)
                    joint = predicted * eta
                    total = joint.sum()

                    if total < 1e-300:
                        alpha[t] = predicted
                    else:
                        alpha[t] = joint / total
                        ll += np.log(total)

                # Check convergence
                if abs(ll - prev_ll) < self.tol and iteration > 0:
                    logger.debug(
                        f"Multivariate Hamilton EM converged at iteration "
                        f"{iteration} (restart {restart})"
                    )
                    break
                prev_ll = ll

                # --- E-step: backward (Kim smoother) ---
                beta = np.zeros((T, K))
                beta[-1] = alpha[-1]

                for t in range(T - 2, -1, -1):
                    predicted_next = alpha[t] @ P
                    predicted_next = np.maximum(predicted_next, 1e-300)
                    ratio = beta[t + 1] / predicted_next
                    beta[t] = alpha[t] * (P @ ratio)
                    beta_sum = beta[t].sum()
                    if beta_sum > 0:
                        beta[t] /= beta_sum

                smoothed = beta

                # --- M-step ---
                gamma_sum = smoothed.sum(axis=0) + 1e-10

                # Update means
                for k in range(K):
                    means[k] = (smoothed[:, k : k + 1] * Y).sum(axis=0) / gamma_sum[k]

                # Update covariances
                for k in range(K):
                    diff = Y - means[k]  # (T, D)
                    weighted_diff = diff * smoothed[:, k : k + 1]  # (T, D)
                    covs[k] = (weighted_diff.T @ diff) / gamma_sum[k]
                    covs[k] += np.eye(D) * 1e-4  # regularise
                    covs[k] = 0.5 * (covs[k] + covs[k].T)  # symmetry

                # Update transition matrix
                for i in range(K):
                    for j in range(K):
                        num = 0.0
                        for t in range(1, T):
                            predicted_j = alpha[t - 1, i] * P[i, j]
                            predicted_all = (alpha[t - 1] @ P)[j]
                            if predicted_all > 1e-300:
                                xi_ij = predicted_j * smoothed[t, j] / predicted_all
                            else:
                                xi_ij = 0.0
                            num += xi_ij
                        P[i, j] = num
                    row_sum = P[i].sum()
                    if row_sum > 0:
                        P[i] /= row_sum

            # Final E-step with the converged parameters.  The loop breaks
            # after the forward pass but before the backward pass, so
            # without this `alpha` and `smoothed` come from different
            # parameter iterations and beta[-1] == alpha[-1] would not
            # hold — which the walk-forward generator depends on.
            final_densities = np.array(
                [_multivariate_gaussian_density(Y[t], means, covs) for t in range(T)]
            )
            alpha, final_ll = _hamilton_forward(final_densities, P, pi0)
            smoothed = _kim_smoother(alpha, P)

            # Check for collapse: min regime occupancy
            avg_occ = smoothed.mean(axis=0)
            min_occ = float(avg_occ.min())
            if min_occ < 0.05:
                logger.debug(
                    f"Restart {restart}: regime collapse detected "
                    f"(min occupancy {min_occ:.3f}), LL={final_ll:.1f}"
                )

            if final_ll > best_ll:
                best_ll = final_ll
                best_state = (
                    means.copy(), covs.copy(), P.copy(),
                    smoothed.copy(), alpha.copy(),
                )

        # Use best result across restarts
        means, covs, P, smoothed, filtered = best_state  # type: ignore[misc]

        # --- Sort regimes by ascending composite mean across ALL dims ---
        # Ordering on means[:, 0] alone identified the "recession" regime
        # by the first factor only.  When the fitted factors disagree in
        # sign — which happens on some training windows — that produced a
        # regime labelled recessionary on the strength of one coordinate
        # while the others said the opposite.  Fitting as of 2019-07, for
        # instance, gave regime means
        #
        #   regime 0: real_activity -0.23, labor_market +1.05
        #   regime 1: real_activity +0.14, labor_market -0.66
        #
        # so dim-0 ordering called regime 0 "recession" even though it was
        # the *strong labour* state.  With labour then running hot, the
        # filter sat in regime 0 and the model reported P(recession) 0.999
        # through a late-cycle expansion.  Averaging over the fitted
        # dimensions makes the ordering reflect the whole state vector.
        #
        # This assumes the factors are oriented so that higher means a
        # stronger economy, which the DFM's loading-sum sign convention
        # and its anchor series provide for the cyclical block the RSM is
        # fitted on by default (real_activity, labor_market).
        regime_score = means.mean(axis=1)
        order = np.argsort(regime_score)

        # A split whose coordinates disagree about which regime is weaker
        # is not cleanly cyclical; say so rather than silently ranking it.
        per_dim_order = [tuple(np.argsort(means[:, d])) for d in range(D)]
        if len(set(per_dim_order)) > 1:
            logger.warning(
                f"RSM regime ordering is ambiguous: the fitted dimensions "
                f"disagree about which regime is weaker "
                f"(per-dimension orderings {per_dim_order}). Ranking on the "
                f"composite mean {np.round(regime_score, 3).tolist()}; the "
                f"regime split may not be a business-cycle split."
            )

        means = means[order]
        covs = covs[order]
        regime_score = regime_score[order]
        P = P[np.ix_(order, order)]
        smoothed = smoothed[:, order]
        filtered = filtered[:, order]

        # --- Numerical guard only ---
        # This used to clip to [0.02, 0.98], which latched the output onto
        # the bounds: in the committed backtest 421 of 434 months sat at a
        # bound, so the "probability" was a step function.  Genuine 0/1
        # confidence is a modelling problem (see the log-likelihood and
        # occupancy warnings below), not something to paper over here.
        smoothed = _clip_probs(smoothed)
        filtered = _clip_probs(filtered)

        # Log collapse warning if applicable
        avg_occ = smoothed.mean(axis=0)
        if avg_occ.min() < 0.05:
            logger.warning(
                f"RSM regime collapse: min average occupancy = "
                f"{avg_occ.min():.3f}. Regime probabilities may be unreliable."
            )

        # ``_means`` is the scalar the regimes are ranked by, so it stays
        # ascending by construction; the full mean vectors are in
        # ``_means_mv``.  It was means[:, 0], which after a composite sort
        # would no longer be ordered.
        self._means = regime_score
        self._variances = np.array([covs[k][0, 0] for k in range(K)])
        self._means_mv = means
        self._covs_mv = covs
        self._transition_matrix = P
        self._set_probs(filtered=filtered, smoothed=smoothed)
