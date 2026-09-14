"""Dynamic Factor Model (DFM) estimated via PCA + EM.

Extracts ``n_factors`` latent common factors from an ``(T × N)`` panel of
macroeconomic indicators.  PCA is used for initialisation, after which
the Kalman filter–based EM algorithm refines the parameters.

The model is:

    y_t = C f_t + ε_t,   ε_t ~ N(0, R)        (observation)
    f_t = A f_{t-1} + η_t, η_t ~ N(0, Q)       (state transition)

where ``f_t`` is a ``K``-dimensional vector of latent factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.data.mixed_frequency import (
    augment_observation_matrix,
    build_cumulator_matrices,
)
from src.models.kalman_filter import KalmanFilter

# ---------------------------------------------------------------------------
# Helper: fill NaNs for PCA initialisation
# ---------------------------------------------------------------------------


def _fill_for_pca(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values column-wise with column means for PCA initialisation."""
    filled = arr.copy()
    for j in range(filled.shape[1]):
        col = filled[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0.0
            col[nan_mask] = col_mean
            filled[:, j] = col
    return filled


def _default_sign_convention(loadings: np.ndarray) -> np.ndarray:
    """Orient each factor so its loadings sum positive.

    The sign of a factor is not identified: flipping ``f_k`` and the
    corresponding column of ``C`` leaves the likelihood unchanged.  Some
    convention therefore has to be imposed, or the orientation is
    effectively random from run to run — and everything downstream that
    assumes "low factor = weak economy" (the RSM's ascending-mean regime
    sort, and hence the recession probability) inherits that coin flip.

    Summing the loadings is the standard choice and is the right one for a
    macro panel, where most series are pro-cyclical and enter positively,
    so the oriented factor rises with activity.  Named anchor series
    override this where they are available.

    Returns
    -------
    np.ndarray of shape ``(K,)`` with values +1 or -1.
    """
    sums = loadings.sum(axis=0)
    signs = np.where(sums < 0, -1.0, 1.0)
    # A factor whose loadings sum to (near) zero has no meaningful
    # orientation from this rule; fall back to the largest single loading.
    ambiguous = np.abs(sums) < 1e-8
    if ambiguous.any():
        dominant = loadings[np.argmax(np.abs(loadings), axis=0), np.arange(loadings.shape[1])]
        signs = np.where(ambiguous, np.where(dominant < 0, -1.0, 1.0), signs)
    return signs


def _varimax(loadings: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Apply varimax rotation to a loading matrix.

    Parameters
    ----------
    loadings : np.ndarray, shape (N, K)
        Unrotated loading matrix.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on rotation change.

    Returns
    -------
    (rotated_loadings, rotation_matrix)
        rotated_loadings : (N, K)
        rotation_matrix  : (K, K) orthogonal matrix ``R`` such that
        ``rotated_loadings = loadings @ R``.
    """
    p, k = loadings.shape
    if k < 2:
        return loadings.copy(), np.eye(k)

    # Start from the identity rotation
    R = np.eye(k)
    L = loadings.copy()
    d = 0.0

    for _ in range(max_iter):
        old_d = d
        B = L @ R
        # Varimax criterion gradient
        # For each column, compute B^3 - B * diag(B'B)/p
        B2 = B ** 2
        col_means_of_B2 = B2.mean(axis=0)  # (K,)
        G = B ** 3 - B * col_means_of_B2[np.newaxis, :]
        # SVD of L' @ G to find the optimal rotation step
        U, S, Vt = np.linalg.svd(L.T @ G, full_matrices=False)
        R = U @ Vt
        d = S.sum()
        if abs(d - old_d) < tol:
            break

    rotated = L @ R
    return rotated, R


# ---------------------------------------------------------------------------
# DynamicFactorModel
# ---------------------------------------------------------------------------


class DynamicFactorModel:
    """PCA + EM Dynamic Factor Model.

    Parameters
    ----------
    n_factors : int
        Number of latent factors ``K``.
    factor_names : list[str] | None
        Human-readable names for the factors.
    max_iter : int
        Maximum EM iterations.
    tol : float
        EM convergence tolerance on log-likelihood change.
    """

    def __init__(
        self,
        n_factors: int = 4,
        factor_names: list[str] | None = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        rotate: bool = True,
        use_filtered: bool = True,
        quarterly_columns: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        use_filtered : bool
            If ``True`` (default), ``factors_`` holds Kalman-**filtered**
            state estimates that use only observations up to time *t*.
            If ``False``, ``factors_`` holds RTS-**smoothed** estimates —
            which incorporate future observations and therefore leak
            look-ahead information into any downstream supervised model.
            Smoothed estimates are always accessible via
            :attr:`smoothed_factors_` regardless of this flag.
        """
        self.n_factors = n_factors
        self.factor_names = factor_names
        self.max_iter = max_iter
        self.tol = tol
        self.rotate = rotate
        self.use_filtered = use_filtered
        self.quarterly_columns = quarterly_columns or []

        # Fitted attributes
        self._is_fitted: bool = False
        self.factors_: np.ndarray | pd.DataFrame | None = None
        self.filtered_factors_: np.ndarray | pd.DataFrame | None = None
        self.smoothed_factors_: np.ndarray | pd.DataFrame | None = None
        self._loadings: np.ndarray | None = None
        self._rotation_matrix: np.ndarray | None = None
        # Factor names in the order varimax actually produced them,
        # matched to anchor loadings rather than assumed positionally.
        self._matched_factor_names: list[str] | None = None
        self._A: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._kf: KalmanFilter | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> DynamicFactorModel:
        """Estimate DFM parameters from an ``(T × N)`` panel.

        Parameters
        ----------
        data : pd.DataFrame
            Panel of macro indicators.  Columns = series, rows = time.

        Returns
        -------
        self
        """
        Y_raw = data.values.astype(float)
        T, N = Y_raw.shape
        K = self.n_factors

        # --- Drop columns that are >50 % NaN (unreliable series) ---
        nan_frac = np.isnan(Y_raw).mean(axis=0)
        good_cols = nan_frac <= 0.5
        if not good_cols.all():
            n_dropped = int((~good_cols).sum())
            logger.info(f"DFM: dropping {n_dropped} columns with >50% NaN")
        Y = Y_raw[:, good_cols]
        N = Y.shape[1]

        # --- Standardise columns (store params for transform) ---
        Y_filled = _fill_for_pca(Y)
        self._col_means = Y_filled.mean(axis=0)
        self._col_stds = Y_filled.std(axis=0)
        self._col_stds[self._col_stds == 0] = 1.0
        Y_std = (Y_filled - self._col_means) / self._col_stds
        self._good_cols = good_cols  # bool mask for transform()

        # For the Kalman filter we use Y_std with NaN re-inserted so that
        # the filter properly skips missing observations.
        Y_kf = (Y - self._col_means) / self._col_stds  # keeps original NaN

        # --- Initialise via PCA on filled standardised data ---
        _, _, Vt = np.linalg.svd(Y_std, full_matrices=False)
        # PCA initialisation: only the loadings are needed, since the
        # first Kalman pass re-derives the factor path from them.
        C_init = Vt[:K, :].T  # (N × K)

        # --- Identify quarterly columns for Mariano–Murasawa cumulator ---
        quarterly_col_indices: list[int] = []
        if self.quarterly_columns and isinstance(data, pd.DataFrame):
            surviving_cols = [
                data.columns[i] for i in range(len(data.columns)) if good_cols[i]
            ]
            quarterly_col_indices = [
                i for i, c in enumerate(surviving_cols)
                if c in self.quarterly_columns
            ]

        use_cumulator = len(quarterly_col_indices) > 0
        if use_cumulator:
            cum_info = build_cumulator_matrices(K, quarterly_col_indices, N)
            n_aug = cum_info["n_states_aug"]
            logger.debug(
                f"DFM: using MM cumulator for {len(quarterly_col_indices)} "
                f"quarterly series (state dim {K} → {n_aug})"
            )
        else:
            n_aug = K

        # --- EM via Kalman smoother ---
        A = np.eye(K) * 0.8  # AR(1) prior
        Q = np.eye(K) * 0.5
        R = np.eye(N) * 1.0
        C = C_init.copy()

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # Build (potentially augmented) Kalman matrices
            if use_cumulator:
                A_aug = cum_info["A_aug_template"].copy()
                A_aug[:K, :K] = A
                Q_aug = np.zeros((n_aug, n_aug))
                Q_aug[:K, :K] = Q
                C_aug = augment_observation_matrix(
                    C, quarterly_col_indices, n_aug, K,
                )
                kf = KalmanFilter(A=A_aug, C=C_aug, Q=Q_aug, R=R)
            else:
                kf = KalmanFilter(A=A, C=C, Q=Q, R=R)

            filt = kf.filter(Y_kf)
            smooth = kf.smooth(filt)

            ll = filt.log_likelihood
            if abs(ll - prev_ll) < self.tol and iteration > 0:
                logger.debug(f"EM converged at iteration {iteration}, LL={ll:.2f}")
                break
            prev_ll = ll

            X_smooth_full = smooth.smoothed_states   # (T, n_aug)
            P_smooth_full = smooth.smoothed_covs     # (T, n_aug, n_aug)

            # Extract only the factor block for M-step parameter updates
            X_smooth = X_smooth_full[:, :K]
            P_smooth = P_smooth_full[:, :K, :K]

            # --- M-step ---
            # Update C (loadings) — only on the K factor states
            sum_xxt = np.zeros((K, K))
            sum_yxt = np.zeros((N, K))
            for t in range(T):
                y_t = Y_kf[t]
                x_t = X_smooth[t]
                xxt = np.outer(x_t, x_t) + P_smooth[t]
                sum_xxt += xxt
                obs_mask = ~np.isnan(y_t)
                for i in range(N):
                    if obs_mask[i]:
                        sum_yxt[i] += y_t[i] * x_t

            C = sum_yxt @ np.linalg.solve(sum_xxt, np.eye(K))

            # Update A (state transition) — on the K factor block
            if T > 1:
                sum_xpxt = np.zeros((K, K))
                sum_xpxp = np.zeros((K, K))
                for t in range(1, T):
                    cross = smooth.smoothed_cross_covs[t - 1, :K, :K]
                    sum_xpxt += np.outer(X_smooth[t], X_smooth[t - 1]) + cross
                    sum_xpxp += np.outer(X_smooth[t - 1], X_smooth[t - 1]) + P_smooth[t - 1]
                A = sum_xpxt @ np.linalg.solve(sum_xpxp, np.eye(K))

            # Update Q
            sum_xxp = np.zeros((K, K))
            for t in range(1, T):
                diff = X_smooth[t] - A @ X_smooth[t - 1]
                sum_xxp += np.outer(diff, diff) + P_smooth[t] - A @ smooth.smoothed_cross_covs[t - 1, :K, :K].T
            Q = sum_xxp / max(T - 1, 1)
            Q = 0.5 * (Q + Q.T)

            # Update R — diagonal only.  For quarterly series with the
            # cumulator, use the full augmented state to compute residuals.
            C_for_resid = (
                augment_observation_matrix(C, quarterly_col_indices, n_aug, K)
                if use_cumulator else C
            )
            diag_R = np.zeros(N)
            counts = np.zeros(N)
            for t in range(T):
                y_t = Y_kf[t]
                obs_mask = ~np.isnan(y_t)
                resid = y_t - C_for_resid @ X_smooth_full[t]
                P_proj = C_for_resid @ P_smooth_full[t] @ C_for_resid.T
                for i in range(N):
                    if obs_mask[i]:
                        diag_R[i] += resid[i] ** 2 + P_proj[i, i]
                        counts[i] += 1
            counts[counts == 0] = 1.0
            R = np.diag(diag_R / counts)

        # --- Store results ---
        self._A = A
        self._Q = Q
        self._R = R
        self._loadings = C
        self._use_cumulator = use_cumulator
        self._quarterly_col_indices = quarterly_col_indices

        if use_cumulator:
            A_aug_final = cum_info["A_aug_template"].copy()
            A_aug_final[:K, :K] = A
            Q_aug_final = np.zeros((n_aug, n_aug))
            Q_aug_final[:K, :K] = Q
            C_aug_final = augment_observation_matrix(
                C, quarterly_col_indices, n_aug, K,
            )
            self._kf = KalmanFilter(A=A_aug_final, C=C_aug_final, Q=Q_aug_final, R=R)
        else:
            self._kf = KalmanFilter(A=A, C=C, Q=Q, R=R)

        # Remember which series survived, for sign alignment
        if isinstance(data, pd.DataFrame):
            all_cols = data.columns.tolist()
            self._series_names = [
                all_cols[i] for i in range(len(all_cols)) if good_cols[i]
            ]
        else:
            self._series_names = None

        # Final E-step for factor estimates (on standardised data).
        # Compute BOTH filtered (no look-ahead) and smoothed (uses future)
        # states so callers can pick the appropriate one.
        # When the cumulator is active the state dim is n_aug > K;
        # extract only the first K entries which are the actual factors.
        filt_final = self._kf.filter(Y_kf)
        smooth_final = self._kf.smooth(filt_final)
        filtered_arr = filt_final.filtered_states[:, :K]   # (T, K) — real-time safe
        smoothed_arr = smooth_final.smoothed_states[:, :K]  # (T, K) — uses t+1..T

        # --- Varimax rotation for interpretable factors ---
        if self.rotate and K >= 2:
            rotated_loadings, R_rot = _varimax(C)
            filtered_arr = filtered_arr @ R_rot
            smoothed_arr = smoothed_arr @ R_rot
            self._loadings = rotated_loadings
            self._rotation_matrix = R_rot
            logger.debug(f"DFM: applied varimax rotation ({K} factors)")

            # Name factors from loading evidence (varimax does not
            # preserve column order), then sign-align so that the
            # dominant economic signal has the intuitive direction.
            matched = self._match_factors_to_names(rotated_loadings)
            if matched is not None:
                matched_names, sign_flips = matched
                self._matched_factor_names = matched_names
                self._sign_flips = sign_flips
                self._loadings = rotated_loadings * sign_flips[np.newaxis, :]
                filtered_arr = filtered_arr * sign_flips[np.newaxis, :]
                smoothed_arr = smoothed_arr * sign_flips[np.newaxis, :]
            else:
                self._matched_factor_names = None
                self._sign_flips = _default_sign_convention(rotated_loadings)
                self._loadings = rotated_loadings * self._sign_flips[np.newaxis, :]
                filtered_arr = filtered_arr * self._sign_flips[np.newaxis, :]
                smoothed_arr = smoothed_arr * self._sign_flips[np.newaxis, :]
        else:
            # No rotation (K == 1, or rotate=False) still needs an
            # orientation, or the factor's sign is left to the arbitrary
            # sign of the SVD and the regime sort becomes a coin flip.
            self._rotation_matrix = np.eye(K)
            self._matched_factor_names = None
            self._sign_flips = _default_sign_convention(C)
            self._loadings = C * self._sign_flips[np.newaxis, :]
            filtered_arr = filtered_arr * self._sign_flips[np.newaxis, :]
            smoothed_arr = smoothed_arr * self._sign_flips[np.newaxis, :]

        # Normalise factors to zero-mean, unit-variance so downstream
        # consumers (regime model, nowcaster) get well-scaled inputs.
        # IMPORTANT: scale from the SMOOTHED arr (a full-sample statistic
        # but applied identically to both so it doesn't introduce a
        # look-ahead at *prediction* time — the scaling is a fixed
        # property of the fitted model, learned at training time).
        f_mean = np.nanmean(smoothed_arr, axis=0)
        f_std = np.nanstd(smoothed_arr, axis=0)
        f_std[f_std == 0] = 1.0
        self._factor_mean = f_mean
        self._factor_std = f_std
        filtered_arr = (filtered_arr - f_mean) / f_std
        smoothed_arr = (smoothed_arr - f_mean) / f_std

        if isinstance(data, pd.DataFrame):
            cols = (
                self._matched_factor_names
                or self.factor_names
                or [f"factor_{i}" for i in range(K)]
            )
            cols = list(cols)[:K]
            self.filtered_factors_ = pd.DataFrame(
                filtered_arr, index=data.index, columns=cols,
            )
            self.smoothed_factors_ = pd.DataFrame(
                smoothed_arr, index=data.index, columns=cols,
            )
        else:
            self.filtered_factors_ = filtered_arr
            self.smoothed_factors_ = smoothed_arr

        # ``factors_`` is the canonical feature input for downstream
        # supervised models: default to filtered so we don't leak t+1..T
        # into the probit / RSM training signal.
        self.factors_ = (
            self.filtered_factors_ if self.use_filtered
            else self.smoothed_factors_
        )

        self._is_fitted = True
        logger.debug(
            f"DFM fitted: {T} obs × {N} series → {K} factors "
            f"(factors_ = {'filtered' if self.use_filtered else 'smoothed'})"
        )
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Project new data onto the fitted factor space.

        Returns filtered estimates when the model was constructed with
        ``use_filtered=True`` (the default) and smoothed otherwise.
        This matches whatever ``factors_`` held after ``fit()``.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before calling transform().")

        Y_raw = data.values.astype(float)
        # Apply same column filter
        if hasattr(self, "_good_cols") and self._good_cols is not None:
            Y_raw = Y_raw[:, self._good_cols]
        # Standardise using training params
        Y_std = (Y_raw - self._col_means) / self._col_stds

        K = self.n_factors
        filt = self._kf.filter(Y_std)
        if self.use_filtered:
            factors = filt.filtered_states[:, :K]
        else:
            smooth = self._kf.smooth(filt)
            factors = smooth.smoothed_states[:, :K]

        # Apply same rotation as training
        if self._rotation_matrix is not None:
            factors = factors @ self._rotation_matrix

        # Apply same sign flips as training
        if hasattr(self, "_sign_flips"):
            factors = factors * self._sign_flips[np.newaxis, :]

        # Apply same normalisation as training
        if hasattr(self, "_factor_mean"):
            factors = (factors - self._factor_mean) / self._factor_std

        if isinstance(data, pd.DataFrame) and isinstance(self.factors_, pd.DataFrame):
            return pd.DataFrame(
                factors,
                index=data.index,
                columns=self.factors_.columns,
            )
        return factors

    def get_loadings(self) -> np.ndarray:
        """Return the observation loading matrix ``C`` of shape ``(N, K)``."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before accessing loadings.")
        return self._loadings

    # ------------------------------------------------------------------
    # Sign alignment
    # ------------------------------------------------------------------

    # Anchor series whose loadings should be *positive* on their factor.
    # For each factor name we list FRED codes that unambiguously move in
    # the "positive" direction of that economic concept.
    _SIGN_ANCHORS: dict[str, list[str]] = {
        "real_activity": ["INDPRO", "PAYEMS", "CFNAI", "W875RX1"],
        "labor_market": ["PAYEMS", "JTSJOL", "CES0500000003"],
        "inflation": ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIFIS"],
        "financial_stress": ["BAA10Y", "TEDRATE", "VIXCLS"],
    }

    def _match_factors_to_names(
        self, loadings: np.ndarray
    ) -> tuple[list[str], np.ndarray] | None:
        """Assign factor names by loading evidence, then compute signs.

        Varimax imposes no ordering on its output columns, so factor *k*
        is not necessarily ``factor_names[k]``.  Naming positionally means
        the column called "inflation" may in fact be the financial-stress
        factor — and it is then sign-aligned against the wrong anchors.

        Instead, score every (factor, name) pair by mean absolute loading
        on that name's anchor series and solve the assignment problem, so
        each name lands on the factor that best evidences it.  The sign is
        taken from the *matched* anchors afterwards.

        Returns
        -------
        (names, signs) or ``None``
            ``names`` is the reordered name list (length K); ``signs``
            holds ±1 per factor.  ``None`` when names or series are
            unavailable.
        """
        if self._series_names is None or self.factor_names is None:
            return None

        K = loadings.shape[1]
        names = list(self.factor_names[:K])
        if len(names) < K:
            names += [f"factor_{i}" for i in range(len(names), K)]
        series_list = self._series_names

        # score[k, j] = evidence that factor k is concept names[j]
        score = np.zeros((K, K))
        anchor_idx_by_name: dict[int, list[int]] = {}
        for j, fname in enumerate(names):
            anchors = self._SIGN_ANCHORS.get(fname, [])
            idx = [i for i, s in enumerate(series_list) if s in anchors]
            anchor_idx_by_name[j] = idx
            if not idx:
                continue
            for k in range(K):
                score[k, j] = np.mean(np.abs(loadings[idx, k]))

        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-score)
        except ImportError:  # pragma: no cover - scipy is a hard dependency
            rows, cols = np.arange(K), np.arange(K)

        assigned = list(range(K))
        for k, j in zip(rows, cols):
            assigned[k] = j

        matched_names = [names[j] for j in assigned]
        signs = _default_sign_convention(loadings)
        for k, j in enumerate(assigned):
            idx = anchor_idx_by_name.get(j, [])
            if not idx:
                continue  # keep the default orientation
            mean_loading = float(np.mean(loadings[idx, k] * signs[k]))
            if mean_loading < 0:
                signs[k] *= -1.0

        if matched_names != names:
            logger.info(
                f"DFM: varimax reordered the factors; naming by loading "
                f"evidence gives {matched_names} (positional would have "
                f"been {names})"
            )
        logger.debug(f"DFM: factor signs {signs.tolist()}")
        return matched_names, signs
