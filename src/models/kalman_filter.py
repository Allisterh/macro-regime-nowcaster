"""Linear-Gaussian Kalman filter and Rauch–Tung–Striebel (RTS) smoother.

Implements the exact Kalman filter/smoother recursions for the state-space
model:

    State:       x_t = A x_{t-1} + η_t,   η_t ~ N(0, Q)
    Observation: y_t = C x_t + ε_t,        ε_t ~ N(0, R)

NaN observations are handled transparently: when a row of y is entirely
``NaN``, the filter skips the update step and propagates the prediction.
When individual elements are ``NaN``, the filter marginalises them out by
masking the corresponding rows of C and R.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Output of :meth:`KalmanFilter.filter`."""

    filtered_states: np.ndarray    # (T, n_states)
    filtered_covs: np.ndarray      # (T, n_states, n_states)
    predicted_states: np.ndarray   # (T, n_states)
    predicted_covs: np.ndarray     # (T, n_states, n_states)
    log_likelihood: float


@dataclass
class SmootherResult:
    """Output of :meth:`KalmanFilter.smooth`."""

    smoothed_states: np.ndarray       # (T, n_states)
    smoothed_covs: np.ndarray         # (T, n_states, n_states)
    smoothed_cross_covs: np.ndarray   # (T-1, n_states, n_states)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Standard linear-Gaussian Kalman filter with NaN support.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix  (n_states × n_states).
    C : np.ndarray
        Observation matrix  (n_obs × n_states).
    Q : np.ndarray
        State noise covariance  (n_states × n_states).
    R : np.ndarray
        Observation noise covariance  (n_obs × n_obs).
    initial_state : np.ndarray, optional
        Mean of initial state belief.  Default: zeros.
    initial_covariance : np.ndarray, optional
        Covariance of initial state belief.  Default: ``10 * I``.
    """

    def __init__(
        self,
        A: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.A = np.atleast_2d(A).astype(float)
        self.C = np.atleast_2d(C).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)

        n_states = self.A.shape[0]
        if initial_state is not None:
            self.x0 = np.atleast_1d(initial_state).astype(float)
        else:
            self.x0 = np.zeros(n_states)

        if initial_covariance is not None:
            self.P0 = np.atleast_2d(initial_covariance).astype(float)
        else:
            self.P0 = np.eye(n_states) * 10.0

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(self, observations: np.ndarray) -> FilterResult:
        """Run the Kalman filter forward pass.

        Parameters
        ----------
        observations : np.ndarray
            Shape ``(T, n_obs)``.  ``NaN`` entries are treated as missing.

        Returns
        -------
        FilterResult
        """
        Y = np.atleast_2d(observations)
        T = Y.shape[0]
        n_s = self.A.shape[0]

        filtered_states = np.zeros((T, n_s))
        filtered_covs = np.zeros((T, n_s, n_s))
        predicted_states = np.zeros((T, n_s))
        predicted_covs = np.zeros((T, n_s, n_s))
        log_lik = 0.0

        x_prev = self.x0.copy()
        P_prev = self.P0.copy()

        for t in range(T):
            # --- Predict ---
            x_pred = self.A @ x_prev
            P_pred = self.A @ P_prev @ self.A.T + self.Q

            predicted_states[t] = x_pred
            predicted_covs[t] = P_pred

            y_t = Y[t]
            obs_mask = ~np.isnan(y_t)

            if not obs_mask.any():
                # All observations missing → no update
                filtered_states[t] = x_pred
                filtered_covs[t] = P_pred
            else:
                # Marginalise to observed subset
                C_obs = self.C[obs_mask]
                R_obs = self.R[np.ix_(obs_mask, obs_mask)]
                y_obs = y_t[obs_mask]

                # Innovation
                innov = y_obs - C_obs @ x_pred
                S = C_obs @ P_pred @ C_obs.T + R_obs

                # Log-likelihood contribution
                n_o = len(y_obs)
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    S_inv = np.linalg.solve(S, np.eye(n_o))
                    log_lik += -0.5 * (
                        n_o * np.log(2 * np.pi) + logdet + innov @ S_inv @ innov
                    )
                else:
                    S_inv = np.linalg.pinv(S)

                # Kalman gain
                K = P_pred @ C_obs.T @ S_inv

                # Update
                x_filt = x_pred + K @ innov
                P_filt = (np.eye(n_s) - K @ C_obs) @ P_pred

                # Enforce symmetry
                P_filt = 0.5 * (P_filt + P_filt.T)

                filtered_states[t] = x_filt
                filtered_covs[t] = P_filt

            x_prev = filtered_states[t]
            P_prev = filtered_covs[t]

        return FilterResult(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            log_likelihood=float(log_lik),
        )

    # ------------------------------------------------------------------
    # RTS Smoother
    # ------------------------------------------------------------------

    def smooth(self, filter_result: FilterResult) -> SmootherResult:
        """Run the Rauch–Tung–Striebel backward smoother.

        Parameters
        ----------
        filter_result : FilterResult
            Output of a prior ``filter()`` call.

        Returns
        -------
        SmootherResult
        """
        fs = filter_result.filtered_states
        fc = filter_result.filtered_covs
        ps = filter_result.predicted_states
        pc = filter_result.predicted_covs

        T, n_s = fs.shape

        smoothed_states = np.zeros_like(fs)
        smoothed_covs = np.zeros_like(fc)
        smoothed_cross_covs = np.zeros((T - 1, n_s, n_s))

        # Initialise from last filtered value
        smoothed_states[-1] = fs[-1]
        smoothed_covs[-1] = fc[-1]

        for t in range(T - 2, -1, -1):
            P_pred_next = pc[t + 1]
            P_pred_inv = np.linalg.solve(P_pred_next, np.eye(n_s))

            # Smoother gain
            L = fc[t] @ self.A.T @ P_pred_inv

            smoothed_states[t] = (
                fs[t] + L @ (smoothed_states[t + 1] - ps[t + 1])
            )
            smoothed_covs[t] = (
                fc[t] + L @ (smoothed_covs[t + 1] - P_pred_next) @ L.T
            )
            # Enforce symmetry
            smoothed_covs[t] = 0.5 * (smoothed_covs[t] + smoothed_covs[t].T)

            # Cross-covariance  Cov(x_{t+1}, x_t | Y)
            smoothed_cross_covs[t] = smoothed_covs[t + 1] @ L.T

        return SmootherResult(
            smoothed_states=smoothed_states,
            smoothed_covs=smoothed_covs,
            smoothed_cross_covs=smoothed_cross_covs,
        )
