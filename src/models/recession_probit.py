"""Probit recession probability model.

Estimates P(recession_t) = Φ(X_t β) via maximum likelihood, where Φ is the
standard normal CDF.  Designed to be trained on NBER recession labels with
DFM factors, yield-curve spread, and CFNAI as features.

Features
--------
- Automatic lag augmentation (1-month lag by default)
- L2 regularisation for small-sample stability
- Analytic gradient for fast L-BFGS-B optimisation
- Handles NaN gracefully (drops rows for training, returns 0.5 for prediction)

Usage
-----
>>> from src.models.recession_probit import RecessionProbit
>>> model = RecessionProbit(add_lags=1, regularization=0.01)
>>> model.fit(X_train, y_train)
>>> proba = model.predict_proba(X_new)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import optimize
from scipy.stats import norm

_PROBA_FLOOR = 1e-3


class _SigmoidCalibrator:
    """Platt scaling wrapper exposing the same ``predict`` as isotonic."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


class RecessionProbit:
    """Probit regression for recession probability estimation.

    Parameters
    ----------
    add_intercept : bool
        Whether to prepend an intercept column.
    add_lags : int
        Number of 1-period lagged copies of each feature to append.
        Lags introduce NaN at the start which are dropped during fit.
    regularization : float
        L2 regularisation parameter (λ).  Larger values shrink
        coefficients toward zero for better generalisation.
    """

    def __init__(
        self,
        add_intercept: bool = True,
        add_lags: int = 1,
        regularization: float = 0.01,
        class_balanced: bool = False,
        max_missing_fraction: float = 0.5,
        max_column_missing: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        class_balanced : bool
            If ``True``, weight positives (recessions) by ``n_neg / n_pos``
            in the negative log-likelihood so that rare-event training
            doesn't collapse the decision boundary toward the majority
            class.  Recessions are ~14% of months, so the default weight
            ratio is ~6:1.  See also scikit-learn's ``class_weight`` behaviour.
        """
        self.add_intercept = add_intercept
        self.add_lags = add_lags
        self.regularization = regularization
        self.class_balanced = class_balanced
        # Rows missing more than this share of their features fall back to
        # 0.5; below it, missing entries are imputed with training means.
        self.max_missing_fraction = max_missing_fraction
        # Feature columns missing more than this share of the training
        # window are excluded from the fit rather than emptying it.
        # 0.5 rather than something laxer because rows are then dropped
        # on any remaining NaN: a quarterly series in a monthly panel is
        # 67% missing, and keeping it would discard two rows in three.
        self.max_column_missing = max_column_missing

        # Fitted attributes
        self._coef: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._n_raw_features: int = 0
        self._is_fitted: bool = False
        self._pos_weight: float = 1.0
        self._neg_weight: float = 1.0
        self._train_means: np.ndarray | None = None
        self._keep_cols: np.ndarray | None = None
        # Monotone score -> probability map, set by fit_calibration().
        # None means predict_proba returns the raw probit output.
        self._calibrator = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> RecessionProbit:
        """Fit probit model via maximum-likelihood estimation.

        Parameters
        ----------
        X : array-like, shape (T, D)
            Feature matrix (DFM factors, CFNAI, yield spread, etc.).
        y : array-like, shape (T,)
            Binary labels: 1 = NBER recession, 0 = expansion.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)

        self._n_raw_features = X_arr.shape[1]
        y_arr = np.asarray(y, dtype=float).ravel()

        # Build augmented feature matrix (lags + intercept)
        X_aug = self._augment_features(X_arr)

        # Drop unusable COLUMNS before unusable rows.
        #
        # Dropping rows first means a single feature with no data in the
        # training window empties the entire training set: fit() raises,
        # Nowcaster catches it, and the signal silently sits at 0.5.  That
        # is what happened when leading indicators starting in 1990 and
        # 2023 were added — the probit stopped training in 64% of months
        # across 1967-2025 and nobody saw an error.
        #
        # A feature with (almost) nothing in the training window carries
        # no information for this fit, so it is excluded and recorded, and
        # predict_proba applies the same selection.
        if X_aug.shape[0] == 0:
            # No rows at all: the column screen below would divide by zero
            # and report every feature as unusable, masking the clearer
            # "too few observations" error raised further down.
            raise ValueError(
                "Too few valid training observations (0). Need at least 20."
            )

        col_missing = np.isnan(X_aug).mean(axis=0)
        self._keep_cols = col_missing <= self.max_column_missing
        if not self._keep_cols.any():
            raise ValueError(
                "Every feature is unusable in this window: all columns "
                f"exceed max_column_missing={self.max_column_missing:.0%}."
            )
        dropped = int((~self._keep_cols).sum())
        if dropped:
            logger.debug(
                f"Probit: dropping {dropped} feature column(s) with more "
                f"than {self.max_column_missing:.0%} missing in this window"
            )
        X_aug = X_aug[:, self._keep_cols]

        # Now drop rows with NaN among the columns we are actually using
        valid = ~(np.isnan(X_aug).any(axis=1) | np.isnan(y_arr))
        X_clean = X_aug[valid]
        y_clean = y_arr[valid]

        if len(y_clean) < 20:
            raise ValueError(
                f"Too few valid training observations ({len(y_clean)}). "
                "Need at least 20."
            )

        n_features = X_clean.shape[1]
        n_pos = int(y_clean.sum())
        n_neg = len(y_clean) - n_pos
        logger.debug(
            f"Probit training: {len(y_clean)} obs, "
            f"{n_pos} recession / {n_neg} expansion, "
            f"{n_features} features"
        )

        # Compute per-class weights for the negative log-likelihood.
        # Default (``class_balanced=False``) gives equal weight to each
        # sample; with rare positives this under-weights the minority
        # class.  The sklearn-style balanced scheme assigns weight
        # ``N / (2 * n_class)`` to each sample of a given class.
        if self.class_balanced and n_pos > 0 and n_neg > 0:
            n_total = len(y_clean)
            self._pos_weight = n_total / (2.0 * n_pos)
            self._neg_weight = n_total / (2.0 * n_neg)
        else:
            self._pos_weight = 1.0
            self._neg_weight = 1.0
        sample_weights = np.where(
            y_clean > 0.5, self._pos_weight, self._neg_weight,
        )

        # Initialise coefficients at zero
        beta0 = np.zeros(n_features)

        # Maximise log-likelihood via L-BFGS-B
        result = optimize.minimize(
            self._neg_log_likelihood,
            beta0,
            args=(X_clean, y_clean, self.regularization, sample_weights),
            method="L-BFGS-B",
            jac=self._neg_log_likelihood_grad,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(f"Probit optimisation warning: {result.message}")

        self._coef = result.x
        self._is_fitted = True
        # Column means of the augmented training matrix, used to impute
        # missing features at prediction time.  Taken from the training
        # fold only, so this introduces no look-ahead.
        self._train_means = np.nanmean(X_clean, axis=0)

        # Training diagnostics
        p_train = norm.cdf(X_clean @ self._coef)
        pred_train = (p_train > 0.5).astype(int)
        acc = float((pred_train == y_clean).mean())
        logger.debug(f"Probit fitted: train accuracy = {acc:.1%}")

        return self

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray, *, calibrated: bool = True
    ) -> np.ndarray:
        """Return P(recession) for each observation.

        Parameters
        ----------
        X : array-like, shape (T, D)
            Same structure as the training features (raw, before
            augmentation — lags and intercept are added automatically).

        Returns
        -------
        np.ndarray, shape (T,)
            Recession probabilities ∈ [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before calling predict_proba.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)

        X_aug = self._augment_features(X_arr)
        if self._keep_cols is not None:
            X_aug = X_aug[:, self._keep_cols]

        # Impute missing features with their training means rather than
        # discarding the whole row.
        #
        # This used to return 0.5 whenever *any* feature was NaN, which is
        # catastrophic with features of differing history: a row is usable
        # only where every single one exists.  Adding leading indicators
        # that start in 1990 and 2023 silently switched the probit off in
        # 64% of months across 1967-2025 — it was not underperforming,
        # it was not running.  The failure is invisible because 0.5 is a
        # plausible-looking probability.
        missing = np.isnan(X_aug)
        frac_missing = missing.mean(axis=1)
        X_filled = X_aug.copy()
        if missing.any() and self._train_means is not None:
            fill = np.broadcast_to(self._train_means, X_aug.shape)
            X_filled = np.where(missing, fill, X_aug)

        # A row that is mostly missing carries too little information to
        # score, so it still falls back to the uninformative 0.5.
        usable = (~np.isnan(X_filled).any(axis=1)) & (
            frac_missing <= self.max_missing_fraction
        )
        proba = np.full(X_aug.shape[0], 0.5)
        if usable.any():
            proba[usable] = norm.cdf(X_filled[usable] @ self._coef)
        # Numerical guard only.  This used to clip to [0.05, 0.95], which
        # turned a near-separable in-sample fit into a step function —
        # 352 of 434 months sat at exactly 0.05 in the committed backtest,
        # leaving the "probability" with no usable gradation.  Genuine
        # overconfidence is a regularisation problem (see `regularization`
        # and `fit_regularization_cv`), not something to clip away.
        proba = np.clip(proba, _PROBA_FLOOR, 1.0 - _PROBA_FLOOR)

        # Monotone, so ordering is preserved and discrimination is
        # essentially unchanged (ties can nudge AUC slightly); the point
        # is to move the values onto the observed frequency scale.
        if calibrated and self._calibrator is not None:
            proba = np.clip(
                self._calibrator.predict(proba), _PROBA_FLOOR, 1.0 - _PROBA_FLOOR
            )
        return proba

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary recession predictions.

        Parameters
        ----------
        X : array-like
            Features.
        threshold : float
            Classification threshold (default 0.5).

        Returns
        -------
        np.ndarray, shape (T,)
            Binary predictions: 1 = recession, 0 = expansion.
        """
        return (self.predict_proba(X) > threshold).astype(int)

    def fit_regularization_cv(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        grid: list[float] | None = None,
        n_splits: int = 4,
    ) -> RecessionProbit:
        """Choose ``regularization`` by forward-chaining CV, then refit.

        The penalty was previously a fixed constant, which on this
        near-separable problem let the coefficients run away and pushed
        every prediction to a clip bound.  Selection uses *forward-chaining*
        splits (train on a prefix, validate on the next block) rather than
        k-fold, because shuffled folds leak across an autocorrelated
        series, and scores by log-loss, which rewards calibration rather
        than just ranking.

        Returns
        -------
        self
        """
        grid = grid or [0.1, 1.0, 5.0, 20.0, 100.0]

        X_arr = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        y_arr = y.values.astype(float) if isinstance(y, pd.Series) else np.asarray(y, float)

        n = len(y_arr)
        # Blocks are contiguous and ordered: fold i trains on everything
        # before block i and validates on block i.
        edges = np.linspace(0, n, n_splits + 2, dtype=int)

        best_lambda, best_score = self.regularization, np.inf
        for lam in grid:
            losses = []
            for i in range(1, n_splits + 1):
                tr_end, va_end = edges[i], edges[i + 1]
                if tr_end < 30 or va_end - tr_end < 6:
                    continue
                trial = RecessionProbit(
                    add_intercept=self.add_intercept,
                    add_lags=self.add_lags,
                    regularization=lam,
                    class_balanced=self.class_balanced,
                )
                try:
                    trial.fit(X_arr[:tr_end], y_arr[:tr_end])
                    p = trial.predict_proba(X_arr[tr_end:va_end])
                except Exception:
                    continue
                yv = y_arr[tr_end:va_end]
                ok = ~np.isnan(yv)
                if not ok.any():
                    continue
                p_ok = np.clip(p[ok], _PROBA_FLOOR, 1 - _PROBA_FLOOR)
                losses.append(
                    -np.mean(yv[ok] * np.log(p_ok) + (1 - yv[ok]) * np.log(1 - p_ok))
                )
            if losses:
                score = float(np.mean(losses))
                logger.debug(f"Probit CV: lambda={lam:g} log-loss={score:.4f}")
                if score < best_score:
                    best_score, best_lambda = score, lam

        logger.info(
            f"Probit: selected regularization={best_lambda:g} "
            f"(CV log-loss={best_score:.4f})"
        )
        self.regularization = best_lambda
        return self.fit(X, y)

    def fit_calibration(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        method: str = "isotonic",
    ) -> RecessionProbit:
        """Learn a monotone map from raw scores to calibrated probabilities.

        A probit can rank months well while its numbers are poor
        probabilities: measured on the walk-forward it reached AUC 0.919
        with a Brier score of 0.0855, worse than a constant forecast at
        the base rate.  Ranking and calibration are different properties,
        and only the second matters when the output is consumed as a
        probability.

        Isotonic regression fits a non-decreasing step function, so it
        cannot *reorder* predictions; it only moves them onto the observed
        frequency scale.  Note it is non-decreasing rather than strictly
        increasing, so it flattens ranges of scores into ties, and AUC can
        move by a little as a result — measured at 0.795 -> 0.787 on a
        synthetic check.  Discrimination is essentially preserved;
        calibration is what improves.

        Fit this on the *training* fold only.  Calibrating on data the
        model is then scored against is circular.

        **Measured on this project's data, this does not help, and it is
        off by default.** Calibrating on 1990-2011 (34 recession months)
        and scoring on 2011-2026 (2) collapsed the output onto 10 distinct
        levels, 123 of 174 months landing on exactly 0.0. AUC fell from
        0.660 to 0.352 — below chance, because the ordering is then decided
        by tie-breaking — while Brier moved 0.0874 -> 0.0403, still worse
        than the 0.0114 of a constant forecast at the test base rate. The
        apparent gain was a move from very bad to less bad, bought by
        predicting near-constant.

        Isotonic needs many observations per step, and 34 positive events
        do not supply them. Revisit once the sample covers more recessions;
        ``method="sigmoid"`` is the two-parameter alternative for short
        samples.

        Parameters
        ----------
        method : {"isotonic", "sigmoid"}
            Isotonic is flexible but needs a few hundred points; sigmoid
            (Platt scaling) is a two-parameter fallback for short samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Fit the probit before calibrating it.")

        raw = self.predict_proba(X, calibrated=False)
        y_arr = (
            y.values.astype(float) if isinstance(y, pd.Series)
            else np.asarray(y, dtype=float)
        )
        ok = ~(np.isnan(raw) | np.isnan(y_arr))
        if ok.sum() < 30 or len(set(y_arr[ok])) < 2:
            logger.warning("Too few usable rows to calibrate; leaving raw.")
            return self

        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(raw[ok], y_arr[ok])
        elif method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression().fit(raw[ok].reshape(-1, 1), y_arr[ok])
            self._calibrator = _SigmoidCalibrator(lr)
        else:
            raise ValueError(f"Unknown calibration method {method!r}")

        logger.debug(f"Probit: fitted {method} calibrator on {int(ok.sum())} rows")
        return self

    def calibration_report(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Brier score, log-loss, and a reliability table.

        Calibration matters more than accuracy for a probability used as
        a downstream feature: a well-calibrated 0.3 is informative, a
        saturated 0.95 that only reflects in-sample separation is not.

        Returns
        -------
        dict
            ``brier``, ``log_loss``, ``base_rate``, ``frac_interior``
            (share of predictions strictly inside [0.1, 0.9] — a
            degeneracy check), and ``bins`` (a reliability DataFrame).
        """
        p = self.predict_proba(X)
        y_arr = y.values.astype(float) if isinstance(y, pd.Series) else np.asarray(y, float)
        ok = ~np.isnan(y_arr)
        p, y_arr = p[ok], y_arr[ok]

        pc = np.clip(p, _PROBA_FLOOR, 1 - _PROBA_FLOOR)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        bins = pd.DataFrame({
            "bin_mid": (edges[:-1] + edges[1:]) / 2,
            "n": [int((idx == b).sum()) for b in range(n_bins)],
            "mean_predicted": [
                float(p[idx == b].mean()) if (idx == b).any() else np.nan
                for b in range(n_bins)
            ],
            "observed_rate": [
                float(y_arr[idx == b].mean()) if (idx == b).any() else np.nan
                for b in range(n_bins)
            ],
        })

        return {
            "brier": float(np.mean((p - y_arr) ** 2)),
            "log_loss": float(
                -np.mean(y_arr * np.log(pc) + (1 - y_arr) * np.log(1 - pc))
            ),
            "base_rate": float(y_arr.mean()),
            "frac_interior": float(np.mean((p > 0.1) & (p < 0.9))),
            "bins": bins,
        }

    def get_coefficients(self) -> dict[str, float]:
        """Return a dictionary mapping feature names to coefficients."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fit first.")

        names = self._build_augmented_names()
        return dict(zip(names, self._coef.tolist()))

    # ------------------------------------------------------------------
    # Estrella–Mishkin yield-curve probit
    # ------------------------------------------------------------------

    @classmethod
    def estrella_mishkin(
        cls,
        spread: pd.Series,
        nber: pd.Series,
        horizon: int = 12,
        regularization: float = 0.01,
    ) -> RecessionProbit:
        """Fit the canonical NY Fed yield-curve recession probit.

        The Estrella & Mishkin (1998) model estimates the probability of a
        recession *horizon* months ahead using only the term spread
        (typically 10Y − 3M).  This serves as both a baseline against
        which the full ensemble can be benchmarked and an additional
        ensemble feature.

        Parameters
        ----------
        spread : pd.Series
            Yield-curve spread (e.g. T10Y3M from FRED), monthly.
        nber : pd.Series
            Binary NBER recession indicator, monthly (1 = recession).
        horizon : int
            Forecast horizon in months (default 12, per the original paper).
        regularization : float
            L2 regularisation strength.

        Returns
        -------
        RecessionProbit
            A fitted model whose ``predict_proba`` gives the *horizon*-month-
            ahead recession probability given the current spread.
        """
        common = spread.dropna().index.intersection(nber.dropna().index)
        spread_aligned = spread.loc[common]
        nber_aligned = nber.loc[common]

        # Target: is there a recession *horizon* months from now?
        y_forward = nber_aligned.shift(-horizon)
        valid = y_forward.notna()
        X = spread_aligned.loc[valid].to_frame("spread")
        y = y_forward.loc[valid].astype(int)

        model = cls(
            add_intercept=True,
            add_lags=0,
            regularization=regularization,
            class_balanced=True,
        )
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Feature augmentation
    # ------------------------------------------------------------------

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Add lagged features and (optionally) an intercept column."""
        parts = [X]

        for lag in range(1, self.add_lags + 1):
            lagged = np.full_like(X, np.nan)
            lagged[lag:] = X[:-lag]
            parts.append(lagged)

        X_aug = np.hstack(parts)

        if self.add_intercept:
            X_aug = np.hstack([X_aug, np.ones((X_aug.shape[0], 1))])

        return X_aug

    def _build_augmented_names(self) -> list[str]:
        """Return feature names after augmentation."""
        base = self._feature_names or [
            f"x{i}" for i in range(self._n_raw_features)
        ]
        names = list(base)
        for lag in range(1, self.add_lags + 1):
            names.extend([f"{n}_lag{lag}" for n in base])
        if self.add_intercept:
            names.append("intercept")
        return names

    # ------------------------------------------------------------------
    # Likelihood and gradient
    # ------------------------------------------------------------------

    @staticmethod
    def _neg_log_likelihood(
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lam: float = 0.01,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """Weighted negative log-likelihood with L2 regularisation."""
        z = X @ beta
        p = norm.cdf(z)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = y * np.log(p) + (1 - y) * np.log(1 - p)
        if sample_weights is not None:
            ll = ll * sample_weights
        reg = 0.5 * lam * np.dot(beta, beta)
        return -ll.sum() + reg

    @staticmethod
    def _neg_log_likelihood_grad(
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lam: float = 0.01,
        sample_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gradient of the weighted negative log-likelihood."""
        z = X @ beta
        p = norm.cdf(z)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        phi = norm.pdf(z)
        w = y * phi / p - (1 - y) * phi / (1 - p)
        if sample_weights is not None:
            w = w * sample_weights
        grad = -(X.T @ w) + lam * beta
        return grad
