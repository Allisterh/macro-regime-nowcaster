"""End-to-end nowcasting orchestrator.

Ties :class:`DataPipeline`, :class:`DynamicFactorModel`, and
:class:`RegimeSwitchingModel` into a single ``Nowcaster.run()`` call that
returns a :class:`NowcastResult`.

When ``use_ensemble=True`` (the default), recession detection combines
four signals:

1. **CFNAI signal** — Chicago Fed MA3 < −0.7 convention, on the published
   index rather than the standardised panel column
2. **Probit model** — supervised probit trained on NBER recession dates
   (restricted to labels the NBER had actually announced) using DFM
   factors, CFNAI, and yield-curve and credit spreads
3. **Sahm rule** — 3-month MA of UNRATE minus 12-month min ≥ 0.50pp, on
   the unemployment level in percentage points
4. **Multivariate RSM** — Markov-switching on the cyclical DFM factors.
   Currently weighted 0.0; see ``DEFAULT_WEIGHTS``.

The ensemble probability is a weighted average.  The weights are not a
judgement call: they were set from a 434-point monthly walk-forward over
four recessions, scoring each candidate out-of-sample.  They are *not*
ordered by how principled each component looks — the simple CFNAI
threshold carries the most weight because it measured best.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.models.dynamic_factor_model import DynamicFactorModel
from src.models.recession_probit import RecessionProbit
from src.models.regime_backtest import (
    NBER_RECESSIONS,
    get_nber_recession_indicator,
)
from src.models.regime_switching import RegimeSwitchingModel
from src.models.sahm_rule import sahm_recession_probability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_nber_cutoff() -> pd.Timestamp:
    """End of the most recent NBER-dated recession.

    Used as the default training-window upper bound for the supervised
    probit: months past this date have no reliable recession label yet,
    so including them in training would mean labelling on what the model
    is supposed to predict.
    """
    last_trough = max(pd.Timestamp(t) for _, t in NBER_RECESSIONS)
    # A month-end tolerance avoids off-by-one from the first-of-month
    # timestamps stored in the NBER table.
    return last_trough + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# NowcastResult
# ---------------------------------------------------------------------------


@dataclass
class NowcastResult:
    """Container for a single nowcast output."""

    gdp_nowcast: float
    gdp_ci_lower: float
    gdp_ci_upper: float
    regime_probabilities: dict[str, float]
    current_regime: str
    factor_values: dict[str, float] = field(default_factory=dict)
    recession_probability: float = 0.0
    ensemble_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Nowcaster
# ---------------------------------------------------------------------------


class Nowcaster:
    """Macro Regime Nowcaster – full pipeline runner.

    Parameters
    ----------
    pipeline : object
        A :class:`DataPipeline` instance (or mock with ``.run()``).
    n_factors : int
        Number of latent factors for the DFM.
    n_regimes : int
        Number of hidden regimes.
    regime_labels : list[str] | None
        Human-readable regime names.
    factor_names : list[str] | None
        Human-readable factor names.
    use_ensemble : bool
        If *True*, combine multivariate RSM, probit, and CFNAI signals
        for recession detection.  Results are stored in
        ``NowcastResult.recession_probability`` and
        ``NowcastResult.ensemble_detail``.
    ensemble_weights : dict[str, float] | None
        Override default ensemble weights.  Keys: ``rsm``, ``probit``,
        ``cfnai``.  Must sum to 1.
    """

    # Default ensemble weights, set from a 434-point monthly walk-forward
    # (1990-2026) covering four recessions, each point a separate refit
    # scored against NBER:
    #
    #   cfnai .50 / probit .375 / sahm .125   AUC 0.944  Brier 0.0544
    #   ...with rsm at .10                    AUC 0.931  Brier 0.0609
    #   ...with rsm at .20                    AUC 0.924  Brier 0.0736
    #   CFNAI signal alone                    AUC 0.947  Brier 0.0521
    #
    # A constant forecast at the 8.3% base rate scores Brier 0.0761.  The
    # CFNAI signal alone still edges out the full ensemble, so the factor
    # machinery earns its keep as a partly-independent second read rather
    # than as a standalone detector.
    #
    # rsm is 0.0 rather than removed: the signal is still computed and
    # published in ensemble_detail.  The regime-ordering bug behind its
    # latching is fixed, but it still reports 42.5% recession probability
    # during expansions and degrades the ensemble at any positive weight.
    DEFAULT_WEIGHTS = {
        "rsm": 0.0, "probit": 0.375, "cfnai": 0.50, "sahm": 0.125,
    }

    # Probit defaults.  ``extra_features`` are series codes used beyond
    # the DFM factors; T10Y2Y and BAA10Y enter both at t and at
    # multi-month lags because yield-curve inversion leads recessions by
    # 6-18 months.  Override via ``probit_config`` / settings.yaml.
    DEFAULT_PROBIT_CONFIG = {
        "add_lags": 3,
        "regularization": 1.0,
        "class_balanced": True,
        "extra_features": ["CFNAI", "T10Y2Y", "BAA10Y"],
    }

    # Factors the Markov-switching model is fitted on.  Hamilton (1989)
    # models the *business cycle* state, so the regime variable should be
    # the cyclical block.  Fitting on all four factors lets the mixture
    # split on inflation or financial-stress regimes instead: measured on
    # the full cached sample, all-four gives AUC 0.705 with 49% average
    # recession occupancy (it fires half the time), while restricting to
    # the cyclical factors gives AUC 0.820 at 33%.  Set to None to use
    # every factor.
    DEFAULT_RSM_FACTORS = ["real_activity", "labor_market"]

    def __init__(
        self,
        pipeline: object,
        n_factors: int = 4,
        n_regimes: int = 2,
        regime_labels: list[str] | None = None,
        factor_names: list[str] | None = None,
        use_ensemble: bool = True,
        ensemble_weights: dict[str, float] | None = None,
        probit_train_end: pd.Timestamp | None = None,
        use_filtered_factors: bool = True,
        probit_config: dict | None = None,
        rsm_factors: list[str] | None = "default",
    ) -> None:
        """
        Parameters
        ----------
        probit_train_end : pd.Timestamp | None
            Upper bound on the training window for the supervised probit
            model.  If ``None``, the probit uses every month where an
            NBER label is known (i.e. all months up to and including the
            end of the most recent NBER-dated recession in
            ``NBER_RECESSIONS``).  Set this explicitly in expanding-window
            backtests to prevent test-fold contamination.
        use_filtered_factors : bool
            When ``True`` (default), pass Kalman-filtered (not RTS-smoothed)
            factor estimates to downstream models.  Smoothed states embed
            look-ahead information from t+1..T and should not be used as
            inputs to a supervised signal evaluated against NBER truth.
        """
        self.pipeline = pipeline
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        # RSM sorts regimes by ascending mean, so index 0 = lowest
        # mean (recession) and last index = highest mean (expansion).
        if regime_labels is not None:
            self.regime_labels = regime_labels
        elif n_regimes == 2:
            self.regime_labels = ["recession", "expansion"]
        elif n_regimes == 3:
            self.regime_labels = ["recession", "slowdown", "expansion"]
        elif n_regimes == 4:
            self.regime_labels = [
                "recession", "slowdown", "recovery", "expansion",
            ]
        else:
            self.regime_labels = [f"regime_{i}" for i in range(n_regimes)]
        self.factor_names = factor_names or [
            "real_activity", "labor_market", "inflation", "financial_stress"
        ][:n_factors]
        self.use_ensemble = use_ensemble
        self.ensemble_weights = ensemble_weights or self.DEFAULT_WEIGHTS.copy()
        self.probit_train_end = probit_train_end
        self.use_filtered_factors = use_filtered_factors
        # Probit hyper-parameters, overridable from settings.yaml.  These
        # used to be hard-coded here while settings.yaml carried a second,
        # silently-ignored copy; the config is now the single source.
        self.probit_config = {**self.DEFAULT_PROBIT_CONFIG, **(probit_config or {})}
        # "default" -> the cyclical block; None -> every factor.
        self.rsm_factors = (
            self.DEFAULT_RSM_FACTORS if rsm_factors == "default" else rsm_factors
        )

        self._dfm: DynamicFactorModel | None = None
        self._rsm: RegimeSwitchingModel | None = None
        self._probit: RecessionProbit | None = None
        self._last_result: NowcastResult | None = None
        self._last_factors: pd.DataFrame | None = None
        self._last_panel: pd.DataFrame | None = None
        self._ensemble_recession_ts: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, end_date: str | None = None) -> NowcastResult:
        """Execute the full nowcast pipeline.

        Steps:
            1. Fetch & transform data via the pipeline.
            2. Fit the Dynamic Factor Model.
            3. Fit the Regime Switching Model (multivariate if ensemble).
            4. *Optionally* fit probit and CFNAI ensemble.
            5. Package results.

        Returns
        -------
        NowcastResult
        """
        # 1. Get data
        logger.info("Nowcaster: fetching data")
        panel = self.pipeline.run(end_date=end_date) if end_date else self.pipeline.run()
        panel = panel.dropna(how="all")
        if panel.empty:
            raise RuntimeError("Data pipeline returned an empty panel")
        self._last_panel = panel

        # 2. Dynamic Factor Model
        # Auto-detect quarterly columns from the pipeline's series config.
        quarterly_cols: list[str] = []
        if hasattr(self.pipeline, "_series_cfg"):
            quarterly_cols = [
                entry["code"]
                for entry in self.pipeline._series_cfg
                if entry.get("frequency") == "quarterly"
            ]
        logger.info(f"Nowcaster: fitting DFM ({self.n_factors} factors)")
        self._dfm = DynamicFactorModel(
            n_factors=self.n_factors,
            factor_names=self.factor_names,
            max_iter=100,
            use_filtered=self.use_filtered_factors,
            quarterly_columns=quarterly_cols,
        )
        self._dfm.fit(panel)
        factors = self._dfm.factors_
        self._last_factors = (
            factors if isinstance(factors, pd.DataFrame)
            else pd.DataFrame(factors)
        )

        # 3. Regime Switching Model (multivariate when ensemble)
        logger.info(f"Nowcaster: fitting RSM ({self.n_regimes} regimes)")
        self._rsm = RegimeSwitchingModel(
            n_regimes=self.n_regimes,
            regime_labels=self.regime_labels,
            use_statsmodels=False,
            multivariate=self.use_ensemble,
        )
        self._rsm.fit(self._rsm_input(self._last_factors))
        regime_info = self._rsm.get_current_regime()

        # 4. Ensemble recession detection
        recession_prob = 0.0
        ensemble_detail: dict[str, float] = {}

        if self.use_ensemble:
            recession_prob, ensemble_detail = self._run_ensemble(
                factors=self._last_factors, panel=panel
            )
            # The headline label must agree with the headline probability.
            # Previously the label came from the RSM's own argmax except
            # in two bands (P > 0.5 or P < 0.2), so a P(recession) of 0.33
            # could be reported next to the label "recession" whenever the
            # RSM disagreed with the other three signals.
            if self.n_regimes == 2:
                regime_info["regime"] = (
                    "recession" if recession_prob > 0.5 else "expansion"
                )
            elif recession_prob > 0.5:
                # For K > 2 the intermediate labels (slowdown, recovery)
                # carry information the binary ensemble cannot express, so
                # the RSM label is kept unless the ensemble calls recession.
                regime_info["regime"] = "recession"

        # 5. GDP nowcast — OLS-calibrated against actual GDPC1 growth
        last_factors = self._last_factors.iloc[-1].values.astype(float)
        intercept, betas, residual_se = self._calibrate_gdp(
            self._last_factors, panel
        )
        gdp_nowcast = float(intercept + betas @ last_factors)

        # 90 % confidence interval from OLS residual standard error
        ci_width = 1.645 * max(residual_se, 0.5)
        gdp_ci_lower = gdp_nowcast - ci_width
        gdp_ci_upper = gdp_nowcast + ci_width

        # Factor dict
        if isinstance(self._last_factors, pd.DataFrame):
            factor_dict = self._last_factors.iloc[-1].to_dict()
        else:
            factor_dict = {
                f"factor_{i}": float(v) for i, v in enumerate(last_factors)
            }

        # Use ensemble probabilities when available (not raw RSM).
        # For K=2 the ensemble fully determines the probability split.
        # For K>2 we inject the ensemble recession probability into the
        # RSM's own regime distribution, replacing the lowest-mean state.
        if self.use_ensemble and recession_prob > 0:
            if self.n_regimes == 2:
                display_probs = {
                    "recession": round(recession_prob, 4),
                    "expansion": round(1.0 - recession_prob, 4),
                }
            else:
                display_probs = dict(regime_info["probabilities"])
                display_probs["recession"] = round(recession_prob, 4)
        else:
            display_probs = regime_info["probabilities"]

        result = NowcastResult(
            gdp_nowcast=gdp_nowcast,
            gdp_ci_lower=gdp_ci_lower,
            gdp_ci_upper=gdp_ci_upper,
            regime_probabilities=display_probs,
            current_regime=regime_info["regime"],
            factor_values=factor_dict,
            recession_probability=recession_prob,
            ensemble_detail=ensemble_detail,
        )

        self._last_result = result
        logger.info(
            f"Nowcast complete: regime={result.current_regime}, "
            f"P(recession)={recession_prob:.1%}"
        )
        return result

    def get_summary(self) -> str:
        """Return a human-readable summary of the latest nowcast."""
        if self._last_result is None:
            return "No nowcast has been run yet."

        r = self._last_result
        lines = [
            f"Regime: {r.current_regime.upper()}",
            f"GDP Nowcast: {r.gdp_nowcast:.2f}% "
            f"(90% CI: [{r.gdp_ci_lower:.2f}%, {r.gdp_ci_upper:.2f}%])",
        ]

        if r.recession_probability > 0 or r.ensemble_detail:
            lines.append(f"Recession Probability: {r.recession_probability:.1%}")

        if r.ensemble_detail:
            lines.append("  Ensemble breakdown:")
            for name, val in r.ensemble_detail.items():
                lines.append(f"    {name}: {val:.1%}")

        lines.append("")
        lines.append("Regime Probabilities:")
        for label, prob in r.regime_probabilities.items():
            lines.append(f"  {label}: {prob:.1%}")

        if r.factor_values:
            lines.append("")
            lines.append("Factor Values:")
            for name, val in r.factor_values.items():
                lines.append(f"  {name}: {val:.3f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # GDP calibration
    # ------------------------------------------------------------------

    # Fallback parameters when GDPC1 is unavailable
    _GDP_TREND_DEFAULT = 2.5
    _GDP_SIGMA_DEFAULT = 1.5

    def _calibrate_gdp(
        self,
        factors: pd.DataFrame,
        panel: pd.DataFrame,
    ) -> tuple[float, np.ndarray, float]:
        """Calibrate GDP nowcast via OLS on actual GDPC1 growth.

        Fetches raw GDPC1 from the FRED client, computes annualised
        quarter-on-quarter growth, averages the monthly DFM factors to
        quarterly, and runs OLS:

            GDP_growth_q = intercept + betas @ factors_q + epsilon

        The quarter being nowcast is held out of the fit, and the returned
        standard error is the *predictive* standard error at that point
        (residual variance inflated by leverage), not the in-sample
        residual standard error.  Both changes widen the reported interval
        relative to the previous implementation, which fitted on the
        target quarter and ignored parameter uncertainty.

        Returns
        -------
        (intercept, betas, predictive_se)
            ``betas`` has shape ``(n_factors,)``.
        """
        n = factors.shape[1]
        try:
            fred_client = getattr(self.pipeline, "fred_client", None)
            if fred_client is None:
                raise RuntimeError("No FRED client on pipeline")

            # Raw real GDP (quarterly, levels, billions of chained $)
            gdpc1 = fred_client.get_series(
                "GDPC1",
                start_date=str(factors.index[0].date()),
            )

            if gdpc1 is None or len(gdpc1) < 20:
                raise RuntimeError("GDPC1 series too short")

            # Annualised Q/Q growth:  4 × 100 × Δlog(GDP)
            gdpc1_growth = 4 * 100 * np.log(gdpc1.astype(float)).diff()
            gdpc1_growth = gdpc1_growth.dropna()

            # Convert monthly factors → quarterly averages
            factors_q = factors.resample("QE").mean().dropna()

            # Align on common quarter-end dates
            common = gdpc1_growth.index.intersection(factors_q.index)
            if len(common) < 20:
                raise RuntimeError(
                    f"Only {len(common)} common quarters (need ≥20)"
                )

            y = gdpc1_growth.loc[common].values.astype(float)
            X = factors_q.loc[common].values.astype(float)

            # OLS:  y = [1 | X] β
            # Exclude the quarter being predicted: the last row of X is
            # the point the caller wants a nowcast for, and fitting on it
            # makes the "prediction" partly a fit to its own target.
            X_fit, y_fit = X[:-1], y[:-1]
            if len(y_fit) < 20:
                raise RuntimeError("Too few quarters after holding out the target")

            X_aug = np.column_stack([np.ones(len(X_fit)), X_fit])
            beta, residuals, _, _ = np.linalg.lstsq(X_aug, y_fit, rcond=None)

            intercept = float(beta[0])
            betas = beta[1:]

            # Residual standard error
            y_hat = X_aug @ beta
            resid = y_fit - y_hat
            dof = max(len(y_fit) - X_aug.shape[1], 1)
            residual_se = float(np.sqrt(np.sum(resid ** 2) / dof))

            # Predictive (not residual) standard error at the point being
            # forecast: residual_se alone omits parameter uncertainty, so
            # the reported interval was systematically too narrow.
            x0 = np.concatenate([[1.0], factors.iloc[-1].values.astype(float)])
            try:
                xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
                leverage = float(x0 @ xtx_inv @ x0)
                residual_se = residual_se * np.sqrt(max(1.0 + leverage, 1.0))
            except np.linalg.LinAlgError:
                pass

            logger.info(
                f"GDP calibration: intercept={intercept:.2f}, "
                f"betas={np.round(betas, 3).tolist()}, "
                f"residual_se={residual_se:.2f}"
            )
            return intercept, betas, residual_se

        except Exception as exc:
            logger.warning(
                f"GDP OLS calibration unavailable ({exc}); "
                f"using trend={self._GDP_TREND_DEFAULT}"
            )
            # Fallback: equal-weighted factors, moderate sensitivity
            betas = np.full(n, self._GDP_SIGMA_DEFAULT / n)
            return self._GDP_TREND_DEFAULT, betas, 1.0

    # ------------------------------------------------------------------
    # Ensemble internals
    # ------------------------------------------------------------------

    # Plausible ranges for the raw levels consumed by threshold signals.
    # These exist to catch the class of bug where a standardised or
    # differenced column is passed to a signal whose threshold is defined
    # in published units — the values then look numerically fine but mean
    # something else.  Ranges are deliberately wide: they must reject a
    # z-score, not police the economics.
    _RAW_LEVEL_RANGES = {
        "UNRATE": (0.0, 30.0),   # percentage points
        "CFNAI": (-25.0, 10.0),  # published index units
    }

    def _rsm_input(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Select the factor block the regime model is fitted on.

        Falls back to every factor when the configured names are absent
        (e.g. a caller supplied custom ``factor_names``), so this never
        silently reduces the model to nothing.
        """
        if not self.rsm_factors:
            return factors
        available = [c for c in self.rsm_factors if c in factors.columns]
        if not available:
            logger.warning(
                f"None of rsm_factors={self.rsm_factors} are present in "
                f"{list(factors.columns)}; fitting the RSM on all factors."
            )
            return factors
        if len(available) < len(self.rsm_factors):
            logger.info(
                f"RSM using {available} (requested {self.rsm_factors})"
            )
        return factors[available]

    def _raw_level(
        self, code: str, idx: pd.DatetimeIndex
    ) -> pd.Series | None:
        """Return an untransformed level series, or ``None`` if absent.

        Reads ``pipeline.raw_levels_`` (populated by
        :meth:`DataPipeline.run`) rather than the modelling panel, whose
        columns are transformed and expanding-standardised.  Raises
        ``ValueError`` if the values fall outside the plausible range for
        published units, which means the wrong frame was wired in.
        """
        raw_levels = getattr(self.pipeline, "raw_levels_", None)
        if raw_levels is None or code not in getattr(raw_levels, "columns", []):
            return None

        series = raw_levels[code].reindex(idx)
        observed = series.dropna()
        if observed.empty:
            return None

        lo, hi = self._RAW_LEVEL_RANGES.get(code, (-np.inf, np.inf))
        if observed.min() < lo or observed.max() > hi:
            raise ValueError(
                f"{code} does not look like a raw level: range "
                f"[{observed.min():.3f}, {observed.max():.3f}] falls outside "
                f"the expected [{lo}, {hi}].  A transformed or standardised "
                f"series was probably passed where published units are required."
            )
        return series

    def _run_ensemble(
        self,
        factors: pd.DataFrame,
        panel: pd.DataFrame,
    ) -> tuple[float, dict[str, float]]:
        """Compute ensemble recession probability from three signals.

        Also builds a full time-series of ensemble recession
        probabilities stored in ``self._ensemble_recession_ts``.

        Returns
        -------
        (recession_probability, detail_dict)
        """
        w = self.ensemble_weights
        idx = factors.index

        # --- Signal 1: RSM recession probability (full time series) ---
        # Align by index only; do NOT forward/back-fill, which would
        # carry stale values into otherwise-missing months at the edge.
        rsm_ts = pd.Series(0.5, index=idx, dtype=float)
        try:
            rsm_probs = self._rsm.get_recession_probability()
            if isinstance(rsm_probs, pd.Series):
                rsm_ts = rsm_probs.reindex(idx).fillna(0.5)
            else:
                rsm_ts = pd.Series(rsm_probs, index=idx, dtype=float).fillna(0.5)
        except Exception:
            pass
        p_rsm = float(rsm_ts.iloc[-1])
        logger.debug(f"Ensemble RSM: P(recession) = {p_rsm:.3f}")

        # --- Signal 2: Probit model (full time series) ---
        probit_ts = pd.Series(0.5, index=idx, dtype=float)
        p_probit = 0.5
        try:
            probit_features = self._build_probit_features(factors, panel)
            nber = get_nber_recession_indicator(
                start=str(panel.index[0].date()),
                end=str(panel.index[-1].date()),
            )
            # Training cutoff: either the user-supplied value or the end
            # of the most recent NBER-dated recession (i.e. the last
            # month where a label is reliably defined).  This replaces
            # the former hard-coded ``2020-04-30`` sentinel.
            if self.probit_train_end is not None:
                train_cutoff = pd.Timestamp(self.probit_train_end)
            else:
                train_cutoff = _default_nber_cutoff()

            common = probit_features.index.intersection(nber.index)
            train_dates = common[common <= train_cutoff]
            if len(train_dates) >= 30:
                X_train = probit_features.loc[train_dates]
                y_train = nber.loc[train_dates]
                self._probit = RecessionProbit(
                    add_lags=self.probit_config["add_lags"],
                    regularization=self.probit_config["regularization"],
                    class_balanced=self.probit_config["class_balanced"],
                )
                self._probit.fit(X_train, y_train)

                # Predict for all dates.  predict_proba returns 0.5 for
                # rows containing any NaN, so we preserve the NaN
                # structure of probit_features rather than forward-filling.
                all_proba = self._probit.predict_proba(probit_features)
                probit_ts = pd.Series(
                    all_proba, index=probit_features.index, dtype=float,
                ).reindex(idx).fillna(0.5)
                p_probit = float(probit_ts.iloc[-1])
            else:
                logger.warning("Probit: too few training samples, using 0.5")
        except Exception as exc:
            logger.warning(f"Probit model failed: {exc}")
        logger.debug(f"Ensemble probit: P(recession) = {p_probit:.3f}")

        # --- Signal 3: CFNAI threshold (full time series) ---
        # Chicago Fed's published convention: the 3-month MA of CFNAI
        # (``CFNAI-MA3``) crossing below ``-0.7`` signals the start of a
        # recession, and back above ``+0.2`` signals the end.  We use a
        # logistic centred at -0.7 with a 0.3σ scale to produce a smooth
        # probability rather than a 0/1 threshold.  This matches
        # https://www.chicagofed.org/publications/cfnai/index
        cfnai_ts = pd.Series(0.5, index=idx, dtype=float)
        p_cfnai = 0.5
        try:
            # The -0.7 convention is defined on the *published* CFNAI, so
            # this must read the untransformed level, not the panel's
            # expanding z-score of it.
            cfnai_raw = self._raw_level("CFNAI", idx)
            if cfnai_raw is not None:
                cfnai_ma3 = cfnai_raw.rolling(window=3, min_periods=2).mean()
                # Logistic: P(rec) = 1 / (1 + exp((MA3 + 0.7) / 0.3))
                cfnai_ts = 1.0 / (1.0 + np.exp((cfnai_ma3 + 0.7) / 0.3))
                cfnai_ts = cfnai_ts.fillna(0.5)
                p_cfnai = float(cfnai_ts.iloc[-1])
            else:
                logger.debug("CFNAI raw level unavailable, using 0.5")
        except Exception:
            pass
        logger.debug(f"Ensemble CFNAI: P(recession) = {p_cfnai:.3f}")

        # --- Signal 4: Sahm rule (full time series) ---
        sahm_ts = pd.Series(0.5, index=idx, dtype=float)
        p_sahm = 0.5
        try:
            # The 0.50pp threshold is in percentage points of the
            # unemployment *level*.  The panel holds the expanding
            # z-score of its first difference, which is a different
            # quantity entirely — read the raw level instead.
            unrate_raw = self._raw_level("UNRATE", idx)
            if unrate_raw is not None:
                sahm_ts = sahm_recession_probability(unrate_raw).fillna(0.5)
                p_sahm = float(sahm_ts.iloc[-1])
            else:
                logger.debug("UNRATE raw level unavailable, Sahm using 0.5")
        except Exception:
            pass
        logger.debug(f"Ensemble Sahm: P(recession) = {p_sahm:.3f}")

        # --- Weighted ensemble (full time series) ---
        ensemble_ts = (
            w.get("rsm", 0.20) * rsm_ts
            + w.get("probit", 0.40) * probit_ts
            + w.get("cfnai", 0.20) * cfnai_ts
            + w.get("sahm", 0.20) * sahm_ts
        ).clip(0.0, 1.0)
        self._ensemble_recession_ts = ensemble_ts

        p_ensemble = float(ensemble_ts.iloc[-1])

        detail = {
            "rsm": p_rsm,
            "probit": p_probit,
            "cfnai": p_cfnai,
            "sahm": p_sahm,
            "ensemble": p_ensemble,
        }
        return p_ensemble, detail

    def get_ensemble_probabilities(self) -> pd.DataFrame:
        """Return full time-series of ensemble regime probabilities.

        Returns a two-column DataFrame (``recession``, ``expansion``)
        built from the weighted ensemble of RSM + Probit + CFNAI.
        Falls back to raw RSM probabilities when the ensemble was not
        computed (``use_ensemble=False``).
        """
        if self._ensemble_recession_ts is not None:
            rec = self._ensemble_recession_ts
            return pd.DataFrame(
                {"recession": rec, "expansion": 1.0 - rec},
                index=rec.index,
            )
        # Fallback: raw RSM probabilities
        return self._rsm.get_regime_probabilities()

    def _build_probit_features(
        self,
        factors: pd.DataFrame,
        panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build feature matrix for the probit model.

        Combines DFM factors with CFNAI, the yield-curve spread, and
        the credit spread from the transformed panel, plus 3-month and
        6-month lagged versions of the leading indicators.

        Does **not** forward-fill: if a feature is missing at the panel
        edge because its release has not yet occurred (see
        ``ragged_edge_mask``) then ``predict_proba`` will fall back to
        its conservative 0.5 instead of consuming a stale value.
        """
        parts = [factors]
        leading_codes = ["T10Y2Y", "BAA10Y"]

        for code in self.probit_config["extra_features"]:
            if code in panel.columns:
                extra = panel[code].reindex(factors.index)
                extra.name = code
                parts.append(extra)

        # Add 3-month and 6-month lags of leading indicators.  The lag
        # shift inherently consumes NaN values from ragged-edge inputs,
        # so no ffill is needed or wanted.
        for code in leading_codes:
            if code in panel.columns:
                series = panel[code].reindex(factors.index)
                for lag_months in [3, 6]:
                    lagged = series.shift(lag_months)
                    lagged.name = f"{code}_lag{lag_months}"
                    parts.append(lagged)

        result = pd.concat(parts, axis=1)
        return result
