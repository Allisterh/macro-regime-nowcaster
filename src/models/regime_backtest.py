"""NBER-based regime backtest and validation framework.

Compares model-detected regimes against NBER recession/expansion dates
to produce accuracy metrics, confusion matrices, and historical regime
timelines.  Supports expanding-window evaluation to avoid look-ahead bias.

Usage
-----
>>> from src.models.regime_backtest import RegimeBacktester
>>> bt = RegimeBacktester(pipeline, fred_client)
>>> report = bt.run(start="2000-01-01", end="2024-12-31")
>>> print(report.summary())
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# NBER reference data
# ---------------------------------------------------------------------------

# Official NBER turning-point dates (peaks & troughs through 2024).
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
# Format: (peak, trough) — recession spans from peak to trough (inclusive).
NBER_RECESSIONS: list[tuple[str, str]] = [
    # Pre-1980 turning points, needed once the panel reaches back to the
    # 1950s via the reconstructed spreads.  These take the sample from
    # four recessions to ten, which is the binding constraint on every
    # confidence interval this project reports.
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# When the NBER's Business Cycle Dating Committee actually *announced*
# each turning point.  The committee waits for data revisions, so a
# turning point becomes public 5-21 months after the month it dates.
# Training on a label before its announcement is look-ahead even though
# the label's own timestamp is in the past.
# Source: https://www.nber.org/research/business-cycle-dating
NBER_ANNOUNCEMENTS: dict[str, str] = {
    # The Business Cycle Dating Committee was only formed in 1978, so
    # pre-1979 turning points were dated retrospectively and have no
    # contemporaneous announcement.  They fall through to
    # _DEFAULT_ANNOUNCE_LAG, which is the honest treatment: a forecaster
    # in 1970 did not have an official label either.
    "1980-01-01": "1980-06-03",
    "1980-07-01": "1981-07-08",
    "1981-07-01": "1982-01-06",
    "1982-11-01": "1983-07-08",
    "1990-07-01": "1991-04-25",
    "1991-03-01": "1992-12-22",
    "2001-03-01": "2001-11-26",
    "2001-11-01": "2003-07-17",
    "2007-12-01": "2008-12-01",
    "2009-06-01": "2010-09-20",
    "2020-02-01": "2020-06-08",
    "2020-04-01": "2021-07-19",
}

# Fallback when a turning point has no recorded announcement date.
_DEFAULT_ANNOUNCE_LAG = pd.DateOffset(months=12)

# Natural cut-off for turning a probability into a 0/1 call.  Fixed on
# purpose: choosing it by sweeping against the evaluation labels inflates
# every metric derived from it.
_DEFAULT_THRESHOLD = 0.5


def nber_label_knowable_at(turning_point: str | pd.Timestamp) -> pd.Timestamp:
    """Date on which an NBER turning point became publicly known."""
    key = pd.Timestamp(turning_point).strftime("%Y-%m-%d")
    announced = NBER_ANNOUNCEMENTS.get(key)
    if announced is not None:
        return pd.Timestamp(announced)
    return pd.Timestamp(turning_point) + _DEFAULT_ANNOUNCE_LAG


def nber_labels_available_at(
    as_of: str | pd.Timestamp,
    start: str = "1980-01-01",
) -> pd.Series:
    """NBER recession indicator restricted to labels announced by *as_of*.

    Months inside a recession whose peak had not yet been announced are
    reported as 0 — that is what an observer at *as_of* would have
    believed — and months after the last announced turning point are
    dropped entirely, since no defensible label existed for them.

    Use this rather than :func:`get_nber_recession_indicator` whenever
    labels feed a model evaluated as if it had run at *as_of*.
    """
    as_of = pd.Timestamp(as_of)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    last_known: pd.Timestamp | None = None

    for peak, trough in NBER_RECESSIONS:
        if nber_label_knowable_at(peak) > as_of:
            continue
        trough_known = nber_label_knowable_at(trough) <= as_of
        # Span end follows the same convention as
        # get_nber_recession_indicator: troughs are first-of-month
        # timestamps compared against a month-end index, so the trough
        # month itself is excluded.  Matching it here keeps this function
        # a pure announcement-lag filter rather than also shifting labels.
        # A peak announced without its trough means "in recession, end not
        # yet declared" — label through as_of and no further.
        end = pd.Timestamp(trough) if trough_known else as_of
        spans.append((pd.Timestamp(peak), end))
        # The *available* range may extend to the trough month-end even
        # though that month is labelled 0.
        known_to = (
            pd.Timestamp(trough) + pd.offsets.MonthEnd(0)
            if trough_known else as_of
        )
        last_known = known_to if last_known is None else max(last_known, known_to)

    idx = pd.date_range(start, as_of, freq="ME")
    rec = pd.Series(0, index=idx, name="nber_recession", dtype=int)
    for peak_dt, end_dt in spans:
        rec.loc[(rec.index >= peak_dt) & (rec.index <= end_dt)] = 1

    if last_known is not None:
        rec = rec.loc[rec.index <= last_known]
    return rec


def get_nber_recession_indicator(
    start: str = "1980-01-01",
    end: str | None = None,
    freq: str = "ME",
) -> pd.Series:
    """Build a monthly binary indicator: 1 = NBER recession, 0 = expansion.

    This uses the hard-coded turning-point table above so the backtest
    works offline without needing a FRED API call for USREC.

    Parameters
    ----------
    start, end : str
        Date range (ISO-8601).
    freq : str
        Pandas frequency string (default ``"ME"`` = month-end).

    Returns
    -------
    pd.Series
        Binary indicator indexed by date.
    """
    if end is None:
        end = str(dt.date.today())
    idx = pd.date_range(start, end, freq=freq)
    rec = pd.Series(0, index=idx, name="nber_recession", dtype=int)

    for peak, trough in NBER_RECESSIONS:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        rec.loc[(rec.index >= peak_dt) & (rec.index <= trough_dt)] = 1

    return rec


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed from rank statistics.

    Threshold-free, so unlike accuracy/F1 it cannot be inflated by
    choosing a cut-off after seeing the labels.  Returns ``nan`` when
    either class is absent.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average ranks within ties so tied scores don't bias the statistic.
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error of a probabilistic forecast (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    if not ok.any():
        return float("nan")
    return float(np.mean((p[ok] - y[ok]) ** 2))


def get_nber_from_fred(fred_client, start: str = "1980-01-01") -> pd.Series:
    """Fetch USREC (NBER recession indicator) from FRED.

    Falls back to the hard-coded table if the fetch fails.
    """
    try:
        usrec = fred_client.get_series("USREC", start_date=start)
        usrec = usrec.resample("ME").last().ffill()
        usrec.name = "nber_recession"
        return usrec.astype(int)
    except Exception as exc:
        logger.warning(f"Could not fetch USREC from FRED ({exc}), using hard-coded dates")
        return get_nber_recession_indicator(start=start)


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------


@dataclass
class RegimeBacktestResult:
    """Container for regime backtest metrics.

    Attributes
    ----------
    accuracy : float
        Overall classification accuracy (fraction correct).
    precision_recession : float
        Precision for recession detection.
    recall_recession : float
        Recall (sensitivity) for recession detection.
    f1_recession : float
        F1 score for recession detection.
    confusion : np.ndarray
        2×2 confusion matrix [[TN, FP], [FN, TP]].
    regime_history : pd.DataFrame
        Full history: model regime probs, predicted label, NBER label.
    n_months : int
        Number of months evaluated.
    detection_lag_months : float
        Average # months after recession start before model detects it.
    false_alarm_rate : float
        Fraction of expansion months mis-classified as recession.
    """

    accuracy: float
    precision_recession: float
    recall_recession: float
    f1_recession: float
    confusion: np.ndarray
    regime_history: pd.DataFrame
    n_months: int
    detection_lag_months: float = np.nan
    false_alarm_rate: float = 0.0
    auc: float = np.nan
    brier: float = np.nan
    threshold: float = 0.5
    threshold_source: str = "fixed"
    in_sample: bool = False

    def summary(self) -> str:
        """Human-readable performance summary."""
        warn_lines = []
        if self.in_sample:
            warn_lines = [
                "  !! IN-SAMPLE: the model saw these labels during fitting.",
                "  !! These numbers overstate real-time accuracy.",
                "",
            ]
        thr_note = (
            "" if self.threshold_source == "fixed"
            else f"  (tuned on {self.threshold_source} — not a live number)"
        )
        lines = [
            "=" * 50,
            "  REGIME BACKTEST — NBER VALIDATION",
            "=" * 50,
            *warn_lines,
            f"  Months evaluated     : {self.n_months}",
            "",
            "  --- Threshold-free (cannot be tuned on labels) ---",
            f"  ROC AUC              : {self.auc:.3f}",
            f"  Brier score          : {self.brier:.4f}",
            "",
            f"  --- At threshold {self.threshold:.2f} ---{thr_note}",
            f"  Overall accuracy     : {self.accuracy:.1%}",
            f"  Precision            : {self.precision_recession:.1%}",
            f"  Recall (sensitivity) : {self.recall_recession:.1%}",
            f"  F1 score             : {self.f1_recession:.3f}",
            f"  Avg detection lag    : {self.detection_lag_months:.1f} months",
            f"  False alarm rate     : {self.false_alarm_rate:.1%}",
            "",
            "  --- Confusion Matrix ---",
            "         Pred Exp  Pred Rec",
            f"  NBER Exp  {int(self.confusion[0, 0]):>6d}  {int(self.confusion[0, 1]):>6d}",
            f"  NBER Rec  {int(self.confusion[1, 0]):>6d}  {int(self.confusion[1, 1]):>6d}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RegimeBacktester
# ---------------------------------------------------------------------------


class RegimeBacktester:
    """Backtest the regime-classification model against NBER dates.

    The backtester supports two modes:

    1. **Full-sample** (default): Fit the model once on the full data
       range and compare the smoothed regime probabilities to NBER dates.
       Fast but includes look-ahead from the smoother.

    2. **Expanding-window**: For each evaluation month *t*, fit the model
       using only data up to *t* and record the **filtered** (real-time)
       regime probability.  No look-ahead bias.

    Parameters
    ----------
    pipeline : DataPipeline | None
        A configured DataPipeline (for live-data mode).
    n_factors : int
        Number of DFM factors.
    n_regimes : int
        Number of regimes.  The model maps its regimes to a binary
        recession/expansion label via the ``recession_labels`` parameter.
    factor_names : list[str] | None
        Names for the factors.
    regime_labels : list[str] | None
        Labels for the regimes.
    recession_labels : list[str] | None
        Which of the ``regime_labels`` correspond to "recession".
        Defaults to ``["recession"]``.
    """

    def __init__(
        self,
        pipeline=None,
        n_factors: int = 4,
        n_regimes: int = 4,
        factor_names: list[str] | None = None,
        regime_labels: list[str] | None = None,
        recession_labels: list[str] | None = None,
        use_ensemble: bool = False,
        ensemble_weights: dict[str, float] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.factor_names = factor_names
        self.regime_labels = regime_labels or [
            "expansion", "slowdown", "recession", "recovery"
        ]
        self.recession_labels = recession_labels or ["recession"]
        self.use_ensemble = use_ensemble
        self.ensemble_weights = ensemble_weights or {
            "rsm": 0.20, "probit": 0.40, "cfnai": 0.20, "sahm": 0.20,
        }

    # ------------------------------------------------------------------
    # Full-sample backtest
    # ------------------------------------------------------------------

    def run(
        self,
        start: str = "1990-01-01",
        end: str | None = None,
        panel: pd.DataFrame | None = None,
        nber: pd.Series | None = None,
        in_sample: bool = False,
    ) -> RegimeBacktestResult:
        """Run a regime backtest.

        By default this dispatches to :meth:`run_expanding` so the
        reported metrics reflect real-time (no look-ahead) performance.
        Set ``in_sample=True`` to keep the legacy full-sample behaviour,
        which fits the DFM/RSM on the whole evaluation window and reads
        back smoothed states — useful for smoke tests and offline
        analysis but **not** a valid measure of live accuracy.

        Parameters
        ----------
        start, end : str
            Evaluation window (ISO-8601).
        panel : pd.DataFrame | None
            Pre-built panel (if ``None``, fetches via ``self.pipeline``).
        nber : pd.Series | None
            NBER binary indicator.  If ``None``, uses the hard-coded table.
        in_sample : bool
            If ``True``, use the look-ahead full-sample path.  Emits a
            ``UserWarning`` because the resulting metrics overstate
            real-time accuracy.

        Returns
        -------
        RegimeBacktestResult
        """
        if not in_sample:
            return self.run_expanding(
                start_eval=start, end=end, panel=panel, nber=nber,
            )
        import warnings
        warnings.warn(
            "RegimeBacktester.run(in_sample=True) uses smoothed, "
            "full-sample-fitted factors and therefore reports "
            "look-ahead-inflated metrics.  Use run_expanding() for "
            "real-time evaluation.",
            UserWarning,
            stacklevel=2,
        )
        from src.models.dynamic_factor_model import DynamicFactorModel
        from src.models.regime_switching import RegimeSwitchingModel

        if end is None:
            end = str(dt.date.today())

        # -- Get data ---
        if panel is None:
            if self.pipeline is None:
                raise ValueError("Either `panel` or `pipeline` must be provided.")
            panel = self.pipeline.run(end_date=end)
            panel = panel.dropna(how="all")

        # -- NBER ground truth ---
        if nber is None:
            nber = get_nber_recession_indicator(start=start, end=end)

        # -- Fit models ---
        logger.info(f"Backtest: fitting DFM ({self.n_factors} factors) on {panel.shape}")
        dfm = DynamicFactorModel(
            n_factors=self.n_factors,
            factor_names=self.factor_names,
            max_iter=100,
        )
        dfm.fit(panel)
        factors = dfm.factors_
        if not isinstance(factors, pd.DataFrame):
            factors = pd.DataFrame(factors)

        logger.info(f"Backtest: fitting RSM ({self.n_regimes} regimes)")
        rsm = RegimeSwitchingModel(
            n_regimes=self.n_regimes,
            regime_labels=self.regime_labels,
            use_statsmodels=False,
            multivariate=self.use_ensemble,
        )
        rsm.fit(factors)
        regime_probs = rsm.get_regime_probabilities()
        if not isinstance(regime_probs, pd.DataFrame):
            regime_probs = pd.DataFrame(
                regime_probs,
                columns=self.regime_labels[:self.n_regimes],
            )

        if self.use_ensemble:
            return self._evaluate_ensemble(
                rsm, factors, panel, nber, start, end,
            )

        # -- Build classification ---
        return self._evaluate(regime_probs, nber, start, end)

    # ------------------------------------------------------------------
    # Expanding-window backtest (no look-ahead)
    # ------------------------------------------------------------------

    def run_expanding(
        self,
        start_eval: str = "2000-01-01",
        end: str | None = None,
        panel: pd.DataFrame | None = None,
        nber: pd.Series | None = None,
        min_train_months: int = 120,
        step_months: int = 3,
    ) -> RegimeBacktestResult:
        """Run an expanding-window backtest.

        For every ``step_months`` period from ``start_eval`` to ``end``,
        fit the model on data up to that date and record the **last**
        regime probability (mimicking real-time usage).

        Parameters
        ----------
        start_eval : str
            First evaluation date.
        end : str
            Last evaluation date.
        panel : pd.DataFrame | None
            Full data panel.
        nber : pd.Series | None
            NBER binary indicator.
        min_train_months : int
            Minimum training sample length.
        step_months : int
            Step size (months) between evaluation points.

        Returns
        -------
        RegimeBacktestResult
        """
        from src.models.dynamic_factor_model import DynamicFactorModel
        from src.models.regime_switching import RegimeSwitchingModel

        if end is None:
            end = str(dt.date.today())

        if panel is None:
            if self.pipeline is None:
                raise ValueError("Either `panel` or `pipeline` must be provided.")
            panel = self.pipeline.run(end_date=end)
            panel = panel.dropna(how="all")

        if nber is None:
            nber = get_nber_recession_indicator(
                start=str(panel.index[0].date()), end=end
            )

        eval_start = pd.Timestamp(start_eval)
        eval_dates = pd.date_range(eval_start, end, freq=f"{step_months}ME")

        # Evaluating the ensemble means refitting the whole nowcaster per
        # step, so route through the shared walk-forward generator rather
        # than maintaining a second copy of the ensemble definition.
        if self.use_ensemble and self.pipeline is not None:
            return self._run_expanding_ensemble(
                eval_dates, nber, start_eval, end, min_train_months,
            )

        results_rows = []
        for eval_dt in eval_dates:
            train = panel.loc[:eval_dt]
            if len(train) < min_train_months:
                continue

            try:
                dfm = DynamicFactorModel(
                    n_factors=self.n_factors,
                    factor_names=self.factor_names,
                    max_iter=60,
                )
                dfm.fit(train)
                factors = dfm.factors_
                if not isinstance(factors, pd.DataFrame):
                    factors = pd.DataFrame(factors)

                rsm = RegimeSwitchingModel(
                    n_regimes=self.n_regimes,
                    regime_labels=self.regime_labels,
                    use_statsmodels=False,
                )
                rsm.fit(factors)
                # Only the final row is real-time safe: there the Kim
                # smoother coincides with the Hamilton filter, so no
                # future observation can reach it.
                info = rsm.get_current_regime()

                row = {"date": eval_dt}
                row.update(info["probabilities"])
                row["predicted_regime"] = info["regime"]
                results_rows.append(row)
            except Exception as exc:
                logger.warning(f"Expanding window failed at {eval_dt}: {exc}")
                continue

        if not results_rows:
            raise RuntimeError("No successful evaluation windows")

        regime_probs = pd.DataFrame(results_rows).set_index("date")
        prob_cols = [c for c in regime_probs.columns if c != "predicted_regime"]
        prob_df = regime_probs[prob_cols]
        prob_df.index = pd.DatetimeIndex(prob_df.index)

        return self._evaluate(prob_df, nber, start_eval, end)

    def _run_expanding_ensemble(
        self,
        eval_dates: pd.DatetimeIndex,
        nber: pd.Series,
        start: str,
        end: str,
        min_train_months: int,
    ) -> RegimeBacktestResult:
        """Expanding-window evaluation of the full four-signal ensemble.

        Each step rebuilds the panel as of that date (so the ragged edge
        is masked correctly), refits every component, and keeps only the
        final row.  The probit sees only NBER labels announced by then.
        """
        from src.models.walk_forward import generate_features_asof

        rows: list[dict] = []
        for i, eval_dt in enumerate(eval_dates, start=1):
            logger.info(
                f"Expanding ensemble [{i}/{len(eval_dates)}]: {eval_dt.date()}"
            )
            try:
                feats = generate_features_asof(
                    self.pipeline,
                    eval_dt,
                    n_factors=self.n_factors,
                    n_regimes=self.n_regimes,
                    factor_names=self.factor_names,
                    regime_labels=self.regime_labels,
                    ensemble_weights=self.ensemble_weights,
                )
            except Exception as exc:
                logger.warning(f"Expanding window failed at {eval_dt}: {exc}")
                continue
            rows.append({
                "date": eval_dt,
                "model_recession_prob": feats["p_recession"],
                "signal_rsm": feats.get("signal_rsm", np.nan),
                "signal_probit": feats.get("signal_probit", np.nan),
                "signal_cfnai": feats.get("signal_cfnai", np.nan),
                "signal_sahm": feats.get("signal_sahm", np.nan),
            })

        if not rows:
            raise RuntimeError("No successful evaluation windows")

        oos = pd.DataFrame(rows).set_index("date")
        oos.index = pd.DatetimeIndex(oos.index)

        common = oos.index.intersection(nber.index)
        if len(common) < 6:
            raise ValueError(f"Too few common dates: {len(common)}")

        y_true = nber.loc[common].astype(int)
        prob = oos.loc[common, "model_recession_prob"]
        pred = (prob > _DEFAULT_THRESHOLD).astype(int)

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        n = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        history = oos.loc[common].copy()
        history["nber_recession"] = y_true
        history["model_pred"] = pred

        return RegimeBacktestResult(
            accuracy=(tp + tn) / n if n else 0.0,
            precision_recession=precision,
            recall_recession=recall,
            f1_recession=f1,
            confusion=np.array([[tn, fp], [fn, tp]]),
            regime_history=history,
            n_months=n,
            detection_lag_months=self._compute_detection_lag(
                pred, y_true, common
            ),
            false_alarm_rate=fp / (fp + tn) if (fp + tn) else 0.0,
            auc=roc_auc(y_true.values, prob.values),
            brier=brier_score(y_true.values, prob.values),
            threshold=_DEFAULT_THRESHOLD,
            threshold_source="fixed",
            in_sample=False,
        )

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        regime_probs: pd.DataFrame,
        nber: pd.Series,
        start: str,
        end: str,
    ) -> RegimeBacktestResult:
        """Compare model regime probabilities to NBER classification."""
        # Align dates
        common = regime_probs.index.intersection(nber.index)
        if len(common) == 0:
            # Try to align by resampling regime_probs; ffill within
            # the target index is OK here because these are modelled
            # probabilities persisting between evaluation points in a
            # historical series (not raw data at the nowcast edge).
            regime_probs = regime_probs.reindex(nber.index).ffill()
            common = regime_probs.index.intersection(nber.index)

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (common >= start_dt) & (common <= end_dt)
        common = common[mask]

        if len(common) < 6:
            raise ValueError(f"Too few common dates for evaluation: {len(common)}")

        probs = regime_probs.loc[common]
        nber_aligned = nber.loc[common]

        # Map model regimes to binary recession probability.
        # The RSM sorts regimes by ascending mean, so the first regime is
        # always the most "recessionary" (lowest composite).  We pick the
        # column(s) whose probabilities best correlate with NBER recessions.
        recession_cols = [
            c for c in probs.columns
            if c.lower() in [r.lower() for r in self.recession_labels]
        ]
        if recession_cols:
            model_recession_prob = probs[recession_cols].sum(axis=1)
        else:
            # Fallback: use first column (lowest-mean regime)
            model_recession_prob = probs.iloc[:, 0]

        # NOTE: an auto-flip used to sit here, inverting the probability
        # whenever it correlated negatively with NBER — i.e. reading the
        # evaluation labels to choose a sign.  It also masked a real bug:
        # the DFM's factor sign was unidentified, so the regime sort was
        # effectively a coin flip.  Orientation is now fixed upstream by
        # the DFM's loading-sum convention and the RSM's ascending-mean
        # sort, so no correction belongs here.
        nber_true = nber_aligned.astype(int)

        # Fixed threshold, for the same reason: sweeping for the
        # F1-maximising cut-off and then reporting metrics at it selects a
        # hyper-parameter on the test labels.  AUC and Brier below are
        # threshold-free.
        model_pred = (model_recession_prob > _DEFAULT_THRESHOLD).astype(int)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        tp = int(((model_pred == 1) & (nber_true == 1)).sum())
        fp = int(((model_pred == 1) & (nber_true == 0)).sum())
        fn = int(((model_pred == 0) & (nber_true == 1)).sum())
        tn = int(((model_pred == 0) & (nber_true == 0)).sum())

        confusion = np.array([[tn, fp], [fn, tp]])
        n = tp + fp + fn + tn

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Detection lag: for each recession episode, how many months before
        # the model flags it?
        detection_lag = self._compute_detection_lag(
            model_pred, nber_true, common
        )

        # Build history DataFrame
        history = pd.DataFrame({
            "nber_recession": nber_true,
            "model_recession_prob": model_recession_prob,
            "model_pred": model_pred,
        }, index=common)
        for col in probs.columns:
            history[f"prob_{col}"] = probs[col].values

        return RegimeBacktestResult(
            accuracy=accuracy,
            precision_recession=precision,
            recall_recession=recall,
            f1_recession=f1,
            confusion=confusion,
            regime_history=history,
            n_months=n,
            detection_lag_months=detection_lag,
            false_alarm_rate=false_alarm,
            auc=roc_auc(nber_true.values, model_recession_prob.values),
            brier=brier_score(nber_true.values, model_recession_prob.values),
            threshold=_DEFAULT_THRESHOLD,
            threshold_source="fixed",
        )

    @staticmethod
    def _compute_detection_lag(
        model_pred: pd.Series,
        nber_true: pd.Series,
        dates: pd.DatetimeIndex,
    ) -> float:
        """Compute average months from recession start to model detection."""
        # Find recession start dates
        nber_shifted = nber_true.shift(1, fill_value=0)
        starts = dates[(nber_true == 1) & (nber_shifted == 0)]

        if len(starts) == 0:
            return np.nan

        lags = []
        for start_date in starts:
            # Find when model first detects recession after this start
            future = model_pred.loc[start_date:]
            detected = future[future == 1]
            if len(detected) > 0:
                first_detect = detected.index[0]
                lag_months = (
                    (first_detect.year - start_date.year) * 12
                    + first_detect.month - start_date.month
                )
                lags.append(lag_months)
            else:
                # Never detected — use the recession duration as penalty
                rec_end = nber_true.loc[start_date:]
                rec_end = rec_end[rec_end == 0]
                if len(rec_end) > 0:
                    duration = (
                        (rec_end.index[0].year - start_date.year) * 12
                        + rec_end.index[0].month - start_date.month
                    )
                    lags.append(duration)

        return float(np.mean(lags)) if lags else np.nan

    # ------------------------------------------------------------------
    # Ensemble evaluation
    # ------------------------------------------------------------------

    def _evaluate_ensemble(
        self,
        rsm,
        factors: pd.DataFrame,
        panel: pd.DataFrame,
        nber: pd.Series,
        start: str,
        end: str,
    ) -> RegimeBacktestResult:
        """Evaluate using the four-signal ensemble.

        Combines: (1) multivariate RSM, (2) probit on factors+indicators,
        (3) CFNAI threshold, (4) Sahm rule.
        """
        from src.models.recession_probit import RecessionProbit
        from src.models.sahm_rule import sahm_recession_probability

        w = self.ensemble_weights

        # --- Align dates ---
        common = factors.index.intersection(nber.index)
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
        mask = (common >= start_dt) & (common <= end_dt)
        common = common[mask]
        if len(common) < 6:
            raise ValueError(f"Too few common dates: {len(common)}")

        nber_aligned = nber.loc[common].astype(int)

        # --- Signal 1: RSM recession probability ---
        rsm_rec = rsm.get_recession_probability()
        if isinstance(rsm_rec, pd.Series):
            rsm_rec = rsm_rec.reindex(common).fillna(0.5)
        else:
            rsm_rec = pd.Series(rsm_rec, index=factors.index).reindex(
                common
            ).fillna(0.5)

        # NOTE: this used to auto-flip rsm_rec when it correlated
        # negatively with NBER — i.e. it read the test labels to pick the
        # sign of a signal.  Orientation is now guaranteed upstream: the
        # RSM sorts regimes by ascending mean, column 0 is the recession
        # state, and RegimeSwitchingModel rejects a contradictory
        # regime_labels ordering at construction.

        # --- Signal 2: Probit ---
        # Build features WITHOUT forward-filling, so ragged-edge NaN
        # propagates into the probit's conservative 0.5 fallback rather
        # than consuming stale last-known values.
        probit_features = factors.copy()
        leading_codes = ["T10Y2Y", "BAA10Y"]
        for code in ["CFNAI", "T10Y2Y", "BAA10Y"]:
            if code in panel.columns:
                probit_features[code] = panel[code].reindex(factors.index)
        for code in leading_codes:
            if code in panel.columns:
                series = panel[code].reindex(factors.index)
                for lag_months in [3, 6]:
                    probit_features[f"{code}_lag{lag_months}"] = series.shift(lag_months)

        probit = RecessionProbit(
            add_lags=3, regularization=1.0, class_balanced=True,
        )
        # Train only on months inside the supplied backtest window AND
        # inside the known-NBER-labels window.  This prevents the
        # supervised component from training on data beyond the test
        # fold's start — the main source of look-ahead in the former
        # full-sample evaluation.
        train_candidates = probit_features.index.intersection(nber.index)
        train_common = train_candidates[
            (train_candidates >= start_dt) & (train_candidates <= end_dt)
        ]
        if len(train_common) >= 30:
            X_train = probit_features.loc[train_common]
            y_train = nber.loc[train_common].astype(float)
            probit.fit(X_train, y_train)
            probit_proba = pd.Series(
                probit.predict_proba(probit_features),
                index=probit_features.index,
            ).reindex(common).fillna(0.5)
        else:
            probit_proba = pd.Series(0.5, index=common)

        # --- Signals 3 & 4: CFNAI and Sahm, on RAW published levels ---
        # Both thresholds (-0.7 on CFNAI, 0.50pp on the Sahm gap) are
        # defined in published units.  `panel` holds expanding z-scores of
        # transformed series, so reading these from it compares a
        # threshold against the wrong quantity entirely.
        raw_levels = getattr(self.pipeline, "raw_levels_", None)

        def _raw(code: str) -> pd.Series | None:
            if raw_levels is None or code not in getattr(raw_levels, "columns", []):
                logger.warning(
                    f"{code} raw level unavailable; its signal falls back to "
                    f"0.5.  Build the panel via DataPipeline.run() so that "
                    f"raw_levels_ is populated."
                )
                return None
            return raw_levels[code].reindex(common)

        cfnai = _raw("CFNAI")
        if cfnai is not None:
            cfnai_ma3 = cfnai.rolling(window=3, min_periods=2).mean()
            cfnai_prob = 1.0 / (1.0 + np.exp((cfnai_ma3 + 0.7) / 0.3))
            cfnai_prob = cfnai_prob.fillna(0.5)
        else:
            cfnai_prob = pd.Series(0.5, index=common)

        unrate = _raw("UNRATE")
        if unrate is not None:
            sahm_prob = sahm_recession_probability(unrate).fillna(0.5)
        else:
            sahm_prob = pd.Series(0.5, index=common)

        # --- Ensemble ---
        ensemble_prob = (
            w.get("rsm", 0.20) * rsm_rec
            + w.get("probit", 0.40) * probit_proba
            + w.get("cfnai", 0.20) * cfnai_prob
            + w.get("sahm", 0.20) * sahm_prob
        )

        logger.info(
            f"Ensemble signal stats — "
            f"RSM mean={rsm_rec.mean():.3f}, "
            f"Probit mean={probit_proba.mean():.3f}, "
            f"CFNAI mean={cfnai_prob.mean():.3f}, "
            f"Sahm mean={sahm_prob.mean():.3f}, "
            f"Ensemble mean={ensemble_prob.mean():.3f}"
        )

        # --- Evaluate at a FIXED threshold ---
        # This previously swept thresholds and kept the F1-maximising one,
        # then reported metrics at that threshold — selecting a
        # hyper-parameter on the test labels, which inflates every headline
        # number.  A probability forecast is scored at its natural 0.5
        # cut-off; AUC and Brier below are threshold-free and are the
        # numbers to compare across model versions.
        nber_true = nber_aligned
        model_pred = (ensemble_prob > _DEFAULT_THRESHOLD).astype(int)

        tp = int(((model_pred == 1) & (nber_true == 1)).sum())
        fp = int(((model_pred == 1) & (nber_true == 0)).sum())
        fn = int(((model_pred == 0) & (nber_true == 1)).sum())
        tn = int(((model_pred == 0) & (nber_true == 0)).sum())

        confusion = np.array([[tn, fp], [fn, tp]])
        n = tp + fp + fn + tn

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        detection_lag = self._compute_detection_lag(
            model_pred, nber_true, common
        )

        history = pd.DataFrame({
            "nber_recession": nber_true,
            "model_recession_prob": ensemble_prob,
            "model_pred": model_pred,
            "signal_rsm": rsm_rec,
            "signal_probit": probit_proba,
            "signal_cfnai": cfnai_prob,
            "signal_sahm": sahm_prob,
        }, index=common)

        return RegimeBacktestResult(
            accuracy=accuracy,
            precision_recession=precision,
            recall_recession=recall,
            f1_recession=f1,
            confusion=confusion,
            regime_history=history,
            n_months=n,
            detection_lag_months=detection_lag,
            false_alarm_rate=false_alarm,
            auc=roc_auc(nber_true.values, ensemble_prob.values),
            brier=brier_score(nber_true.values, ensemble_prob.values),
            threshold=_DEFAULT_THRESHOLD,
            threshold_source="fixed",
            # This path fits on the window it scores; the caller was
            # warned, and the flag makes it visible on the result too.
            in_sample=True,
        )


# ---------------------------------------------------------------------------
# Standalone baseline: CFNAI threshold
# ---------------------------------------------------------------------------


def cfnai_baseline_backtest(
    panel: pd.DataFrame,
    nber: pd.Series | None = None,
    threshold: float = -0.7,
    start: str = "1990-01-01",
    end: str | None = None,
) -> RegimeBacktestResult:
    """Run a simple CFNAI threshold baseline for comparison.

    The Chicago Fed National Activity Index (CFNAI) is specifically
    designed as a coincident recession indicator.  CFNAI < -0.7 signals
    recession (per the Chicago Fed's own guidance).

    This acts as an upper-bound benchmark: if the DFM+RSM model can't
    beat a simple CFNAI threshold, the factor model needs more work.

    Parameters
    ----------
    panel : pd.DataFrame
        Transformed data panel (must include ``"CFNAI"`` column).
    nber : pd.Series | None
        NBER binary indicator.
    threshold : float
        CFNAI threshold for recession classification (default -0.7σ).
    start, end : str
        Evaluation window.
    """
    if end is None:
        end = str(dt.date.today())

    if nber is None:
        nber = get_nber_recession_indicator(start=start, end=end)

    if "CFNAI" not in panel.columns:
        raise ValueError("Panel must contain 'CFNAI' column for baseline")

    cfnai = panel["CFNAI"]

    # The pipeline standardises CFNAI, so threshold in σ-space
    # CFNAI < -0.7σ ≈ bottom ~25% of distribution
    common = cfnai.index.intersection(nber.index)
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    mask = (common >= start_dt) & (common <= end_dt)
    common = common[mask]

    cfnai_aligned = cfnai.loc[common]
    nber_aligned = nber.loc[common]

    # Optimise threshold like the main backtest
    nber_true = nber_aligned.astype(int)
    best_f1, best_thr = 0.0, threshold
    for thr_candidate in np.arange(-2.0, 0.5, 0.1):
        pred = (cfnai_aligned < thr_candidate).astype(int)
        tp_ = int(((pred == 1) & (nber_true == 1)).sum())
        fp_ = int(((pred == 1) & (nber_true == 0)).sum())
        fn_ = int(((pred == 0) & (nber_true == 1)).sum())
        pr_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        re_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f1_ = 2 * pr_ * re_ / (pr_ + re_) if (pr_ + re_) > 0 else 0.0
        if f1_ > best_f1:
            best_f1, best_thr = f1_, thr_candidate

    logger.debug(f"CFNAI baseline optimal threshold: {best_thr:.2f} (F1={best_f1:.3f})")

    model_pred = (cfnai_aligned < best_thr).astype(int)
    model_recession_prob = -cfnai_aligned  # lower CFNAI → higher recession prob

    tp = int(((model_pred == 1) & (nber_true == 1)).sum())
    fp = int(((model_pred == 1) & (nber_true == 0)).sum())
    fn = int(((model_pred == 0) & (nber_true == 1)).sum())
    tn = int(((model_pred == 0) & (nber_true == 0)).sum())

    confusion = np.array([[tn, fp], [fn, tp]])
    n = tp + fp + fn + tn

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    history = pd.DataFrame({
        "nber_recession": nber_true,
        "model_recession_prob": model_recession_prob,
        "model_pred": model_pred,
        "cfnai": cfnai_aligned,
    }, index=common)

    # Detection lag
    nber_shifted = nber_true.shift(1, fill_value=0)
    starts = common[(nber_true == 1) & (nber_shifted == 0)]
    lags = []
    for sd in starts:
        future = model_pred.loc[sd:]
        detected = future[future == 1]
        if len(detected) > 0:
            fd = detected.index[0]
            lag = (fd.year - sd.year) * 12 + fd.month - sd.month
            lags.append(lag)
    detection_lag = float(np.mean(lags)) if lags else np.nan

    return RegimeBacktestResult(
        accuracy=accuracy,
        precision_recession=precision,
        recall_recession=recall,
        f1_recession=f1,
        confusion=confusion,
        regime_history=history,
        n_months=n,
        detection_lag_months=detection_lag,
        false_alarm_rate=false_alarm,
    )
