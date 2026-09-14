"""Deep validation: recession-by-recession breakdown + expanding-window OOS backtest.

Usage
-----
    python scripts/run_validation.py                # both analyses
    python scripts/run_validation.py --breakdown    # recession breakdown only
    python scripts/run_validation.py --expanding    # expanding-window only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# NBER recession episodes (peak, trough, label)
RECESSION_EPISODES = [
    ("1990-07-01", "1991-03-01", "1990-91 recession"),
    ("2001-03-01", "2001-11-01", "2001 dot-com recession"),
    ("2007-12-01", "2009-06-01", "2007-09 Great Recession"),
    ("2020-02-01", "2020-04-01", "2020 COVID recession"),
]


def recession_breakdown(history: pd.DataFrame) -> None:
    """Analyse model performance on each recession episode."""
    print()
    print("=" * 70)
    print("  RECESSION-BY-RECESSION BREAKDOWN")
    print("=" * 70)

    # Columns we need
    nber_col = "nber_recession"
    pred_col = "model_pred"
    prob_col = "model_recession_prob"

    for peak, trough, label in RECESSION_EPISODES:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)

        # Look at a window: 6 months before peak to 6 months after trough
        window_start = peak_dt - pd.DateOffset(months=6)
        window_end = trough_dt + pd.DateOffset(months=6)

        window = history.loc[
            (history.index >= window_start) & (history.index <= window_end)
        ]

        if len(window) == 0:
            print(f"\n  {label}: No data in evaluation window")
            continue

        # Recession months only
        rec_months = window[window[nber_col] == 1]
        exp_months_before = window[
            (window.index < peak_dt) & (window[nber_col] == 0)
        ]
        exp_months_after = window[
            (window.index > trough_dt) & (window[nber_col] == 0)
        ]

        n_rec = len(rec_months)
        if n_rec == 0:
            print(f"\n  {label}: No recession months in evaluation window")
            continue

        # Detection metrics for this episode
        detected = rec_months[rec_months[pred_col] == 1]
        n_detected = len(detected)
        recall = n_detected / n_rec if n_rec > 0 else 0.0

        # Detection lag: months from peak to first detection
        first_detect = None
        if n_detected > 0:
            first_detect = detected.index[0]
            lag = (first_detect.year - peak_dt.year) * 12 + first_detect.month - peak_dt.month
        else:
            lag = n_rec  # penalty: full duration

        # False alarms in surrounding expansion months
        pre_false = (
            (exp_months_before[pred_col] == 1).sum()
            if len(exp_months_before) > 0 else 0
        )
        post_false = (
            (exp_months_after[pred_col] == 1).sum()
            if len(exp_months_after) > 0 else 0
        )

        # Average ensemble probability during recession
        avg_prob_rec = rec_months[prob_col].mean() if n_rec > 0 else 0.0
        max_prob_rec = rec_months[prob_col].max() if n_rec > 0 else 0.0

        # Average ensemble probability in the 6 months before
        avg_prob_pre = (
            exp_months_before[prob_col].mean()
            if len(exp_months_before) > 0 else float("nan")
        )

        # Per-signal breakdown if available
        signal_cols = [c for c in history.columns if c.startswith("signal_")]

        print(f"\n  --- {label} ---")
        print(f"  Duration         : {peak[:7]} to {trough[:7]} ({n_rec} months)")
        print(f"  Months detected  : {n_detected}/{n_rec} (recall={recall:.0%})")
        print(f"  Detection lag    : {lag} month(s)")
        if first_detect is not None:
            print(f"  First detection  : {str(first_detect.date())[:7]}")
        else:
            print("  First detection  : NEVER DETECTED")
        print(f"  Avg P(rec) during: {avg_prob_rec:.1%}")
        print(f"  Max P(rec) during: {max_prob_rec:.1%}")
        print(f"  Avg P(rec) before: {avg_prob_pre:.1%}")
        print(f"  False alarms     : {pre_false} before, {post_false} after")

        if signal_cols:
            print("  Signal averages during recession:")
            for sc in signal_cols:
                if sc in rec_months.columns:
                    avg = rec_months[sc].mean()
                    print(f"    {sc.replace('signal_', ''):>10s}: {avg:.1%}")

    # Overall summary
    print("\n  --- Overall False Alarm Analysis ---")
    exp_total = history[history[nber_col] == 0]
    if len(exp_total) > 0:
        fa = (exp_total[pred_col] == 1).sum()
        print(f"  Expansion months   : {len(exp_total)}")
        print(f"  False alarms       : {fa} ({fa/len(exp_total):.1%})")
    print("=" * 70)


def expanding_window_ensemble(
    pipeline,
    nber: pd.Series,
    panel: pd.DataFrame,
    start_eval: str = "2000-01-01",
    end: str | None = None,
    step_months: int = 3,
    min_train_months: int = 120,
    ensemble_weights: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window out-of-sample ensemble backtest.

    Delegates to :func:`src.models.walk_forward.generate_feature_panel`,
    which refits the *production* Nowcaster at each date and keeps only
    its final row.

    This function used to re-implement the ensemble inline and drifted
    away from what the model actually serves: different weights
    (0.25/0.50/0.25 vs 0.20/0.40/0.20/0.20), no Sahm signal, a different
    CFNAI mapping (``norm.cdf(-x)`` rather than the logistic at -0.7),
    and ``.ffill()`` on features that production deliberately leaves as
    NaN.  Whatever it measured, it was not the shipped model.
    """
    from src.models.walk_forward import generate_feature_panel

    if end is None:
        end = str(pd.Timestamp.today().date())

    feats = generate_feature_panel(
        pipeline,
        start=start_eval,
        end=end,
        step_months=step_months,
        cache_path="data/oos_validation_interim.csv",
        ensemble_weights=ensemble_weights,
    )

    out = pd.DataFrame({
        "p_rsm": feats.get("signal_rsm"),
        "p_probit": feats.get("signal_probit"),
        "p_cfnai": feats.get("signal_cfnai"),
        "p_sahm": feats.get("signal_sahm"),
        "p_ensemble": feats["p_recession"],
    }, index=feats.index)
    out.index.name = "date"
    return out


def evaluate_expanding(oos_probs: pd.DataFrame, nber: pd.Series) -> None:
    """Evaluate expanded-window OOS results against NBER."""
    common = oos_probs.index.intersection(nber.index)
    if len(common) < 6:
        print("Too few OOS evaluation points for metrics")
        return

    nber_aligned = nber.loc[common].astype(int)
    ensemble = oos_probs["p_ensemble"].loc[common]

    # Fixed threshold.  Sweeping for the F1-maximising cut-off and then
    # reporting metrics at it selects a hyper-parameter on the evaluation
    # labels, which inflates every number that follows.  AUC and Brier
    # below are threshold-free and are what to compare across versions.
    from src.models.regime_backtest import brier_score, roc_auc

    thr = 0.5
    pred = (ensemble > thr).astype(int)
    auc = roc_auc(nber_aligned.values, ensemble.values)
    brier = brier_score(nber_aligned.values, ensemble.values)

    tp = int(((pred == 1) & (nber_aligned == 1)).sum())
    fp = int(((pred == 1) & (nber_aligned == 0)).sum())
    fn = int(((pred == 0) & (nber_aligned == 1)).sum())
    tn = int(((pred == 0) & (nber_aligned == 0)).sum())

    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Detection lag
    nber_shifted = nber_aligned.shift(1, fill_value=0)
    starts = common[(nber_aligned == 1) & (nber_shifted == 0)]
    lags = []
    for sd in starts:
        future = pred.loc[sd:]
        detected = future[future == 1]
        if len(detected) > 0:
            fd = detected.index[0]
            lag = (fd.year - sd.year) * 12 + fd.month - sd.month
            lags.append(lag)
    det_lag = float(np.mean(lags)) if lags else float("nan")

    print()
    print("=" * 60)
    print("  EXPANDING-WINDOW OUT-OF-SAMPLE BACKTEST")
    print("=" * 60)
    print(f"  Evaluation points  : {len(common)}")
    print()
    print("  --- Threshold-free (cannot be tuned on labels) ---")
    print(f"  ROC AUC            : {auc:.3f}")
    print(f"  Brier score        : {brier:.4f}")
    print()
    print(f"  --- At fixed threshold {thr:.2f} ---")
    print(f"  Overall accuracy   : {accuracy:.1%}")
    print("  --- Recession Detection ---")
    print(f"  Precision          : {precision:.1%}")
    print(f"  Recall             : {recall:.1%}")
    print(f"  F1 score           : {f1:.3f}")
    print(f"  Avg detection lag  : {det_lag:.1f} months")
    print(f"  False alarm rate   : {false_alarm:.1%}")
    print()
    print("  --- Confusion Matrix ---")
    print("         Pred Exp  Pred Rec")
    print(f"  NBER Exp  {tn:>6d}  {fp:>6d}")
    print(f"  NBER Rec  {fn:>6d}  {tp:>6d}")
    print()

    # Per-signal mean during OOS recession months
    rec_mask = nber_aligned == 1
    if rec_mask.any():
        print("  --- Signal Averages during OOS Recession Months ---")
        for col in ["p_rsm", "p_probit", "p_cfnai", "p_ensemble"]:
            avg = oos_probs[col].loc[common[rec_mask]].mean()
            print(f"    {col:>12s}: {avg:.1%}")
    print("=" * 60)

    # Per-recession OOS breakdown
    print()
    print("  --- Per-Recession OOS Results ---")
    for peak, trough, label in RECESSION_EPISODES:
        peak_dt = pd.Timestamp(peak)
        trough_dt = pd.Timestamp(trough)
        ep = common[(common >= peak_dt) & (common <= trough_dt)]
        if len(ep) == 0:
            print(f"  {label}: no OOS eval points during this recession")
            continue
        ep_pred = pred.loc[ep]
        ep_nber = nber_aligned.loc[ep]
        n_rec_pts = int(ep_nber.sum())
        n_detected = int(((ep_pred == 1) & (ep_nber == 1)).sum())
        avg_p = oos_probs["p_ensemble"].loc[ep].mean()
        print(
            f"  {label}: {n_detected}/{n_rec_pts} detected, "
            f"avg P(rec)={avg_p:.1%}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Model Validation")
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Run recession-by-recession breakdown only",
    )
    parser.add_argument(
        "--expanding", action="store_true",
        help="Run expanding-window OOS backtest only",
    )
    parser.add_argument(
        "--step", type=int, default=3,
        help="Step size in months for expanding-window (default: 3)",
    )
    parser.add_argument(
        "--start-eval", default="2000-01-01",
        help="Start of expanding-window evaluation",
    )
    args = parser.parse_args()

    run_both = not args.breakdown and not args.expanding

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not set.")
        sys.exit(1)

    from src.data.data_pipeline import DataPipeline
    from src.data.fred_client import FREDClient
    from src.models.regime_backtest import (
        RegimeBacktester,
        get_nber_recession_indicator,
    )

    logger.info("Initialising data pipeline...")
    client = FREDClient(api_key=api_key, cache_dir="data/cache")
    pipeline = DataPipeline(
        fred_client=client,
        start_date="1980-01-01",
        series_config_path="config/fred_series.yaml",
    )
    panel = pipeline.run()
    panel = panel.dropna(how="all")
    nber = get_nber_recession_indicator(
        start=str(panel.index[0].date()),
    )

    # ------------------------------------------------------------------
    # 1. Recession-by-recession breakdown (uses full-sample backtest)
    # ------------------------------------------------------------------
    if args.breakdown or run_both:
        logger.info("Running full-sample ensemble backtest for breakdown...")
        bt = RegimeBacktester(
            pipeline=pipeline,
            n_factors=4,
            n_regimes=2,
            factor_names=[
                "real_activity", "labor_market",
                "inflation", "financial_conditions",
            ],
            regime_labels=["recession", "expansion"],
            recession_labels=["recession"],
            use_ensemble=True,
        )
        report = bt.run(start="1990-01-01", panel=panel, nber=nber)
        print(report.summary())
        recession_breakdown(report.regime_history)

    # ------------------------------------------------------------------
    # 2. Expanding-window OOS backtest
    # ------------------------------------------------------------------
    if args.expanding or run_both:
        logger.info(
            f"Running expanding-window OOS backtest "
            f"(start={args.start_eval}, step={args.step}mo)..."
        )
        logger.info("This will take several minutes (re-fits model at each window)...")
        oos_probs = expanding_window_ensemble(
            pipeline=pipeline,
            nber=nber,
            panel=panel,
            start_eval=args.start_eval,
            step_months=args.step,
        )

        # Save OOS results
        out_path = Path("data") / "oos_validation.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        oos_probs.to_csv(out_path)
        logger.info(f"OOS results saved to {out_path}")

        evaluate_expanding(oos_probs, nber)


if __name__ == "__main__":
    main()
