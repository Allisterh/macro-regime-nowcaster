"""Does the regime nowcast beat CFNAI alone on a downstream equity task?

The point of this project, for a downstream user, is that its output is
worth more than the free indicator it is partly built from.  This script
tests that directly, under purged and embargoed cross-validation.

Three targets, all forward-looking from the date the features were
*knowable*:

- ``fwd_return``   — total return over the next H months
- ``fwd_vol``      — annualised realised volatility over the next H months
- ``fwd_drawdown`` — worst peak-to-trough move over the next H months

Two rules make this an honest test:

1. **Join on ``knowable_at``, never the reference month.** A January
   reading is not usable on 31 January; it is usable once its slowest
   input has printed. The feature panel carries that timestamp.
2. **Purged, embargoed, expanding-window CV.** Overlapping forward
   windows mean adjacent samples share outcome data, and macro state is
   slow, so shuffled k-fold reports large skill on targets that are pure
   noise (see ``tests/test_purged_cv.py``).

Usage
-----
    python scripts/benchmark_features.py --features data/features.csv
    python scripts/benchmark_features.py --horizon 6 --ticker NASDAQCOM
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.evaluation.purged_cv import PurgedWalkForward
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/features.csv",
                   help="Point-in-time panel from scripts/build_features.py")
    p.add_argument("--horizon", type=int, default=3, help="Months ahead")
    p.add_argument("--embargo", type=int, default=3, help="Embargo in months")
    p.add_argument("--splits", type=int, default=5, help="CV folds")
    p.add_argument("--ticker", default="NASDAQCOM",
                   help="FRED equity series. NASDAQCOM reaches back to 1971; "
                        "FRED's SP500 only covers the last 10 years.")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def build_targets(
    features: pd.DataFrame, prices: pd.Series, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align features to their knowable date and build forward targets."""
    month_end = prices.resample("ME").last()

    pairs = []
    for ref, knowable in features["knowable_at"].items():
        pos = month_end.index.searchsorted(knowable)
        if pos >= len(month_end) - horizon:
            continue
        pairs.append((ref, month_end.index[pos]))
    if not pairs:
        raise RuntimeError("No feature rows could be aligned to price history")

    align = pd.DataFrame(pairs, columns=["ref", "entry"]).set_index("ref")
    X = features.drop(columns=["knowable_at"]).loc[align.index]

    ret, vol, dd = [], [], []
    for entry in align["entry"]:
        i = month_end.index.get_loc(entry)
        window = month_end.iloc[i : i + horizon + 1]
        daily = prices.loc[entry : window.index[-1]]
        ret.append(window.iloc[-1] / window.iloc[0] - 1.0)
        d = daily.pct_change().dropna()
        vol.append(d.std() * np.sqrt(252) if len(d) > 5 else np.nan)
        peak = daily.cummax()
        dd.append(float(((daily - peak) / peak).min()) if len(daily) > 1 else np.nan)

    targets = pd.DataFrame(
        {"fwd_return": ret, "fwd_vol": vol, "fwd_drawdown": dd}, index=X.index
    )
    return X, targets


def evaluate(X: pd.DataFrame, y: pd.Series, model_fn, cv) -> tuple[float, float, int]:
    """Mean information coefficient and out-of-sample R² across folds."""
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]

    ics, r2s = [], []
    for train_idx, test_idx in cv.split(X):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx].values
        if np.std(pred) > 1e-12 and np.std(actual) > 1e-12:
            ics.append(float(np.corrcoef(pred, actual)[0, 1]))
        # R² against the *training* mean: the honest naive forecast, since
        # the test mean is not knowable at prediction time.
        ss_res = float(((actual - pred) ** 2).sum())
        ss_tot = float(((actual - y.iloc[train_idx].mean()) ** 2).sum())
        if ss_tot > 0:
            r2s.append(1.0 - ss_res / ss_tot)
    return (
        float(np.mean(ics)) if ics else np.nan,
        float(np.mean(r2s)) if r2s else np.nan,
        len(ics),
    )




def build_models() -> dict:
    """The estimators the benchmark compares, as name -> factory.

    Module level rather than inside main() so the scale-invariance
    property can be asserted in a test.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Ridge is scaled and has its penalty chosen, not fixed.
    #
    # This was `Ridge(alpha=1.0)` on raw columns. The L2 penalty is not
    # scale-invariant, so with features spanning probabilities (sd ~ 0.1)
    # and `expected_recession_duration` (sd ~ 3e10, a 7.7e11x ratio) the
    # effective regularisation was set by units rather than by the data.
    # Measured on the old panel, standardising alone moved the full-panel
    # R² from -0.299 to -3.355 on returns: the unstandardised fit had been
    # accidentally shrinking the wide columns to nothing, which is
    # regularisation by accident, and not reproducible under a change of
    # units. The verdict did not change — every R² is negative either way
    # — but the numbers were an artifact of scale.
    #
    # StandardScaler and RidgeCV both sit inside the pipeline, so they are
    # refitted on each training fold and never see the test fold. Alpha is
    # chosen by RidgeCV's leave-one-out GCV on the training fold only;
    # with overlapping forward windows that can favour a slightly small
    # alpha, which is a conservative direction here (it can only make the
    # features look worse, never better).
    alphas = np.logspace(-2, 4, 13)
    return {
        "ridge": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=alphas)
        ),
        # Trees split on order, not magnitude, so the GBM was never
        # affected by the scaling problem and is left as it was.
        "gbm": lambda: GradientBoostingRegressor(
            random_state=0, n_estimators=100, max_depth=2
        ),
    }


def main() -> int:
    args = parse_args()
    load_dotenv()
    setup_logging(level=args.log_level)

    path = Path(args.features)
    if not path.exists():
        logger.error(
            f"{path} not found. Generate it first:\n"
            f"  python scripts/build_features.py --step 1 --start 1990-01-31"
        )
        return 1

    features = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "knowable_at" not in features.columns:
        logger.error(f"{path} has no knowable_at column; regenerate it.")
        return 1
    features["knowable_at"] = pd.to_datetime(features["knowable_at"])

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.error("FRED_API_KEY not set.")
        return 1
    from src.data.fred_client import FREDClient

    client = FREDClient(api_key=api_key, cache_dir="data/cache")
    prices = pd.to_numeric(
        client.get_series(args.ticker), errors="coerce"
    ).dropna()

    X, targets = build_targets(features, prices, args.horizon)

    feature_sets = {
        "CFNAI only": ["signal_cfnai"],
        "p_recession only": ["p_recession"],
        "regime signals (4)": [
            c for c in ["signal_cfnai", "signal_probit", "signal_sahm", "signal_rsm"]
            if c in X.columns
        ],
        "factors only": [c for c in X.columns if c.startswith("factor_")],
        "full regime panel": list(X.columns),
    }
    models = build_models()

    cv = PurgedWalkForward(
        n_splits=args.splits, label_horizon=args.horizon,
        embargo=args.embargo, min_train=80,
    )

    print()
    print(f"Equity series : {args.ticker}  ({prices.index[0]:%Y-%m} to "
          f"{prices.index[-1]:%Y-%m})")
    print(f"Horizon       : {args.horizon} months, embargo {args.embargo}, "
          f"{args.splits} purged walk-forward folds")
    print("Alignment     : features joined on knowable_at, not reference month")

    for target_name in ["fwd_return", "fwd_vol", "fwd_drawdown"]:
        y = targets[target_name]
        print()
        print("=" * 78)
        print(f"TARGET: {target_name}   n={int(y.notna().sum())}  "
              f"mean={y.mean():+.4f}  std={y.std():.4f}")
        print("=" * 78)
        header = f"  {'feature set':26s}"
        for m in models:
            header += f" {'IC(' + m + ')':>11s} {'R2(' + m + ')':>11s}"
        print(header + "  folds")
        for name, cols in feature_sets.items():
            if not cols:
                continue
            line, folds = f"  {name:26s}", 0
            for model_fn in models.values():
                ic, r2, folds = evaluate(X[cols], y, model_fn, cv)
                line += f" {ic:+11.3f} {r2:+11.3f}"
            print(line + f"  {folds}")

    print()
    print("R² is measured against the training mean — the naive forecast that")
    print("was actually available. Negative means the model is worse than")
    print("predicting the historical average, whatever its IC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
