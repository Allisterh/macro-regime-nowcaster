"""Put a candidate downstream signal through the checks a real one must pass.

Two volatility findings in this project were reported and later
retracted. Each looked convincing on a mean out-of-sample IC, and each
failed on something a mean hides: the first disappeared when the sample
was extended; the second was an artefact of a diverging factor model and
turned out to be era-dependent. This script runs the checks that would
have caught them, so a new lead faces them *before* it is written up.

For one feature set, target and estimator it reports:

1. **Per-fold IC at each horizon.** A mean can be carried by one fold.
2. **Era split.** Folds are grouped before and after a cutoff year; a
   relationship that holds in one era and reverses in the other is not
   one to build on.
3. **Estimator agreement.** The same features under the benchmark's
   ridge. A signal only a nonlinear model finds may be real
   nonlinearity, or may be the nonlinear model fitting noise.
4. **A null distribution.** The target is circularly shifted by a random
   offset and the whole purged CV is rerun, many times. Shifting keeps
   the autocorrelation of both series — which is what makes spurious
   ICs large on overlapping forward targets — while destroying their
   alignment. The observed IC is then placed against what the same
   estimator finds on the same data with the relationship removed.
5. **Hyperparameter sensitivity** (GBM only). A real signal should not
   appear at one configuration and vanish at its neighbours.

Usage
-----
    python scripts/scrutinise_signal.py
    python scripts/scrutinise_signal.py --features signal_cfnai signal_probit \\
        --target fwd_vol --horizons 3 6 12 --n-null 200
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scripts.benchmark_features import build_targets
from src.evaluation.downstream_models import build_models
from src.evaluation.purged_cv import PurgedWalkForward

DEFAULT_FEATURES = [
    "signal_cfnai", "signal_probit", "signal_sahm", "signal_rsm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--target", default="fwd_vol",
                   choices=["fwd_return", "fwd_vol", "fwd_drawdown"])
    p.add_argument("--model", default="gbm", choices=["gbm", "ridge"])
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
    p.add_argument("--panel", default="data/features.csv")
    p.add_argument("--ticker", default="NASDAQCOM")
    p.add_argument("--embargo", type=int, default=3)
    p.add_argument("--era-split", type=int, default=1996,
                   help="Year dividing the early and late folds")
    p.add_argument("--n-null", type=int, default=100,
                   help="Circular-shift draws for the null distribution")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fold_ics(X: pd.DataFrame, y: pd.Series, model_fn, cv) -> list[tuple]:
    """(test start, test end, IC) for each purged fold."""
    out = []
    for train_idx, test_idx in cv.split(X):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx].to_numpy()
        if np.std(pred) > 1e-12 and np.std(actual) > 1e-12:
            ic = float(np.corrcoef(pred, actual)[0, 1])
        else:
            ic = float("nan")
        out.append((y.index[test_idx[0]], y.index[test_idx[-1]], ic))
    return out


def null_ics(
    X: pd.DataFrame, y: pd.Series, model_fn, cv, n: int, rng
) -> np.ndarray:
    """Mean fold IC after circularly shifting *y* by random offsets.

    Offsets avoid the first and last five years so the shifted target is
    never close to realigned with the features.
    """
    T = len(y)
    margin = min(60, T // 4)
    values = y.to_numpy()
    means = []
    for _ in range(n):
        k = int(rng.integers(margin, T - margin))
        shifted = pd.Series(np.roll(values, k), index=y.index)
        ics = [ic for *_, ic in fold_ics(X, shifted, model_fn, cv)]
        ics = [v for v in ics if np.isfinite(v)]
        if ics:
            means.append(float(np.mean(ics)))
    return np.asarray(means)


def gbm_factory(n_estimators: int, max_depth: int):
    from sklearn.ensemble import GradientBoostingRegressor

    return lambda: GradientBoostingRegressor(
        random_state=0, n_estimators=n_estimators, max_depth=max_depth
    )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    load_dotenv(PROJECT_ROOT / ".env")

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    from src.data.fred_client import FREDClient

    feats = pd.read_csv(args.panel, index_col=0, parse_dates=True).sort_index()
    feats["knowable_at"] = pd.to_datetime(feats["knowable_at"])
    client = FREDClient(
        api_key=os.environ["FRED_API_KEY"], cache_dir="data/cache"
    )
    prices = pd.to_numeric(
        client.get_series(args.ticker), errors="coerce"
    ).dropna()
    rng = np.random.default_rng(args.seed)

    print(f"features : {args.features}")
    print(f"target   : {args.target}   model: {args.model}   "
          f"null draws: {args.n_null}\n")

    for h in args.horizons:
        X_all, targets = build_targets(feats, prices, h)
        X, y = X_all[args.features], targets[args.target]
        ok = X.notna().all(axis=1) & y.notna()
        X, y = X[ok], y[ok]

        cv = PurgedWalkForward(
            n_splits=5, label_horizon=h, embargo=args.embargo, min_train=80
        )
        models = build_models(label_horizon=h, embargo=args.embargo)
        model_fn = models[args.model]
        other = "ridge" if args.model == "gbm" else "gbm"

        folds = fold_ics(X, y, model_fn, cv)
        ics = np.array([ic for *_, ic in folds])
        mean_ic = float(np.nanmean(ics))

        print(f"=== horizon {h} months  (n={len(y)}) ===")
        for start, end, ic in folds:
            era = "late " if start.year >= args.era_split else "early"
            print(f"  {start:%Y-%m}..{end:%Y-%m}  [{era}]  IC {ic:+.3f}")

        early = [ic for s, _, ic in folds if s.year < args.era_split]
        late = [ic for s, _, ic in folds if s.year >= args.era_split]
        print(f"  mean IC {mean_ic:+.3f}   positive {int((ics > 0).sum())}/"
              f"{len(ics)}   early mean {np.mean(early):+.3f}   "
              f"late mean {np.mean(late):+.3f}")

        other_ics = [ic for *_, ic in fold_ics(X, y, models[other], cv)]
        print(f"  {other} on the same features: mean IC "
              f"{np.nanmean(other_ics):+.3f}")

        null = null_ics(X, y, model_fn, cv, args.n_null, rng)
        p = float((null >= mean_ic).mean())
        print(f"  null (circular shift): mean {null.mean():+.3f}, "
              f"95th pct {np.quantile(null, 0.95):+.3f}, "
              f"max {null.max():+.3f}")
        print(f"  share of null draws >= observed: {p:.2f}")

        if args.model == "gbm":
            grid = []
            for depth in (1, 2, 3):
                for n_est in (50, 100, 200):
                    g = [ic for *_, ic in
                         fold_ics(X, y, gbm_factory(n_est, depth), cv)]
                    grid.append((depth, n_est, float(np.nanmean(g))))
            spread = [v for *_, v in grid]
            print("  GBM sensitivity (depth x trees):")
            for depth in (1, 2, 3):
                row = [f"{v:+.3f}" for d, _, v in grid if d == depth]
                print(f"    depth {depth}: " + "  ".join(row))
            print(f"    range {min(spread):+.3f} .. {max(spread):+.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
