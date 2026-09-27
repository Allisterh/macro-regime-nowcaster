"""Recession-detection metrics from the point-in-time feature panel.

Produces the numbers in the README's *Measured Real-Time Performance*
section: per-signal AUC and Brier against NBER, the ensemble-minus-CFNAI
comparison with a block bootstrap over contiguous label runs, and the
AUC-by-horizon curve.

It reads ``data/features.csv`` rather than refitting anything, so it
runs in seconds once the panel exists. This used to live in a scratch
script, which meant the README's headline numbers could not be
reproduced from the repository.

Usage
-----
    python scripts/measure_recession.py
    python scripts/measure_recession.py --features data/other.csv --n-boot 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.horizon_curve import _label_blocks, horizon_auc_curve
from src.models.regime_backtest import (
    brier_score,
    get_nber_recession_indicator,
    roc_auc,
)

SIGNALS = [
    ("Ensemble", "p_recession"),
    ("Probit", "signal_probit"),
    ("CFNAI signal alone", "signal_cfnai"),
    ("Sahm rule", "signal_sahm"),
    ("Markov-switching (RSM)", "signal_rsm"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/features.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.features)
    if not path.exists():
        print(
            f"{path} not found. Build it with:\n"
            f"  python scripts/build_features.py --start 1967-01-31 --step 1"
        )
        return 1

    feats = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    nber = get_nber_recession_indicator(
        start=str(feats.index[0].date()), end=str(feats.index[-1].date())
    )
    common = feats.index.intersection(nber.index)
    y = nber.loc[common].astype(int).to_numpy()

    print(f"panel        : {path}")
    print(f"span         : {common[0]:%Y-%m} to {common[-1]:%Y-%m}  "
          f"({len(common)} months)")
    print(f"recession mo : {int(y.sum())}  (base rate {y.mean():.1%})")
    print()

    print(f"{'Signal':26s} {'AUC':>7s} {'Brier':>9s}")
    scores: dict[str, np.ndarray] = {}
    for label, col in SIGNALS:
        s = feats[col].reindex(common).to_numpy(dtype=float)
        ok = np.isfinite(s)
        scores[col] = s
        print(f"{label:26s} {roc_auc(y[ok], s[ok]):7.3f} "
              f"{brier_score(y[ok], s[ok]):9.4f}")
    base = np.full(len(y), y.mean())
    print(f"{'Constant at base rate':26s} {0.5:7.3f} "
          f"{brier_score(y, base):9.4f}")
    print()

    # Ensemble vs CFNAI, resampling whole recession/expansion runs so the
    # interval reflects how few independent episodes there are.
    ens, cf = scores["p_recession"], scores["signal_cfnai"]
    blocks = _label_blocks(pd.Series(y, index=common))
    rng = np.random.default_rng(args.seed)
    d_auc, d_brier = [], []
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        a = roc_auc(yb, ens[idx]) - roc_auc(yb, cf[idx])
        if np.isfinite(a):
            d_auc.append(a)
        d_brier.append(brier_score(yb, ens[idx]) - brier_score(yb, cf[idx]))

    point_auc = roc_auc(y, ens) - roc_auc(y, cf)
    point_brier = brier_score(y, ens) - brier_score(y, cf)
    lo_a, hi_a = np.percentile(d_auc, [2.5, 97.5])
    lo_b, hi_b = np.percentile(d_brier, [2.5, 97.5])
    print(f"Ensemble vs CFNAI  ({len(blocks)} blocks, {len(d_auc)} draws)")
    print(f"  AUC   {point_auc:+.3f}  95% CI [{lo_a:+.3f}, {hi_a:+.3f}]  "
          f"P(>0) = {np.mean(np.array(d_auc) > 0):.2f}")
    print(f"  Brier {point_brier:+.4f}  95% CI [{lo_b:+.4f}, {hi_b:+.4f}]  "
          f"P(<0) = {np.mean(np.array(d_brier) < 0):.2f}")
    print(f"  Brier improvement: {-point_brier / brier_score(y, cf):.1%}")
    print()

    curve = horizon_auc_curve(
        feats, nber,
        {"Ensemble": "p_recession", "CFNAI": "signal_cfnai",
         "Probit": "signal_probit"},
        n_boot=args.n_boot, seed=args.seed,
    ).to_frame()
    wide = curve.pivot(index="horizon", columns="signal", values="auc")
    print("AUC by horizon (months ahead)")
    print(wide[["Ensemble", "CFNAI", "Probit"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
