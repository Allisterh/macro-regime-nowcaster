"""Benchmark the ensemble against the SPF's professional recession odds.

The Philadelphia Fed's Survey of Professional Forecasters has asked, every
quarter since 1968Q4, for the probability that real GDP declines in the
survey quarter (RECESS1) and the next (RECESS2, the "Anxious Index").
It is a consensus probability recorded at the time — the idea prediction
markets offer, with the 55 years of history they lack.

Questions, fixed before any result was seen
-------------------------------------------
Two volatility findings in this project were reported and later
retracted, partly because a best-looking variant was picked from many.
So the comparisons are declared here and nothing else is searched over:

1. **Benchmark.** Is SPF RECESS1 a better recession detector than the
   ensemble? Primary metric AUC. The SPF prices a *different event* —
   P(GDP falls this quarter), not P(this month is in an NBER recession)
   — so it is not calibrated to this target and its standalone Brier is
   expected to lose; that is not evidence against it.
2. **Added value.** Does a fixed 50/50 average of the ensemble and SPF
   RECESS1 beat the ensemble alone? AUC and Brier. The weight is not
   tuned.
3. **Horizons.** RECESS1 and RECESS2 on the AUC-by-horizon curve.

Not searched: RECESS3-5, blend weights, median instead of mean.

Alignment
---------
Each panel row is scored with the latest survey *published* by that
row's reference month-end — the model's own information cutoff, since
the walk-forward masks every input against it. (Not ``knowable_at``,
which adds a 60-day buffer for downstream joins; see ``build_frame``.)
Surveys before 1990Q2 have no published release date, so results are reported
under three assumed lags from the start of the survey quarter, plus the
post-1990 subsample where every date is known. A conclusion that holds
only under the optimistic lag would be one that look-ahead is carrying.

Usage
-----
    python scripts/measure_spf.py
    python scripts/measure_spf.py --download       # refresh the SPF files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.spf import (
    DEFAULT_DIR,
    FIRST_KNOWN_RELEASE,
    as_of,
    download,
    knowable_dates,
    load_recess,
    load_release_dates,
)
from src.evaluation.horizon_curve import _label_blocks, horizon_auc_curve
from src.evaluation.paired_bootstrap import paired_block_bootstrap
from src.models.regime_backtest import (
    brier_score,
    get_nber_recession_indicator,
    roc_auc,
)

# Days after the start of the survey quarter at which a pre-1990Q2 survey
# is assumed knowable. Post-1990 surveys publish a median 45 days in
# (minimum 39), so 45 matches modern practice, 90 is the end of the
# survey quarter, and 120 is a month beyond that.
LAG_SCENARIOS = {"optimistic": 45, "central": 90, "conservative": 120}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/features.csv")
    p.add_argument("--spf-dir", default=str(DEFAULT_DIR))
    p.add_argument("--download", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_frame(
    feats: pd.DataFrame, recess: pd.DataFrame, releases: pd.Series,
    lag_days: int,
) -> pd.DataFrame:
    """Signals for each panel row, all at the model's information cutoff.

    That cutoff is the reference month-end itself. The walk-forward runs
    the pipeline with ``end_date`` set to the reference month, and the
    pipeline masks the ragged edge against ``end_date``, so the model's
    row for month T sees only data published by T.

    A first version aligned the SPF to ``knowable_at`` instead — T plus
    60 days. That column is a conservative buffer for joining downstream
    targets, not the model's cutoff, and using it handed the SPF up to 60
    days the model never had. With surveys publishing about 45 days into
    each quarter, that routinely meant the *next* survey, and it inflated
    the SPF in every comparison it was part of.
    """
    knowable = knowable_dates(recess.index, releases, lag_days)
    when = pd.DatetimeIndex(feats.index)
    frame = pd.DataFrame(index=feats.index)
    frame["ensemble"] = feats["p_recession"]
    frame["cfnai"] = feats["signal_cfnai"]
    for col in ("RECESS1", "RECESS2"):
        frame[col.lower()] = as_of(recess[col], knowable, when).to_numpy()
    frame["blend"] = 0.5 * frame["ensemble"] + 0.5 * frame["recess1"]
    return frame


def evaluate(frame: pd.DataFrame, nber: pd.Series, args, label: str) -> None:
    rows = frame.dropna(subset=["recess1", "ensemble"]).index
    rows = rows.intersection(nber.index)
    y = nber.loc[rows].astype(int).to_numpy()
    f = frame.loc[rows]
    blocks = _label_blocks(pd.Series(y, index=rows))

    print(f"--- {label}: {rows[0]:%Y-%m} to {rows[-1]:%Y-%m}, "
          f"{len(rows)} months, {int(y.sum())} in recession, "
          f"{len(blocks)} label runs ---")
    print(f"  {'signal':26s} {'AUC':>7s} {'Brier':>8s}")
    names = [("ensemble", "Ensemble"), ("cfnai", "CFNAI signal"),
             ("recess1", "SPF RECESS1 (this qtr)"),
             ("recess2", "SPF RECESS2 (Anxious)"),
             ("blend", "50/50 ensemble + RECESS1")]
    for col, name in names:
        s = f[col].to_numpy(dtype=float)
        ok = np.isfinite(s)
        print(f"  {name:26s} {roc_auc(y[ok], s[ok]):7.3f} "
              f"{brier_score(y[ok], s[ok]):8.4f}")

    def compare(a, b, metric, better):
        d = paired_block_bootstrap(
            y, f[a].to_numpy(float), f[b].to_numpy(float), blocks, metric,
            n_boot=args.n_boot, seed=args.seed,
        )
        p = d.p_positive if better == "higher" else d.p_negative
        verdict = "excludes 0" if d.excludes_zero() else "includes 0"
        return (f"{d.point:+.4f}  CI [{d.lo:+.4f}, {d.hi:+.4f}]  "
                f"P(better) = {p:.2f}  ({verdict})")

    print(f"  Q1  SPF - ensemble,   AUC  : "
          f"{compare('recess1', 'ensemble', roc_auc, 'higher')}")
    print(f"  Q2  blend - ensemble, AUC  : "
          f"{compare('blend', 'ensemble', roc_auc, 'higher')}")
    print(f"  Q2  blend - ensemble, Brier: "
          f"{compare('blend', 'ensemble', brier_score, 'lower')}")
    print()


def main() -> int:
    args = parse_args()
    spf_dir = Path(args.spf_dir)
    if args.download or not (spf_dir / "Mean_RECESS_Level.xlsx").exists():
        print(f"Downloading SPF files into {spf_dir} ...")
        download(spf_dir)

    feats = pd.read_csv(args.features, index_col=0, parse_dates=True).sort_index()
    feats["knowable_at"] = pd.to_datetime(feats["knowable_at"])
    recess = load_recess(spf_dir / "Mean_RECESS_Level.xlsx")
    releases = load_release_dates(spf_dir / "spf-release-dates.txt")
    nber = get_nber_recession_indicator(
        start=str(feats.index[0].date()), end=str(feats.index[-1].date())
    )

    print(f"SPF: {len(recess)} surveys {recess.index[0]:%Y}Q"
          f"{recess.index[0].quarter} to {recess.index[-1]:%Y}Q"
          f"{recess.index[-1].quarter}; {len(releases)} with published "
          f"release dates (from {FIRST_KNOWN_RELEASE:%Y}Q2)\n")

    for name, lag in LAG_SCENARIOS.items():
        frame = build_frame(feats, recess, releases, lag)
        evaluate(frame, nber, args,
                 f"full sample, pre-1990 lag {lag}d ({name})")

    frame = build_frame(feats, recess, releases, LAG_SCENARIOS["central"])
    late = frame.loc[frame.index >= FIRST_KNOWN_RELEASE]
    evaluate(late, nber, args, "post-1990 only, every release date known")

    curve = horizon_auc_curve(
        frame.dropna(subset=["recess1"]), nber,
        {"Ensemble": "ensemble", "SPF RECESS1": "recess1",
         "SPF RECESS2": "recess2", "Blend": "blend"},
        n_boot=args.n_boot, seed=args.seed,
    ).to_frame()
    wide = curve.pivot(index="horizon", columns="signal", values="auc")
    print("AUC by horizon (central lag)")
    print(wide[["Ensemble", "SPF RECESS1", "SPF RECESS2", "Blend"]]
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
