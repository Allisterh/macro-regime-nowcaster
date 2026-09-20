"""Build a point-in-time feature panel for downstream models.

Refits the nowcaster once per as-of date and keeps only the final row of
each fit, so every value is one an observer could actually have computed
at that date.  See ``src/models/walk_forward.py`` for why slicing a
single full-sample fit is not equivalent.

This is slow — one DFM/RSM/probit fit per step — so results are cached
incrementally and re-runs resume rather than recompute.

Usage
-----
    python scripts/build_features.py                        # monthly, 2000→today
    python scripts/build_features.py --step 3 --start 1995-01-31
    python scripts/build_features.py --verify               # point-in-time check only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate point-in-time macro regime features",
    )
    parser.add_argument("--start", default="2000-01-31", help="First as-of date")
    parser.add_argument("--end", default=None, help="Last as-of date")
    parser.add_argument(
        "--pipeline-start",
        default=None,
        help=(
            "First date the data pipeline fetches. Defaults to 20 years "
            "before --start, so the first window has history to fit on. "
            "This used to be pinned at 1980-01-01, which silently skipped "
            "every as-of date before then."
        ),
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Months between as-of dates",
    )
    parser.add_argument(
        "--output", default="data/features.csv", help="Output CSV path",
    )
    parser.add_argument(
        "--vintages",
        action="store_true",
        help=(
            "Fetch ALFRED point-in-time vintages instead of latest revised "
            "values. Much slower (one API call per series per date) but the "
            "only way to reproduce what was actually knowable."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run the point-in-time check instead of generating: rebuilds a "
            "short panel at two cutoffs and asserts the overlap is identical."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    setup_logging(level=args.log_level)

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.error("FRED_API_KEY not set.")
        return 1

    from src.data.data_pipeline import DataPipeline
    from src.data.fred_client import FREDClient
    from src.models.walk_forward import (
        assert_point_in_time,
        generate_feature_panel,
    )

    # The pipeline must hold data well before the first as-of date: the
    # DFM needs a history to estimate on and the expanding
    # standardisation needs its minimum window. This was pinned at
    # 1980-01-01 with no way to override it, so `--start 1967-01-31`
    # failed its first 120 windows with "empty panel" — warned, skipped,
    # and reported as a successful run 13 years and two recessions short.
    # Those are the 1969-70 and 1973-75 recessions, which is most of what
    # the reconstructed spreads were added to recover.
    pipeline_start = args.pipeline_start or (
        pd.Timestamp(args.start) - pd.DateOffset(years=20)
    ).strftime("%Y-%m-%d")
    logger.info(f"Data pipeline starts {pipeline_start} (as-of from {args.start})")

    client = FREDClient(api_key=api_key, cache_dir="data/cache")
    pipeline = DataPipeline(
        fred_client=client,
        start_date=pipeline_start,
        series_config_path="config/fred_series.yaml",
        use_vintages=args.vintages,
    )

    if args.verify:
        logger.info("Verifying point-in-time stability (two short runs)...")
        diffs = assert_point_in_time(
            pipeline,
            early_cutoff="2016-12-31",
            late_cutoff="2019-12-31",
            start="2012-01-31",
            step_months=6,
        )
        print()
        print("Point-in-time check PASSED")
        print(f"  columns compared : {len(diffs)}")
        print(f"  max difference   : {float(diffs['max_abs_diff'].max()):.2e}")
        return 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Building features {args.start} → {args.end or 'today'} "
        f"every {args.step} month(s). This refits the model at each step."
    )
    panel = generate_feature_panel(
        pipeline,
        start=args.start,
        end=args.end,
        step_months=args.step,
        cache_path=out_path.with_suffix(".partial.csv"),
    )
    panel.to_csv(out_path)

    print()
    print(f"Wrote {len(panel)} rows x {panel.shape[1]} features to {out_path}")
    print()
    print("Columns:")
    for col in panel.columns:
        print(f"  {col}")
    print()
    print(
        "Join downstream targets on `knowable_at`, not on the index: the "
        "index is the reference month, `knowable_at` is when the row could "
        "first have been computed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
