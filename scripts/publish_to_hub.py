"""Publish the nowcast into the Dashboard Hub (market-compass).

The hub's ``run_jobs.py`` runs this with *this* project's venv: it sets
``HUB_DATA_DIR`` and puts the hub folder on ``PYTHONPATH`` so
``from core import store`` resolves to the hub's storage module.

    python scripts/publish_to_hub.py           # run a live nowcast, publish it + the panel
    python scripts/publish_to_hub.py --demo    # no network: publish the panel only

Written to ``<HUB_DATA_DIR>/macro/``:

* ``walk_forward``  the point-in-time panel from ``data/features.csv`` (one refit per
                    month, final row kept), numeric columns only. This is the only
                    history published: a single ``Nowcaster.run()`` returns an
                    in-sample history that embeds later data.
* ``summary``       (JSON) the live nowcast (same configuration as
                    ``scripts/run_nowcast.py``) and the panel's coverage.

The panel is *not* rebuilt here (``scripts/build_features.py`` refits once per month
and takes a while); the hub page shows how old it is.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.utils.logging_config import setup_logging  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "macro"
PANEL_PATH = PROJECT_ROOT / "data" / "features.csv"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def _iso(v):
    return v.strftime("%Y-%m-%d") if isinstance(v, (pd.Timestamp, datetime)) else v


def load_panel() -> pd.DataFrame | None:
    if not PANEL_PATH.exists():
        log(f"No walk-forward panel at {PANEL_PATH}; run scripts/build_features.py to create one")
        return None
    panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True).sort_index()
    return panel.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").astype(np.float64)


def live_nowcast() -> dict:
    """One nowcast with the measured configuration in config/settings.yaml (as run_nowcast.py does)."""
    from src.data.data_pipeline import DataPipeline
    from src.data.fred_client import FREDClient
    from src.models.nowcaster import Nowcaster

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set (hub .env or this project's .env)")
    settings = yaml.safe_load(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    model_cfg = settings.get("model", {})
    ensemble_cfg = model_cfg.get("ensemble", {})
    data_cfg = settings.get("data", {})

    client = FREDClient(api_key=api_key, cache_dir=str(PROJECT_ROOT / data_cfg.get("cache_dir", "data/cache")))
    pipeline = DataPipeline(fred_client=client, start_date=data_cfg.get("start_date", "1980-01-01"))
    nowcaster = Nowcaster(
        pipeline=pipeline,
        n_factors=model_cfg.get("n_factors", 4),
        n_regimes=model_cfg.get("n_regimes", 2),
        regime_labels=model_cfg.get("regime_labels"),
        factor_names=model_cfg.get("factor_names"),
        use_ensemble=ensemble_cfg.get("enabled", True),
        ensemble_weights=ensemble_cfg.get("weights"),
        probit_config=ensemble_cfg.get("probit"),
    )
    result = nowcaster.run()
    out = result.to_dict()
    out["weights"] = dict(nowcaster.ensemble_weights)
    out["signal_as_of"] = {k: _iso(v) for k, v in (getattr(nowcaster, "_signal_as_of", {}) or {}).items()}
    out["factor_quality"] = dict(getattr(nowcaster._dfm, "_factor_match_quality", {}) or {})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="no network: publish the walk-forward panel only")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")  # real env vars (e.g. from the hub) take precedence
    setup_logging(level="WARNING")

    panel = load_panel()
    nowcast = None
    if not args.demo:
        t = time.time()
        log("Running the live nowcast (FRED pull, DFM + ensemble fit) ...")
        nowcast = live_nowcast()
        log(f"  done in {time.time() - t:.0f}s")

    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "demo": args.demo,
        "nowcast": nowcast,
        "panel": None if panel is None else {
            "rows": len(panel),
            "start": _iso(panel.index.min()),
            "through": _iso(panel.index.max()),
            "built": datetime.fromtimestamp(PANEL_PATH.stat().st_mtime).isoformat(timespec="seconds"),
            "latest": {k: v for k, v in panel.iloc[-1].items() if pd.notna(v)},
        },
    }
    if panel is not None:
        store.save_frame(MODULE, "walk_forward", panel)
    store.save_json(MODULE, "summary", summary)

    if nowcast:
        log(f"Published macro nowcast: {nowcast['current_regime']}, P(recession) "
            f"{nowcast['recession_probability']:.1%}, GDP nowcast {nowcast['gdp_nowcast']:.2f}%")
    if panel is not None:
        log(f"Published walk-forward panel: {len(panel)} months through {summary['panel']['through']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
