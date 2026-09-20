"""Relative forward-volatility outlook from the latent factors.

What the evidence supports, and what it does not
------------------------------------------------
Measured over 1967-2026 on the point-in-time panel, under purged and
embargoed walk-forward CV (``scripts/benchmark_features.py``), the
latent factors carry information about forward NASDAQ volatility:

    horizon   IC       R²      folds with R² > 0
    3 months  +0.162   +0.063  4 of 5
    6 months  +0.173   +0.018  3 of 5
    12 months +0.161   +0.063  4 of 5

The IC is positive in **all 15** fold-horizon combinations, and the
gradient-boosted model agrees on the sign, so the *ranking* is not an
artifact of one estimator or one era.

The level is a different matter. R² is small, and at every horizon the
negative fold is the same one — 2006-2016, which contains 2008. A linear
model ranks that period correctly and still misses its magnitude,
because the realised volatility of 2008 is outside anything in its
training range. A point forecast would therefore be least trustworthy in
exactly the conditions that would make anyone want one.

This module is built around that asymmetry. It reports a **percentile**
— where the current reading sits against the model's own history — and
exposes the level only as supporting detail, clearly bounded. Do not
promote the level to a headline number without new evidence.

Both the estimator and the CV splitter are imported rather than
redefined, so what this serves is what the benchmark measured.

One deliberate difference. The benchmark's "factors only" set is every
``factor_*`` column, which includes the 1/3/6-month momentum columns —
20 in all, scoring +0.162. This module uses the five factor *levels*
only, for two reasons: a current reading then needs nothing but the
current factor row, and the panel's momentum columns are point-in-time
differences between separate fits, which a live single-fit dashboard
cannot reproduce faithfully. Rather than quote a number measured on a
different specification, ``fit_volatility_outlook`` measures the IC of
the specification it actually serves (+0.157 on the current panel) and
reports that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.downstream_models import build_ridge
from src.evaluation.purged_cv import PurgedWalkForward

DEFAULT_HORIZON = 3
DEFAULT_TICKER = "NASDAQCOM"
DEFAULT_PANEL = "data/features.csv"

# Below this many aligned observations the percentile reference is too
# thin to be meaningful and fitting is refused outright.
MIN_OBSERVATIONS = 120


@dataclass
class VolatilityOutlook:
    """A fitted relative-volatility model and its own historical scale."""

    model: Any
    horizon: int
    ticker: str
    # Model predictions across history; the percentile reference.
    history: pd.Series
    # Realised forward volatility on the same index, for context.
    realised: pd.Series
    factor_columns: list[str]
    # Out-of-sample IC measured on this panel, by the same purged CV the
    # benchmark uses. None when too few folds were available.
    oos_ic: float | None = None
    oos_folds: int = 0
    notes: list[str] = field(default_factory=list)

    def percentile_of(self, prediction: float) -> float:
        """Where *prediction* sits in the model's own historical range."""
        hist = self.history.dropna().to_numpy()
        if hist.size == 0 or not np.isfinite(prediction):
            return float("nan")
        return float((hist < prediction).mean() * 100.0)

    def predict(self, factors: pd.Series | pd.DataFrame) -> float:
        """Predicted forward volatility for one observation.

        Accepts either a row indexed by this model's factor columns, or
        a one-row frame. Missing columns raise rather than defaulting:
        a silent zero here would be a plausible-looking reading.
        """
        if isinstance(factors, pd.Series):
            frame = factors.to_frame().T
        else:
            frame = factors.tail(1)

        missing = [c for c in self.factor_columns if c not in frame.columns]
        if missing:
            raise KeyError(
                f"volatility outlook needs {missing}, which the supplied "
                f"factors do not carry; refusing to substitute a value"
            )
        row = frame[self.factor_columns].astype(float)
        if not np.isfinite(row.to_numpy()).all():
            raise ValueError("factor row contains non-finite values")
        return float(self.model.predict(row)[0])

    def assess(self, factors: pd.Series | pd.DataFrame) -> dict[str, float]:
        """Prediction, percentile, and the realised range it maps to."""
        pred = self.predict(factors)
        pct = self.percentile_of(pred)

        # What volatility actually did, historically, when the model
        # predicted around this level. This is the honest way to give a
        # magnitude: an empirical spread rather than a point estimate.
        hist = self.history.dropna()
        lo_q, hi_q = max(0.0, pct - 10.0) / 100.0, min(100.0, pct + 10.0) / 100.0
        band = hist[(hist >= hist.quantile(lo_q)) & (hist <= hist.quantile(hi_q))]
        realised_near = self.realised.reindex(band.index).dropna()

        return {
            "prediction": pred,
            "percentile": pct,
            "realised_p25": float(realised_near.quantile(0.25))
            if len(realised_near) else float("nan"),
            "realised_median": float(realised_near.median())
            if len(realised_near) else float("nan"),
            "realised_p75": float(realised_near.quantile(0.75))
            if len(realised_near) else float("nan"),
            "n_comparable": int(len(realised_near)),
        }


def realised_forward_volatility(
    prices: pd.Series, index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    """Annualised realised volatility over the *horizon* months ahead."""
    month_end = prices.resample("ME").last()
    out = {}
    for ts in index:
        pos = month_end.index.searchsorted(ts)
        if pos >= len(month_end) - horizon:
            continue
        window = month_end.iloc[pos : pos + horizon + 1]
        daily = prices.loc[month_end.index[pos] : window.index[-1]]
        d = daily.pct_change().dropna()
        if len(d) > 5:
            out[ts] = float(d.std() * np.sqrt(252))
    return pd.Series(out, name="realised_vol")


def fit_volatility_outlook(
    prices: pd.Series,
    panel_path: str | Path = DEFAULT_PANEL,
    *,
    horizon: int = DEFAULT_HORIZON,
    ticker: str = DEFAULT_TICKER,
    measure_ic: bool = True,
) -> VolatilityOutlook:
    """Fit the outlook on the point-in-time panel.

    Raises
    ------
    FileNotFoundError
        The panel has not been built. It is gitignored and costs several
        hours, so this is the normal state of a fresh clone and the
        caller should say so rather than showing an empty panel.
    ValueError
        Too few aligned observations to form a percentile reference.
    """
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            f"  python scripts/build_features.py --start 1967-01-31 --step 1"
        )

    panel = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "knowable_at" not in panel.columns:
        raise ValueError(f"{path} has no knowable_at column; regenerate it")
    knowable = pd.to_datetime(panel["knowable_at"])

    factor_columns = [
        c for c in panel.columns
        if c.startswith("factor_") and not c.endswith(("_d1m", "_d3m", "_d6m"))
    ]
    if not factor_columns:
        raise ValueError(f"{path} carries no factor columns")

    # Align each row to the date it was actually usable, exactly as the
    # benchmark does — never to the reference month.
    month_end = prices.resample("ME").last()
    entries = {}
    for ref, know in knowable.items():
        pos = month_end.index.searchsorted(know)
        if pos < len(month_end) - horizon:
            entries[ref] = month_end.index[pos]
    if not entries:
        raise ValueError("no panel rows could be aligned to price history")

    aligned = pd.Series(entries)  # reference month -> entry month
    X = panel.loc[aligned.index, factor_columns].astype(float)

    # Realised volatility is keyed by entry date; map it back onto the
    # reference month so features and target share one index. Entries
    # the price history cannot cover simply drop out.
    by_entry = realised_forward_volatility(
        prices, pd.DatetimeIndex(aligned.to_numpy()), horizon
    )
    y = aligned.map(by_entry).rename("realised_vol")

    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    if len(X) < MIN_OBSERVATIONS:
        raise ValueError(
            f"only {len(X)} aligned observations; need at least "
            f"{MIN_OBSERVATIONS} for a usable percentile reference"
        )

    notes: list[str] = []
    oos_ic, folds = None, 0
    if measure_ic:
        oos_ic, folds = _measure_oos_ic(X, y, horizon)
        if oos_ic is None:
            notes.append("out-of-sample IC could not be measured on this panel")

    model = build_ridge(label_horizon=horizon, embargo=3).fit(X, y)
    history = pd.Series(model.predict(X), index=X.index, name="predicted_vol")

    logger.info(
        f"Volatility outlook fitted: {len(X)} observations, horizon {horizon}m, "
        f"OOS IC {oos_ic if oos_ic is None else round(oos_ic, 3)}"
    )
    return VolatilityOutlook(
        model=model,
        horizon=horizon,
        ticker=ticker,
        history=history,
        realised=y,
        factor_columns=factor_columns,
        oos_ic=oos_ic,
        oos_folds=folds,
        notes=notes,
    )


def _measure_oos_ic(
    X: pd.DataFrame, y: pd.Series, horizon: int
) -> tuple[float | None, int]:
    """Out-of-sample IC under the same purged CV the benchmark uses.

    Reported next to the reading so the panel states its own reliability
    rather than leaving the viewer to assume it.
    """
    cv = PurgedWalkForward(
        n_splits=5, label_horizon=horizon, embargo=3, min_train=80
    )
    ics = []
    for train_idx, test_idx in cv.split(X):
        model = build_ridge(label_horizon=horizon, embargo=3)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx].to_numpy()
        if np.std(pred) > 1e-12 and np.std(actual) > 1e-12:
            ics.append(float(np.corrcoef(pred, actual)[0, 1]))
    if not ics:
        return None, 0
    return float(np.mean(ics)), len(ics)
