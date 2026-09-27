"""Relative forward-volatility outlook from the latent factors.

The evidence this was built on did not survive
----------------------------------------------
This module was written when the latent factors appeared to rank forward
NASDAQ volatility consistently: out-of-sample IC of +0.157 to +0.187 at
3, 6 and 12 months, positive in every fold of a purged walk-forward.

That panel was produced by a DFM whose EM diverged on long windows,
leaving two of five factors with no loadings. Rebuilt with the EM fixed,
the same code on the same five factor levels measures:

    horizon    old panel   rebuilt panel
    3 months   +0.157      +0.009
    6 months   +0.187      -0.010
    12 months  +0.180      +0.031

Effectively zero. The benchmark's wider "factors only" set, which adds
the 1/3/6-month momentum columns, does somewhat better (+0.03 to +0.08)
but splits by era: positive in every fold after 1996, negative or
inverted in most before it. That is not a relationship this module can
stand on.

So nothing is hard-coded here any more. ``fit_volatility_outlook``
measures the out-of-sample IC of the exact specification it serves, and
:attr:`VolatilityOutlook.has_usable_signal` decides from that
measurement whether a reading may be shown at all. On the current panel
it says no, and the dashboard shows why instead of a percentile of
noise.

If a future panel restores the signal, the gate lets it back through
without a code change — which is the point of measuring it rather than
writing it down.

Design, for when it does apply
------------------------------
It reports a **percentile** — where the current reading sits against
the model's own history — rather than a level, and gives magnitude only
as the empirical spread of what volatility did in comparable months.
Both the estimator and the CV splitter are imported from the benchmark
rather than redefined. It uses the five factor *levels*, not the
momentum columns, because a live single-fit dashboard cannot reproduce
momentum computed as differences between separate point-in-time fits.
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

# Evidence required before a reading is shown. Both must hold:
#
# - The mean out-of-sample IC must reach MIN_USABLE_IC. The panel that
#   justified building this measured +0.157; the rebuilt one measures
#   +0.009. 0.05 sits well clear of both, so the gate separates the two
#   cases this module has actually encountered rather than splitting
#   hairs near either.
# - The IC must be positive in all but at most one fold. The original
#   claim rested on sign consistency across eras, not on the mean, and a
#   mean can be carried by a single strong fold — the momentum-inclusive
#   benchmark set reaches +0.08 at 3 months while inverting in 1976-86.
MIN_USABLE_IC = 0.05
MAX_NEGATIVE_FOLDS = 1


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
    # Per-fold out-of-sample ICs, so sign consistency can be judged and
    # shown rather than inferred from the mean.
    oos_fold_ics: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_usable_signal(self) -> bool:
        """Whether the measured evidence supports showing a reading at all.

        Decided from this panel's own out-of-sample measurement, never
        from a number written into the code — the number written here
        before was +0.157, and it stopped being true when the panel was
        rebuilt.
        """
        if self.oos_ic is None or not self.oos_fold_ics:
            return False
        negative = sum(ic <= 0 for ic in self.oos_fold_ics)
        return self.oos_ic >= MIN_USABLE_IC and negative <= MAX_NEGATIVE_FOLDS

    @property
    def evidence_summary(self) -> str:
        """One line describing the measurement behind the gate."""
        if self.oos_ic is None:
            return "out-of-sample IC could not be measured"
        positive = sum(ic > 0 for ic in self.oos_fold_ics)
        return (
            f"out-of-sample IC {self.oos_ic:+.3f}, positive in {positive} "
            f"of {len(self.oos_fold_ics)} purged folds, "
            f"{self.horizon}-month horizon"
        )

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
    oos_ic, fold_ics = None, []
    if measure_ic:
        fold_ics = _measure_oos_fold_ics(X, y, horizon)
        oos_ic = float(np.mean(fold_ics)) if fold_ics else None
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
        oos_folds=len(fold_ics),
        oos_fold_ics=fold_ics,
        notes=notes,
    )


def _measure_oos_fold_ics(
    X: pd.DataFrame, y: pd.Series, horizon: int
) -> list[float]:
    """Per-fold out-of-sample IC under the benchmark's purged CV.

    Returned per fold rather than averaged, because the gate needs sign
    consistency and a mean can be carried by one strong fold.
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
    return ics
