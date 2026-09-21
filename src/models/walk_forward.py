"""Point-in-time feature generation for downstream models.

The rest of the codebase fits a model once over the full sample and
reads back a history.  That history is **not** what the model would have
produced in real time: the DFM's EM parameters, its varimax rotation and
factor normalisation, the RSM's transition matrix, and the probit's
coefficients are all estimated using the whole sample, so a value at
month *t* embeds information from months after *t*.  Measured on this
repository's own cached data, re-running the nowcaster six years later
moved 21% of historical months by more than 0.10.

This module builds the history the other way round: one fit per as-of
date, keeping only the final row of each.  That row is safe because at
*t = T* the Kalman smoother coincides with the filter and the Kim
smoother coincides with the Hamilton filter, so no future information can
reach it.

Usage
-----
>>> from src.models.walk_forward import generate_feature_panel
>>> feats = generate_feature_panel(pipeline, start="2000-01-31", end="2024-12-31")
>>> feats.to_parquet("data/features.parquet")

The result is suitable as an input to a downstream model.  Join it to a
target on ``knowable_at``, never on the index: the index is the reference
month, while ``knowable_at`` is when the row could first have been
computed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.nowcaster import Nowcaster
from src.models.regime_backtest import nber_labels_available_at

# Longest publication lag in fred_series.yaml (CSUSHPISA, 60 days).  A row
# dated month-end T is not fully computable until every input series for
# month T has printed, so this is the offset from the reference month to
# the date the row could first have been built.
MAX_PUBLICATION_LAG = pd.Timedelta(days=60)

# Momentum horizons, in months, for factor and probability changes.
_DELTA_HORIZONS = (1, 3, 6)

# The number of factors belongs to the model, not to this module.  A
# literal here read 4 after the validated default moved to 5, so the
# documented way to rebuild the downstream panel produced a
# configuration nobody had measured — and with five names competing for
# four factors the assignment quietly dropped `long_rates` altogether.
_DEFAULT_N_FACTORS = len(Nowcaster.DEFAULT_FACTOR_NAMES)


def _safe_delta(series: pd.Series, months: int) -> float:
    """Change over *months*, or NaN when the history is too short."""
    if len(series) <= months:
        return float("nan")
    current, past = series.iloc[-1], series.iloc[-1 - months]
    if pd.isna(current) or pd.isna(past):
        return float("nan")
    return float(current - past)


def _regime_age(prob_recession: pd.Series, threshold: float = 0.5) -> float:
    """Months since the recession probability last crossed *threshold*.

    A slow-moving state variable is more informative when the model also
    knows how long the state has held — hazard rates are not flat.
    """
    if prob_recession.empty:
        return float("nan")
    state = (prob_recession > threshold).astype(int)
    current = state.iloc[-1]
    age = 0
    for value in reversed(state.values):
        if value != current:
            break
        age += 1
    return float(age)


def generate_features_asof(
    pipeline: Any,
    as_of: str | pd.Timestamp,
    *,
    n_factors: int = _DEFAULT_N_FACTORS,
    n_regimes: int = 2,
    factor_names: list[str] | None = None,
    regime_labels: list[str] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    probit_config: dict | None = None,
    respect_nber_announcement_lag: bool = True,
) -> dict[str, float]:
    """Build one row of point-in-time features for *as_of*.

    Everything here is computed from a model fitted on data through
    *as_of* only, and only the final row of that fit is read.

    Parameters
    ----------
    pipeline : DataPipeline
        Must rebuild its panel per call so the ragged edge is masked
        against *as_of* rather than against today.
    respect_nber_announcement_lag : bool
        When ``True`` (default), the supervised probit may only train on
        NBER labels that had actually been announced by *as_of*.  The
        turning points themselves are published 5-21 months late, so
        using the final dated table is look-ahead even when the label's
        own timestamp precedes *as_of*.

    Returns
    -------
    dict[str, float]
        Feature name → value.  Always includes ``knowable_at``.
    """
    as_of = pd.Timestamp(as_of)

    probit_train_end: pd.Timestamp | None = None
    if respect_nber_announcement_lag:
        available = nber_labels_available_at(as_of)
        probit_train_end = (
            available.index[-1] if len(available) else None
        )

    nowcaster = Nowcaster(
        pipeline=pipeline,
        n_factors=n_factors,
        n_regimes=n_regimes,
        factor_names=factor_names,
        regime_labels=regime_labels,
        use_ensemble=True,
        ensemble_weights=ensemble_weights,
        probit_config=probit_config,
        probit_train_end=probit_train_end,
        use_filtered_factors=True,
    )

    result = nowcaster.run(end_date=str(as_of.date()))

    row: dict[str, float] = {}

    # --- Latent factors and their momentum ---
    factors = nowcaster._last_factors

    # A fit that failed must not be written as a row of zeros.
    #
    # 25 of the 690 rows in the previous panel had every factor exactly
    # 0.0, `signal_rsm` exactly 0.5 and `p_stay_recession` exactly 0.0:
    # the DFM had returned zeros instead of raising, the RSM had then
    # been fitted on a degenerate input, and the row was written with a
    # complete set of plausible-looking numbers.  They clustered in 2020
    # and later — 11 of the last 25 months — so the fabricated rows sat
    # exactly where a downstream model would weight them most.
    #
    # Standardised factors are never all exactly zero on real data, so
    # this costs nothing and turns a silent fabrication into a skipped
    # window that `generate_feature_panel` logs.
    final_factors = factors.iloc[-1].to_numpy(dtype=float)
    if np.all(np.abs(np.nan_to_num(final_factors)) < 1e-12):
        raise RuntimeError(
            f"DFM returned all-zero factors at {as_of.date()}; the fit "
            f"failed without raising. Refusing to emit a fabricated row."
        )

    for name in factors.columns:
        row[f"factor_{name}"] = float(factors[name].iloc[-1])
        for h in _DELTA_HORIZONS:
            row[f"factor_{name}_d{h}m"] = _safe_delta(factors[name], h)

    # --- Ensemble components, unweighted ---
    # The blend in Nowcaster.DEFAULT_WEIGHTS is a judgement call; a downstream
    # model is better placed to learn the combination, and to learn that
    # one of the components is uninformative.
    detail = result.ensemble_detail or {}
    for signal in ("rsm", "probit", "cfnai", "sahm"):
        row[f"signal_{signal}"] = float(detail.get(signal, np.nan))

    component_values = [
        v for k, v in detail.items()
        if k != "ensemble" and not pd.isna(v)
    ]
    # Disagreement across signals is a genuine model-uncertainty proxy,
    # and often more useful than the mean for sizing decisions.
    row["signal_dispersion"] = (
        float(np.std(component_values)) if len(component_values) > 1
        else float("nan")
    )

    # --- Ensemble probability, level and change ---
    row["p_recession"] = float(result.recession_probability)
    ensemble_ts = nowcaster._ensemble_recession_ts
    if ensemble_ts is not None and len(ensemble_ts):
        for h in _DELTA_HORIZONS:
            row[f"p_recession_d{h}m"] = _safe_delta(ensemble_ts, h)
        row["regime_age_months"] = _regime_age(ensemble_ts)
    else:
        for h in _DELTA_HORIZONS:
            row[f"p_recession_d{h}m"] = float("nan")
        row["regime_age_months"] = float("nan")

    # --- Regime persistence from the Markov chain ---
    try:
        P = nowcaster._rsm.get_transition_matrix()
        stay_rec = float(P[0, 0])
        row["p_stay_recession"] = stay_rec
        # Expected remaining duration of a geometric holding time,
        # saturated at the length of the sample.
        #
        # `1 / (1 - p)` was guarded only against p == 1.0 exactly, which
        # float64 almost never produces: an estimated p of
        # 0.9999999999946 cleared the guard and returned 1.9e11 months.
        # Eight rows of the previous panel exceeded 1e6 and the largest
        # was 7.5e11 — one feature whose scale would have swamped every
        # other column under any standardisation a downstream model
        # applied, while reading as a finite number throughout.
        #
        # The cap is substantive, not cosmetic. A holding time longer
        # than the sample is not identified by the sample: with T monthly
        # observations, p = 1 - 1/T and p = 1 - 1/(100T) imply the same
        # data. Saturating at T states the most the estimate can support.
        max_duration = float(max(len(factors), 1))
        row["expected_recession_duration"] = (
            min(1.0 / (1.0 - stay_rec), max_duration)
            if stay_rec < 1.0
            else max_duration
        )
        row["p_enter_recession"] = float(P[-1, 0])
    except Exception:
        row["p_stay_recession"] = float("nan")
        row["expected_recession_duration"] = float("nan")
        row["p_enter_recession"] = float("nan")

    # --- GDP nowcast ---
    row["gdp_nowcast"] = float(result.gdp_nowcast)
    row["gdp_ci_width"] = float(result.gdp_ci_upper - result.gdp_ci_lower)

    # --- Timing ---
    # Join downstream targets on this, not on the index: the index is the
    # reference month, this is when the row could first have been built.
    row["knowable_at"] = as_of + MAX_PUBLICATION_LAG

    return row


def generate_feature_panel(
    pipeline: Any,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    step_months: int = 1,
    cache_path: str | Path | None = None,
    on_error: str = "warn",
    max_failure_fraction: float = 0.05,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
    **feature_kwargs: Any,
) -> pd.DataFrame:
    """Walk forward from *start* to *end*, one model fit per date.

    This is deliberately slow — it refits the DFM at every step — so the
    result is cached incrementally.  Re-running with the same
    ``cache_path`` resumes rather than recomputing.

    Parameters
    ----------
    step_months : int
        Months between evaluation dates.  Use 1 for a monthly feature
        panel; larger steps are for quick diagnostic runs.
    cache_path : str | Path | None
        CSV written after every row.  Existing rows are reused.
    on_error : {"warn", "raise"}
        Whether a failed window aborts the run.
    max_failure_fraction : float
        Raise at the end if more than this share of windows were skipped.
        With ``on_error="warn"`` a failing window is a logged warning and
        nothing else, so a run can drop a large contiguous stretch of
        history and still finish with a cheerful row count: pointing the
        pipeline at 1980 while asking for as-of dates from 1967 skipped
        120 consecutive windows — two recessions — and reported success.
        A panel missing a sixth of its span is a different object from
        the one that was asked for, and should say so.

    Returns
    -------
    pd.DataFrame
        Indexed by reference month-end, with a ``knowable_at`` column.
    """
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today()
    dates = pd.date_range(pd.Timestamp(start), end, freq=f"{step_months}ME")

    cached: dict[pd.Timestamp, dict] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None and cache_file.exists():
        prior = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # A cache from a superseded schema must not be resumed from.
        #
        # Every row this function emits carries `knowable_at`, so its
        # absence means the file was written by different code. One was
        # sitting in the repository: data/oos_validation_interim.csv, in
        # the old `p_rsm, p_probit, p_cfnai, p_ensemble` format, with an
        # inverted RSM at 0.98 and a probit pinned to its since-removed
        # 0.05 clip floor. Resuming from it would have merged 28 rows of
        # known-bad numbers, under column names nothing downstream reads,
        # into a fresh run — and the metrics computed over the result
        # would have looked entirely ordinary.
        if "knowable_at" not in prior.columns:
            logger.warning(
                f"Ignoring {cache_file}: it has no knowable_at column, so it "
                f"was written by a superseded version of this function "
                f"(columns: {list(prior.columns)[:6]}). Recomputing from "
                f"scratch; delete the file to silence this."
            )
        else:
            prior["knowable_at"] = pd.to_datetime(prior["knowable_at"])
            cached = {ts: r for ts, r in prior.to_dict("index").items()}
            logger.info(
                f"Resuming from {len(cached)} cached rows in {cache_file}"
            )

    rows: dict[pd.Timestamp, dict] = dict(cached)
    total = len(dates)
    failures: list[pd.Timestamp] = []

    for i, as_of in enumerate(dates, start=1):
        if as_of in rows:
            continue
        if progress is not None:
            progress(i, total, as_of)
        logger.info(f"Walk-forward [{i}/{total}]: fitting as of {as_of.date()}")
        try:
            rows[as_of] = generate_features_asof(
                pipeline, as_of, **feature_kwargs
            )
        except Exception as exc:
            if on_error == "raise":
                raise
            logger.warning(f"Walk-forward failed at {as_of.date()}: {exc}")
            failures.append(as_of)
            continue

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(rows, orient="index").sort_index().to_csv(
                cache_file
            )

    if not rows:
        raise RuntimeError("No successful walk-forward windows")

    if failures:
        share = len(failures) / max(total, 1)
        logger.warning(
            f"Walk-forward skipped {len(failures)} of {total} windows "
            f"({share:.1%}), from {failures[0].date()} to "
            f"{failures[-1].date()}"
        )
        if share > max_failure_fraction:
            raise RuntimeError(
                f"{len(failures)} of {total} walk-forward windows failed "
                f"({share:.1%} > {max_failure_fraction:.1%}), spanning "
                f"{failures[0].date()} to {failures[-1].date()}. The panel "
                f"would be missing that history without saying so. Check the "
                f"first warning above; a long run of failures at the start "
                f"usually means the pipeline's start_date is later than the "
                f"first as-of date. Pass max_failure_fraction to accept it."
            )

    panel = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    panel.index.name = "reference_month"
    return panel


def assert_point_in_time(
    pipeline: Any,
    early_cutoff: str | pd.Timestamp,
    late_cutoff: str | pd.Timestamp,
    *,
    step_months: int = 3,
    start: str | pd.Timestamp = "2000-01-31",
    tolerance: float = 1e-8,
    **feature_kwargs: Any,
) -> pd.DataFrame:
    """Verify that generated features do not change as data is appended.

    Generates the panel twice — once through *early_cutoff*, once through
    *late_cutoff* — and compares the overlap.  A point-in-time generator
    produces identical values; anything else means future information is
    reaching the history.

    This is the gate a feature table should pass before a downstream
    model consumes it.

    Returns
    -------
    pd.DataFrame
        Per-column maximum absolute difference over the overlap.

    Raises
    ------
    AssertionError
        If any column differs by more than *tolerance*.
    """
    early = generate_feature_panel(
        pipeline, start, early_cutoff, step_months=step_months, **feature_kwargs
    )
    late = generate_feature_panel(
        pipeline, start, late_cutoff, step_months=step_months, **feature_kwargs
    )

    common_idx = early.index.intersection(late.index)
    numeric = [
        c for c in early.columns
        if c != "knowable_at" and pd.api.types.is_numeric_dtype(early[c])
    ]
    diffs = (
        (early.loc[common_idx, numeric] - late.loc[common_idx, numeric])
        .abs()
        .max()
    )

    offenders = diffs[diffs > tolerance]
    if len(offenders):
        raise AssertionError(
            f"Features are not point-in-time: {len(offenders)} column(s) "
            f"changed when data through {pd.Timestamp(late_cutoff).date()} "
            f"was appended to a panel ending "
            f"{pd.Timestamp(early_cutoff).date()}.\n"
            f"{offenders.sort_values(ascending=False).head(10).to_string()}"
        )
    return diffs.to_frame("max_abs_diff")
