"""Time-series transformations for macroeconomic indicators.

Each transform takes a ``pd.Series`` and returns a ``pd.Series`` of the same
length, with the first value as ``NaN`` where a lag is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Elementary transforms
# ---------------------------------------------------------------------------


def log_difference(series: pd.Series) -> pd.Series:
    """Compute ln(x_t / x_{t-1})."""
    result = np.log(series / series.shift(1))
    return result


def first_difference(series: pd.Series) -> pd.Series:
    """Compute x_t - x_{t-1}."""
    return series.diff()


def percent_change(series: pd.Series) -> pd.Series:
    """Compute (x_t - x_{t-1}) / x_{t-1}."""
    return series.pct_change()


def standardize(series: pd.Series) -> pd.Series:
    """Standardise to zero mean and unit variance (ddof=1) using full-sample stats.

    WARNING: introduces look-ahead when used on temporal data intended for
    backtesting or real-time inference, because the mean and std are computed
    over the entire series including observations that post-date any given
    point.  Use :func:`standardize_expanding` in the modelling pipeline;
    reserve this helper for exploratory analysis, tests, or static panels.
    """
    mean = series.mean()
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return series - mean
    return (series - mean) / std


def standardize_expanding(
    series: pd.Series,
    min_periods: int = 24,
) -> pd.Series:
    """Standardise using only data available up to each point in time.

    For index *t*, the mean and std are computed over observations
    in ``series.iloc[:t+1]`` only.  This removes the look-ahead bias
    introduced by :func:`standardize` when the series is used in a
    time-series model.

    Values before ``min_periods`` observations are returned as NaN so
    downstream models do not consume a scale estimated from too few points.

    Parameters
    ----------
    series : pd.Series
        Series to standardise.  NaN entries are skipped in the rolling
        statistics but preserved in the output.
    min_periods : int
        Minimum valid observations required before producing a z-score.

    Returns
    -------
    pd.Series
        Expanding-window standardised series.
    """
    # pandas' expanding() skips NaN when computing mean/std — we shift by 1
    # so that each t uses statistics from [0, t] inclusive but NEVER from t+1.
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std(ddof=1)
    # Replace zero / NaN std with 1 to avoid division errors; the
    # resulting value will be (x - mean) which is still well-defined.
    std = std.where((std != 0) & std.notna(), 1.0)
    z = (series - mean) / std
    # Mask entries before min_periods have accumulated to NaN explicitly
    # (the expanding mean already returns NaN there, which propagates).
    return z


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_TRANSFORM_MAP: dict[str, callable] = {
    "log_diff": log_difference,
    "diff": first_difference,
    "pct_change": percent_change,
    "standardize": standardize,
    "none": lambda s: s,
}


def apply_transform(series: pd.Series, transform_name: str) -> pd.Series:
    """Apply a named transform to a single series.

    Raises ``ValueError`` for unknown transform names.
    """
    func = _TRANSFORM_MAP.get(transform_name)
    if func is None:
        raise ValueError(
            f"Unknown transform '{transform_name}'. "
            f"Choose from: {sorted(_TRANSFORM_MAP.keys())}"
        )
    return func(series)


def apply_all_transforms(
    df: pd.DataFrame,
    transform_map: dict[str, str],
    *,
    standardize_after: bool = True,
    standardize_mode: str = "expanding",
    min_periods: int = 24,
) -> pd.DataFrame:
    """Apply per-column transforms according to *transform_map*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data panel (columns = series).
    transform_map : dict[str, str]
        Mapping ``{column_name: transform_name}``.
    standardize_after : bool
        If ``True``, standardise each column after its transform.
    standardize_mode : {"expanding", "full"}
        ``"expanding"`` (default) uses :func:`standardize_expanding` to avoid
        look-ahead bias; ``"full"`` uses full-sample statistics (only
        appropriate for static analysis, not for time-series modelling).
    min_periods : int
        Forwarded to :func:`standardize_expanding` when applicable.
    """
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        t_name = transform_map.get(col, "none")
        transformed = apply_transform(df[col].copy(), t_name)
        if standardize_after:
            if standardize_mode == "expanding":
                transformed = standardize_expanding(
                    transformed, min_periods=min_periods,
                )
            elif standardize_mode == "full":
                transformed = standardize(transformed)
            else:
                raise ValueError(
                    f"Unknown standardize_mode '{standardize_mode}'. "
                    "Choose 'expanding' or 'full'."
                )
        result[col] = transformed
    return result
