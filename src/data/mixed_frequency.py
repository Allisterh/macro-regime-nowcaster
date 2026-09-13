"""Mixed-frequency panel construction for the DFM.

Implements the Mariano–Murasawa (2003) approach for handling quarterly
series within a monthly state-space model, and frequency-aware alignment
for weekly/daily series.

The key insight: quarterly GDP growth is the (approximate) sum of three
unobserved monthly contributions.  By augmenting the state vector with
two lags of the monthly latent GDP component, the quarterly observation
equation becomes a partial-sum that the Kalman filter handles naturally
via NaN padding on non-quarter-end months.

For weekly/daily series, the module provides two aggregation strategies:
- ``last``: use the last observation within each month (current default)
- ``mean``: use the within-month average (preserves more information)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_mixed_frequency(
    raw: dict[str, pd.Series],
    series_config: list[dict],
    method: str = "last",
) -> pd.DataFrame:
    """Align mixed-frequency series to a monthly panel.

    Parameters
    ----------
    raw : dict[str, pd.Series]
        Raw series at their native frequency.
    series_config : list[dict]
        Per-series configuration dicts with ``code`` and ``frequency``.
    method : str
        Aggregation method for sub-monthly data: ``"last"`` or ``"mean"``.

    Returns
    -------
    pd.DataFrame
        Monthly-frequency panel (rows = month-end dates, columns = codes).
    """
    freq_map = {entry["code"]: entry.get("frequency", "monthly") for entry in series_config}
    frames: list[pd.Series] = []

    for code, s in raw.items():
        freq = freq_map.get(code, "monthly")
        monthly = _to_monthly(s, freq, method)
        monthly.name = code
        frames.append(monthly)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1)
    panel.sort_index(inplace=True)
    return panel


def _to_monthly(
    s: pd.Series,
    freq: str,
    method: str = "last",
) -> pd.Series:
    """Convert a single series to monthly frequency."""
    if freq in ("daily", "weekly"):
        if method == "mean":
            return s.resample("ME").mean()
        return s.resample("ME").last()
    elif freq == "quarterly":
        return _quarterly_to_monthly(s)
    else:
        return s.resample("ME").last()


def _quarterly_to_monthly(s: pd.Series) -> pd.Series:
    """Expand a quarterly series to monthly with NaN padding.

    The quarterly value is placed at the quarter-end month; the two
    preceding months are NaN.  This is the observation-side preparation
    for the Mariano–Murasawa cumulator: the Kalman filter sees the
    quarterly value only at t=3,6,9,12 and skips the NaN months via
    its standard missing-data logic.
    """
    idx = pd.date_range(
        start=s.index.min() - pd.offsets.MonthBegin(2),
        end=s.index.max(),
        freq="ME",
    )
    monthly = pd.Series(np.nan, index=idx, name=s.name, dtype=float)
    for date, val in s.items():
        month_end = date + pd.offsets.MonthEnd(0)
        if month_end in monthly.index:
            monthly.loc[month_end] = val
    return monthly


def build_cumulator_matrices(
    n_factors: int,
    quarterly_columns: list[int],
    n_series: int,
) -> dict:
    """Build the augmented state-space matrices for the MM cumulator.

    The standard DFM state is ``f_t`` (K-dimensional).  For each
    quarterly series, we augment the state with two cumulator lags so
    the quarterly observation at month 3 of the quarter equals
    ``f_t + f_{t-1} + f_{t-2}`` projected through that series' loading.

    This function returns the augmented transition and observation
    structure that the DFM should use when quarterly series are present.

    Parameters
    ----------
    n_factors : int
        Number of DFM factors (K).
    quarterly_columns : list[int]
        Column indices (in the observation panel) that are quarterly.
    n_series : int
        Total number of observed series.

    Returns
    -------
    dict with keys:
        ``n_states_aug`` : int — augmented state dimension
        ``A_aug_template`` : np.ndarray — template for augmented A
        ``C_quarterly_map`` : dict — maps quarterly col → cumulator indices
    """
    K = n_factors
    n_quarterly = len(quarterly_columns)

    # Augmented state: [f_t, f_{t-1}, f_{t-2}] for the cumulator
    # We add 2*K extra states for the two factor lags
    n_states_aug = K + 2 * K if n_quarterly > 0 else K

    # Augmented transition: shift lags
    A_aug = np.zeros((n_states_aug, n_states_aug))
    # f_t block is filled by the DFM's own A matrix
    # f_{t-1} = f_t from previous step
    if n_quarterly > 0:
        A_aug[K:2*K, :K] = np.eye(K)        # f_{t-1} ← f_t
        A_aug[2*K:3*K, K:2*K] = np.eye(K)   # f_{t-2} ← f_{t-1}

    # For quarterly columns, the observation equation at quarter-end is:
    # y_q = C_q @ (f_t + f_{t-1} + f_{t-2})
    # = [C_q  C_q  C_q] @ [f_t; f_{t-1}; f_{t-2}]
    quarterly_map = {}
    for col_idx in quarterly_columns:
        quarterly_map[col_idx] = {
            "factor_slice": slice(0, K),
            "lag1_slice": slice(K, 2*K),
            "lag2_slice": slice(2*K, 3*K),
        }

    return {
        "n_states_aug": n_states_aug,
        "A_aug_template": A_aug,
        "C_quarterly_map": quarterly_map,
    }


def augment_observation_matrix(
    C: np.ndarray,
    quarterly_columns: list[int],
    n_states_aug: int,
    n_factors: int,
) -> np.ndarray:
    """Expand C to the augmented state dimension.

    Monthly series load on ``f_t`` only (first K states).
    Quarterly series load on ``f_t + f_{t-1} + f_{t-2}`` via the
    cumulator structure.

    Parameters
    ----------
    C : np.ndarray, shape (N, K)
        Original loading matrix.
    quarterly_columns : list[int]
        Indices of quarterly series in the panel.
    n_states_aug : int
        Augmented state dimension (K + 2K if quarterly present).
    n_factors : int
        Number of factors K.

    Returns
    -------
    np.ndarray, shape (N, n_states_aug)
    """
    N, K = C.shape
    C_aug = np.zeros((N, n_states_aug))
    C_aug[:, :K] = C

    for col_idx in quarterly_columns:
        C_aug[col_idx, K:2*K] = C[col_idx, :K]
        C_aug[col_idx, 2*K:3*K] = C[col_idx, :K]

    return C_aug
