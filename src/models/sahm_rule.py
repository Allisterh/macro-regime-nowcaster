"""Sahm Rule recession indicator.

The Sahm rule (Claudia Sahm, 2019) triggers when the 3-month moving
average of the national unemployment rate rises by 0.50 percentage
points or more relative to its minimum over the prior 12 months.

This module provides both the raw Sahm indicator value and a logistic-
smoothed recession probability suitable for ensemble use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sahm_indicator(unrate: pd.Series) -> pd.Series:
    """Compute the Sahm rule indicator (continuous value).

    Parameters
    ----------
    unrate : pd.Series
        Monthly unemployment rate (e.g. FRED series ``UNRATE``).
        Must be in percentage-point units (e.g. 3.5, not 0.035).

    Returns
    -------
    pd.Series
        Sahm indicator value at each month.  Values >= 0.50 signal
        recession onset in real time.
    """
    ma3 = unrate.rolling(window=3, min_periods=2).mean()
    min12 = unrate.rolling(window=12, min_periods=6).min()
    return ma3 - min12


def sahm_recession_signal(
    unrate: pd.Series,
    threshold: float = 0.50,
) -> pd.Series:
    """Binary Sahm rule signal: 1 when triggered, 0 otherwise."""
    indicator = sahm_indicator(unrate)
    return (indicator >= threshold).astype(int)


def sahm_recession_probability(
    unrate: pd.Series,
    threshold: float = 0.50,
    scale: float = 0.15,
) -> pd.Series:
    """Logistic-smoothed recession probability from the Sahm indicator.

    Uses a logistic function centred at ``threshold`` with width
    ``scale`` to map the Sahm indicator to a [0, 1] probability.
    This avoids a hard 0/1 jump and provides a smooth signal for
    the ensemble weighting.

    Parameters
    ----------
    unrate : pd.Series
        Monthly unemployment rate in percentage points.
    threshold : float
        Sahm rule threshold (default 0.50 pp).
    scale : float
        Logistic scale parameter.  Smaller values → sharper transition.

    Returns
    -------
    pd.Series
        P(recession) at each month, in [0, 1].
    """
    indicator = sahm_indicator(unrate)
    prob = 1.0 / (1.0 + np.exp(-(indicator - threshold) / scale))
    return prob
