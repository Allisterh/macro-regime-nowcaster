"""Estimators for downstream tasks built on the regime feature panel.

One definition, used by both ``scripts/benchmark_features.py`` and the
dashboard's volatility panel.  A second copy of these specifications
would be the same defect this codebase has been fixing all along: a
literal restating something another module owns, going stale in silence.
Whatever the benchmark measured is therefore what the dashboard serves.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.purged_cv import PurgedWalkForward

# Penalty grid. Widening this to 1e-4..1e7 moves out-of-sample R² by
# 0.001 on the volatility target, so the range is not binding even
# though some folds select at its edge.
ALPHA_GRID = np.logspace(-2, 4, 13)


def build_models(label_horizon: int = 3, embargo: int = 3) -> dict:
    """The estimators the benchmark compares, as name -> factory.

    Parameters
    ----------
    label_horizon, embargo : int
        Passed to the *inner* splitter so hyper-parameter selection
        respects the same label overlap as the outer evaluation.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    return {
        "ridge": lambda: build_ridge(label_horizon, embargo),
        # Trees split on order, not magnitude, so the GBM was never
        # affected by the scaling problem described below.
        "gbm": lambda: GradientBoostingRegressor(
            random_state=0, n_estimators=100, max_depth=2
        ),
    }


def build_ridge(label_horizon: int = 3, embargo: int = 3):
    """Standardised ridge whose penalty is chosen by a purged inner loop.

    Two things were wrong with this before, and both changed the numbers.

    It was ``Ridge(alpha=1.0)`` on raw columns. An L2 penalty is not
    scale-invariant — a coefficient scales as ``1/sd``, so the penalty
    bites hardest on the *narrowest* columns — and the panel mixes
    probabilities (sd ~ 0.1) with ``expected_recession_duration``
    (sd ~ 3e10). How hard each feature was shrunk was decided by the
    units it happened to be recorded in. Standardising alone moved the
    full-panel R² on returns from -0.299 to -3.355.

    Then alpha was selected by ``RidgeCV``'s leave-one-out GCV. LOO is
    the wrong instrument for an overlapping forward target: the held-out
    point's label is computed from prices it shares with its immediate
    neighbours, and those neighbours stay in the training set, so the
    held-out error is optimistic and the search is pulled toward too
    small a penalty. On the volatility target that under-regularisation
    cost real accuracy — twelve-month R² was -0.034 under LOO and +0.063
    once the inner search was purged.

    Everything sits inside the ``Pipeline``, so the scaler and the
    search are refitted per training fold and neither sees the test
    fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    inner = PurgedWalkForward(
        n_splits=3,
        label_horizon=label_horizon,
        embargo=embargo,
        # Inner folds are carved out of one outer training fold, so this
        # has to sit well below the outer min_train or the early folds
        # yield no inner splits at all.
        min_train=30,
    )
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": ALPHA_GRID},
        cv=inner,
        scoring="neg_mean_squared_error",
    )
