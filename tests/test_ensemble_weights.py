"""Consistency checks on the ensemble weighting.

The weights were set from a 105-point quarterly walk-forward rather than
by judgement, and they live in two places (``settings.yaml`` and
``Nowcaster.DEFAULT_WEIGHTS``) because the dashboard reads the code
default while the CLI reads the config.  These tests keep the two from
drifting apart, which is how the dashboard came to advertise a
three-signal 0.25/0.50/0.25 split long after the model had moved on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.models.nowcaster import Nowcaster

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


@pytest.fixture(scope="module")
def config_weights() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    return settings["model"]["ensemble"]["weights"]


def test_config_and_code_defaults_agree(config_weights):
    """settings.yaml and Nowcaster.DEFAULT_WEIGHTS must not drift apart."""
    assert config_weights == Nowcaster.DEFAULT_WEIGHTS, (
        "config/settings.yaml and Nowcaster.DEFAULT_WEIGHTS disagree:\n"
        f"  settings.yaml : {config_weights}\n"
        f"  DEFAULT_WEIGHTS: {Nowcaster.DEFAULT_WEIGHTS}\n"
        "The CLI reads the config and the dashboard reads the code default, "
        "so they must be kept in step."
    )


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param(Nowcaster.DEFAULT_WEIGHTS, id="code-default"),
    ],
)
def test_weights_sum_to_one(weights):
    assert abs(sum(weights.values()) - 1.0) < 1e-9, (
        f"ensemble weights must sum to 1, got {sum(weights.values())}"
    )


def test_config_weights_sum_to_one(config_weights):
    assert abs(sum(config_weights.values()) - 1.0) < 1e-9


def test_weights_are_non_negative(config_weights):
    negative = {k: v for k, v in config_weights.items() if v < 0}
    assert not negative, f"negative ensemble weights: {negative}"


def test_all_four_signals_are_declared(config_weights):
    """Every signal keeps an explicit weight, including zeroed ones.

    The Markov-switching signal is weighted 0.0 rather than deleted: it is
    still computed and published in ``ensemble_detail`` so the latching
    defect stays visible, and so restoring a weight is a one-line change
    once it is diagnosed.
    """
    assert set(config_weights) == {"rsm", "probit", "cfnai", "sahm"}


def test_rsm_is_not_silently_reweighted(config_weights):
    """The RSM stays at zero until its real-time latching is fixed.

    Measured over 105 real-time refits: >0.99 in 54 of them, including
    deep expansions (AUC 0.600, Brier 0.574).  At the previous 0.20 weight
    the ensemble scored Brier 0.0842, worse than the 0.0784 of a constant
    forecast at the 8.6% base rate.  If someone restores a positive weight
    they should have to change this test, and re-run the walk-forward.
    """
    assert config_weights["rsm"] == 0.0, (
        "The RSM weight is non-zero again. It was zeroed because the "
        "real-time signal is latched near 1.0 and degrades ensemble "
        "calibration. Re-run scripts/build_features.py and confirm the "
        "Brier score before restoring it."
    )


# ---------------------------------------------------------------------------
# Factor configuration
# ---------------------------------------------------------------------------


def test_factor_names_agree_between_config_and_code(config_weights):  # noqa: ARG001
    """settings.yaml and Nowcaster.DEFAULT_FACTOR_NAMES must not drift."""
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        model_cfg = yaml.safe_load(fh)["model"]

    assert model_cfg["factor_names"] == Nowcaster.DEFAULT_FACTOR_NAMES
    assert model_cfg["n_factors"] == len(Nowcaster.DEFAULT_FACTOR_NAMES)


def test_no_labour_factor_is_claimed_by_default():
    """The default names must not include one the panel cannot support.

    There is no distinct labour factor at any K from 3 to 6 — it scores
    0.19-0.27 against the 90th-percentile loading, because the labour
    series load on real activity. Listing it anyway meant the label was
    assigned regardless and landed on a yield-curve factor, and the
    regime model then tracked interest rates while calling them labour.
    """
    assert "labor_market" not in Nowcaster.DEFAULT_FACTOR_NAMES, (
        "labor_market is back in the default factor names; re-check the "
        "match-quality scores before restoring it"
    )


def test_every_default_factor_name_has_anchors():
    """A name with no anchors cannot be validated, so it must not be a default.

    Unanchored names score NaN and bypass the evidence check entirely —
    which is how the yield-curve and long-rates factors went unnamed and
    were absorbed by whatever label was left over.
    """
    from src.models.dynamic_factor_model import DynamicFactorModel

    anchors = DynamicFactorModel._SIGN_ANCHORS
    missing = [n for n in Nowcaster.DEFAULT_FACTOR_NAMES if not anchors.get(n)]
    assert not missing, f"default factor names without anchor series: {missing}"


def test_anchor_sets_are_disjoint():
    """Overlapping anchors make the name assignment ill-posed.

    PAYEMS once anchored both real_activity and labor_market. When one
    factor dominates both names' anchors the two pairings score
    identically and the tie is broken arbitrarily, so the label lands on
    whichever factor the solver happened to pick.
    """
    from itertools import combinations

    from src.models.dynamic_factor_model import DynamicFactorModel

    anchors = DynamicFactorModel._SIGN_ANCHORS
    for a, b in combinations(anchors, 2):
        shared = set(anchors[a]) & set(anchors[b])
        assert not shared, f"{a} and {b} share anchor series {shared}"


def test_rsm_fits_on_the_cyclical_block_only():
    """The regime variable is the business cycle, not rates or prices."""
    for name in Nowcaster.DEFAULT_RSM_FACTORS:
        assert name in Nowcaster.DEFAULT_FACTOR_NAMES, (
            f"RSM is configured to use '{name}', which is not a default factor"
        )
    assert "yield_curve" not in Nowcaster.DEFAULT_RSM_FACTORS
    assert "inflation" not in Nowcaster.DEFAULT_RSM_FACTORS
