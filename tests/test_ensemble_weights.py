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
