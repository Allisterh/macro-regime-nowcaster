"""The dashboard must not hard-code values the model owns.

This has now gone wrong twice. The sidebar advertised ensemble weights
of 0.25/0.50/0.25 with no Sahm signal long after the model had moved to
four signals at different weights. Then the "Latent Factors" slider kept
``value=4`` after the validated default became 5, so the dashboard ran a
configuration nobody had measured — and at 4 it also truncates
``DEFAULT_FACTOR_NAMES``, silently dropping the long-rates factor.

Neither failed: both rendered a plausible number. These tests read the
dashboard source rather than importing it, because importing executes
Streamlit at module scope.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.models.nowcaster import Nowcaster

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def test_factor_slider_default_comes_from_the_model(source):
    """The slider must derive its default, not restate it."""
    match = re.search(r'st\.slider\(\s*"Latent Factors".*?\)', source, re.S)
    assert match, "could not find the Latent Factors slider"
    slider = match.group(0)

    # The default may be computed a few lines above the call, so look at
    # the surrounding block rather than the call alone.
    block = source[max(0, match.start() - 600) : match.end()]

    assert "DEFAULT_FACTOR_NAMES" in block, (
        "the Latent Factors slider does not derive its default from "
        "Nowcaster.DEFAULT_FACTOR_NAMES; a literal here goes stale silently"
    )
    # Anchor so min_value=/max_value= are not mistaken for the default.
    assert not re.search(r"(?<![_\w])value\s*=\s*\d", slider), (
        f"the slider hard-codes a numeric default: {slider}"
    )


def test_sidebar_weights_come_from_the_model(source):
    """Weights are read from the model rather than restated in the UI."""
    assert "DEFAULT_WEIGHTS" in source, (
        "the sidebar does not read Nowcaster.DEFAULT_WEIGHTS"
    )
    # The superseded hard-coded values must not reappear as literals.
    for stale in ("RSM weight:** 0.25", "Probit weight:** 0.50"):
        assert stale not in source, f"stale hard-coded weight text: {stale}"


def test_signal_count_excludes_the_ensemble_entry(source):
    """'N active' once counted the ensemble itself, reporting 4 signals as 5."""
    assert 'len(result.ensemble_detail)} active' not in source, (
        "the signal count is back to len(ensemble_detail), which includes "
        "the 'ensemble' key and counts zero-weighted components as active"
    )


def test_default_factor_names_are_not_duplicated_in_the_dashboard(source):
    """Factor names belong to the model; the dashboard must not restate them."""
    literal = '"' + '", "'.join(Nowcaster.DEFAULT_FACTOR_NAMES) + '"'
    assert literal not in source, (
        "the dashboard hard-codes the factor name list; it should pass "
        "nothing and let Nowcaster apply its own defaults"
    )
