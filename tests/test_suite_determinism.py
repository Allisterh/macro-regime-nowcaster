"""The test suite's own results must not depend on collection order.

``tests/conftest.py`` defined one module-level ``RNG = default_rng(42)``
and every session-scoped fixture drew from it.  Session fixtures are
built lazily, in the order tests first request them, so how many draws
had already been consumed — and therefore what data a fixture contained
— depended on which tests were selected.

``synthetic_panel`` was one panel under ``pytest tests/`` and a
different one under ``pytest tests/test_dynamic_factor_model.py``.
``test_dfm_rotated_factors_are_distinct`` passed in the first case and
failed at |r| = 0.785 in the second, and it had been asserting the
opposite of what varimax does for as long as the full-suite ordering
happened to hide it.

A green run was partly a statement about collection order.  These tests
are cheap and they protect every other result in the suite.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"


@pytest.fixture(scope="module")
def conftest_source() -> str:
    return CONFTEST.read_text(encoding="utf-8")


def test_no_module_level_generator_is_shared_between_fixtures(conftest_source):
    """A generator created at import time is shared state across fixtures."""
    tree = ast.parse(conftest_source)

    offenders: list[str] = []
    for node in tree.body:  # module level only — nested ones are private
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        if "default_rng" in name or "RandomState" in name:
            targets = ", ".join(ast.unparse(t) for t in node.targets)
            offenders.append(f"line {node.lineno}: {targets} = {name}(...)")

    assert not offenders, (
        "conftest.py creates a random generator at module scope. Every "
        "fixture reading it shares one stream, so fixture contents depend "
        "on which tests were collected and in what order. Give each "
        "fixture its own seeded generator instead:\n  "
        + "\n  ".join(offenders)
    )


def test_every_fixture_that_randomises_seeds_itself(conftest_source):
    """Each fixture using randomness must create its own generator."""
    tree = ast.parse(conftest_source)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_fixture = any(
            "fixture" in ast.unparse(dec) for dec in node.decorator_list
        )
        if not is_fixture:
            continue

        body = ast.unparse(node)
        uses_randomness = re.search(r"\brng\b|\bnp\.random\b", body)
        if not uses_randomness:
            continue

        seeds_itself = re.search(r"=\s*(_rng\(|np\.random\.default_rng\()", body)
        if not seeds_itself:
            offenders.append(f"{node.name} (line {node.lineno})")

    assert not offenders, (
        "these fixtures use randomness without creating their own "
        "generator, so their contents depend on collection order:\n  "
        + "\n  ".join(offenders)
    )
