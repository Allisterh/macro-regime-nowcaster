"""requirements.txt must not contradict pyproject.toml.

The project declares its dependencies twice: pyproject.toml, which the
README, Makefile and CI install from, and requirements.txt, which
deployment targets such as Streamlit Community Cloud read by default.

When the dashboard moved to Streamlit's ``width="stretch"`` — available
only from 1.50 — pyproject.toml's floor was raised and requirements.txt
was not. It kept ``streamlit>=1.28``, so an install from it could pull a
Streamlit that crashes at the first chart. Nothing failed, because
nothing compared the two files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# tomllib is standard library from Python 3.11; the project supports
# 3.10, where this check skips rather than failing at import.
tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[1]


def _requirements_txt() -> dict[str, Requirement]:
    out = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        req = Requirement(line)
        out[canonicalize_name(req.name)] = req
    return out


def _pyproject() -> dict[str, Requirement]:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    specs = list(project["dependencies"])
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    return {canonicalize_name(Requirement(s).name): Requirement(s) for s in specs}


def test_shared_packages_have_the_same_version_constraint():
    txt, toml = _requirements_txt(), _pyproject()
    shared = sorted(set(txt) & set(toml))
    assert shared, "no packages in common; the parser is probably broken"

    mismatched = [
        f"{name}: requirements.txt '{txt[name].specifier}' vs "
        f"pyproject.toml '{toml[name].specifier}'"
        for name in shared
        if txt[name].specifier != toml[name].specifier
    ]
    assert not mismatched, (
        "requirements.txt and pyproject.toml disagree:\n  "
        + "\n  ".join(mismatched)
    )


def test_every_runtime_dependency_is_in_requirements_txt():
    """A deploy that installs from requirements.txt must get a working app."""
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    runtime = {canonicalize_name(Requirement(s).name) for s in project["dependencies"]}
    missing = sorted(runtime - set(_requirements_txt()))
    assert not missing, (
        f"runtime dependencies missing from requirements.txt: {missing}"
    )
