"""Pytest hook: auto-parametrize (path, attr) for any subclass of
_StandardChecks. Reads ARTIFACT_NAME / SCHEMA / FILES from the
collected test class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))

from acceptance.model._template import _StandardChecks, ARTIFACTS  # noqa: E402


def pytest_generate_tests(metafunc):
    """If the test class subclasses _StandardChecks, expand the
    (path × attr) parametrize matrix at collection time."""
    cls = getattr(metafunc, "cls", None)
    if cls is None or not issubclass(cls, _StandardChecks):
        return
    files = [ARTIFACTS / f for f in cls.FILES]
    attrs = list(cls.SCHEMA.keys())
    if "path" in metafunc.fixturenames:
        metafunc.parametrize(
            "path",
            [pytest.param(p, id=p.name) for p in files],
        )
    if "attr" in metafunc.fixturenames:
        metafunc.parametrize("attr", attrs)
