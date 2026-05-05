"""Standard artifact-acceptance test template.

User mandate (2026-05-04): every attribute on every artifact gets a
test for every applicable check kind. Every artifact must follow the
SAME standard template so coverage is uniform.

Usage in subclasses (e.g. test_panel_ltr_template.py):

    from acceptance.model._template import StandardArtifactTests
    from acceptance.model.schemas import PANEL_LTR_SCHEMA

    class TestPanelLTR(StandardArtifactTests):
        ARTIFACT_NAME = "panel-ltr"
        SCHEMA        = PANEL_LTR_SCHEMA
        FILES         = ["panel-ltr.json", "panel-ltr.golden-daily.json"]
        SKIP_KIND     = {"panel_transformer"}   # transformer shim has different schema

The harness uses pytest_generate_tests to expand the (file × attribute)
matrix at collection time, producing one test_* per check kind per
attribute per file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "backtesting" / "renquant_104" / "artifacts"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


class StandardArtifactTests:
    """Mixin: subclass with ARTIFACT_NAME, SCHEMA, FILES.

    Every subclass automatically receives 9 standard check methods:
      test_presence, test_type, test_finite, test_lower_bound,
      test_upper_bound, test_non_empty, test_unique, test_min_len,
      test_allowed, test_validator
    each parametrized over (file × attribute), so a schema with N
    attributes × F files yields up to 10×N×F test cases.

    A subclass may override SKIP_KIND with kinds to skip (e.g. shim
    variants with different schemas).
    """

    ARTIFACT_NAME: ClassVar[str] = "<unset>"
    SCHEMA:        ClassVar[dict[str, dict]] = {}
    FILES:         ClassVar[list[str]] = []
    SKIP_KIND:     ClassVar[set[str]] = set()

    # ── helpers ────────────────────────────────────────────────────────────

    @classmethod
    def _resolved_paths(cls) -> list[Path]:
        return [ARTIFACTS / f for f in cls.FILES]

    @classmethod
    def _payload_or_skip(cls, path: Path) -> dict:
        d = _load(path)
        if d is None:
            pytest.skip(f"{path.name} missing/corrupt")
        if d.get("kind") in cls.SKIP_KIND:
            pytest.skip(f"{path.name} kind={d.get('kind')} excluded")
        return d

    # ── parametrize matrix at collection-time via pytest hook ─────────────

    def pytest_generate_tests(self, metafunc):
        # Filled in by the conftest helper below.
        pass


def parametrize_path_attr(metafunc, files: list[Path], attrs: list[str]):
    """Helper invoked from a conftest's pytest_generate_tests hook:
    parametrizes `path` and `attr` arguments on any test that needs them."""
    if "path" in metafunc.fixturenames:
        metafunc.parametrize(
            "path",
            [pytest.param(p, id=p.name) for p in files],
        )
    if "attr" in metafunc.fixturenames:
        metafunc.parametrize("attr", attrs)


# ── The 10 standard check methods ────────────────────────────────────────

class _StandardChecks(StandardArtifactTests):
    """Concrete test methods. Subclass StandardArtifactTests with this mixin
    to get all the standard assertions auto-generated."""

    def test_presence(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if spec.get("required", False):
            assert attr in d, f"{path.name}: required attribute {attr!r} missing"

    def test_type(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if attr not in d:
            pytest.skip(f"{attr} absent")
        v = d[attr]
        assert isinstance(v, spec["type"]), (
            f"{path.name}: {attr} type {type(v).__name__} ≠ "
            f"expected {spec['type']}"
        )

    def test_finite(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if not spec.get("finite", False):
            pytest.skip(f"{attr} not flagged finite")
        if attr not in d or d[attr] is None:
            pytest.skip(f"{attr} absent or None")
        v = d[attr]
        assert math.isfinite(float(v)), \
            f"{path.name}: {attr}={v} is NaN/inf"

    def test_lower_bound(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if "bounds" not in spec or attr not in d or d[attr] is None:
            pytest.skip(f"{attr} no bounds or absent")
        v = d[attr]
        if not isinstance(v, (int, float)):
            pytest.skip(f"{attr} non-numeric")
        lo, _ = spec["bounds"]
        assert v >= lo, \
            f"{path.name}: {attr}={v} below lower bound {lo}"

    def test_upper_bound(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if "bounds" not in spec or attr not in d or d[attr] is None:
            pytest.skip(f"{attr} no bounds or absent")
        v = d[attr]
        if not isinstance(v, (int, float)):
            pytest.skip(f"{attr} non-numeric")
        _, hi = spec["bounds"]
        assert v <= hi, \
            f"{path.name}: {attr}={v} above upper bound {hi}"

    def test_non_empty(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if not spec.get("non_empty", False):
            pytest.skip(f"{attr} not flagged non_empty")
        if attr not in d:
            pytest.skip(f"{attr} absent")
        assert len(d[attr]) > 0, f"{path.name}: {attr} is empty"

    def test_unique(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if not spec.get("unique", False):
            pytest.skip(f"{attr} not flagged unique")
        if attr not in d:
            pytest.skip(f"{attr} absent")
        v = d[attr]
        # Convert to hashable for uniqueness check
        try:
            uniq = len(set(v))
        except TypeError:
            uniq = len({tuple(x) if isinstance(x, list) else x for x in v})
        assert uniq == len(v), \
            f"{path.name}: {attr} has duplicates"

    def test_min_len(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        if "min_len" not in spec or attr not in d:
            pytest.skip(f"{attr} no min_len or absent")
        m = spec["min_len"]
        assert len(d[attr]) >= m, \
            f"{path.name}: {attr} len={len(d[attr])} below min_len={m}"

    def test_allowed(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        allowed = spec.get("allowed") or spec.get("allowed_optional")
        if allowed is None or attr not in d:
            pytest.skip(f"{attr} no allowed set or absent")
        v = d[attr]
        assert v in allowed, \
            f"{path.name}: {attr}={v!r} not in allowed {sorted(str(x) for x in allowed)}"

    def test_validator(self, path, attr):
        d = self._payload_or_skip(path)
        spec = self.SCHEMA[attr]
        validator = spec.get("validator")
        if validator is None or attr not in d or d[attr] is None:
            pytest.skip(f"{attr} no validator or absent")
        validator(d[attr])


__all__ = [
    "StandardArtifactTests",
    "_StandardChecks",
    "parametrize_path_attr",
    "_load",
    "ARTIFACTS",
]
