"""Exhaustive panel-rank-calibration.json attribute tests.

Special focus: probability and expected_return sub-dicts have their own
internal contracts (x/y monotonic + length match).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "backtesting" / "renquant_104" / "artifacts"
sys.path.insert(0, str(REPO / "tests"))
from acceptance.model.schemas import CALIBRATOR_SCHEMA   # noqa: E402


CALIB_FILES = [
    ARTIFACTS / "panel-rank-calibration.json",
    ARTIFACTS / "panel-calibration-BULL_CALM.json",
    ARTIFACTS / "panel-calibration-BULL_VOLATILE.json",
    ARTIFACTS / "panel-calibration-CHOPPY.json",
    ARTIFACTS / "panel-calibration-BEAR.json",
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


_FILE_PARAMS = [pytest.param(p, id=p.name) for p in CALIB_FILES]
_ATTR_PARAMS = list(CALIBRATOR_SCHEMA.keys())


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestCalibratorPresence:
    def test_presence(self, path, attr):
        p = _load(path)
        if p is None:
            pytest.skip(f"{path.name} missing")
        spec = CALIBRATOR_SCHEMA[attr]
        if spec.get("required", False):
            assert attr in p, f"{path.name}: required {attr} missing"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestCalibratorType:
    def test_type(self, path, attr):
        p = _load(path)
        if p is None:
            pytest.skip(f"{path.name} missing")
        spec = CALIBRATOR_SCHEMA[attr]
        if attr not in p:
            pytest.skip("absent")
        v = p[attr]
        assert isinstance(v, spec["type"]), \
            f"{path.name}: {attr} type {type(v).__name__} ≠ {spec['type']}"


# ── Sub-structure tests for probability / expected_return ──────────────────

@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("subdict_name", ["probability", "expected_return"])
class TestCalibratorSubdict:
    """Each calibration head has internal x/y arrays. Test each property."""

    def _subdict(self, path: Path, subdict_name: str):
        p = _load(path)
        if p is None:
            pytest.skip(f"{path.name} missing")
        if subdict_name not in p:
            pytest.skip(f"{subdict_name} absent")
        return p[subdict_name]

    def test_has_x_array(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        assert "x" in d, f"{path.name}.{subdict_name}: missing x"

    def test_has_y_array(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        assert "y" in d, f"{path.name}.{subdict_name}: missing y"

    def test_x_is_list(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "x" not in d:
            pytest.skip("no x")
        assert isinstance(d["x"], list)

    def test_y_is_list(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "y" not in d:
            pytest.skip("no y")
        assert isinstance(d["y"], list)

    def test_x_y_same_length(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "x" not in d or "y" not in d:
            pytest.skip()
        assert len(d["x"]) == len(d["y"]), \
            f"{path.name}.{subdict_name}: x/y length mismatch"

    def test_x_non_empty(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "x" not in d:
            pytest.skip()
        assert len(d["x"]) >= 2, f"{path.name}.{subdict_name}: x has <2 thresholds"

    def test_x_monotonic(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "x" not in d or len(d["x"]) < 2:
            pytest.skip()
        xs = d["x"]
        # Monotonic non-decreasing
        for i in range(1, len(xs)):
            assert xs[i] >= xs[i - 1], \
                f"{path.name}.{subdict_name}: x not monotonic at i={i}: " \
                f"x[{i-1}]={xs[i-1]} > x[{i}]={xs[i]}"

    def test_x_all_finite(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "x" not in d:
            pytest.skip()
        for i, v in enumerate(d["x"]):
            assert math.isfinite(float(v)), \
                f"{path.name}.{subdict_name}: x[{i}]={v} non-finite"

    def test_y_all_finite(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "y" not in d:
            pytest.skip()
        for i, v in enumerate(d["y"]):
            assert math.isfinite(float(v)), \
                f"{path.name}.{subdict_name}: y[{i}]={v} non-finite"

    def test_y_in_range(self, path, subdict_name):
        d = self._subdict(path, subdict_name)
        if "y" not in d:
            pytest.skip()
        if subdict_name == "probability":
            for i, v in enumerate(d["y"]):
                assert 0.0 <= v <= 1.0, \
                    f"{path.name}.probability.y[{i}]={v} not in [0,1]"
        else:
            # expected_return — must be in plausible range [-1.0, +1.0]
            for i, v in enumerate(d["y"]):
                assert -1.0 <= v <= 1.0, \
                    f"{path.name}.expected_return.y[{i}]={v} outside [-1,+1]"

    def test_y_not_collapsed(self, path, subdict_name):
        """The 2026-05-04 NaN-leaf incident: calibrator collapsed to
        a near-constant output. Catch this — require ≥3 unique values
        for adequately-sized pools.

        Small-pool regime calibrators (n_rows < 1000, e.g. CHOPPY regime
        with ~600 rows) legitimately produce few unique y values because
        Isotonic only fits a handful of knots. Relax to ≥2 unique for
        these (still catches true constant-collapse, allows small-data
        smoothing)."""
        import json as _json
        d = self._subdict(path, subdict_name)
        if "y" not in d:
            pytest.skip()
        # Read parent payload metadata for n_rows
        try:
            full = _json.loads(path.read_text())
            n_rows = full.get("metadata", {}).get("n_rows", float("inf"))
        except Exception:
            n_rows = float("inf")
        uniq = len({round(float(v), 6) for v in d["y"]})
        threshold = 2 if n_rows < 1000 else 3
        assert uniq >= threshold, (
            f"{path.name}.{subdict_name}.y has only {uniq} unique values "
            f"across {len(d['y'])} thresholds (n_rows={n_rows}) — calibrator "
            f"collapsed below threshold {threshold}. "
            f"This is the 2026-05-04 NaN-leaf incident class."
        )
