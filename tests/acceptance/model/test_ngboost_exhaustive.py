"""Exhaustive ngboost-head.json attribute tests.

Same structure as test_panel_ltr_exhaustive — one test per
(file × attribute × check_kind).
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
from acceptance.model.schemas import NGBOOST_SCHEMA   # noqa: E402


NGBOOST_FILES = [
    ARTIFACTS / "ngboost-head.json",
    ARTIFACTS / "ngboost-head.golden-daily.json",
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


_FILE_PARAMS = [pytest.param(p, id=p.name) for p in NGBOOST_FILES]
_ATTR_PARAMS = list(NGBOOST_SCHEMA.keys())


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostPresence:
    def test_presence(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if spec.get("required", False):
            assert attr in payload, f"{path.name}: required attr {attr!r} missing"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostType:
    def test_type(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if attr not in payload:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        assert isinstance(v, spec["type"]), (
            f"{path.name}: {attr} type {type(v).__name__} ≠ {spec['type']}"
        )


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostFinite:
    def test_finite(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if not spec.get("finite", False):
            pytest.skip("not flagged finite")
        if attr not in payload or payload[attr] is None:
            pytest.skip("absent")
        assert math.isfinite(float(payload[attr])), \
            f"{path.name}: {attr}={payload[attr]} non-finite"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostBoundsLow:
    def test_lower(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if "bounds" not in spec or attr not in payload or payload[attr] is None:
            pytest.skip("no bounds / absent")
        v = payload[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric")
        lo, _ = spec["bounds"]
        assert v >= lo, f"{path.name}: {attr}={v} < {lo}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostBoundsHigh:
    def test_upper(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if "bounds" not in spec or attr not in payload or payload[attr] is None:
            pytest.skip("no bounds / absent")
        v = payload[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric")
        _, hi = spec["bounds"]
        assert v <= hi, f"{path.name}: {attr}={v} > {hi}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostNonEmpty:
    def test_non_empty(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if not spec.get("non_empty", False):
            pytest.skip("not flagged")
        if attr not in payload:
            pytest.skip("absent")
        assert len(payload[attr]) > 0, f"{path.name}: {attr} empty"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostUnique:
    def test_unique(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if not spec.get("unique", False):
            pytest.skip("not flagged")
        if attr not in payload:
            pytest.skip("absent")
        v = payload[attr]
        assert len(v) == len(set(v)), f"{path.name}: {attr} duplicates"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostMinLen:
    def test_min_len(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        if "min_len" not in spec:
            pytest.skip("no min_len")
        if attr not in payload:
            pytest.skip("absent")
        m = spec["min_len"]
        assert len(payload[attr]) >= m, \
            f"{path.name}: {attr} len={len(payload[attr])} < {m}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestNGBoostValidator:
    def test_validator(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        spec = NGBOOST_SCHEMA[attr]
        v = spec.get("validator")
        if v is None:
            pytest.skip("no validator")
        if attr not in payload or payload[attr] is None:
            pytest.skip("absent")
        v(payload[attr])
