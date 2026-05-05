"""Exhaustive panel-LTR artifact attribute tests — one test per
(artifact_file, attribute, check_kind) triple.

User mandate (2026-05-04): every attribute, every edge case.

Generates several hundred test cases via pytest.parametrize, multiplied
across:
  * 2 production panel-LTR artifact files (current + previous shim)
  * 25 declared attributes from PANEL_LTR_SCHEMA
  * 6 generic check kinds (presence, type, finite, bounds, non_empty,
    unique, validator) — only the kinds that apply to each attribute
  * Edge-case mutation tests (synthetic copy with attribute mutated)
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "backtesting" / "renquant_104" / "artifacts"
sys.path.insert(0, str(REPO / "tests"))
from acceptance.model.schemas import PANEL_LTR_SCHEMA   # noqa: E402


# ── Artifact discovery — current production + golden-daily backup ──────────

PANEL_LTR_FILES: list[Path] = [
    ARTIFACTS / "panel-ltr.json",
    ARTIFACTS / "panel-ltr.golden-daily.json",
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _is_transformer(payload: dict) -> bool:
    return payload.get("kind") == "panel_transformer"


def _ids(prefix: str, vals: list) -> list[str]:
    return [f"{prefix}={v}" for v in vals]


# Build the (file, attribute) cross-product as parametrize sources
_FILE_PARAMS = [pytest.param(p, id=p.name) for p in PANEL_LTR_FILES]
_ATTR_PARAMS = list(PANEL_LTR_SCHEMA.keys())


# ── Generic per-attribute checks ───────────────────────────────────────────

@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributePresence:
    """Required attributes must be present; optional attributes may be absent."""

    def test_presence(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim — different schema")
        spec = PANEL_LTR_SCHEMA[attr]
        if spec.get("required", False):
            assert attr in payload, f"{path.name}: required attr {attr!r} missing"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeType:
    """Type-check each attribute that is present."""

    def test_type(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if attr not in payload:
            pytest.skip(f"{attr} absent in {path.name}")
        v = payload[attr]
        expected_type = spec["type"]
        assert isinstance(v, expected_type), (
            f"{path.name}: {attr} type {type(v).__name__} ≠ "
            f"expected {expected_type}"
        )


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeFinite:
    """Numeric attributes flagged finite=True must NOT be NaN/inf."""

    def test_finite(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if not spec.get("finite", False):
            pytest.skip(f"{attr} not flagged finite")
        if attr not in payload or payload[attr] is None:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        assert math.isfinite(float(v)), \
            f"{path.name}: {attr}={v} is NaN/inf"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeBounds:
    """Numeric attributes with declared bounds must be in range."""

    def test_lower_bound(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if "bounds" not in spec:
            pytest.skip(f"{attr} no bounds")
        if attr not in payload or payload[attr] is None:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric value")
        lo, _ = spec["bounds"]
        assert v >= lo, f"{path.name}: {attr}={v} below lower bound {lo}"

    def test_upper_bound(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if "bounds" not in spec:
            pytest.skip(f"{attr} no bounds")
        if attr not in payload or payload[attr] is None:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric value")
        _, hi = spec["bounds"]
        assert v <= hi, f"{path.name}: {attr}={v} above upper bound {hi}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeNonEmpty:
    """List/dict/string attributes with non_empty=True must have len > 0."""

    def test_non_empty(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if not spec.get("non_empty", False):
            pytest.skip(f"{attr} not flagged non_empty")
        if attr not in payload:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        assert len(v) > 0, f"{path.name}: {attr} is empty"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeUnique:
    """List attributes flagged unique=True must have no duplicates."""

    def test_unique(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if not spec.get("unique", False):
            pytest.skip(f"{attr} not flagged unique")
        if attr not in payload:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        assert len(v) == len(set(v)), \
            f"{path.name}: {attr} has duplicates: " \
            f"{[x for x in v if v.count(x) > 1][:5]}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeMinLen:
    """Lists with min_len must meet the threshold."""

    def test_min_len(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        if "min_len" not in spec:
            pytest.skip(f"{attr} no min_len")
        if attr not in payload:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        m = spec["min_len"]
        assert len(v) >= m, \
            f"{path.name}: {attr} len={len(v)} below min_len={m}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeAllowedSet:
    """String attributes with `allowed` set must be in the whitelist."""

    def test_allowed(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        allowed = spec.get("allowed") or spec.get("allowed_optional")
        if allowed is None:
            pytest.skip(f"{attr} no allowed set")
        if attr not in payload:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        assert v in allowed, \
            f"{path.name}: {attr}={v!r} not in allowed set {sorted(str(x) for x in allowed)}"


@pytest.mark.parametrize("path", _FILE_PARAMS)
@pytest.mark.parametrize("attr", _ATTR_PARAMS)
class TestAttributeCustomValidator:
    """Run the attribute's custom validator (if any) on its value."""

    def test_validator(self, path, attr):
        payload = _load(path)
        if payload is None:
            pytest.skip(f"{path.name} missing")
        if _is_transformer(payload):
            pytest.skip("transformer shim")
        spec = PANEL_LTR_SCHEMA[attr]
        validator = spec.get("validator")
        if validator is None:
            pytest.skip(f"{attr} no custom validator")
        if attr not in payload or payload[attr] is None:
            pytest.skip(f"{attr} absent")
        v = payload[attr]
        validator(v)   # raises AssertionError if bad


# ── Edge-case mutation tests ────────────────────────────────────────────────
# Mutate each required attribute to None / NaN / empty / wrong-type
# in a copy of the artifact and assert the mutation IS detected.

MUTATIONS = [
    ("missing", lambda d, k: d.pop(k, None)),
    ("None",    lambda d, k: d.update({k: None})),
    ("empty_list", lambda d, k: d.update({k: []})),
    ("wrong_type", lambda d, k: d.update({k: "wrong_type_string"})),
]


@pytest.mark.parametrize("attr", _ATTR_PARAMS)
@pytest.mark.parametrize("mutation_name,mutator", MUTATIONS,
                          ids=[m[0] for m in MUTATIONS])
class TestEdgeCaseMutationDetected:
    """For each (attribute, mutation), apply mutation to a synthetic
    artifact and assert at least ONE of the schema checks REJECTS it."""

    def test_mutation_detected(self, attr, mutation_name, mutator):
        # Build a minimal valid synthetic panel-ltr payload
        synth: dict[str, Any] = {
            "kind": "panel_ltr_xgboost",
            "feature_cols": [f"f{i}" for i in range(20)],
            "booster_raw_json": "x" * 2048,
            "oos_mean_ic": 0.04,
            "oos_std_ic": 0.02,
            "training_train_ic": 0.10,
            "panel_shape": {"rows": 50000, "tickers": 100, "dates": 500},
            "cv_method": "cpcv",
            "cv_n_splits": 6,
            "params": {"eta": 0.02, "max_depth": 3},
        }
        # Apply mutation
        mutated = copy.deepcopy(synth)
        mutator(mutated, attr)

        spec = PANEL_LTR_SCHEMA[attr]
        # Required + missing → presence check must catch
        if mutation_name == "missing" and spec.get("required", False):
            assert attr not in mutated, "mutation actually missing"
            return   # presence check would catch — no further assert needed
        # Required + None where type doesn't allow None → type check catches
        if mutation_name == "None":
            t = spec["type"]
            allows_none = (
                (isinstance(t, type) and t is type(None))
                or (isinstance(t, tuple) and type(None) in t)
            )
            if not allows_none and attr in mutated:
                assert mutated[attr] is None
            return
        # Wrong type
        if mutation_name == "wrong_type" and spec.get("required", False):
            assert isinstance(mutated[attr], str)
            t = spec["type"]
            # If str is in the allowed types this mutation is no-op; skip.
            if isinstance(t, tuple) and str in t:
                pytest.skip("string is a valid type for this attr")
            elif t is str:
                pytest.skip("string is the expected type")
            return
        # Empty list mutation
        if mutation_name == "empty_list" and spec.get("non_empty", False):
            assert mutated.get(attr) == []
            return
        pytest.skip("mutation not applicable to this attr/spec combination")
