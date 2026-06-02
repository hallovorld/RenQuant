"""Wrapper-level guard for weekly WF manifest recipe parity."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
WRAPPER = REPO / "scripts" / "weekly_wf_promote.sh"
RUN_WF_GATE = REPO / "scripts" / "run_wf_gate.py"

_ASSIGN = re.compile(
    r'^\s*([A-Z_][A-Z0-9_]*)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))\s*(?:#.*)?$',
    re.MULTILINE,
)
_VAR = re.compile(r"\$(?:\{([A-Z_][A-Z0-9_]*)\}|([A-Z_][A-Z0-9_]*))")


def _load_run_wf_gate():
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_wf_gate_recipe_guard", RUN_WF_GATE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assignments(src: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _ASSIGN.finditer(src):
        out[match.group(1)] = next(
            part for part in match.groups()[1:] if part is not None
        )
    return out


def _expand(value: str, assignments: dict[str, str]) -> str:
    expanded = value
    for _ in range(6):
        prior = expanded
        expanded = _VAR.sub(
            lambda m: assignments.get(m.group(1) or m.group(2), m.group(0)),
            expanded,
        )
        if expanded == prior:
            break
    return expanded


def _strategy_path(value: str, assignments: dict[str, str]) -> Path:
    expanded = Path(_expand(value, assignments))
    return expanded if expanded.is_absolute() else STRATEGY_DIR / expanded


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    assert isinstance(rows, list) and rows, f"manifest has no retrains: {path}"
    return rows


def test_weekly_wrapper_manifest_recipe_guard_is_fail_fast() -> None:
    """The wrapper must validate the manifest against the staged candidate.

    This pins the 2026-06-02 failure mode where manifest cuts existed and were
    internally consistent, but still differed from the weekly candidate/prod
    artifact recipe. The gate would then fail closed on every bar.
    """
    src = WRAPPER.read_text()
    assignments = _assignments(src)

    assert '--reference-artifact "$STAGING_ART"' in src
    assert '--artifact "$STAGING_ART"' in src
    assert "--no-drop-sentiment" in src
    assert "WF manifest stamping/recipe validation FAILED" in src
    assert "the gate's own contract check will handle it" not in src
    assert "WF_MANIFEST" in assignments

    manifest_path = _strategy_path(assignments["WF_MANIFEST"], assignments)
    assert manifest_path.exists(), f"WF_MANIFEST does not exist: {manifest_path}"

    mod = _load_run_wf_gate()
    rows = _manifest_rows(manifest_path)
    missing: list[str] = []
    recipe_fingerprints: set[str] = set()
    feature_counts: set[int] = set()
    for idx, row in enumerate(rows):
        artifact_uri = row.get("artifact_uri")
        assert artifact_uri, f"retrains[{idx}] missing artifact_uri"
        artifact_path = _strategy_path(str(artifact_uri), assignments)
        if not artifact_path.exists():
            missing.append(f"retrains[{idx}].artifact_uri={artifact_uri}")
        else:
            artifact = mod._load_artifact_payload(artifact_path)
            recipe_fingerprints.add(mod._recipe_fingerprint(artifact))
            feature_counts.add(len(artifact.get("feature_cols") or []))

        calibrator_uri = row.get("calibrator_uri") or row.get("calibration_uri")
        assert calibrator_uri, f"retrains[{idx}] missing calibrator_uri"
        calibrator_path = _strategy_path(str(calibrator_uri), assignments)
        if not calibrator_path.exists():
            missing.append(f"retrains[{idx}].calibrator_uri={calibrator_uri}")
    assert not missing, "manifest references missing files:\n" + "\n".join(missing[:10])

    assert len(recipe_fingerprints) == 1, json.dumps(sorted(recipe_fingerprints))
    assert feature_counts == {172}
