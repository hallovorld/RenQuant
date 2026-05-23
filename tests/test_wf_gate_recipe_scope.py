"""Unit tests for WF gate scope and recipe matching."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_wf_gate.py"


def _load_module():
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_wf_gate_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(features: list[str]) -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "feature_cols": features,
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {"objective": "rank:pairwise", "eta": 0.05},
    }


def test_manifest_recipe_usage_accepts_matching_samples(tmp_path: Path):
    mod = _load_module()
    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(_artifact(["a", "b"])))
    sample.write_text(json.dumps(_artifact(["a", "b"])))
    manifest.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(sample), "cutoff_date": "2024-01-01"},
            {"artifact_uri": str(sample), "cutoff_date": "2024-02-01"},
        ]
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is True
    assert usage["candidate_n_features"] == 2


def test_manifest_recipe_usage_rejects_feature_drift(tmp_path: Path):
    mod = _load_module()
    candidate = tmp_path / "candidate.json"
    sample = tmp_path / "sample.json"
    manifest = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(_artifact(["a", "b", "sentiment"])))
    sample.write_text(json.dumps(_artifact(["a", "b"])))
    manifest.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(sample), "cutoff_date": "2024-01-01"},
            {"artifact_uri": str(sample), "cutoff_date": "2024-02-01"},
        ]
    }))

    usage = mod._manifest_recipe_usage(manifest, candidate)

    assert usage["recipe_validated"] is False
    report = usage["manifest_sample_reports"][0]
    assert report["missing_features_vs_candidate"] == ["sentiment"]


def test_recipe_fingerprint_ignores_execution_only_xgb_params() -> None:
    """Hardware/threading changes must not invalidate historical WF recipes."""
    mod = _load_module()
    old_hw = _artifact(["a", "b"])
    new_hw = _artifact(["a", "b"])
    old_hw["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.05,
        "max_depth": 5,
        "nthread": 8,
        "verbosity": 0,
    }
    new_hw["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.05,
        "max_depth": 5,
        "nthread": 14,
        "verbosity": 2,
    }

    assert mod._recipe_fingerprint(old_hw) == mod._recipe_fingerprint(new_hw)

    changed_learning_param = _artifact(["a", "b"])
    changed_learning_param["params"] = {
        "objective": "rank:pairwise",
        "eta": 0.10,
        "max_depth": 5,
        "nthread": 14,
        "verbosity": 0,
    }
    assert (
        mod._recipe_fingerprint(old_hw)
        != mod._recipe_fingerprint(changed_learning_param)
    )
