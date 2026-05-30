"""Bug D regression — model fingerprint is invariant to metadata edits.

Pins the architectural invariant: ``compute_model_fingerprint`` returns the
SAME hash when only metadata changes (cv_method update, P-PANEL-CONTRACT
stamps, doc-string edits, …) but a DIFFERENT hash when the model itself
changes (booster_raw_json mutated, .pt re-trained).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.panel_pipeline.model_fingerprint import (
    compute_model_fingerprint,
    model_fingerprint_kind,
)


@pytest.fixture
def gbdt_artifact(tmp_path):
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-05-18",
        "config_fingerprint": "sha256:abc",
        "cv_method": "purged_walk_forward",
        "booster_raw_json": '{"version": [1,2,3], "tree_id": 7, "splits": "..."}',
        "feature_cols": ["a", "b", "c"],
    }))
    return p


@pytest.fixture
def pt_artifact(tmp_path):
    p = tmp_path / "model.pt"
    # Fake torch checkpoint bytes
    p.write_bytes(b"\x80\x05\x95...torch pickled state_dict bytes...")
    return p


def test_gbdt_fingerprint_stable_across_metadata_edits(gbdt_artifact):
    """Editing cv_method / adding P-PANEL-CONTRACT stamps must NOT change fp."""
    fp1 = compute_model_fingerprint(gbdt_artifact)

    d = json.loads(gbdt_artifact.read_text())
    d["cv_method"] = "different_value"
    d["new_stamp_field"] = "added today"
    d["train_run_id"] = "synthetic_xyz"
    gbdt_artifact.write_text(json.dumps(d, indent=2))

    fp2 = compute_model_fingerprint(gbdt_artifact)
    assert fp1 == fp2, (
        f"metadata edit must not change model fingerprint; "
        f"before={fp1!r}, after={fp2!r}"
    )


def test_gbdt_fingerprint_changes_when_booster_changes(gbdt_artifact):
    """Mutating booster_raw_json (the actual model) MUST change fp."""
    fp1 = compute_model_fingerprint(gbdt_artifact)
    d = json.loads(gbdt_artifact.read_text())
    d["booster_raw_json"] = '{"version": [1,2,3], "tree_id": 8, "splits": "DIFFERENT"}'
    gbdt_artifact.write_text(json.dumps(d, indent=2))
    fp2 = compute_model_fingerprint(gbdt_artifact)
    assert fp1 != fp2


def test_gbdt_legacy_json_without_booster_falls_back_to_file_hash(tmp_path):
    """Old artifacts may not have booster_raw_json; fall back to file hash + report kind."""
    p = tmp_path / "old_panel-ltr.json"
    p.write_text(json.dumps({"kind": "panel_ltr_xgboost", "no_booster_field": True}))
    fp = compute_model_fingerprint(p)
    assert fp.startswith("sha256:")
    assert model_fingerprint_kind(p) == "file_bytes_legacy_json"


def test_pt_fingerprint_stable_across_sidecar_edits(pt_artifact):
    """For .pt artifacts, metadata is in the SIDECAR not the .pt itself.
    Sidecar edits don't touch .pt bytes, so fp is naturally stable."""
    fp1 = compute_model_fingerprint(pt_artifact)
    # Edit a sibling sidecar — doesn't affect the .pt hash
    sidecar = pt_artifact.with_name(pt_artifact.name + ".metadata.json")
    sidecar.write_text(json.dumps({"metadata": {"wf_gate_metadata": {"passed": False}}}))
    fp2 = compute_model_fingerprint(pt_artifact)
    assert fp1 == fp2
    assert model_fingerprint_kind(pt_artifact) == "file_bytes_binary"


def test_pt_fingerprint_changes_when_pt_retrained(pt_artifact):
    fp1 = compute_model_fingerprint(pt_artifact)
    pt_artifact.write_bytes(b"\x80\x05\x95...different state_dict bytes...")
    fp2 = compute_model_fingerprint(pt_artifact)
    assert fp1 != fp2


def test_missing_path_returns_empty():
    assert compute_model_fingerprint(Path("/nonexistent/path.pt")) == ""


def test_fingerprint_returns_sha256_prefix(gbdt_artifact, pt_artifact):
    for p in (gbdt_artifact, pt_artifact):
        fp = compute_model_fingerprint(p)
        assert fp.startswith("sha256:")
        assert len(fp) == 7 + 64
