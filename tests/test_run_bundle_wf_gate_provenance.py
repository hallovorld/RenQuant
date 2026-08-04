"""GOAL-5 AC6 R4 on the SERVING bundle producer (orch#564).

Measured 2026-08-04: the successful full run's persisted bundle (9,154 B)
carried no `wf_gate_provenance` key — R4 had landed only in
renquant_orchestrator/daily.py, whose output surface stopped being exercised
2026-05-07. These tests pin the block on the producer that actually serves:
kernel/artifact_contract.build_run_bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.artifact_contract import _wf_gate_provenance_block, build_run_bundle  # noqa: E402


def _artifact(tmp_path: Path, *, gate: dict | None, basis=None, trained=None) -> Path:
    meta: dict = {}
    if gate is not None:
        meta["wf_gate_metadata"] = gate
    if basis is not None:
        meta["promotion_basis"] = basis
    payload: dict = {"kind": "panel_ltr_xgboost", "metadata": meta}
    if trained is not None:
        payload["trained_date"] = trained
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_governance_served_artifact_yields_present_block_with_license_fields(tmp_path):
    p = _artifact(
        tmp_path,
        gate={"passed": False, "gate_version": "v2", "diagnostic_only": False},
        basis="freshness_fallback_rfc210", trained="2026-08-02")
    out = _wf_gate_provenance_block(p)
    assert out["status"] == "present"
    assert out["source_key"] == "metadata.wf_gate_metadata"
    assert out["passed"] is False
    assert out["promotion_basis"] == "freshness_fallback_rfc210"
    assert out["trained_date"] == "2026-08-02"
    assert "operator_authorized_override" in out["fields_absent"]


def test_no_artifact_and_no_stamp_are_distinct_statuses(tmp_path):
    assert _wf_gate_provenance_block(None)["status"] == "no_artifact_manifest"
    p = _artifact(tmp_path, gate=None)
    assert _wf_gate_provenance_block(p)["status"] == "artifact_carries_no_gate_stamp"


def test_empty_canonical_block_is_no_stamp_not_legacy_fallthrough(tmp_path):
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({
        "metadata": {"wf_gate_metadata": {}},
        "wf_gate_metadata": {"passed": True},   # legacy decoy must NOT answer
    }), encoding="utf-8")
    out = _wf_gate_provenance_block(p)
    assert out["status"] == "artifact_carries_no_gate_stamp"


def test_unreadable_artifact_never_raises(tmp_path):
    p = tmp_path / "panel-ltr.json"
    p.write_text("{not json", encoding="utf-8")
    out = _wf_gate_provenance_block(p)
    assert out["status"] == "provenance_read_failed"


def test_build_run_bundle_carries_the_block(tmp_path):
    art = _artifact(
        tmp_path,
        gate={"passed": False, "gate_version": "v2"},
        basis="freshness_fallback_rfc210", trained="2026-08-02")
    cfg = {
        "watchlist": ["AAPL"],
        "ranking": {"panel_scoring": {
            "enabled": True, "kind": "xgb",
            "artifact_path": art.name,
        }},
    }
    bundle = build_run_bundle(cfg, tmp_path, run_id="t-1", run_type="live")
    block = bundle["wf_gate_provenance"]
    assert block["status"] == "present"
    assert block["promotion_basis"] == "freshness_fallback_rfc210"
