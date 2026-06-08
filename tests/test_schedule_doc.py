"""Regression guards for the operator schedule documentation."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEDULE = REPO / "doc" / "ops" / "schedule.md"


def test_schedule_doc_records_multirepo_inventory_status() -> None:
    doc = SCHEDULE.read_text(encoding="utf-8")

    assert "2026-06-08" in doc
    assert "renquant_orchestrator.cli scheduled-jobs" in doc
    assert "8 of 10 scheduled jobs are `native_multirepo`" in doc
    assert "`daily_live_runner_bridge` and `live_runner_bridge`" in doc
    assert "--fail-on-umbrella-bridge" in doc


def test_schedule_doc_records_patchtst_scheduled_retrain() -> None:
    doc = SCHEDULE.read_text(encoding="utf-8")

    assert "scripts/weekly_retrain_patchtst.sh" in doc
    assert "renquant_orchestrator.build_patchtst_wf_manifest" in doc
    assert "RQ_PATCHTST_FULL_MANIFEST=1" in doc
    assert "operator-directed PatchTST promotion" in doc
