"""Regression guards for the pre-open cancel gate multirepo wrapper."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_umbrella_preopen_gate_implementation_remains_present() -> None:
    src = (REPO / "scripts" / "preopen_cancel_gate.py").read_text()
    assert "def compute_overnight_severity(" in src
    assert "def cancel_stale_market_orders(" in src
    assert "def main()" in src


def test_preopen_wrapper_defaults_to_execution_subrepo_with_rollback() -> None:
    src = (REPO / "scripts" / "preopen_cancel_gate.sh").read_text()
    assert 'RQ_PREOPEN_GATE_RUNNER:-multirepo' in src
    assert "RQ_PREOPEN_GATE_RUNNER=umbrella" in src
    assert "../renquant-execution/src" in src
    assert "../renquant-common/src" in src
    assert "python -m renquant_execution.preopen_cancel_gate" in src
    assert "renquant_execution.preopen_cancel_gate={m.__file__}" in src
    assert "exec python scripts/preopen_cancel_gate.py" in src
    assert "RQ_PREOPEN_GATE_STRICT" in src
    assert "PREOPEN_GATE_FALLBACK" in src
    assert "Priority: low" in src
    assert ".subrepo_fallback_alert_stamp" in src
