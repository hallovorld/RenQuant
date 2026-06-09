"""RFC #259 P0b — the umbrella active-scorer check is a thin multi-repo delegate.

The gate/acceptance logic lives in renquant-backtesting
(`renquant_backtesting.wf_gate.check_active_scorer`); the umbrella only resolves
the subrepo env + PYTHONPATH and delegates (RenQuant CLAUDE.md §3.5). This guards
against the regression of re-introducing the gate logic as an umbrella script.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "scripts" / "check_active_scorer_gated.sh"


def test_wrapper_exists_and_is_bash() -> None:
    assert WRAPPER.exists(), "umbrella delegate wrapper missing"
    assert WRAPPER.read_text().startswith("#!/usr/bin/env bash")


def test_wrapper_delegates_through_subrepo_env() -> None:
    src = WRAPPER.read_text()
    assert "scripts/subrepo_env.sh" in src
    assert 'renquant_load_subrepo_env "$REPO_DIR"' in src
    assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in src
    assert 'renquant_subrepo_pythonpath "$SUBREPO_ROOT"' in src
    assert "renquant-backtesting" in src


def test_wrapper_delegates_to_gate_owner_module() -> None:
    src = WRAPPER.read_text()
    assert "renquant_backtesting.wf_gate.check_active_scorer" in src
    assert "python -m renquant_backtesting.wf_gate.check_active_scorer" in src.replace(
        '"$PYTHON"', "python"
    )


def test_no_umbrella_canonical_gate_logic() -> None:
    """The umbrella must NOT carry the check logic itself (§3.5)."""
    assert not (REPO / "scripts" / "check_active_scorer_gated.py").exists()
    src = WRAPPER.read_text()
    # the wrapper resolves config paths only via the delegate, not inline
    assert "panel_scoring" not in src
    assert "assert_artifact_gated" not in src
