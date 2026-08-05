"""The WF gate's production reference must be the PINNED config or nothing.

orch#799, measured 2026-08-04: after the full-book z-blend switch made the
pinned primary `kind=blend`, `_find_gbdt_config`'s search fell through to the
umbrella WORKING COPY (`backtesting/renquant_104/strategy_config.shadow.json`
— the A8 registry's known-diverged, hf_patchtst-era file). The gate then
derived "production semantics" from it and simulated a strategy nobody runs:
same booster sha, same data day, Sharpe 0.6018 -> 0.0524, greedy
Selection+TopUp -> joint QP, 373 -> 104 simulated trades.

These are SOURCE-LEVEL guards on the search contract (the function runs inside
the promote wrapper's environment; the properties below are what must hold).
"""
from __future__ import annotations

import pathlib
import re

WRAPPER = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weekly_wf_promote.sh"


def _finder_block() -> str:
    src = WRAPPER.read_text()
    start = src.index("_find_gbdt_config()")
    end = src.index("if ! GBDT_PROD_CONFIG=", start)
    return src[start:end]


def test_working_copy_is_not_a_candidate_reference():
    """The umbrella working copy may be NAMED (so the exclusion is visible)
    but must never enter the candidates array the search iterates."""
    block = _finder_block()
    assert "workingcopy_path=" in block, "the path is still named for clarity"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("candidates=("):
            assert "$workingcopy_path" not in stripped, (
                f"working copy must not be a candidate reference: {stripped}"
            )


def test_pinned_path_is_the_only_candidate_in_every_mode():
    """Round 2 (codex on #580): the sibling/multirepo path resolved through
    renquant_subrepo_root defaults to a DEVELOPER CHECKOUT absent an assembly
    override — a locally-edited checkout would recreate this incident. The
    lock-aligned runtime config (.subrepo_runtime, what the daily run loads)
    is the ONLY candidate, in every runner mode."""
    block = _finder_block()
    cand_lines = [l.strip() for l in block.splitlines() if l.strip().startswith("candidates=(")]
    assert cand_lines, "no candidates array found"
    for line in cand_lines:
        assert "$pinned_path" in line, line
        assert "$multirepo_path" not in line, (
            f"unpinned sibling checkout must not be a candidate: {line}"
        )
        assert "$workingcopy_path" not in line, line
    assert len(cand_lines) == 1, (
        f"one unconditional candidates array expected (no mode branch that could "
        f"reintroduce an unpinned path): {cand_lines}"
    )


def test_pinned_path_resolves_under_subrepo_runtime():
    """The 'pinned' path must be the lock-aligned runtime checkout, not any
    other tree that happens to be called pinned."""
    block = _finder_block()
    assert '.subrepo_runtime/repos/renquant-strategy-104/configs/' in block


def test_no_match_fails_closed_with_the_blend_explanation_and_alert():
    """A missing kind match is a REAL state (prod is a blend). The wrapper
    must exit nonzero, explain it, name the decision issue, and page — never
    fall back to a stale file."""
    src = WRAPPER.read_text()
    idx = src.index("if ! GBDT_PROD_CONFIG=")
    block = src[idx: idx + 1400]
    assert "no PINNED strategy config declares kind=xgb" in block
    assert "orch#799" in block
    assert "blend" in block
    assert "WEEKLY-BLOCKED" in block          # pages the operator
    assert re.search(r"\n\s*exit 2\b", block)  # fails closed
    assert "RFC#210 freshness governance unaffected" in block


def test_the_alert_states_production_is_unchanged():
    """A blocked gate must never read as a production mutation."""
    src = WRAPPER.read_text()
    idx = src.index("WEEKLY-BLOCKED")
    assert "Production unchanged" in src[idx: idx + 400]
