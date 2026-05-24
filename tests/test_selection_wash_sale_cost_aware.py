"""Regression test: run_selection_loop wash-sale path is cost-aware (FIX-A).

Bug: kernel/selection.py::run_selection_loop line 403 called
is_wash_sale_blocked (BINARY) — the only wash-sale path that hadn't been
upgraded to cost-aware. WashSaleFilterTask, ComputeWashSaleMaskTask,
RotationTask, JointActionTask all used cost-aware. The greedy selection
path was the last binary holdout.

Production effect: the greedy path is the legacy `solver=greedy`. Production
uses `solver=qp` (per CLAUDE.md), so this fix is non-blocking but required
for cleanliness — eliminates the last wash-sale inconsistency surface.

Reference: doc/AUDIT_2026-05-09.md §2.3 finding #24.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.selection import (
    CandidateResult,
    SelectionContext,
    run_selection_loop,
)


def _make_candidate(ticker: str, rank_score: float = 0.5):
    return CandidateResult(
        ticker=ticker,
        raw_score=rank_score,
        rank_score=rank_score,
        rs_score=0.5,
    )


def _make_ctx(*, today, last_sells, last_pls, wash_days=30, slots=5):
    return SelectionContext(
        today=today,
        held_tickers=[],
        last_sell_dates=last_sells,
        last_sell_pls=last_pls,
        earnings_calendar={},
        corr_matrix=None,
        sector_map={},
        defensive_set=set(),
        wash_sale_days=wash_days,
        earnings_buffer=3,
        corr_threshold=0.99,
        max_per_sector=0,
        tiered_thresholds=[{"min_model_score": 0.0}],
        open_slots=slots,
        bear_only=False,
    )


# ── Cost-aware behavior tests ────────────────────────────────────────────────

class TestSelectionWashSaleCostAware:

    def test_gain_sale_within_window_admitted(self):
        """A ticker sold 14d ago for +$500 GAIN passes selection (§1091 N/A)."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            today=today,
            last_sells={"AAPL": sold},
            last_pls={"AAPL": 500.0},   # GAIN
        )
        cands = [_make_candidate("AAPL", 0.8)]
        selected, blocks = run_selection_loop(cands, ctx)
        assert "AAPL" in selected, \
            "Cost-aware wash-sale: gain sale must NOT block re-entry"
        assert blocks["wash_sale"] == 0

    def test_loss_sale_within_window_blocked(self):
        """A ticker sold 14d ago for -$200 LOSS is blocked (§1091 applies)."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            today=today,
            last_sells={"AAPL": sold},
            last_pls={"AAPL": -200.0},   # LOSS
        )
        cands = [_make_candidate("AAPL", 0.8)]
        selected, blocks = run_selection_loop(cands, ctx)
        assert "AAPL" not in selected, \
            "Cost-aware wash-sale: loss sale must block re-entry"
        assert blocks["wash_sale"] == 1

    def test_outside_window_admitted(self):
        """Ticker sold 35d ago is admitted regardless of P/L sign."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=35)
        for pl in [500.0, -200.0]:
            ctx = _make_ctx(
                today=today,
                last_sells={"AAPL": sold},
                last_pls={"AAPL": pl},
            )
            cands = [_make_candidate("AAPL", 0.8)]
            selected, blocks = run_selection_loop(cands, ctx)
            assert "AAPL" in selected, \
                f"Outside-window pl={pl} must be admitted"

    def test_unknown_pl_blocked_conservatively(self):
        """No P/L data → fail-conservative block (mirrors WashSaleFilterTask)."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            today=today,
            last_sells={"AAPL": sold},
            last_pls={},   # P/L unknown
        )
        cands = [_make_candidate("AAPL", 0.8)]
        selected, blocks = run_selection_loop(cands, ctx)
        assert "AAPL" not in selected
        assert blocks["wash_sale"] == 1

    def test_no_sale_history_admitted(self):
        today = datetime.date(2026, 5, 9)
        ctx = _make_ctx(
            today=today,
            last_sells={},
            last_pls={},
        )
        cands = [_make_candidate("AAPL", 0.8)]
        selected, _ = run_selection_loop(cands, ctx)
        assert "AAPL" in selected


# ── Single-source-of-truth invariant ─────────────────────────────────────────

class TestSelectionUsesCostAwareHelper:
    """All 4 wash-sale paths (candidates, rotation, joint, selection, QP) MUST
    call is_wash_sale_blocked_with_cost. This pins the contract."""

    def test_run_selection_loop_uses_cost_aware(self):
        src = (REPO / "backtesting" / "renquant_104"
               / "kernel" / "selection.py").read_text()
        # Find run_selection_loop body
        idx = src.find("def run_selection_loop(")
        assert idx > 0
        next_def = src.find("\ndef ", idx + 1)
        body = src[idx:next_def if next_def > 0 else len(src)]
        assert "is_wash_sale_blocked_with_cost" in body, \
            "AUDIT REGRESSION (FIX-A): run_selection_loop reverted to binary " \
            "wash-sale. The greedy path is now the only inconsistent surface."


# ── Pipeline-task wiring ─────────────────────────────────────────────────────

class TestBuildSelectionContextPropagatesPls:
    """BuildSelectionContextTask must populate ctx._sel_ctx.last_sell_pls
    from the InferenceContext.last_sell_pls so the greedy path can use it."""

    def test_task_propagates_last_sell_pls(self):
        src = (REPO / "backtesting" / "renquant_104"
               / "kernel" / "pipeline" / "task_selection.py").read_text()
        ctx_block_start = src.find("SelectionContext(")
        assert ctx_block_start > 0
        ctx_block_end = src.find("        )", ctx_block_start)
        ctx_block = src[ctx_block_start:ctx_block_end]
        assert "last_sell_pls" in ctx_block, \
            "AUDIT REGRESSION: BuildSelectionContextTask no longer passes " \
            "last_sell_pls — greedy wash-sale will fall back to binary block " \
            "(no last_sell_pls in selection context = unknown for every ticker)."
