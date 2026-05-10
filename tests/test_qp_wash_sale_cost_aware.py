"""Regression test: QP wash-sale mask must be cost-aware (Audit Phase 2.2).

Bug: kernel/portfolio_qp/tasks.py::ComputeWashSaleMaskTask used a binary
30-day block, ignoring the gain-vs-loss distinction. WashSaleFilterTask
upstream (in candidate path) was correctly cost-aware via
is_wash_sale_blocked_with_cost. Result: a ticker that just sold for a
GAIN passed candidate filter (§1091 N/A on gains) BUT got Δw≤0 in the
QP — architecturally locked from any post-gain re-entry.

Two wash-sale paths must use the same helper. This test pins both at
is_wash_sale_blocked_with_cost so future drift fails loudly.

Reference:
- doc/AUDIT_2026-05-09.md §2.2
- IRC §1091 wash-sale: applies only to LOSSES, not gains
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


def _make_ctx(*, tickers, today, last_sells, last_pls, wash_days=30):
    """Minimal ctx for ComputeWashSaleMaskTask."""
    ctx = SimpleNamespace()
    ctx._qp_tickers = list(tickers)
    ctx.last_sell_dates = dict(last_sells)
    ctx.last_sell_pls = dict(last_pls)
    ctx.config = {"wash_sale_days": wash_days}
    ctx.today = today
    return ctx


def _run_mask(ctx):
    from kernel.portfolio_qp.tasks import ComputeWashSaleMaskTask
    ComputeWashSaleMaskTask().run(ctx)
    return ctx._qp_wash_mask


# ── Cost-aware behavior tests ────────────────────────────────────────────────

class TestQPWashSaleCostAware:

    def test_gain_sale_not_blocked(self):
        """A ticker sold 14 days ago for +$500 gain → §1091 N/A → not blocked."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            tickers=["AAPL", "MSFT"],
            today=today,
            last_sells={"AAPL": sold},
            last_pls={"AAPL": 500.0},   # GAIN
        )
        mask = _run_mask(ctx)
        # AAPL: gain → §1091 N/A → mask = False
        assert mask[0] == False, \
            "Gain sale within window: §1091 does NOT apply, mask must be False"
        # MSFT: never sold → False
        assert mask[1] == False

    def test_loss_sale_blocked(self):
        """A ticker sold 14 days ago for -$200 loss → §1091 applies → blocked."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            tickers=["AAPL"],
            today=today,
            last_sells={"AAPL": sold},
            last_pls={"AAPL": -200.0},   # LOSS
        )
        mask = _run_mask(ctx)
        assert mask[0] == True, \
            "Loss sale within window: §1091 applies, mask must be True"

    def test_outside_window_not_blocked(self):
        """A ticker sold 35 days ago (outside 30d) → not blocked regardless of P/L."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=35)
        for pl in [500.0, -200.0, None]:
            ctx = _make_ctx(
                tickers=["AAPL"],
                today=today,
                last_sells={"AAPL": sold},
                last_pls={"AAPL": pl} if pl is not None else {},
            )
            mask = _run_mask(ctx)
            assert mask[0] == False, \
                f"Outside 30d window: never blocked (got mask=True for pl={pl})"

    def test_unknown_pl_blocked_conservatively(self):
        """Unknown P/L (sale recorded but no realized-pnl data) → fail-conservative
        → block. Mirrors WashSaleFilterTask binary-fallback behavior."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            tickers=["AAPL"],
            today=today,
            last_sells={"AAPL": sold},
            last_pls={},   # Unknown — fail conservative
        )
        mask = _run_mask(ctx)
        assert mask[0] == True, \
            "Unknown P/L within window: must fail-conservative (block)"

    def test_no_sale_history_not_blocked(self):
        """No sale history at all → not blocked."""
        today = datetime.date(2026, 5, 9)
        ctx = _make_ctx(
            tickers=["AAPL", "MSFT"],
            today=today,
            last_sells={},
            last_pls={},
        )
        mask = _run_mask(ctx)
        assert all(m == False for m in mask)

    def test_wash_days_zero_disabled(self):
        """wash_sale_days=0 → mask all False (gate disabled)."""
        today = datetime.date(2026, 5, 9)
        sold = today - datetime.timedelta(days=14)
        ctx = _make_ctx(
            tickers=["AAPL"],
            today=today,
            last_sells={"AAPL": sold},
            last_pls={"AAPL": -200.0},
            wash_days=0,
        )
        mask = _run_mask(ctx)
        assert mask[0] == False


# ── Single-source-of-truth invariant ─────────────────────────────────────────

class TestWashSalePathsConsistent:
    """Audit invariant: WashSaleFilterTask (candidate path) and
    ComputeWashSaleMaskTask (QP path) MUST call the same helper. If a future
    refactor splits them, this test fails — preventing silent drift."""

    def test_qp_mask_uses_is_wash_sale_blocked_with_cost(self):
        src = (REPO / "backtesting" / "renquant_104"
               / "kernel" / "portfolio_qp" / "tasks.py").read_text()
        # Find ComputeWashSaleMaskTask body
        cls_idx = src.find("class ComputeWashSaleMaskTask")
        assert cls_idx > 0
        next_cls = src.find("\nclass ", cls_idx + 1)
        body = src[cls_idx:next_cls if next_cls > 0 else len(src)]
        assert "is_wash_sale_blocked_with_cost" in body, \
            "AUDIT REGRESSION: ComputeWashSaleMaskTask no longer uses " \
            "is_wash_sale_blocked_with_cost. The QP wash-sale path will " \
            "silently diverge from the candidate path again."

    def test_candidate_filter_uses_is_wash_sale_blocked_with_cost(self):
        src = (REPO / "backtesting" / "renquant_104"
               / "kernel" / "pipeline" / "task_candidates.py").read_text()
        assert "is_wash_sale_blocked_with_cost" in src, \
            "AUDIT REGRESSION: task_candidates.py no longer uses cost-aware " \
            "wash-sale helper. Reverted to binary block."


# ── Boundary tests ───────────────────────────────────────────────────────────

class TestQPWashSaleBoundary:

    def test_empty_qp_tickers_returns_empty_mask(self):
        today = datetime.date(2026, 5, 9)
        ctx = _make_ctx(tickers=[], today=today, last_sells={}, last_pls={})
        mask = _run_mask(ctx)
        assert isinstance(mask, np.ndarray)
        assert len(mask) == 0

    def test_string_date_format(self):
        """last_sell_dates entries can be ISO date strings (from JSON state).
        The cost-aware helper handles parsing."""
        today = datetime.date(2026, 5, 9)
        ctx = _make_ctx(
            tickers=["AAPL"],
            today=today,
            last_sells={"AAPL": "2026-04-25"},   # 14d ago
            last_pls={"AAPL": -100.0},   # loss
        )
        mask = _run_mask(ctx)
        # Loss within window → blocked
        assert mask[0] == True
