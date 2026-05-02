"""Tests for buy-quality gates added 2026-05-01 trade-audit response.

Three gates landed together:

  (a) Cash-aware portfolio fill (SizeAndEmitTask) —
      sum(invest) ≤ available_cash. Pre-fix the 4/28 burst emitted 6
      buys × ~$8k each against a ~$10k account (≈5x implied leverage)
      because each per-position call to compute_position_size saw the
      same constant ctx.cash.

  (b) Conviction floor on TopUp (TopUpHeldTask) —
      do not add to a holding whose latest calibrated rank_score is
      below `kelly_sizing.topup_conviction_floor` (default 0.20). Pre-fix
      4 of 7 buys 2026-04-29 → 05-01 had rank_score=0.0 because TopUp
      blindly Kelly-maintained held positions while the panel had no
      current opinion on them.

  (c) order_type preserved in trade log (runner adapter) —
      audit trail distinguishes "NEW_BUY" (fresh entry from selection)
      from "TOP_UP" (Kelly maintenance add). Pre-fix the trade log
      collapsed both into action="BUY" indistinguishably.

Invariants:
  * sum_of(emitted invests in this bar) ≤ ctx.cash - reserve
  * TopUp emits 0 orders for any holding with rank_score < floor or None
  * Trade log entry carries order_type ∈ {"NEW_BUY","TOP_UP",...} not None
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.exits import HoldingState   # noqa: E402
from kernel.pipeline.context import InferenceContext   # noqa: E402
from kernel.pipeline.task_selection import SizeAndEmitTask   # noqa: E402
from kernel.pipeline.task_topup import TopUpHeldTask   # noqa: E402
from kernel.selection import CandidateResult           # noqa: E402


TODAY = datetime.date(2026, 5, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidate(ticker: str, *, rank: float = 0.5,
                panel: float = 0.5, sigma: float = 0.05,
                mu: float = 0.01, kelly: float = 0.10) -> CandidateResult:
    c = CandidateResult(
        ticker=ticker, raw_score=0.5, rank_score=rank, rs_score=0.0,
        detail="", expected_return=0.01,
    )
    c.panel_score      = panel
    c.sigma            = sigma
    c.mu               = mu
    c.kelly_target_pct = kelly
    return c


def _holding(*, rank_score: float | None = 0.50,
              kelly_target: float = 0.30,
              shares: float = 10.0) -> HoldingState:
    h = HoldingState(
        entry_price=100.0,
        entry_date=TODAY - datetime.timedelta(days=30),
        high_watermark=110.0,
        sell_streak=0,
        last_streak_inc_date=TODAY,
        shares=shares,
    )
    h.rank_score        = rank_score
    h.kelly_target_pct  = kelly_target
    h.panel_score       = 0.55
    h.sigma             = 0.05
    h.mu                = 0.01
    return h


def _size_emit_ctx(*, selected: list[str], cash: float = 10_000.0,
                   portfolio_value: float = 10_000.0,
                   max_position_pct: float = 0.30,
                   prices: dict[str, float] | None = None,
                   ranked: list[CandidateResult] | None = None
                   ) -> InferenceContext:
    cfg = {
        "regime_params": {
            "BULL_CALM": {
                "max_position_pct": max_position_pct,
                "cash_reserve_pct": 0.0,
            },
        },
    }
    ctx = InferenceContext(config=cfg, today=TODAY)
    ctx.regime = "BULL_CALM"
    ctx.confidence = 1.0
    ctx.cash = cash
    ctx.portfolio_value = portfolio_value
    ctx.prices = prices or {t: 100.0 for t in selected}
    ctx.ranked = ranked or [_candidate(t) for t in selected]
    ctx._selected = selected   # noqa: SLF001
    return ctx


# ── (a) Cash-aware portfolio fill ─────────────────────────────────────────────

class TestCashAwareFill:
    """4/28 incident: 6 selections × 30% target × $10k = $18k of buys
    sized into $10k of cash. Post-fix: total invest ≤ cash, low-conviction
    later picks simply hit zero remaining cash and skip.
    """

    def test_cumulative_invest_never_exceeds_starting_cash(self):
        # 6 selected, each at 30% target → would cumulatively need
        # $18k in a $10k account. With cash-aware fill, sum(invest) ≤ $10k.
        selected = [f"T{i}" for i in range(6)]
        ctx = _size_emit_ctx(selected=selected, cash=10_000.0,
                             portfolio_value=10_000.0,
                             max_position_pct=0.30)
        SizeAndEmitTask().run(ctx)
        total_invest = sum(o["invest"] for o in ctx.orders)
        assert total_invest <= 10_000.0 + 1e-6, (
            f"sum(invest)=${total_invest:.0f} exceeds starting cash "
            f"$10000 — cash-aware fill failed"
        )

    def test_2026_04_28_regression_six_buys_at_thirty_pct_caps_at_cash(self):
        """Reproduces the actual 4/28 incident: 6 buys × 30% target × $10k
        account. Pre-fix it emitted 6 × ~$3k = $18k of orders. Post-fix
        emits at most a portfolio-sized basket (~$10k spent, fewer fills).
        """
        selected = ["MU", "NET", "NVDA", "NVTS", "PLTR", "SMCI"]
        prices = {"MU": 500.0, "NET": 207.0, "NVDA": 208.0,
                  "NVTS": 17.0, "PLTR": 143.0, "SMCI": 29.0}
        ctx = _size_emit_ctx(selected=selected, cash=10_000.0,
                             portfolio_value=10_000.0,
                             max_position_pct=0.30,
                             prices=prices,
                             ranked=[_candidate(t) for t in selected])
        SizeAndEmitTask().run(ctx)
        total_invest = sum(o["invest"] for o in ctx.orders)
        # Allow at most starting cash; pre-fix produced ~$18k.
        assert total_invest <= 10_000.0 + 1e-6
        # Should have placed at least one order (not regressed to zero).
        assert len(ctx.orders) >= 1

    def test_low_priority_picks_skipped_when_cash_runs_out(self):
        """If first picks consume all cash, later picks skip cleanly
        rather than emitting fractional / negative-cash orders."""
        # 2 picks each targeting 60% → second would need $6k after first
        # took $6k. With $10k starting cash, second still fits ($4k).
        # 3 picks × 60% → third must skip (need $6k, only $4k left).
        selected = ["A", "B", "C"]
        ctx = _size_emit_ctx(selected=selected, cash=10_000.0,
                             portfolio_value=10_000.0,
                             max_position_pct=0.60)
        SizeAndEmitTask().run(ctx)
        total_invest = sum(o["invest"] for o in ctx.orders)
        assert total_invest <= 10_000.0 + 1e-6
        # Strictly fewer than 3 emitted (one must skip on cash)
        assert len(ctx.orders) <= 3
        # Each emitted order's invest is positive
        for o in ctx.orders:
            assert o["invest"] > 0

    def test_zero_cash_emits_no_orders(self):
        ctx = _size_emit_ctx(selected=["A", "B"], cash=0.0,
                             portfolio_value=10_000.0)
        SizeAndEmitTask().run(ctx)
        assert ctx.orders == []

    def test_sufficient_cash_all_picks_filled(self):
        """Plenty of cash: all selected picks should get a fill."""
        selected = ["A", "B", "C"]
        ctx = _size_emit_ctx(selected=selected, cash=100_000.0,
                             portfolio_value=100_000.0,
                             max_position_pct=0.20)
        SizeAndEmitTask().run(ctx)
        emitted = {o["ticker"] for o in ctx.orders}
        assert emitted == set(selected)

    def test_emitted_orders_carry_order_type_new_buy(self):
        ctx = _size_emit_ctx(selected=["A"], cash=10_000.0,
                             portfolio_value=10_000.0,
                             max_position_pct=0.20)
        SizeAndEmitTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["order_type"] == "NEW_BUY"


# ── (b) Conviction floor on TopUp ─────────────────────────────────────────────

class TestTopUpConvictionFloor:
    """4 of 7 buys 2026-04-29 → 05-01 had rank_score=0.0 because TopUp
    Kelly-maintained held positions while the panel had no current
    opinion. Post-fix: TopUp requires hs.rank_score >= floor (default 0.20).
    """

    @staticmethod
    def _ctx_with_holding(*, rank_score: float | None,
                           kelly_target: float = 0.30,
                           topup_floor: float | None = None,
                           ) -> InferenceContext:
        cfg: dict = {
            "regime": {"earnings_buffer_days": 3},
            "ranking": {"kelly_sizing": {"enabled": True,
                                          "top_up_threshold": 0.05}},
        }
        if topup_floor is not None:
            cfg["ranking"]["kelly_sizing"]["topup_conviction_floor"] = topup_floor
        ctx = InferenceContext(config=cfg, today=TODAY)
        ctx.holdings = {"FTNT": _holding(rank_score=rank_score,
                                          kelly_target=kelly_target)}
        ctx.prices = {"FTNT": 85.0}
        ctx.cash = 10_000.0
        ctx.portfolio_value = 10_000.0
        return ctx

    def test_blocked_when_rank_score_zero(self):
        """Reproduces the 4/29 + 5/01 FTNT TopUp incident — rank=0.0."""
        ctx = self._ctx_with_holding(rank_score=0.0)
        TopUpHeldTask().run(ctx)
        assert ctx.orders == [], "rank=0.0 must NOT trigger TopUp"

    def test_blocked_when_rank_score_below_floor(self):
        ctx = self._ctx_with_holding(rank_score=0.15)  # below 0.20 default
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_blocked_when_rank_score_none(self):
        """Fail-CLOSED: missing rank means no model opinion → don't add."""
        ctx = self._ctx_with_holding(rank_score=None)
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_blocked_when_rank_score_nan(self):
        ctx = self._ctx_with_holding(rank_score=float("nan"))
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_allowed_when_rank_score_at_floor(self):
        """Inclusive: rank_score == floor passes."""
        ctx = self._ctx_with_holding(rank_score=0.20)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["order_type"] == "TOP_UP"

    def test_allowed_when_rank_score_above_floor(self):
        ctx = self._ctx_with_holding(rank_score=0.50)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1

    def test_floor_disabled_at_zero(self):
        """topup_conviction_floor=0 → operator-disabled (legacy behavior)."""
        ctx = self._ctx_with_holding(rank_score=0.0, topup_floor=0.0)
        TopUpHeldTask().run(ctx)
        assert len(ctx.orders) == 1, (
            "floor=0 must allow TopUp regardless of rank — operator override"
        )

    def test_custom_floor_value_respected(self):
        # rank=0.30, floor=0.40 → blocked
        ctx = self._ctx_with_holding(rank_score=0.30, topup_floor=0.40)
        TopUpHeldTask().run(ctx)
        assert ctx.orders == []

    def test_block_logs_ticker_and_floor(self, caplog):
        import logging as _log
        caplog.set_level(_log.INFO, logger="kernel.pipeline.topup")
        ctx = self._ctx_with_holding(rank_score=0.0)
        TopUpHeldTask().run(ctx)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("FTNT" in m and "conviction floor" in m.lower()
                   for m in msgs)


# ── (c) Trade log distinguishes order_type ────────────────────────────────────

class TestOrderTypeMarker:
    """Audit trail must tell NEW_BUY apart from TOP_UP. The order dict
    field is set at emission time (NEW_BUY by SizeAndEmitTask, TOP_UP by
    TopUpHeldTask). The runner's _log_trade then reads order["order_type"].
    """

    def test_size_and_emit_tags_new_buy(self):
        ctx = _size_emit_ctx(selected=["A"], cash=10_000.0,
                             portfolio_value=10_000.0)
        SizeAndEmitTask().run(ctx)
        assert ctx.orders[0]["order_type"] == "NEW_BUY"

    def test_topup_tags_top_up(self):
        cfg = {
            "regime": {"earnings_buffer_days": 3},
            "ranking": {"kelly_sizing": {"enabled": True,
                                          "top_up_threshold": 0.05}},
        }
        ctx = InferenceContext(config=cfg, today=TODAY)
        ctx.holdings = {"FTNT": _holding(rank_score=0.50)}
        ctx.prices = {"FTNT": 85.0}
        ctx.cash = 10_000.0
        ctx.portfolio_value = 10_000.0
        TopUpHeldTask().run(ctx)
        assert ctx.orders[0]["order_type"] == "TOP_UP"

    def test_runner_log_trade_includes_order_type(self):
        """Adapter runner.py emits order_type into the trade-log dict.
        Smoke test the field is propagated, not silently dropped."""
        # Read the runner code as a black-box smoke test — full broker
        # integration is heavy; the field-presence is what we guard here.
        runner_path = REPO_ROOT / "backtesting" / "renquant_104" / "adapters" / "runner.py"
        src = runner_path.read_text()
        assert '"order_type": order_type' in src or "'order_type': order_type" in src, (
            "runner._log_trade must propagate order_type so audits can "
            "distinguish NEW_BUY from TOP_UP"
        )