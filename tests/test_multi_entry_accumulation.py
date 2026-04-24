"""Multi-entry accumulation — per-session buy cap.

User request 2026-04-24: "65% OK, but not from one session — allow
model to buy same stock multiple times and accumulate to 65%".

Implementation: `ranking.kelly_sizing.per_session_buy_cap` caps any
single-bar BUY order at that fraction of portfolio. kelly_target can
still reach max_concentration over multiple sessions via TopUpHeldTask.

Default None = no cap (preserves v4 behaviour).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.task_selection import SizeAndEmitTask  # noqa: E402
from kernel.pipeline.task_topup import TopUpHeldTask  # noqa: E402


# ── SizeAndEmitTask caps new-buy orders ───────────────────────────────────────

def _size_ctx(kelly_target: float, per_session: float | None):
    class Cand:
        ticker          = "NVDA"
        panel_score     = 0.60
        sigma           = 0.02
        rank_score      = 0.60
        rs_score        = 0.0
        detail          = ""
        expected_return = 0.08
        kelly_target_pct = kelly_target
    return SimpleNamespace(
        regime = "BULL_CALM", confidence = 1.0,
        ranked = [Cand()], _selected = ["NVDA"],
        prices = {"NVDA": 100.0}, cash = 100_000, portfolio_value = 100_000,
        orders = [], bear_only = False, skip_buys = False,
        today = None, regime_state = None, counters = {},
        config = {
            "regime_params": {"BULL_CALM": {"max_position_pct": 0.70,
                                             "cash_reserve_pct": 0.0}},
            "ranking": {
                "kelly_sizing": {
                    "enabled":               True,
                    "disable_extra_multipliers": True,   # pure Kelly for crisp test
                    "per_session_buy_cap":   per_session,
                },
                "panel_scoring": {
                    "sizing":        {"enabled": True, "floor": 0.5, "ceiling": 1.5,
                                       "min_mult": 0.3},
                    "sigma_sizing":  {"enabled": True, "floor": 0.5, "ceiling": 1.5},
                },
            },
        },
    )


class TestSizeAndEmitPerSessionCap:
    def test_cap_absent_is_no_op(self):
        """Default: no cap, kelly_target = 0.60 → sizes full 0.60 portfolio fraction."""
        ctx = _size_ctx(kelly_target=0.60, per_session=None)
        SizeAndEmitTask().run(ctx)
        o = ctx.orders[0]
        # 100k × 0.60 / $100 = ~600 shares
        assert 580 <= o["shares"] <= 600

    def test_cap_trims_big_buy(self):
        """kelly_target=0.60 capped to per_session=0.35 → only ~350 shares."""
        ctx = _size_ctx(kelly_target=0.60, per_session=0.35)
        SizeAndEmitTask().run(ctx)
        o = ctx.orders[0]
        assert 340 <= o["shares"] <= 350

    def test_cap_above_kelly_is_inert(self):
        """kelly_target=0.20 < cap=0.35 → cap doesn't fire, full kelly size."""
        ctx = _size_ctx(kelly_target=0.20, per_session=0.35)
        SizeAndEmitTask().run(ctx)
        o = ctx.orders[0]
        assert 195 <= o["shares"] <= 200

    def test_cap_zero_or_negative_noop(self):
        """0 or negative cap is treated as disabled."""
        ctx = _size_ctx(kelly_target=0.60, per_session=0.0)
        SizeAndEmitTask().run(ctx)
        o = ctx.orders[0]
        assert o["shares"] >= 500   # uncapped


# ── TopUpHeldTask also honours the cap ───────────────────────────────────────

def _hs(shares, kelly_target, entry_price=100.0):
    import datetime
    from kernel.exits import HoldingState
    h = HoldingState(
        entry_price=entry_price, entry_date=datetime.date(2026, 1, 15),
        shares=shares, high_watermark=entry_price,
    )
    h.kelly_target_pct = kelly_target
    return h


def _topup_ctx(holding_shares, kelly_target, per_session):
    return SimpleNamespace(
        holdings = {"NVDA": _hs(holding_shares, kelly_target)},
        prices = {"NVDA": 100.0},
        portfolio_value = 100_000,
        orders = [], exits = [], rotations = [],
        bear_only = False, skip_buys = False,
        regime = "BULL_CALM", confidence = 1.0,
        config = {"ranking": {"kelly_sizing": {
            "enabled":             True,
            "top_up_threshold":    0.05,
            "per_session_buy_cap": per_session,
        }}},
    )


class TestTopUpPerSessionCap:
    def test_topup_uncapped_uses_full_delta(self):
        """No cap: delta=0.50 (kelly 0.60 - current 0.10) → 500 extra shares."""
        ctx = _topup_ctx(holding_shares=100, kelly_target=0.60, per_session=None)
        TopUpHeldTask().run(ctx)
        o = ctx.orders[0]
        # 0.50 delta × 100k / 100 = 500
        assert 490 <= o["shares"] <= 500

    def test_topup_capped_buys_at_most_cap(self):
        """cap=0.20: even with delta=0.50, only buy 0.20 this session."""
        ctx = _topup_ctx(holding_shares=100, kelly_target=0.60, per_session=0.20)
        TopUpHeldTask().run(ctx)
        o = ctx.orders[0]
        # 0.20 × 100k / 100 = 200
        assert 195 <= o["shares"] <= 200

    def test_topup_under_cap_passes_through(self):
        """delta=0.08 < cap=0.20 → buy the full delta."""
        ctx = _topup_ctx(holding_shares=120, kelly_target=0.20, per_session=0.20)
        TopUpHeldTask().run(ctx)
        o = ctx.orders[0]
        # 0.08 × 100k / 100 = 80
        assert 75 <= o["shares"] <= 80
