"""Atomicity regression: rotation pair sell+buy must commit together.

Bug found in deep audit 2026-04-24: EmitRotationsTask appended the
sell exit BEFORE confirming the buy could execute. If buy failed
(no price / shares<1), the held position closed with no replacement
— effectively forced cash exit on what looked like a "rotation".
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


@dataclass
class _FakePair:
    sell_ticker: str
    buy_ticker:  str
    sell_score:  float = 0.30
    buy_score:   float = 0.50
    sell_er:     float = 0.0
    buy_er:      float = 0.05
    horizon_days: int = 20
    raw_advantage: float = 0.05
    tax_drag:    float = 0.0
    transaction_cost: float = 0.0
    net_advantage: float = 0.05
    threshold:   float = 0.03
    margin_realized: float = 0.02


def _ctx_with_pair(prices, cash=100_000.0, portfolio_value=100_000.0):
    cand = SimpleNamespace(
        ticker="B",
        rank_score=0.50, panel_score=0.5,
        mu=0.05, sigma=0.02, expected_return=0.05,
    )
    return SimpleNamespace(
        config = {
            "regime_params": {"BULL_CALM": {
                "max_position_pct": 0.10, "cash_reserve_pct": 0.0,
            }},
            "ranking": {"panel_scoring": {
                "sizing":       {"enabled": False},
                "sigma_sizing": {"enabled": False},
            }},
        },
        regime          = "BULL_CALM",
        confidence      = 1.0,
        rotations       = [_FakePair(sell_ticker="A", buy_ticker="B")],
        ranked          = [cand],
        exits           = [],
        orders          = [],
        prices          = prices,
        portfolio_value = portfolio_value,
        cash            = cash,
        counters        = {},
    )


class TestAtomicRotation:
    def test_buy_succeeds_exit_committed(self):
        from kernel.pipeline.task_rotation import EmitRotationsTask

        ctx = _ctx_with_pair(prices={"B": 50.0})
        EmitRotationsTask().run(ctx)

        # Buy confirmed (cash=100k, price=50, max 10% of pv=10k → 200 shares)
        assert len(ctx.orders) == 1
        assert ctx.orders[0]["ticker"] == "B"
        # Exit committed too
        assert len(ctx.exits) == 1
        assert ctx.exits[0][0] == "A"

    def test_no_price_skips_entire_pair(self):
        """Bug-fix invariant: if buy can't price, the SELL must not commit."""
        from kernel.pipeline.task_rotation import EmitRotationsTask

        ctx = _ctx_with_pair(prices={})   # B has no price
        EmitRotationsTask().run(ctx)

        # Buy skipped → no order
        assert ctx.orders == []
        # CRITICAL: sell must also be skipped (atomic rotation)
        assert ctx.exits == [], (
            "Atomic rotation invariant violated: sell exit committed "
            "without a corresponding buy. The position would close to "
            "cash with no replacement.")

    def test_insufficient_cash_skips_entire_pair(self):
        from kernel.pipeline.task_rotation import EmitRotationsTask

        # cash=$100, price=$200/share → can't afford 1 share
        ctx = _ctx_with_pair(prices={"B": 200.0}, cash=100.0,
                              portfolio_value=100.0)
        EmitRotationsTask().run(ctx)

        assert ctx.orders == []
        assert ctx.exits == [], (
            "Insufficient-cash buy must skip the entire pair, including "
            "the sell side.")
