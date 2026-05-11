"""Slice 3 of ExecutionPipeline refactor — :class:`SimBackend` parity.

The sim backend owns:
* cash (mutated by every buy/sell)
* per-ticker position quantity (long-only)
* per-ticker volume-weighted avg cost basis (for unrealized P&L)
* per-bar price cache (mark-to-market + fill price)
* fee schedule via :mod:`kernel.execution.fees`
* slippage via :mod:`kernel.execution.slippage` (mid-price adjustment)
* T+2 settlement queue (sell proceeds become available T+2 bars later)

It does NOT own: tax accounting, lot disposal (FIFO/HIFO), trade log,
equity curve, wash-sale stamping. Those stay in :class:`SimAdapter`'s
post-pipeline hook (slice 3b) since they're strategy-level concerns,
not broker-level ones.

The tests below pin behavioural parity with the existing
``SimAdapter._apply_buy`` / ``_apply_sell`` cash + position math
(ignoring tax / lot ledger, which slice 3b validates separately).
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from kernel.execution import OrderIntent, OrderSide
from kernel.execution.backend_sim import SimBackend


def _make_backend(cash=100_000.0, **kw) -> SimBackend:
    return SimBackend(starting_cash=cash, **kw)


# ─── Construction + config ─────────────────────────────────────────────────


class TestSimBackendConstruction:
    def test_implements_execution_backend_protocol(self):
        """All 6 ABC methods must be present on SimBackend."""
        from kernel.execution.backend import ExecutionBackend
        b = _make_backend()
        assert isinstance(b, ExecutionBackend)
        for m in ("place_market_order", "get_position_quantity",
                  "get_unrealized_pnl", "get_cash", "get_portfolio_value",
                  "get_last_price"):
            assert hasattr(b, m)

    def test_rejects_negative_starting_cash(self):
        with pytest.raises(ValueError):
            SimBackend(starting_cash=-1.0)

    def test_starting_cash_finite_guard(self):
        with pytest.raises(ValueError):
            SimBackend(starting_cash=float("nan"))


# ─── Price marking + mark-to-market ────────────────────────────────────────


class TestSimBackendMarkToMarket:
    def test_update_bar_prices_overwrites_cache(self):
        b = _make_backend()
        b.update_bar_prices({"AAPL": 100.0, "MSFT": 300.0},
                             today=pd.Timestamp("2025-01-15"))
        assert b.get_last_price("AAPL") == 100.0
        assert b.get_last_price("MSFT") == 300.0

    def test_skips_non_finite_prices_silently(self):
        """A delisted ticker arriving as NaN must NOT poison the cache."""
        b = _make_backend()
        b.update_bar_prices({"AAPL": float("nan"), "MSFT": 300.0},
                             today=pd.Timestamp("2025-01-15"))
        with pytest.raises(KeyError):
            b.get_last_price("AAPL")
        assert b.get_last_price("MSFT") == 300.0


# ─── BUY parity vs SimAdapter._apply_buy ────────────────────────────────────


class TestSimBackendBuyParity:
    def test_buy_cash_math_matches_legacy(self):
        """Identical to ``SimAdapter._apply_buy`` line 1087:
        ``self._cash -= invest`` where invest = shares * fill_price + fees.
        """
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="initial", exit_type=None,
        )
        fill = b.place_market_order(intent)
        # With slippage off (default exec_enabled=False) fill_price == market.
        assert math.isclose(fill.price, 100.0)
        assert fill.fees == 0.0  # buy custom_bps=0 default
        assert b.get_cash() == 90_000.0
        assert b.get_position_quantity("AAPL") == 100

    def test_buy_with_slippage_increases_fill_price(self):
        """Slippage on: BUY pays ask (mid + half-spread). Default 2bps."""
        b = _make_backend(cash=100_000.0, exec_enabled=True)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        assert fill.price > 100.0  # slipped above mid
        # 2 bps = 0.02% → fill ≈ 100.02
        assert math.isclose(fill.price, 100.02, rel_tol=1e-6)

    def test_buy_topup_updates_avg_cost(self):
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=50,
            target_pct=0.05, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        b.update_bar_prices({"AAPL": 200.0}, today=pd.Timestamp("2025-02-01"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=50,
            target_pct=0.15, today=pd.Timestamp("2025-02-01"),
            reason="topup", exit_type=None,
        ))
        # avg cost = (50*100 + 50*200) / 100 = 150
        assert math.isclose(b.get_unrealized_pnl("AAPL"), 100 * (200 - 150))

    def test_buy_rejects_insufficient_cash(self):
        """Matches sim._apply_buy:1083 — invest > cash + ε → warn + skip."""
        b = _make_backend(cash=100.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        with pytest.raises(ValueError, match="insufficient cash"):
            b.place_market_order(OrderIntent(
                ticker="AAPL", side=OrderSide.BUY, shares=100,
                target_pct=1.0, today=pd.Timestamp("2025-01-02"),
                reason="x", exit_type=None,
            ))


# ─── SELL parity vs SimAdapter._apply_sell ──────────────────────────────────


class TestSimBackendSellParity:
    def test_sell_proceeds_credited_T0_when_t2_disabled(self):
        """Default exec_enabled=False ⇒ T+2 OFF ⇒ proceeds immediate."""
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        b.update_bar_prices({"AAPL": 105.0}, today=pd.Timestamp("2025-02-01"))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-02-01"),
            reason="exit", exit_type="model_sell",
        ))
        # default fees == 0 (exec off), proceeds full T+0
        assert fill.fees == 0.0
        assert math.isclose(b.get_cash(), 100_000.0 - 100*100 + 100*105)
        assert b.get_position_quantity("AAPL") == 0

    def test_sell_with_t2_settlement_queues_proceeds(self):
        """exec_enabled=True ⇒ T+2 ⇒ proceeds NOT in cash until 2 bars later."""
        b = _make_backend(cash=100_000.0, exec_enabled=True, t2_days=2)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        cash_after_buy = b.get_cash()
        b.update_bar_prices({"AAPL": 105.0}, today=pd.Timestamp("2025-02-01"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-02-01"),
            reason="exit", exit_type="model_sell",
        ))
        # Cash NOT yet credited (T+2 queue holds proceeds).
        assert math.isclose(b.get_cash(), cash_after_buy)
        # Drain after settlement date.
        b.drain_settled(pd.Timestamp("2025-02-04"))
        assert b.get_cash() > cash_after_buy

    def test_sell_with_slippage_decreases_fill_price(self):
        b = _make_backend(cash=100_000.0, exec_enabled=True)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-15"))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="x", exit_type="model_sell",
        ))
        # SELL hits bid (mid - half-spread). Default 2bps below mid.
        assert fill.price < 100.0
        assert math.isclose(fill.price, 99.98, rel_tol=1e-6)

    def test_partial_sell_decrements_position(self):
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-15"))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=30,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="trim", exit_type="qp_sell",
        ))
        assert fill.shares == 30
        assert b.get_position_quantity("AAPL") == 70

    def test_sell_without_position_raises(self):
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        with pytest.raises(ValueError, match="no position"):
            b.place_market_order(OrderIntent(
                ticker="AAPL", side=OrderSide.SELL, shares=None,
                target_pct=0.0, today=pd.Timestamp("2025-01-02"),
                reason="x", exit_type="stop_loss",
            ))


# ─── Audit regression guards ───────────────────────────────────────────────


class TestSimBackendAuditRegressionGuards:
    """Pin §5.13.11 NaN-guard invariants from sim's existing _apply_*."""

    def test_buy_nan_price_seeded_raises_at_seed_not_at_buy(self):
        """update_bar_prices already filters NaN; an attempt to buy a
        ticker with no seeded price must fail loudly."""
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": float("nan")},
                             today=pd.Timestamp("2025-01-02"))
        with pytest.raises(KeyError):
            b.place_market_order(OrderIntent(
                ticker="AAPL", side=OrderSide.BUY, shares=100,
                target_pct=0.10, today=pd.Timestamp("2025-01-02"),
                reason="x", exit_type=None,
            ))

    def test_cash_never_becomes_nan(self):
        """Pin the invariant that broke sim's equity curve 3 times in
        Q2 2026: a non-finite mutation poisoning self._cash. Every code
        path must finite-guard."""
        b = _make_backend(cash=100_000.0)
        b.update_bar_prices({"AAPL": 100.0}, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        assert math.isfinite(b.get_cash())
        b.update_bar_prices({"AAPL": 105.0}, today=pd.Timestamp("2025-02-01"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-02-01"),
            reason="x", exit_type="model_sell",
        ))
        assert math.isfinite(b.get_cash())
