"""Slice 4a of ExecutionPipeline refactor — :class:`LeanBackend` contract.

LEAN's brokerage layer is the source of truth for cash + position state
in the backtest path; the adapter must NOT maintain a parallel mirror
(§5.13.5). :class:`LeanBackend` is therefore a thin proxy: every read
delegates to ``algo.Portfolio`` / ``algo.Securities``, every order
placement delegates to ``algo.MarketOrder`` / ``algo.Liquidate``.

The tests below use a minimal duck-typed ``MockAlgo`` so we can verify
behaviour without spinning up the LEAN Docker. The contract MockAlgo
mimics matches the QC Algorithm surface the production adapter touches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from kernel.execution import OrderIntent, OrderSide
from kernel.execution.backend_lean import LeanBackend


# ─── MockAlgo (test-only QC harness) ──────────────────────────────────────


@dataclass
class _MockPosition:
    Quantity: float = 0.0
    UnrealizedProfit: float = 0.0

    # Compat aliases for older test code that used lowercase attributes.
    @property
    def quantity(self):
        return self.Quantity

    @quantity.setter
    def quantity(self, value):
        self.Quantity = value


@dataclass
class _MockSecurity:
    Price: float = 0.0

    @property
    def price(self):
        return self.Price

    @price.setter
    def price(self, value):
        self.Price = value


class MockAlgo:
    """A minimal in-memory stand-in for ``QCAlgorithm`` for unit tests.

    Records every order placement call so tests can assert the LEAN API
    is invoked correctly. State mutation on fill is modelled in-process
    — LeanBackend assumes synchronous fills at last close, which matches
    LEAN's behaviour for `MarketOrder` at end-of-bar.
    """

    def __init__(self, starting_cash: float = 100_000.0):
        self._cash = starting_cash
        self._positions: dict[str, _MockPosition] = {}
        self._securities: dict[str, _MockSecurity] = {}
        self.symbols: dict[str, str] = {}  # ticker → symbol (identity here)
        self.order_log: list[tuple] = []   # records (method, ticker, *args)

    # ── set up prices + symbols ──

    def set_price(self, ticker: str, price: float) -> None:
        self._securities.setdefault(ticker, _MockSecurity()).price = price
        self.symbols.setdefault(ticker, ticker)

    # ── QC-flavoured accessors ──

    class _PortfolioFacade:
        def __init__(self, owner: "MockAlgo"):
            self._o = owner

        def __getitem__(self, sym: str) -> _MockPosition:
            return self._o._positions.setdefault(sym, _MockPosition())

        @property
        def Cash(self) -> float:
            return self._o._cash

        @property
        def TotalPortfolioValue(self) -> float:
            total = self._o._cash
            for ticker, pos in self._o._positions.items():
                sec = self._o._securities.get(ticker)
                if sec is None:
                    continue
                total += pos.quantity * sec.price
            return total

    @property
    def Portfolio(self) -> "MockAlgo._PortfolioFacade":
        return MockAlgo._PortfolioFacade(self)

    class _SecuritiesFacade:
        def __init__(self, owner: "MockAlgo"):
            self._o = owner

        def __getitem__(self, sym: str) -> _MockSecurity:
            return self._o._securities.setdefault(sym, _MockSecurity())

    @property
    def Securities(self) -> "MockAlgo._SecuritiesFacade":
        return MockAlgo._SecuritiesFacade(self)

    # ── Order placement (records + applies fill at last close) ──

    def MarketOrder(self, sym: str, quantity: int) -> None:
        self.order_log.append(("MarketOrder", sym, quantity))
        price = self._securities[sym].price
        # quantity < 0 means sell, > 0 means buy
        pos = self._positions.setdefault(sym, _MockPosition())
        pos.quantity += quantity
        self._cash -= quantity * price  # debit on buy, credit on sell

    def Liquidate(self, sym: str) -> None:
        self.order_log.append(("Liquidate", sym))
        pos = self._positions.get(sym)
        if pos is None or pos.quantity == 0:
            return
        price = self._securities[sym].price
        self._cash += pos.quantity * price
        pos.quantity = 0.0

    def SetHoldings(self, sym: str, target_pct: float) -> None:
        self.order_log.append(("SetHoldings", sym, target_pct))
        # Approximate LEAN behaviour: compute target shares from target_pct
        # over current portfolio value, place market order for the delta.
        price = self._securities[sym].price
        pv = self.Portfolio.TotalPortfolioValue
        target_value = target_pct * pv
        target_shares = int(target_value / price)
        pos = self._positions.setdefault(sym, _MockPosition())
        delta = target_shares - int(pos.quantity)
        pos.quantity = target_shares
        self._cash -= delta * price


# ─── LeanBackend tests ────────────────────────────────────────────────────


def _algo_with_aapl(starting_cash=100_000.0, aapl_price=100.0):
    a = MockAlgo(starting_cash=starting_cash)
    a.set_price("AAPL", aapl_price)
    return a


class TestLeanBackendABCConformance:
    def test_is_execution_backend(self):
        from kernel.execution.backend import ExecutionBackend
        a = _algo_with_aapl()
        b = LeanBackend(a)
        assert isinstance(b, ExecutionBackend)
        for m in ("place_market_order", "get_position_quantity",
                  "get_unrealized_pnl", "get_cash", "get_portfolio_value",
                  "get_last_price"):
            assert hasattr(b, m)


class TestLeanBackendReadsDelegateToAlgo:
    def test_cash_reads_portfolio_cash(self):
        a = _algo_with_aapl(starting_cash=12_345.0)
        b = LeanBackend(a)
        assert b.get_cash() == 12_345.0

    def test_position_quantity_reads_portfolio(self):
        a = _algo_with_aapl()
        # Seed a holding via the algo's portfolio facade
        a._positions["AAPL"] = _MockPosition(Quantity=50.0)
        b = LeanBackend(a)
        assert b.get_position_quantity("AAPL") == 50.0
        assert b.get_position_quantity("ZZZZ") == 0.0  # unheld

    def test_last_price_reads_securities(self):
        a = _algo_with_aapl(aapl_price=187.40)
        b = LeanBackend(a)
        assert math.isclose(b.get_last_price("AAPL"), 187.40)

    def test_last_price_unknown_ticker_raises(self):
        a = _algo_with_aapl()
        b = LeanBackend(a)
        with pytest.raises(KeyError):
            b.get_last_price("ZZZZ")

    def test_portfolio_value_reads_total(self):
        a = _algo_with_aapl(starting_cash=50_000.0, aapl_price=100.0)
        a._positions["AAPL"] = _MockPosition(Quantity=500.0)
        b = LeanBackend(a)
        # 50_000 cash + 500 × 100 = 100_000
        assert math.isclose(b.get_portfolio_value(), 100_000.0)


class TestLeanBackendOrderPlacement:
    def test_buy_routes_to_exact_share_market_order(self):
        """BUY intents execute the pipeline-sized whole-share order."""
        a = _algo_with_aapl()
        b = LeanBackend(a)
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        )
        fill = b.place_market_order(intent)
        assert ("MarketOrder", "AAPL", 100) in a.order_log
        assert fill.ticker == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.shares == 100
        assert math.isclose(fill.price, 100.0)
        # LEAN tracks fees natively; the synchronous Fill carries 0
        # (post-bar Portfolio.TotalFees will reflect the real number).
        assert fill.fees == 0.0

    def test_full_sell_routes_to_liquidate(self):
        a = _algo_with_aapl()
        a._positions["AAPL"] = _MockPosition(Quantity=100.0)
        b = LeanBackend(a)
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="stop", exit_type="stop_loss",
        )
        fill = b.place_market_order(intent)
        assert ("Liquidate", "AAPL") in a.order_log
        assert fill.shares == 100  # resolved from current position
        assert math.isclose(fill.price, 100.0)
        # Position closed
        assert b.get_position_quantity("AAPL") == 0.0

    def test_partial_sell_routes_to_market_order_negative(self):
        a = _algo_with_aapl()
        a._positions["AAPL"] = _MockPosition(Quantity=100.0)
        b = LeanBackend(a)
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=30,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="qp_sell", exit_type="qp_sell",
        )
        fill = b.place_market_order(intent)
        assert ("MarketOrder", "AAPL", -30) in a.order_log
        assert fill.shares == 30
        assert b.get_position_quantity("AAPL") == 70

    def test_sell_unheld_raises(self):
        """LEAN's brokerage will reject a sell for zero position; we
        surface this loudly to catch pipeline-state-mismatch bugs."""
        a = _algo_with_aapl()
        b = LeanBackend(a)
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="zombie", exit_type="stop_loss",
        )
        with pytest.raises(ValueError, match="no position"):
            b.place_market_order(intent)


class TestLeanBackendRegressionGuards:
    """AUDIT REGRESSION GUARD: pin §5.13.5 + §5.13.11 invariants."""

    def test_buy_intent_uses_pipeline_shares_not_lean_recompute(self):
        """The Fill records the pipeline's REQUESTED shares; LEAN may
        internally round to a different lot count via SetHoldings, but
        the OrderIntent carries the authoritative quantity (matches
        sim behaviour). Asserting this prevents a future "let LEAN
        decide" change from silently desyncing sim and LEAN sizing."""
        a = _algo_with_aapl(starting_cash=100_000.0, aapl_price=187.40)
        b = LeanBackend(a)
        intent = OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.1874, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        )
        fill = b.place_market_order(intent)
        assert fill.shares == 100

    def test_get_last_price_finite(self):
        """A NaN price on algo.Securities[sym].Price would silently
        propagate into Fill.price → __post_init__ raises (defensive
        against a delisted ticker leaking through)."""
        a = _algo_with_aapl()
        a._securities["AAPL"].price = float("nan")
        b = LeanBackend(a)
        with pytest.raises(ValueError, match="finite"):
            b.get_last_price("AAPL")
