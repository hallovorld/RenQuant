"""Slice 1 of ExecutionPipeline refactor (CLAUDE.md §5.13.5 / §1c).

Pins the **execution-backend contract** that sim/LEAN/live will all
route through. Today there are three monolithic ``commit()`` bodies
(``adapters/sim.py``:616, ``adapters/runner.py``:797, ``adapters/lean.py``:202)
each ≥150 lines doing the same logical work — placing exit orders,
computing tax, mutating holdings, placing buy orders, updating telemetry.
The divergence between them is the root cause of the sim/LEAN/live
inconsistency surfaced by the 2026-05-10 audit (Z9 broker-side stops
only present in runner; partial-trim wash-sale exemption only in sim).

This slice introduces only the *interface*:

* :class:`OrderIntent` — immutable request the pipeline emits ("BUY 100
  AAPL at <today> targeting 5% of equity, reason=...").
* :class:`Fill` — immutable response the backend returns once an order
  executes ("filled 100 AAPL at $187.40 with $0.18 fees on <today>").
* :class:`ExecutionBackend` — abstract base every concrete backend
  (sim/Alpaca/LEAN) implements. Methods are intentionally minimal: the
  pipeline does ALL business logic, the backend ONLY translates intents
  to broker API calls and reports back state.
* :class:`FakeBackend` — in-memory reference implementation used by the
  pipeline tests so they can run without a broker or sim harness.

Per §5.13.10 every ``ExecutionBackend`` subclass MUST be greppable in
prod code before slice 1 is considered shipped. We assert FakeBackend
is *only* used by tests via the marker comment ``# fake-backend-test-only``.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

# All three symbols live in kernel.execution to match the existing
# fees / slippage / t2_settlement single-source-of-truth pattern.
from kernel.execution.backend import ExecutionBackend, FakeBackend
from kernel.execution.types import Fill, OrderIntent, OrderSide


# ─── OrderIntent: immutability + required fields ──────────────────────────


class TestOrderIntentContract:
    """OrderIntent is a frozen dataclass with field-level finite guards."""

    def test_frozen_dataclass(self):
        intent = OrderIntent(
            ticker="AAPL",
            side=OrderSide.BUY,
            shares=100,
            target_pct=0.05,
            today=pd.Timestamp("2025-01-02"),
            reason="initial entry",
            exit_type=None,
        )
        with pytest.raises((AttributeError, Exception)):
            intent.shares = 200  # type: ignore[misc]

    def test_buy_requires_positive_shares_and_target_pct(self):
        with pytest.raises(ValueError, match="shares"):
            OrderIntent(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=0,
                target_pct=0.05,
                today=pd.Timestamp("2025-01-02"),
                reason="initial",
                exit_type=None,
            )
        with pytest.raises(ValueError, match="target_pct"):
            OrderIntent(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=100,
                target_pct=0.0,
                today=pd.Timestamp("2025-01-02"),
                reason="initial",
                exit_type=None,
            )

    def test_full_sell_allows_shares_none(self):
        """Full liquidate: shares=None means 'close entire position'."""
        intent = OrderIntent(
            ticker="AAPL",
            side=OrderSide.SELL,
            shares=None,
            target_pct=0.0,
            today=pd.Timestamp("2025-01-02"),
            reason="stop_loss",
            exit_type="stop_loss",
        )
        assert intent.shares is None
        assert intent.is_full_liquidate

    def test_partial_sell_requires_positive_shares(self):
        intent = OrderIntent(
            ticker="AAPL",
            side=OrderSide.SELL,
            shares=50,
            target_pct=0.0,
            today=pd.Timestamp("2025-01-02"),
            reason="qp_sell partial",
            exit_type="qp_sell",
        )
        assert intent.shares == 50
        assert not intent.is_full_liquidate

    def test_nan_inf_finite_guard_on_floats(self):
        """§5.13.11: NaN/inf on target_pct must raise, not silently pass."""
        with pytest.raises(ValueError, match="finite"):
            OrderIntent(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=100,
                target_pct=float("nan"),
                today=pd.Timestamp("2025-01-02"),
                reason="bad",
                exit_type=None,
            )
        with pytest.raises(ValueError, match="finite"):
            OrderIntent(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=100,
                target_pct=float("inf"),
                today=pd.Timestamp("2025-01-02"),
                reason="bad",
                exit_type=None,
            )


# ─── Fill: immutability + required fields ────────────────────────────────


class TestFillContract:
    def test_frozen_dataclass(self):
        fill = Fill(
            ticker="AAPL",
            side=OrderSide.BUY,
            shares=100,
            price=187.40,
            fees=0.0,
            today=pd.Timestamp("2025-01-02"),
        )
        with pytest.raises((AttributeError, Exception)):
            fill.price = 999.0  # type: ignore[misc]

    def test_finite_price_and_fees_required(self):
        with pytest.raises(ValueError):
            Fill(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=100,
                price=float("nan"),
                fees=0.0,
                today=pd.Timestamp("2025-01-02"),
            )

    def test_zero_shares_disallowed(self):
        """A Fill is a *confirmed execution*; zero shares is a non-fill."""
        with pytest.raises(ValueError):
            Fill(
                ticker="AAPL",
                side=OrderSide.BUY,
                shares=0,
                price=187.40,
                fees=0.0,
                today=pd.Timestamp("2025-01-02"),
            )


# ─── ExecutionBackend ABC ────────────────────────────────────────────────


class TestExecutionBackendABC:
    """ABC contract: subclasses MUST implement these methods or fail to
    instantiate. We pin the method set so adding a method to one backend
    without updating others is caught at import-time, not at first use.
    """

    REQUIRED_METHODS = (
        "place_market_order",
        "get_position_quantity",
        "get_unrealized_pnl",
        "get_cash",
        "get_portfolio_value",
        "get_last_price",
    )

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ExecutionBackend()  # type: ignore[abstract]

    def test_method_set_is_pinned(self):
        for m in self.REQUIRED_METHODS:
            assert hasattr(ExecutionBackend, m), (
                f"ExecutionBackend ABC missing required method {m!r}; "
                "every adapter migration assumes this method set."
            )


# ─── FakeBackend ─────────────────────────────────────────────────────────


def _mk_backend(starting_cash=100_000.0) -> FakeBackend:
    return FakeBackend(starting_cash=starting_cash)


class TestFakeBackendBehavioural:
    def test_starts_with_no_positions(self):
        b = _mk_backend()
        assert b.get_cash() == 100_000.0
        assert b.get_position_quantity("AAPL") == 0.0
        assert b.intents == ()

    def test_seed_price_then_buy_records_intent_and_fill(self):
        b = _mk_backend()
        b.seed_price("AAPL", 187.40, today=pd.Timestamp("2025-01-02"))
        intent = OrderIntent(
            ticker="AAPL",
            side=OrderSide.BUY,
            shares=100,
            target_pct=0.18740,  # 100*187.40 / 100_000
            today=pd.Timestamp("2025-01-02"),
            reason="initial",
            exit_type=None,
        )
        fill = b.place_market_order(intent)
        assert fill is not None
        assert fill.ticker == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.shares == 100
        assert math.isclose(fill.price, 187.40)
        # FakeBackend uses fee config defaults from kernel.execution.fees;
        # buy-side custom_bps=0 by default → fees should be 0.0.
        assert fill.fees == 0.0
        assert len(b.intents) == 1
        assert b.intents[0] is intent
        # Cash debited by notional (no fees on buy at default config).
        assert math.isclose(b.get_cash(), 100_000.0 - 100 * 187.40)
        assert b.get_position_quantity("AAPL") == 100

    def test_buy_then_full_sell_zeros_position(self):
        b = _mk_backend()
        b.seed_price("AAPL", 187.40, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.187, today=pd.Timestamp("2025-01-02"),
            reason="initial", exit_type=None,
        ))
        b.seed_price("AAPL", 200.00, today=pd.Timestamp("2025-01-15"))
        # full liquidate (shares=None)
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="stop_loss", exit_type="stop_loss",
        ))
        assert fill.shares == 100  # full liquidate resolves to held qty
        assert math.isclose(fill.price, 200.00)
        assert b.get_position_quantity("AAPL") == 0.0
        # Unrealized P&L is now zero (no position).
        assert b.get_unrealized_pnl("AAPL") == 0.0

    def test_unrealized_pnl_tracks_mark_to_market(self):
        b = _mk_backend()
        b.seed_price("AAPL", 100.0, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=10,
            target_pct=0.01, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        b.seed_price("AAPL", 110.0, today=pd.Timestamp("2025-01-05"))
        assert math.isclose(b.get_unrealized_pnl("AAPL"), 100.0)  # 10 * +$10

    def test_sell_proceeds_credit_cash_minus_fees(self):
        """Sell-side: SEC §31 + FINRA TAF fees come out of cash credit."""
        b = _mk_backend()
        b.seed_price("AAPL", 100.0, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        cash_after_buy = b.get_cash()
        b.seed_price("AAPL", 105.0, today=pd.Timestamp("2025-02-01"))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=None,
            target_pct=0.0, today=pd.Timestamp("2025-02-01"),
            reason="exit", exit_type="model_sell",
        ))
        # Sell fees are non-zero (SEC + TAF). Cash credit = revenue - fees.
        assert fill.fees > 0
        cash_after_sell = b.get_cash()
        assert math.isclose(
            cash_after_sell, cash_after_buy + 100 * 105.0 - fill.fees,
            rel_tol=1e-9,
        )

    def test_portfolio_value_equals_cash_plus_positions(self):
        b = _mk_backend()
        b.seed_price("AAPL", 100.0, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        # cash = 90000, position value = 10000 → PV = 100000
        assert math.isclose(b.get_portfolio_value(), 100_000.0, rel_tol=1e-9)
        b.seed_price("AAPL", 110.0, today=pd.Timestamp("2025-01-05"))
        # cash = 90000, position value = 11000 → PV = 101000
        assert math.isclose(b.get_portfolio_value(), 101_000.0, rel_tol=1e-9)

    def test_sell_without_position_raises(self):
        """A SELL intent for a ticker with no position is a pipeline bug;
        backend must surface it, not silently no-op (which would mask
        the real issue in tests / live)."""
        b = _mk_backend()
        b.seed_price("AAPL", 100.0, today=pd.Timestamp("2025-01-02"))
        with pytest.raises(ValueError, match="no position"):
            b.place_market_order(OrderIntent(
                ticker="AAPL", side=OrderSide.SELL, shares=None,
                target_pct=0.0, today=pd.Timestamp("2025-01-02"),
                reason="x", exit_type="stop_loss",
            ))

    def test_partial_sell_records_actual_quantity(self):
        b = _mk_backend()
        b.seed_price("AAPL", 100.0, today=pd.Timestamp("2025-01-02"))
        b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="x", exit_type=None,
        ))
        fill = b.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.SELL, shares=30,
            target_pct=0.0, today=pd.Timestamp("2025-01-15"),
            reason="trim", exit_type="qp_sell",
        ))
        assert fill.shares == 30
        assert b.get_position_quantity("AAPL") == 70

    def test_get_last_price_returns_seeded_value(self):
        b = _mk_backend()
        b.seed_price("AAPL", 187.40, today=pd.Timestamp("2025-01-02"))
        assert math.isclose(b.get_last_price("AAPL"), 187.40)

    def test_get_last_price_unknown_ticker_raises(self):
        b = _mk_backend()
        with pytest.raises(KeyError):
            b.get_last_price("ZZZZ")
