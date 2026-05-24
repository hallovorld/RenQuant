"""Slice 2 of ExecutionPipeline refactor (CLAUDE.md §1b / §1c / §5.13.5).

Pins the **pipeline-level execution contract**: given an
:class:`InferenceContext` populated with ``exits`` + ``orders`` and an
attached :class:`ExecutionBackend`, the :class:`ExecutionPipeline` must:

1. Run exit orders BEFORE buy orders (cash from sells funds same-bar buys).
2. Emit one :class:`Fill` per executed order, in execution order, to
   ``ctx.fills``.
3. Prune full-liquidate tickers from ``ctx.holdings`` (matches the
   current ``sim/runner/lean.commit()`` semantics).
4. Upsert :class:`HoldingState` entries on new buys, using
   volume-weighted average cost for top-ups.
5. Stamp ``ctx.last_sell_dates`` + ``ctx.last_stop_exit_dates`` for the
   appropriate exits (full liquidates for wash-sale, path-rule exits
   for post-stop cooldown).
6. Be no-op for any ctx field NOT in {fills, holdings,
   last_sell_dates, last_stop_exit_dates, last_sell_pls}.

The intent is to be the **single source of truth** that the existing
sim / runner / LEAN ``commit()`` monoliths all collapse to in slice 3+.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from kernel.exits import ExitSignal, HoldingState
from kernel.execution import FakeBackend, OrderSide
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pp_execution import ExecutionPipeline


def _empty_ctx(today: str = "2025-01-15") -> InferenceContext:
    """Minimal ctx populated with the fields ExecutionPipeline reads."""
    return InferenceContext(
        config={},
        today=pd.Timestamp(today).date(),
    )


def _hs(entry_price: float, entry_date: str, hwm: float | None = None) -> HoldingState:
    return HoldingState(
        entry_price=entry_price,
        entry_date=pd.Timestamp(entry_date).date(),
        high_watermark=hwm if hwm is not None else entry_price,
    )


class TestExecutionPipelineExitsOnly:
    """Exits-only flow: full liquidate + partial trim."""

    def test_full_liquidate_emits_fill_and_prunes_holdings(self):
        from kernel.execution import OrderIntent
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        # Open an AAPL position via the same backend path the pipeline uses.
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("AAPL", 110.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02", hwm=110.0)}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="stop_loss fired", exit_type="stop_loss",
            quantity=None,
        ))]
        ctx.orders = []

        ExecutionPipeline().run(ctx)

        # 1 fill: full liquidate, 100 shares
        assert len(ctx.fills) == 1
        f = ctx.fills[0]
        assert f.ticker == "AAPL"
        assert f.side == OrderSide.SELL
        assert f.shares == 100
        assert math.isclose(f.price, 110.0)

        # ctx.holdings AAPL pruned
        assert "AAPL" not in ctx.holdings

        # stamp last_stop_exit_dates AND last_sell_dates (full liquidate)
        assert ctx.last_stop_exit_dates.get("AAPL") == ctx.today
        assert ctx.last_sell_dates.get("AAPL") == ctx.today

    def test_partial_trim_keeps_holding_open(self):
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("AAPL", 110.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02", hwm=110.0)}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="qp_sell trim", exit_type="qp_sell",
            quantity=30.0,
        ))]
        ctx.orders = []

        ExecutionPipeline().run(ctx)

        # 1 fill: partial, 30 shares
        assert len(ctx.fills) == 1
        f = ctx.fills[0]
        assert f.shares == 30

        # Holding stays open with original HWM
        assert "AAPL" in ctx.holdings
        assert ctx.holdings["AAPL"].high_watermark == 110.0
        # Partial trims do NOT stamp last_sell_dates (matches existing
        # sim/runner/lean partial-trim wash-sale exemption).
        assert "AAPL" not in ctx.last_sell_dates
        # qp_sell is NOT a path-rule exit (stop_loss / trailing_stop /
        # single_day_loss), so post-stop cooldown stamp does NOT fire.
        assert "AAPL" not in ctx.last_stop_exit_dates

    def test_quantity_at_or_above_held_is_full_liquidate(self):
        """ExecutionPipeline must match legacy sim semantics:
        quantity >= current shares means full liquidation, not a rejected
        oversized partial sell."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("AAPL", 110.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02", hwm=110.0)}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="full by quantity",
            exit_type="max_hold", quantity=100.0,
        ))]
        ctx.orders = []

        ExecutionPipeline().run(ctx)

        assert len(ctx.fills) == 1
        assert ctx.fills[0].shares == 100
        assert "AAPL" not in ctx.holdings
        assert ctx.last_sell_dates.get("AAPL") == ctx.today

    def test_path_rule_exit_stamps_stop_exit_dates_even_when_partial(self):
        """G8: stop_loss/trailing_stop/single_day_loss stamp post-stop
        cooldown date regardless of partial-vs-full."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("AAPL", 95.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02")}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="trailing fired", exit_type="trailing_stop",
            quantity=40.0,  # partial trim with path-rule exit
        ))]
        ctx.orders = []

        ExecutionPipeline().run(ctx)

        # Cooldown stamp fired even though it was partial
        assert ctx.last_stop_exit_dates.get("AAPL") == ctx.today
        # last_sell_dates still NOT stamped (partial trim wash-sale exempt)
        assert "AAPL" not in ctx.last_sell_dates

    def test_duplicate_exits_dedupe_full_wins(self):
        """If a ticker appears twice with one full and one partial, prefer
        the full liquidate (matches sim.commit:628-642 semantics)."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("AAPL", 110.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02")}
        ctx.exits = [
            ("AAPL", ExitSignal(should_exit=True, reason="qp trim",
                                exit_type="qp_sell", quantity=20.0)),
            ("AAPL", ExitSignal(should_exit=True, reason="stop fired",
                                exit_type="stop_loss", quantity=None)),
        ]
        ctx.orders = []

        ExecutionPipeline().run(ctx)

        # Exactly one fill — for the full liquidate (100 shares)
        assert len(ctx.fills) == 1
        assert ctx.fills[0].shares == 100
        assert "AAPL" not in ctx.holdings


class TestExecutionPipelineBuysOnly:
    def test_new_buy_creates_holding_state(self):
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("MSFT", 300.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {}
        ctx.exits = []
        ctx.orders = [{
            "ticker": "MSFT",
            "shares": 50,
            "target_pct": 0.15,
            "price": 300.0,
            "regime": "BULL_CALM",
            "confidence": 0.7,
            "rank_score": 1.23,
            "rs_score": 0.45,
            "panel_score": 0.18,
            "kelly_target_pct": 0.15,
            "detail": "test",
        }]

        ExecutionPipeline().run(ctx)

        assert len(ctx.fills) == 1
        f = ctx.fills[0]
        assert f.ticker == "MSFT"
        assert f.side == OrderSide.BUY
        assert f.shares == 50

        assert "MSFT" in ctx.holdings
        hs = ctx.holdings["MSFT"]
        assert math.isclose(hs.entry_price, 300.0)
        assert hs.entry_date == ctx.today
        assert math.isclose(hs.high_watermark, 300.0)
        # Thesis-baseline scores propagate from order dict
        assert math.isclose(hs.entry_rank_score, 1.23)
        assert hs.entry_regime == "BULL_CALM"

    def test_topup_uses_volume_weighted_avg_cost(self):
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("MSFT", 200.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="MSFT", side=OrderSide.BUY, shares=50,
            target_pct=0.10, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        backend.seed_price("MSFT", 300.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"MSFT": _hs(200.0, "2025-01-02", hwm=300.0)}
        ctx.exits = []
        ctx.orders = [{
            "ticker": "MSFT",
            "shares": 50,
            "target_pct": 0.25,
            "price": 300.0,
            "regime": "BULL_CALM",
            "confidence": 0.7,
            "rank_score": 1.0,
            "rs_score": 0.5,
            "panel_score": 0.18,
            "kelly_target_pct": 0.25,
            "detail": "topup",
        }]

        ExecutionPipeline().run(ctx)

        # New cost basis = (50*200 + 50*300) / 100 = 250
        hs = ctx.holdings["MSFT"]
        assert math.isclose(hs.entry_price, 250.0)
        # entry_date PRESERVED on top-up (not reset to today)
        assert hs.entry_date == pd.Timestamp("2025-01-02").date()
        # HWM is max(old=300, today=300) = 300
        assert math.isclose(hs.high_watermark, 300.0)


class TestExecutionPipelineOrderingAndIsolation:
    def test_exits_run_before_buys(self):
        """Same-bar sells must execute before buys so cash funds the buys
        (matches existing commit() ordering invariant)."""
        backend = FakeBackend(starting_cash=10_000.0)  # tight cash
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-02"))
        from kernel.execution import OrderIntent
        backend.place_market_order(OrderIntent(
            ticker="AAPL", side=OrderSide.BUY, shares=100,
            target_pct=1.0, today=pd.Timestamp("2025-01-02"),
            reason="setup", exit_type=None,
        ))
        # After setup: cash = 0, AAPL = 100 shares @ $100
        assert backend.get_cash() == 0.0
        backend.seed_price("AAPL", 105.0, pd.Timestamp("2025-01-15"))
        backend.seed_price("MSFT", 50.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {"AAPL": _hs(100.0, "2025-01-02")}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="rotate", exit_type="rotation",
            quantity=None,
        ))]
        ctx.orders = [{
            "ticker": "MSFT", "shares": 100, "target_pct": 0.50,
            "price": 50.0, "regime": "BULL_CALM", "confidence": 0.7,
            "rank_score": 1.0, "rs_score": 0.5, "panel_score": 0.18,
            "kelly_target_pct": 0.50, "detail": "rotate-in",
        }]

        ExecutionPipeline().run(ctx)

        # Exit fill first, buy fill second
        assert len(ctx.fills) == 2
        assert ctx.fills[0].side == OrderSide.SELL
        assert ctx.fills[1].side == OrderSide.BUY

    def test_fills_cleared_between_runs(self):
        """A stale Fill from the previous bar must not survive."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-15"))

        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.fills = ["STALE"]  # type: ignore[list-item]
        ctx.holdings = {}
        ctx.exits = []
        ctx.orders = []

        ExecutionPipeline().run(ctx)
        assert ctx.fills == []

    def test_runs_with_no_backend_raises(self):
        """A missing execution_backend is a wiring bug; surface loudly."""
        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = None
        ctx.exits = []
        ctx.orders = []
        with pytest.raises(ValueError, match="execution_backend"):
            ExecutionPipeline().run(ctx)

    def test_empty_exits_and_orders_is_noop(self):
        backend = FakeBackend(starting_cash=100_000.0)
        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.exits = []
        ctx.orders = []

        ExecutionPipeline().run(ctx)
        assert ctx.fills == []
        assert backend.intents == ()


class TestExecutionPipelineRegressionGuards:
    """AUDIT REGRESSION GUARD: pin invariants from §5.13.5 single-source
    discipline so a future patch can't re-introduce divergence.
    """

    def test_no_buy_intent_when_order_has_nonfinite_fields(self):
        """§5.13.11: NaN price/shares/target_pct in an order is silently
        skipped (matches lean.commit:371-378 + sim defensive guard)."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("MSFT", 300.0, pd.Timestamp("2025-01-15"))
        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.exits = []
        ctx.orders = [{
            "ticker": "MSFT", "shares": 50, "target_pct": float("nan"),
            "price": 300.0, "regime": "BULL_CALM", "confidence": 0.7,
            "rank_score": 1.0, "rs_score": 0.5, "panel_score": 0.18,
            "kelly_target_pct": 0.25, "detail": "bad",
        }]
        ExecutionPipeline().run(ctx)
        assert ctx.fills == []
        assert backend.intents == ()

    def test_sell_intent_for_unheld_ticker_is_skipped_not_raised(self):
        """If pipeline emits a SELL for a ticker the broker shows as flat
        (rare: race condition between bar data and broker state), the
        ExecutionPipeline drops it with a warning rather than crashing the
        whole bar. Matches sim._apply_sell:781 guard."""
        backend = FakeBackend(starting_cash=100_000.0)
        backend.seed_price("AAPL", 100.0, pd.Timestamp("2025-01-15"))
        ctx = _empty_ctx("2025-01-15")
        ctx.execution_backend = backend
        ctx.holdings = {}
        ctx.exits = [("AAPL", ExitSignal(
            should_exit=True, reason="zombie", exit_type="stop_loss",
            quantity=None,
        ))]
        ctx.orders = []
        ExecutionPipeline().run(ctx)
        assert ctx.fills == []  # silently dropped
