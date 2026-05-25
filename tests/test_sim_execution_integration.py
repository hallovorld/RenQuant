"""SimAdapter end-to-end execution-model integration tests.

Per CLAUDE.md §5.13.1 — tests walk real SimAdapter via its public API
(not hand-constructed fixtures), exercising the actual production
codepath that production sim runs through.

What we pin:
1. ``execution.enabled=False`` → byte-identical pre-2026-05-10 behavior
   (no fees, no slippage, T+0 settlement) — preserves backward compat
   for the legacy regression suite.
2. ``execution.enabled=True`` (default) → equity curve diverges from
   legacy by the expected fee + slippage delta on a 1-trade sim.
3. T+1 settlement by default, matching current SEC Rule 15c6-1 timing.
4. Buying-power modes are explicit: default non-marginable buying power can
   reuse executed sell proceeds before settlement; settled-cash mode cannot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Synthetic fixture helpers ─────────────────────────────────────────────


def _ramp_ohlcv(start: str = "2024-01-02", days: int = 30,
                start_price: float = 100.0, daily_drift: float = 0.0) -> pd.DataFrame:
    """Deterministic flat-or-ramping OHLCV so trade P&L is exact."""
    idx = pd.bdate_range(start, periods=days)
    closes = np.array([start_price * (1.0 + daily_drift) ** i for i in range(days)])
    return pd.DataFrame({
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": np.full(days, 1e7),
    }, index=idx)


def _build_min_adapter(*, execution_enabled: bool, legacy_no_fees: bool = False,
                       initial_cash: float = 100_000.0,
                       execution_overrides: dict | None = None):
    """Construct a minimal SimAdapter wired with controlled execution config."""
    from adapters.sim import SimAdapter
    spy = _ramp_ohlcv(days=30)
    cfg = {
        "watchlist": [],
        "sector_etf_map": {},
        "tax": {"short_term_rate": 0.0,  # zero tax → isolate fee+slip delta
                "long_term_rate": 0.0,
                "long_term_threshold_days": 365},
        "regime": {},
        "execution": {
            "enabled": execution_enabled,
            "legacy_no_fees": legacy_no_fees,
            **(execution_overrides or {}),
        },
    }
    adapter = SimAdapter(
        config=cfg, strategy_dir=_STRATEGY_DIR,
        ohlcv={"SPY": spy}, spy_df=spy,
        sector_etf_map={}, initial_cash=initial_cash,
    )
    return adapter, spy


# ── Tests ──────────────────────────────────────────────────────────────────


class TestExecutionFlagWiring:
    """Per §5.13.1 — config flows through to per-bar behavior."""

    def test_init_reads_execution_block(self):
        adapter, _ = _build_min_adapter(execution_enabled=True)
        assert adapter._exec_enabled is True  # noqa: SLF001
        # Defaults baked in
        assert adapter._fee_cfg.sec_fee_rate == pytest.approx(27.0e-6)  # noqa: SLF001
        assert adapter._slip_cfg.half_spread_bps == pytest.approx(2.0)  # noqa: SLF001
        assert adapter._t2_queue.settlement_days == 1  # noqa: SLF001
        assert adapter._buying_power_mode == "non_marginable_buying_power"  # noqa: SLF001

    def test_legacy_no_fees_flag_disables_model(self):
        adapter, _ = _build_min_adapter(execution_enabled=True, legacy_no_fees=True)
        assert adapter._exec_enabled is False  # noqa: SLF001

    def test_enabled_false_disables_model(self):
        adapter, _ = _build_min_adapter(execution_enabled=False)
        assert adapter._exec_enabled is False  # noqa: SLF001


class TestBuyAppliesSlippageAndFees:
    """Slippage + buy fees actually deducted from cash on _apply_buy.

    Walks the real `_apply_buy` codepath (not a hand-stubbed shim) per
    §5.13.1. The order dict mimics what `SizeAndEmitTask` would emit.
    """

    def _execute_buy(self, adapter, ticker, shares, price, today):
        from kernel.pipeline.context import InferenceContext
        # Real ctx — same constructor the pipeline uses.
        ctx = InferenceContext(
            config=adapter._config, today=today.date(),  # noqa: SLF001
            ohlcv={}, spy_returns=[], models={}, gmm=None, corr_matrix={},
            earnings_calendar={}, holdings={}, last_sell_dates={},
            portfolio_value=adapter._cash, cash=adapter._cash, prices={},  # noqa: SLF001
            hwm=adapter._cash, skip_buys=False,                              # noqa: SLF001
            regime_state=adapter._regime_state,                              # noqa: SLF001
            regime_counts=adapter._regime_counts,                            # noqa: SLF001
        )
        order = {
            "ticker": ticker, "shares": shares, "price": price,
            "target_pct": 0.1, "regime": "BULL_CALM",
            "confidence": 0.8, "rank_score": 0.5, "rs_score": 0.5,
            "detail": "test", "panel_score": 0.5, "kelly_target_pct": 0.1,
            "sigma": None, "mu": None, "sigma_mult": None,
        }
        adapter._apply_buy(order, today, ctx)  # noqa: SLF001
        return ctx

    def test_buy_with_execution_costs_cash_above_notional(self):
        adapter, spy = _build_min_adapter(execution_enabled=True)
        today = spy.index[5]
        cash_before = adapter._cash  # noqa: SLF001
        # 100 shares @ $100 = $10k notional. With 2 bps half-spread:
        # fill price = 100.02 → invest = 10,002.00. Custom bps default 0.
        self._execute_buy(adapter, "AAA", shares=100.0, price=100.0, today=today)
        cash_after = adapter._cash  # noqa: SLF001
        invest = cash_before - cash_after
        # $10,002 expected (2 bps slip on the price)
        assert invest == pytest.approx(10_002.0, rel=1e-4)

    def test_buy_without_execution_costs_exact_notional(self):
        adapter, spy = _build_min_adapter(execution_enabled=False)
        today = spy.index[5]
        cash_before = adapter._cash  # noqa: SLF001
        self._execute_buy(adapter, "AAA", shares=100.0, price=100.0, today=today)
        cash_after = adapter._cash  # noqa: SLF001
        invest = cash_before - cash_after
        # Exactly $10,000 — legacy path.
        assert invest == pytest.approx(10_000.0, rel=1e-9)


class TestSellWithT1Settlement:
    """Sell with execution model: fees deducted, proceeds queued to T+1.

    The key behavior: ``self._cash`` is NOT credited on sell-date when
    T+N is active. Cash arrives via ``make_context`` drain on settlement day.
    """

    def _put_a_lot(self, adapter, ticker, shares, price, today):
        """Stuff a fully-owned lot into the adapter so _apply_sell finds it."""
        from kernel.exits import HoldingState, TaxLot
        hs = HoldingState(
            entry_price=price, entry_date=today.date(),
            high_watermark=price, prev_close=price, shares=shares,
        )
        hs.lots.append(TaxLot(shares=shares, price=price, date=today.date()))
        adapter._holdings[ticker] = hs        # noqa: SLF001
        adapter._pos_shares[ticker] = shares  # noqa: SLF001

    def _ctx_for(self, adapter, today, prices):
        from kernel.pipeline.context import InferenceContext
        return InferenceContext(
            config=adapter._config, today=today.date(),  # noqa: SLF001
            ohlcv={}, spy_returns=[], models={}, gmm=None, corr_matrix={},
            earnings_calendar={}, holdings={}, last_sell_dates={},
            portfolio_value=adapter._cash, cash=adapter._cash, prices=prices,  # noqa: SLF001
            hwm=adapter._cash, skip_buys=False,                                  # noqa: SLF001
            regime_state=adapter._regime_state,                                  # noqa: SLF001
            regime_counts=adapter._regime_counts,                                # noqa: SLF001
        )

    def test_full_exit_without_executable_price_does_not_drop_position(self):
        """A skipped sell is not a fill; sim must not ghost-delete holdings."""
        from kernel.exits import ExitSignal
        from kernel.pipeline.context import InferenceContext

        adapter, spy = _build_min_adapter(execution_enabled=False)
        today = spy.index[10]
        self._put_a_lot(adapter, "AAA", shares=10.0, price=100.0,
                        today=spy.index[5])
        ctx = InferenceContext(
            config=adapter._config, today=today.date(),  # noqa: SLF001
            ohlcv={}, spy_returns=[], models={}, gmm=None, corr_matrix={},
            earnings_calendar={}, holdings=dict(adapter._holdings),  # noqa: SLF001
            last_sell_dates={}, portfolio_value=1_000.0, cash=0.0,
            prices={}, hwm=1_000.0, skip_buys=False,
            regime_state=adapter._regime_state,      # noqa: SLF001
            regime_counts=adapter._regime_counts,    # noqa: SLF001
        )
        ctx.exits = [
            ("AAA", ExitSignal(
                should_exit=True,
                reason="unit_missing_price",
                exit_type="model_sell",
                quantity=None,
            ))
        ]
        ctx.orders = []
        ctx.candidates = []
        ctx.counters = {}
        ctx.rotations = []
        ctx.monitor_state = {}
        ctx.regime = "BULL_CALM"
        ctx.confidence = 0.8
        ctx.buy_blocked = False
        ctx.bear_only = False
        before_cash = adapter._cash  # noqa: SLF001

        adapter.commit(ctx)

        assert "AAA" in adapter._holdings  # noqa: SLF001
        assert adapter._pos_shares["AAA"] == pytest.approx(10.0)  # noqa: SLF001
        assert adapter._cash == pytest.approx(before_cash)  # noqa: SLF001
        assert adapter._trade_log == []  # noqa: SLF001

    def test_sell_with_tn_queues_proceeds_not_cash(self):
        from kernel.exits import ExitSignal
        adapter, spy = _build_min_adapter(execution_enabled=True)
        sell_date = spy.index[10]
        # Holding bought 5 bars earlier at $100, now sold at $100 flat
        self._put_a_lot(adapter, "AAA", shares=100.0, price=100.0,
                        today=spy.index[5])
        cash_before = adapter._cash  # noqa: SLF001
        ctx = self._ctx_for(adapter, sell_date, prices={"AAA": 100.0})
        sig = ExitSignal(should_exit=True, reason="test_exit", exit_type="rotation", quantity=None)
        adapter._apply_sell("AAA", sig, sell_date, ctx)  # noqa: SLF001
        cash_after = adapter._cash  # noqa: SLF001
        # T+N active: cash should NOT have grown by proceeds. Tax is
        # zero, so cash should be cash_before EXACTLY (no immediate credit).
        # Slippage discounts the fill price by 2 bps; SEC + TAF + custom
        # fees deducted are immediate? No — they net out of proceeds.
        # Per implementation: cash -= tax (=0); proceeds queued, NOT cash.
        assert cash_after == pytest.approx(cash_before)
        # Pending queue should hold (notional - fees)
        assert adapter._t2_queue.pending_total() > 0   # noqa: SLF001

    def test_default_proceeds_settle_next_trading_day(self):
        from kernel.exits import ExitSignal
        adapter, spy = _build_min_adapter(execution_enabled=True)
        # Pick a sell date with no nearby holidays to keep T+1 = simple +1.
        sell_date = spy.index[10]
        self._put_a_lot(adapter, "AAA", 100.0, 100.0, spy.index[5])
        cash_before = adapter._cash  # noqa: SLF001
        ctx = self._ctx_for(adapter, sell_date, {"AAA": 100.0})
        sig = ExitSignal(should_exit=True, reason="test", exit_type="rotation", quantity=None)
        adapter._apply_sell("AAA", sig, sell_date, ctx)  # noqa: SLF001

        # Look up the actual queued settle date (NYSE-aware T+1).
        settle = adapter._t2_queue._pending[0].settle_date  # noqa: SLF001

        # On sale date: still no cash credit.
        adapter.make_context(sell_date)
        assert adapter._cash == pytest.approx(cash_before)  # noqa: SLF001

        # On settle_date: drain settles proceeds.
        adapter.make_context(settle)
        # Sell at $100 × 100 shares ≈ $9998 (2 bps slip) - fees.
        # Sell fees: SEC = 9998 * 27e-6 ≈ $0.270, TAF = 100 * 1.19e-4 ≈ $0.012
        # Net proceeds ≈ 9998 - 0.282 ≈ $9997.72
        gained = adapter._cash - cash_before  # noqa: SLF001
        assert gained == pytest.approx(9997.72, abs=1.0)

    def test_explicit_legacy_t2_proceeds_settle_two_trading_days_later(self):
        from kernel.exits import ExitSignal
        adapter, spy = _build_min_adapter(
            execution_enabled=True,
            execution_overrides={"t2_settlement_days": 2},
        )
        sell_date = spy.index[10]
        self._put_a_lot(adapter, "AAA", 100.0, 100.0, spy.index[5])
        cash_before = adapter._cash  # noqa: SLF001
        ctx = self._ctx_for(adapter, sell_date, {"AAA": 100.0})
        sig = ExitSignal(should_exit=True, reason="test", exit_type="rotation", quantity=None)
        adapter._apply_sell("AAA", sig, sell_date, ctx)  # noqa: SLF001

        settle = adapter._t2_queue._pending[0].settle_date  # noqa: SLF001
        assert settle == spy.index[12]
        adapter.make_context(spy.index[11])
        assert adapter._cash == pytest.approx(cash_before)  # noqa: SLF001
        adapter.make_context(settle)
        assert adapter._cash > cash_before  # noqa: SLF001

    def test_same_bar_sell_proceeds_can_fund_buy_in_non_marginable_mode(self):
        from kernel.exits import ExitSignal
        from kernel.pipeline.context import InferenceContext

        adapter, spy = _build_min_adapter(
            execution_enabled=True,
            initial_cash=0.0,
        )
        today = spy.index[10]
        self._put_a_lot(adapter, "SPY", 10.0, 100.0, spy.index[5])
        ctx = InferenceContext(
            config=adapter._config, today=today.date(),  # noqa: SLF001
            ohlcv={}, spy_returns=[], models={}, gmm=None, corr_matrix={},
            earnings_calendar={}, holdings=dict(adapter._holdings),  # noqa: SLF001
            last_sell_dates={}, portfolio_value=1_000.0, cash=0.0,
            prices={"SPY": 100.0, "BBB": 100.0},
            hwm=1_000.0, skip_buys=False,
            regime_state=adapter._regime_state,  # noqa: SLF001
            regime_counts=adapter._regime_counts,  # noqa: SLF001
        )
        ctx.exits = [("SPY", ExitSignal(should_exit=True, reason="fund", exit_type="rotation"))]
        ctx.orders = [{
            "ticker": "BBB", "shares": 5.0, "price": 100.0,
            "target_pct": 0.5, "regime": "BULL_CALM",
            "confidence": 1.0, "rank_score": 0.7, "rs_score": 0.0,
            "detail": "test", "panel_score": 1.0, "kelly_target_pct": 0.5,
            "sigma": 0.1, "mu": 0.05, "sigma_mult": 1.0,
        }]

        adapter.commit(ctx)

        assert "SPY" not in adapter._holdings  # noqa: SLF001
        assert "BBB" in adapter._holdings  # noqa: SLF001
        assert adapter._cash < 0  # noqa: SLF001
        assert adapter._t2_queue.pending_total() > abs(adapter._cash)  # noqa: SLF001

    def test_settled_cash_mode_blocks_same_bar_unsettled_reinvestment(self):
        from kernel.exits import ExitSignal
        from kernel.pipeline.context import InferenceContext

        adapter, spy = _build_min_adapter(
            execution_enabled=True,
            initial_cash=0.0,
            execution_overrides={"buying_power_mode": "settled_cash"},
        )
        today = spy.index[10]
        self._put_a_lot(adapter, "SPY", 10.0, 100.0, spy.index[5])
        ctx = InferenceContext(
            config=adapter._config, today=today.date(),  # noqa: SLF001
            ohlcv={}, spy_returns=[], models={}, gmm=None, corr_matrix={},
            earnings_calendar={}, holdings=dict(adapter._holdings),  # noqa: SLF001
            last_sell_dates={}, portfolio_value=1_000.0, cash=0.0,
            prices={"SPY": 100.0, "BBB": 100.0},
            hwm=1_000.0, skip_buys=False,
            regime_state=adapter._regime_state,  # noqa: SLF001
            regime_counts=adapter._regime_counts,  # noqa: SLF001
        )
        ctx.exits = [("SPY", ExitSignal(should_exit=True, reason="fund", exit_type="rotation"))]
        ctx.orders = [{
            "ticker": "BBB", "shares": 5.0, "price": 100.0,
            "target_pct": 0.5, "regime": "BULL_CALM",
            "confidence": 1.0, "rank_score": 0.7, "rs_score": 0.0,
            "detail": "test", "panel_score": 1.0, "kelly_target_pct": 0.5,
            "sigma": 0.1, "mu": 0.05, "sigma_mult": 1.0,
        }]

        adapter.commit(ctx)

        assert "SPY" not in adapter._holdings  # noqa: SLF001
        assert "BBB" not in adapter._holdings  # noqa: SLF001
        assert adapter._cash == pytest.approx(0.0)  # noqa: SLF001
        assert adapter._t2_queue.pending_total() > 0  # noqa: SLF001


class TestExecutionModelDeltaVsLegacy:
    """The headline number: equity curve diverges by expected fee+slip delta.

    On a single round-trip (buy then sell same price), the legacy sim
    breaks even (no fees, T+0). The execution model gives back the
    buy-side slip + sell-side slip + sell-side fees.
    """

    def test_single_trade_pnl_legacy_vs_execution(self):
        from kernel.exits import ExitSignal
        # Two adapters identical except for the execution toggle.
        leg_adapter, spy = _build_min_adapter(execution_enabled=False)
        exec_adapter, _ = _build_min_adapter(execution_enabled=True)

        buy_date = spy.index[5]
        sell_date = spy.index[10]
        sell_date_plus_5 = spy.index[15]  # comfortably > settlement date

        from kernel.pipeline.context import InferenceContext

        def _run_round_trip(adapter):
            cash_initial = adapter._cash  # noqa: SLF001

            # Buy 100 shares @ $100 on buy_date
            ctx_b = InferenceContext(
                config=adapter._config, today=buy_date.date(),  # noqa: SLF001
                ohlcv={}, spy_returns=[], models={}, gmm=None,
                corr_matrix={}, earnings_calendar={}, holdings={},
                last_sell_dates={}, portfolio_value=adapter._cash,  # noqa: SLF001
                cash=adapter._cash, prices={"AAA": 100.0},          # noqa: SLF001
                hwm=adapter._cash, skip_buys=False,                  # noqa: SLF001
                regime_state=adapter._regime_state,                  # noqa: SLF001
                regime_counts=adapter._regime_counts,                # noqa: SLF001
            )
            order = {
                "ticker": "AAA", "shares": 100.0, "price": 100.0,
                "target_pct": 0.1, "regime": "BULL_CALM",
                "confidence": 0.8, "rank_score": 0.5, "rs_score": 0.5,
                "detail": "test", "panel_score": 0.5, "kelly_target_pct": 0.1,
                "sigma": None, "mu": None, "sigma_mult": None,
            }
            adapter._apply_buy(order, buy_date, ctx_b)  # noqa: SLF001

            # Sell on sell_date at the same $100 (flat market)
            ctx_s = InferenceContext(
                config=adapter._config, today=sell_date.date(),    # noqa: SLF001
                ohlcv={}, spy_returns=[], models={}, gmm=None,
                corr_matrix={}, earnings_calendar={}, holdings={},
                last_sell_dates={}, portfolio_value=adapter._cash,  # noqa: SLF001
                cash=adapter._cash, prices={"AAA": 100.0},          # noqa: SLF001
                hwm=adapter._cash, skip_buys=False,                  # noqa: SLF001
                regime_state=adapter._regime_state,                  # noqa: SLF001
                regime_counts=adapter._regime_counts,                # noqa: SLF001
            )
            sig = ExitSignal(should_exit=True, reason="test", exit_type="rotation", quantity=None)
            adapter._apply_sell("AAA", sig, sell_date, ctx_s)  # noqa: SLF001

            # Advance to T+5 so any pending settlement has drained.
            adapter.make_context(sell_date_plus_5)
            final_cash = adapter._cash  # noqa: SLF001
            return final_cash - cash_initial   # net P&L from initial

        leg_pnl = _run_round_trip(leg_adapter)
        exec_pnl = _run_round_trip(exec_adapter)
        # Legacy: round-trip at flat $100 → cash unchanged (0 P&L, 0 tax)
        assert leg_pnl == pytest.approx(0.0, abs=0.001)
        # Execution model: lose buy-slip + sell-slip + sell-fees
        # buy_slip = 100 sh × (100.02 - 100) = $2.00
        # sell_slip = 100 sh × (100 - 99.98) = $2.00
        # sell_fees ≈ $0.28 (SEC + TAF)
        # Total drag ≈ $4.28
        assert exec_pnl < 0  # always negative on a flat round-trip
        assert exec_pnl == pytest.approx(-4.28, abs=0.5)


class TestExecutionAuditRegressionGuard:
    """AUDIT REGRESSION GUARD — invariants for the entire execution model.

    Per CLAUDE.md §5.13.3 — every fix names the bug-class invariant.
    Reverting any of these silently re-inflates sim APY.
    """

    def test_make_context_drains_t2_queue_first(self):
        """The T+N drain MUST happen at top of make_context, not somewhere later."""
        adapter, spy = _build_min_adapter(execution_enabled=True)
        # Inject a pending entry that settles today: settle_date == today.
        # We compute the actual settle date from a sale at spy.index[8],
        # then read today off the queue so the test is robust to whichever
        # NYSE holidays land in the fixture window.
        adapter._t2_queue.add_pending(spy.index[8], 5_000.0)  # noqa: SLF001
        settle_day = adapter._t2_queue._pending[0].settle_date  # noqa: SLF001
        cash_before = adapter._cash  # noqa: SLF001
        adapter.make_context(settle_day)
        cash_after = adapter._cash  # noqa: SLF001
        assert cash_after == pytest.approx(cash_before + 5_000.0)

    def test_disabled_execution_skips_t2_drain(self):
        adapter, spy = _build_min_adapter(execution_enabled=False)
        adapter._t2_queue.add_pending(spy.index[8], 5_000.0)  # noqa: SLF001
        cash_before = adapter._cash  # noqa: SLF001
        adapter.make_context(spy.index[10])
        # No drain → cash unchanged when execution model is off
        assert adapter._cash == pytest.approx(cash_before)  # noqa: SLF001
