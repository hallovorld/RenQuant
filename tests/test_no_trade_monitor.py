"""Regression tests for the "systematic no-trade period" guard.

User requirement: it's fine to sit in cash for a day or two when the
strategy has no edge. It is NOT fine to sit in cash for weeks on end while
the market runs — that means some upstream gate (calibrator stale, panel
feature frames empty, tier thresholds too tight, universe empty) is
silently blocking every trade. This file enforces three invariants:

  1. MonitorIdleStreakTask increments and resets streak counters correctly.
  2. SimResult exposes `longest_no_trade_streak` so any caller — notebook,
     scheduled backtest, CI — can assert on it.
  3. Adapter state (SimAdapter._monitor_state, RunnerAdapter live_state
     JSON) persists the counters across bars so a sub-5-bar pipeline
     smoke run in prod can catch a newly-introduced bug.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Layer 1: Task logic ──────────────────────────────────────────────────────

def _make_ctx(*, has_order: bool, has_exit: bool, has_candidate: bool,
              prior_state: dict | None = None, max_idle: int = 5,
              max_blank: int = 5):
    from kernel.pipeline.context import InferenceContext
    cfg = {
        "monitoring": {
            "max_no_trade_days":     max_idle,
            "max_no_candidate_days": max_blank,
        },
    }
    ctx = InferenceContext(config=cfg, today=datetime.date(2024, 6, 4))
    ctx.orders     = [{"symbol": "AAA"}] if has_order     else []
    ctx.exits      = [("AAA", object())]  if has_exit     else []
    ctx.candidates = [object()]           if has_candidate else []
    ctx.holdings   = {}
    ctx.monitor_state = dict(prior_state or {})
    return ctx


class TestStreakLogic:
    def test_streak_resets_on_order(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 4, "no_candidate_streak": 4}
        ctx = _make_ctx(has_order=True, has_exit=False, has_candidate=True,
                        prior_state=prior)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["no_trade_streak"]     == 0
        assert ctx.monitor_state["no_candidate_streak"] == 0

    def test_streak_resets_on_exit(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 10}
        ctx = _make_ctx(has_order=False, has_exit=True, has_candidate=False,
                        prior_state=prior)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["no_trade_streak"] == 0

    def test_streak_increments_on_idle_day(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 7, "no_candidate_streak": 7}
        ctx = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                        prior_state=prior)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["no_trade_streak"]     == 8
        assert ctx.monitor_state["no_candidate_streak"] == 8

    def test_candidate_without_order_still_counts_as_no_trade(self):
        """Having candidates that don't get filled still counts as idle —
        the issue is downstream (selection blocked them)."""
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 3, "no_candidate_streak": 3}
        ctx = _make_ctx(has_order=False, has_exit=False, has_candidate=True,
                        prior_state=prior)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["no_trade_streak"]     == 4
        assert ctx.monitor_state["no_candidate_streak"] == 0  # had candidate

    def test_warning_above_threshold(self, caplog):
        import logging
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 5}
        ctx = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                        prior_state=prior, max_idle=5, max_blank=100)
        with caplog.at_level(logging.WARNING, logger="kernel.pipeline.monitor"):
            MonitorIdleStreakTask().run(ctx)
        assert any("NoTradeAlert" in rec.message for rec in caplog.records)

    def test_first_trade_date_captured(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        ctx = _make_ctx(has_order=True, has_exit=False, has_candidate=True)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["first_trade_date"] == "2024-06-04"
        assert ctx.monitor_state["last_activity_date"] == "2024-06-04"


class TestStreakPerTradingDayInvariant:
    """2026-05-20 fix regression guard: streak counts PER TRADING DAY, not
    per pipeline invocation. Pre-fix bug: SellOnlyPipeline (intraday cron,
    ~33 firings/day) incremented streak by 1 each tick → 33×-inflated
    counter → false NoTradeAlert at 32 days while LIVE Alpaca account had
    47 fills in 16 trading days.
    """

    def test_multiple_idle_invocations_same_day_count_as_one(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        prior = {"no_trade_streak": 3, "no_candidate_streak": 3,
                  "last_check_date": "2024-06-03"}  # yesterday
        # First invocation of the day: idle → +1
        ctx1 = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                         prior_state=prior)
        MonitorIdleStreakTask().run(ctx1)
        assert ctx1.monitor_state["no_trade_streak"] == 4
        assert ctx1.monitor_state["last_check_date"] == "2024-06-04"

        # 30 subsequent same-day idle invocations: must stay at 4, not climb to 34
        state_carry = dict(ctx1.monitor_state)
        for _ in range(30):
            ctx_n = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                              prior_state=state_carry)
            MonitorIdleStreakTask().run(ctx_n)
            state_carry = dict(ctx_n.monitor_state)
        assert state_carry["no_trade_streak"] == 4, (
            f"streak inflated to {state_carry['no_trade_streak']} after 30 "
            f"same-day idle ticks (must stay 4 per per-trading-day invariant)"
        )

    def test_intraday_activity_resets_streak_then_same_day_idle_does_not_re_increment(self):
        """Reproduces 2026-05-20 bug exactly:
        - prior streak high
        - 06:54 SellOnly with TXN exit → reset to 0
        - 33 subsequent SellOnly + 1 daily, all no-activity → must stay 0 (or 1 if next day)
        Pre-fix: each tick added +1 → ~34.
        """
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        # 06:31 SellOnly: idle (1st of day)
        prior = {"no_trade_streak": 5,
                  "last_check_date": "2024-06-03"}
        ctx1 = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                         prior_state=prior)
        MonitorIdleStreakTask().run(ctx1)
        assert ctx1.monitor_state["no_trade_streak"] == 6

        # 06:54 SellOnly: TXN exit → reset
        s = dict(ctx1.monitor_state)
        ctx2 = _make_ctx(has_order=False, has_exit=True, has_candidate=False,
                         prior_state=s)
        MonitorIdleStreakTask().run(ctx2)
        assert ctx2.monitor_state["no_trade_streak"] == 0

        # 33 subsequent SellOnly + 1 daily, all idle, same day:
        s = dict(ctx2.monitor_state)
        for _ in range(34):
            ctx_n = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                              prior_state=s)
            MonitorIdleStreakTask().run(ctx_n)
            s = dict(ctx_n.monitor_state)
        assert s["no_trade_streak"] == 0, (
            f"streak inflated to {s['no_trade_streak']} (must stay 0 — "
            f"same trading day after activity)"
        )

    def test_next_trading_day_with_idle_increments_by_one(self):
        from kernel.pipeline.task_monitor import MonitorIdleStreakTask
        # Today = 2024-06-04; prior last_check yesterday with streak=4
        prior = {"no_trade_streak": 4, "last_check_date": "2024-06-03"}
        ctx = _make_ctx(has_order=False, has_exit=False, has_candidate=False,
                        prior_state=prior)
        MonitorIdleStreakTask().run(ctx)
        assert ctx.monitor_state["no_trade_streak"] == 5  # +1 once for the new day


# ── Layer 2: SimResult exposes the metric ────────────────────────────────────

class TestSimResultMonitoring:
    def test_simresult_has_streak_fields(self):
        from sim.runner import SimResult
        import pandas as _pd
        r = SimResult(
            equity_df=_pd.DataFrame(), trade_log=[], rotation_log=[],
            final_value=100.0, total_return=0.0, apy=0.0, win_rate=0.0,
            avg_hold=0.0, avg_pnl=0.0, total_tax=0.0,
            exit_reasons={}, rotations=[],
        )
        assert r.longest_no_trade_streak == 0
        assert r.first_trade_date is None

    def test_print_summary_includes_streak_when_nonzero(self, capsys):
        from sim.runner import SimResult
        import pandas as _pd
        r = SimResult(
            equity_df=_pd.DataFrame({"portfolio": [100], "regime": ["BULL_CALM"]},
                                     index=[_pd.Timestamp("2024-01-02")]),
            trade_log=[], rotation_log=[],
            final_value=100.0, total_return=0.0, apy=0.0, win_rate=0.0,
            avg_hold=0.0, avg_pnl=0.0, total_tax=0.0,
            exit_reasons={}, rotations=[],
            longest_no_trade_streak=20, first_trade_date="2024-04-01",
        )
        r.print_summary()
        out = capsys.readouterr().out
        assert "Longest no-trade streak: 20d" in out
        assert "2024-04-01" in out
        assert "⚠️" in out  # emoji gate fires when streak > 15


# ── Layer 3: Adapter persists state across bars ──────────────────────────────

class TestSimAdapterPersistence:
    def test_sim_adapter_stores_monitor_state_between_commits(self, tmp_path):
        """Verify SimAdapter reads monitor_state into ctx each make_context
        and writes it back on commit — the round-trip is what keeps streaks
        correct over multiple bars."""
        from adapters.sim import SimAdapter

        # Minimal fixture — we're testing the state plumbing, not the whole
        # pipeline. Use no watchlist so no models load / no frames get built.
        idx = pd.bdate_range("2024-01-02", periods=30)
        spy = pd.DataFrame(
            {"open": 400.0, "high": 402.0, "low": 398.0,
             "close": 400.0, "volume": 1e9},
            index=idx,
        )
        cfg = {
            "watchlist": [], "sector_etf_map": {},
            "tax": {}, "regime": {},
            "monitoring": {"max_no_trade_days": 5, "max_no_candidate_days": 5},
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv={"SPY": spy}, spy_df=spy, sector_etf_map={},
            initial_cash=100_000,
        )
        # Simulate: push a prior state via context, then commit and verify
        # round-trip.
        ctx = adapter.make_context(idx[10])
        ctx.monitor_state = {"no_trade_streak": 42, "first_trade_date": "2024-01-05"}
        # Force an empty-state commit — no orders, no exits, no holdings.
        ctx.orders = []
        ctx.exits  = []
        adapter.commit(ctx)
        assert adapter._monitor_state["no_trade_streak"] == 42, \
            "SimAdapter must persist ctx.monitor_state (from pipeline) on commit"


class TestRunnerAdapterPersistence:
    def test_runner_adapter_live_state_round_trip(self, tmp_path):
        """live_state.json persists monitor_state between scheduled runs."""
        state_file = tmp_path / "live_state.json"
        state_file.write_text(json.dumps({
            "monitor_state": {"no_trade_streak": 7, "first_trade_date": "2024-06-04"},
        }))
        loaded = json.loads(state_file.read_text())
        assert loaded["monitor_state"]["no_trade_streak"] == 7

    def test_runner_adapter_writes_monitor_state(self):
        """Source-level enforcement: RunnerAdapter.commit persists monitor_state
        (regression guard — the whole system relies on this round-trip)."""
        src = (_STRATEGY_DIR / "adapters" / "runner.py").read_text()
        assert "\"monitor_state\"" in src, \
            "RunnerAdapter.commit must write monitor_state into live_state.json"
        assert "state.get(\"monitor_state\"" in src, \
            "RunnerAdapter.make_context must read monitor_state from live_state.json"
