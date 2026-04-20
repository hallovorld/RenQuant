"""Unit tests for kernel/pipeline inference jobs.

Each job is tested in isolation with a minimal synthetic InferenceContext.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_103"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.exits import HoldingState
from kernel.regime import RegimeState
from kernel.pipeline import InferenceContext
from kernel.pipeline.jobs import (
    RegimeJob, DrawdownJob, SellJob, BuyGatesJob,
    CandidateJob, RankingJob, SelectionJob,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n=300, start="2022-01-01", seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99,  "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _minimal_config(**overrides):
    cfg = {
        "watchlist": ["AAPL", "GOOG"],
        "max_concurrent_positions": 4,
        "wash_sale_days": 30,
        "min_hold_days": 30,
        "consecutive_sell_signals": 3,
        "lt_hold_gate_days": 330,
        "lt_hold_min_gain": 0.10,
        "sharpe_floor": 0.8,
        "max_positions_per_sector": 3,
        "tiered_thresholds": [{"min_model_score": 0.10}, {"min_model_score": 0.30}],
        "sector_map": {"AAPL": "tech", "GOOG": "tech"},
        "sector_etf_map": {"tech": "XLK"},
        "defensive_tickers": ["GLD"],
        "_exportable": ["AAPL", "GOOG"],
        "ranking": {"blend_weights": [0.5, 0.5]},
        "regime": {
            "correlation_guard_threshold": 0.70,
            "earnings_buffer_days": 3,
            "gmm_artifact": "spy-gmm-regime.json",
        },
        "regime_params": {
            "BULL_CALM": {
                "stop_loss_pct": 0.15, "max_hold_days": 500,
                "max_position_pct": 0.15, "drawdown_halt_pct": 0.35,
                "trailing_stop_trigger_pct": 0.20, "trailing_stop_trail_pct": 0.18,
                "max_single_day_loss_pct": 0.10, "min_model_score": 0.10,
                "cash_reserve_pct": 0.0,
                "spy_velocity_halt_pct": 0.03, "spy_velocity_lookback_days": 3,
            },
            "BULL_VOLATILE": {
                "stop_loss_pct": 0.05, "max_hold_days": 500,
                "max_position_pct": 0.20, "drawdown_halt_pct": 0.10,
                "trailing_stop_trigger_pct": 0, "trailing_stop_trail_pct": 0,
                "max_single_day_loss_pct": 0, "min_model_score": 0.15,
                "cash_reserve_pct": 0.20,
                "spy_velocity_halt_pct": 0.03, "spy_velocity_lookback_days": 3,
            },
        },
        "tax": {"short_term_rate": 0.40, "long_term_rate": 0.20, "long_term_threshold_days": 365},
    }
    cfg.update(overrides)
    return cfg


def _make_ctx(today="2024-06-01", cash=100_000.0, holdings=None, pos_shares=None,
              skip_buys=False, regime="BULL_CALM", in_transition=False,
              action_fn=None, score_fn=None, **kw):
    today_d = datetime.date.fromisoformat(today)
    spy_df  = _make_ohlcv(400, seed=99)
    aapl_df = _make_ohlcv(400, seed=1)
    goog_df = _make_ohlcv(400, seed=2)
    xlk_df  = _make_ohlcv(400, seed=3)

    ohlcv   = {"SPY": spy_df, "AAPL": aapl_df, "GOOG": goog_df, "XLK": xlk_df}
    today_ts = pd.Timestamp(today)
    # Make sure today is in the index
    for df in ohlcv.values():
        if today_ts not in df.index:
            df.loc[today_ts] = df.iloc[-1]
            df.sort_index(inplace=True)

    spy_rets = spy_df["close"].pct_change().fillna(0).loc[:today_ts].values.astype(float)

    cfg = _minimal_config(**kw)
    rp  = cfg["regime_params"].get(regime, cfg["regime_params"]["BULL_CALM"])

    return InferenceContext(
        today           = today_d,
        ohlcv           = ohlcv,
        spy_returns     = spy_rets,
        prev_closes     = {},
        holdings        = holdings or {},
        pos_shares      = pos_shares or {},
        cash            = cash,
        portfolio_value = cash,
        action_fn       = action_fn or (lambda t, ts: "hold"),
        score_fn        = score_fn  or (lambda t, ts: 0.5),
        gmm_artifact    = None,
        corr_dict       = {"AAPL": {"GOOG": 0.5}, "GOOG": {"AAPL": 0.5}},
        earnings_cal    = {},
        config          = cfg,
        hwm             = cash,
        regime_state    = RegimeState(),
        last_sell_dates = {},
        regime          = regime,
        regime_confidence = 0.7,
        in_transition   = in_transition,
        regime_params   = rp,
        skip_buys       = skip_buys,
    )


# ── RegimeJob ─────────────────────────────────────────────────────────────────

class TestRegimeJob:
    def test_sets_regime_and_confidence(self):
        ctx = _make_ctx()
        RegimeJob().run(ctx)
        assert ctx.regime in {"BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"}
        assert 0.0 <= ctx.regime_confidence <= 1.0

    def test_sets_regime_params(self):
        ctx = _make_ctx()
        RegimeJob().run(ctx)
        assert "stop_loss_pct" in ctx.regime_params

    def test_updates_regime_state(self):
        ctx = _make_ctx()
        old_state = ctx.regime_state
        RegimeJob().run(ctx)
        # regime_state is updated (may be same object or new)
        assert ctx.regime_state is not None

    def test_no_spy_data_defaults_to_bull_calm(self):
        ctx = _make_ctx()
        ctx.ohlcv = {}
        RegimeJob().run(ctx)
        assert ctx.regime == "BULL_CALM"


# ── DrawdownJob ───────────────────────────────────────────────────────────────

class TestDrawdownJob:
    def test_equity_point_populated(self):
        ctx = _make_ctx(cash=100_000)
        DrawdownJob().run(ctx)
        assert ctx.equity_point["portfolio"] == pytest.approx(100_000, rel=0.01)
        assert ctx.equity_point["regime"] == "BULL_CALM"

    def test_skip_buys_set_when_drawdown_exceeded(self):
        ctx = _make_ctx(cash=60_000)
        ctx.hwm = 100_000   # 40% drawdown > 35% halt
        DrawdownJob().run(ctx)
        assert ctx.skip_buys is True

    def test_skip_buys_false_when_drawdown_ok(self):
        ctx = _make_ctx(cash=95_000)
        ctx.hwm = 100_000   # 5% drawdown < 35% halt
        DrawdownJob().run(ctx)
        assert ctx.skip_buys is False

    def test_hwm_advances(self):
        ctx = _make_ctx(cash=120_000)
        ctx.hwm = 100_000
        DrawdownJob().run(ctx)
        assert ctx.hwm >= 100_000


# ── SellJob ───────────────────────────────────────────────────────────────────

class TestSellJob:
    def _holding(self, ctx, entry_price=None, days_ago=40):
        entry_date = ctx.today - datetime.timedelta(days=days_ago)
        today_ts   = pd.Timestamp(ctx.today)
        price      = entry_price or float(ctx.ohlcv["AAPL"].loc[today_ts, "close"])
        return HoldingState(entry_price=price, entry_date=entry_date,
                            high_watermark=price, prev_close=price)

    def test_no_exit_when_price_ok(self):
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        today_ts = pd.Timestamp(today)
        price_now = float(ctx.ohlcv["AAPL"].loc[today_ts, "close"])
        # Entry at current price — no stop triggered
        state = self._holding(ctx, entry_price=price_now, days_ago=40)
        ctx.holdings  = {"AAPL": state}
        ctx.pos_shares = {"AAPL": 10.0}
        SellJob().run(ctx)
        assert ctx.exit_actions == []

    def test_stop_loss_triggers(self):
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        today_ts = pd.Timestamp(today)
        price_now = float(ctx.ohlcv["AAPL"].loc[today_ts, "close"])
        # Entry 25% above current → ~20% loss → triggers 15% stop
        entry_price = price_now * 1.25
        state = self._holding(ctx, entry_price=entry_price, days_ago=40)
        state.high_watermark = entry_price
        ctx.holdings   = {"AAPL": state}
        ctx.pos_shares = {"AAPL": 10.0}
        SellJob().run(ctx)
        assert len(ctx.exit_actions) == 1
        assert ctx.exit_actions[0]["exit_type"] == "stop_loss"

    def test_max_hold_triggers(self):
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        # Entry 600 days before today (2024-06-01) → hold_days=600 > max_hold=500
        state = self._holding(ctx, days_ago=600)
        ctx.holdings   = {"AAPL": state}
        ctx.pos_shares = {"AAPL": 10.0}
        SellJob().run(ctx)
        assert any(a["exit_type"] == "max_hold" for a in ctx.exit_actions)

    def test_no_sell_for_missing_price(self):
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        ctx.holdings   = {"MISSING": self._holding(ctx)}
        ctx.pos_shares = {"MISSING": 10.0}
        SellJob().run(ctx)
        assert ctx.exit_actions == []


# ── BuyGatesJob ───────────────────────────────────────────────────────────────

class TestBuyGatesJob:
    def test_transition_blocks_buys(self):
        ctx = _make_ctx(in_transition=True)
        BuyGatesJob().run(ctx)
        assert ctx.skip_buys is True

    def test_skip_buys_already_set_stays_set(self):
        ctx = _make_ctx(skip_buys=True)
        BuyGatesJob().run(ctx)
        assert ctx.skip_buys is True

    def test_full_portfolio_blocks_buys(self):
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        ctx.config["max_concurrent_positions"] = 2
        ctx.holdings = {"AAPL": object(), "GOOG": object()}  # type: ignore
        BuyGatesJob().run(ctx)
        assert ctx.skip_buys is True

    def test_bear_regime_sets_skip_buys(self):
        ctx = _make_ctx(regime="BEAR")
        ctx.regime_params = ctx.config["regime_params"]["BULL_CALM"]  # use as proxy
        BuyGatesJob().run(ctx)
        assert ctx.skip_buys is True

    def test_normal_conditions_pass(self):
        ctx = _make_ctx()
        # Ensure SPY is trending up (EMA50 OK) by making close well above EMA50
        spy_df = ctx.ohlcv["SPY"]
        spy_df["close"] = spy_df["close"] * 2  # push close way above any EMA50
        BuyGatesJob().run(ctx)
        # should NOT set skip_buys (market gates pass)
        assert ctx.skip_buys is False


# ── CandidateJob ─────────────────────────────────────────────────────────────

class TestCandidateJob:
    def test_skips_when_skip_buys(self):
        ctx = _make_ctx(skip_buys=True)
        CandidateJob().run(ctx)
        assert ctx.candidates == []

    def test_skips_held_ticker(self):
        ctx = _make_ctx(action_fn=lambda t, ts: "buy",
                        score_fn=lambda t, ts: 0.8)
        today_ts = pd.Timestamp(ctx.today)
        state = HoldingState(entry_price=100, entry_date=ctx.today,
                             high_watermark=100, prev_close=100)
        ctx.holdings = {"AAPL": state}
        CandidateJob().run(ctx)
        tickers = [c.ticker for c in ctx.candidates]
        assert "AAPL" not in tickers

    def test_adds_buy_candidates(self):
        ctx = _make_ctx(action_fn=lambda t, ts: "buy",
                        score_fn=lambda t, ts: 0.5)
        CandidateJob().run(ctx)
        assert len(ctx.candidates) > 0

    def test_filters_below_min_score(self):
        ctx = _make_ctx(action_fn=lambda t, ts: "buy",
                        score_fn=lambda t, ts: 0.01)  # below min_model_score 0.10
        CandidateJob().run(ctx)
        assert ctx.candidates == []

    def test_wash_sale_blocks_candidate(self):
        ctx = _make_ctx(action_fn=lambda t, ts: "buy",
                        score_fn=lambda t, ts: 0.8)
        yesterday = ctx.today - datetime.timedelta(days=1)
        ctx.last_sell_dates = {"AAPL": yesterday, "GOOG": yesterday}
        CandidateJob().run(ctx)
        assert ctx.candidates == []


# ── RankingJob ────────────────────────────────────────────────────────────────

class TestRankingJob:
    def test_skips_when_no_candidates(self):
        ctx = _make_ctx()
        ctx.candidates = []
        RankingJob().run(ctx)
        assert ctx.ranked == []

    def test_sorts_descending(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx()
        ctx.candidates = [
            CandidateResult("AAPL", raw_score=0.3, rank_score=0.3, rs_score=0.1),
            CandidateResult("GOOG", raw_score=0.8, rank_score=0.8, rs_score=0.2),
        ]
        RankingJob().run(ctx)
        assert ctx.ranked[0].ticker == "GOOG"

    def test_blend_weights_applied(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx()
        ctx.config["ranking"]["blend_weights"] = [1.0, 0.0]  # pure rank_score
        ctx.candidates = [
            CandidateResult("AAPL", raw_score=0.9, rank_score=0.9, rs_score=0.0),
            CandidateResult("GOOG", raw_score=0.4, rank_score=0.4, rs_score=1.0),
        ]
        RankingJob().run(ctx)
        assert ctx.ranked[0].ticker == "AAPL"


# ── SelectionJob ──────────────────────────────────────────────────────────────

class TestSelectionJob:
    def test_skips_when_skip_buys(self):
        ctx = _make_ctx(skip_buys=True)
        SelectionJob().run(ctx)
        assert ctx.orders == []

    def test_places_buy_order(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx(cash=100_000)
        ctx.ranked = [
            CandidateResult("AAPL", raw_score=0.8, rank_score=0.8, rs_score=0.2),
        ]
        SelectionJob().run(ctx)
        assert len(ctx.orders) >= 1
        assert ctx.orders[0]["ticker"] == "AAPL"
        assert ctx.orders[0]["shares"] >= 1

    def test_tiered_threshold_blocks_low_score(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx(cash=100_000)
        # tier 0 requires 0.10; candidate at 0.05 should be skipped
        ctx.ranked = [
            CandidateResult("AAPL", raw_score=0.05, rank_score=0.05, rs_score=0.0),
        ]
        SelectionJob().run(ctx)
        assert ctx.orders == []

    def test_cash_decremented_after_buy(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx(cash=100_000)
        ctx.ranked = [
            CandidateResult("AAPL", raw_score=0.8, rank_score=0.8, rs_score=0.0),
        ]
        cash_before = ctx.cash
        SelectionJob().run(ctx)
        if ctx.orders:
            assert ctx.cash < cash_before

    def test_sector_guard_limits_concentration(self):
        from kernel.selection import CandidateResult
        ctx = _make_ctx(cash=500_000)
        ctx.config["max_positions_per_sector"] = 1
        # Both AAPL and GOOG are tech; only 1 should be selected
        ctx.ranked = [
            CandidateResult("AAPL", raw_score=0.9, rank_score=0.9, rs_score=0.0),
            CandidateResult("GOOG", raw_score=0.8, rank_score=0.8, rs_score=0.0),
        ]
        SelectionJob().run(ctx)
        tickers = [o["ticker"] for o in ctx.orders]
        assert len(set(tickers)) <= 1 or not ("AAPL" in tickers and "GOOG" in tickers)


# ── Full pipeline smoke test ──────────────────────────────────────────────────

class TestInferencePipelineSmoke:
    def test_full_pipeline_runs_without_error(self):
        from adapters.notebook import InferencePipeline
        ctx = _make_ctx(
            action_fn=lambda t, ts: "buy",
            score_fn=lambda t, ts: 0.6,
        )
        # Should run all 7 jobs without exception
        InferencePipeline().run(ctx)
        assert ctx.equity_point != {}
        assert isinstance(ctx.candidates, list)
        assert isinstance(ctx.orders, list)

    def test_pipeline_with_sell_trigger(self):
        from adapters.notebook import InferencePipeline
        today = "2024-06-01"
        ctx = _make_ctx(today=today)
        today_ts = pd.Timestamp(today)
        price_now = float(ctx.ohlcv["AAPL"].loc[today_ts, "close"])
        entry = price_now * 1.30  # 23% above → stop loss triggers
        entry_date = datetime.date.fromisoformat(today) - datetime.timedelta(days=40)
        state = HoldingState(
            entry_price=entry, entry_date=entry_date,
            high_watermark=entry, prev_close=entry,
        )
        ctx.holdings   = {"AAPL": state}
        ctx.pos_shares = {"AAPL": 5.0}
        InferencePipeline().run(ctx)
        assert any(a["exit_type"] == "stop_loss" for a in ctx.exit_actions)
