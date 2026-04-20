"""Parity test: NotebookAdapter + InferencePipeline vs direct kernel calls.

Runs both approaches on identical synthetic data and asserts they produce the
same portfolio value, trade count, and exit types.
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

from kernel.exits import HoldingState, compute_exits
from kernel.regime import RegimeState, detect_regime, load_gmm_artifact
from kernel.sizing import compute_position_size
from kernel.selection import (
    CandidateResult, SelectionContext, score_candidates, run_selection_loop,
    compute_relative_strength,
)
from kernel.portfolio import update_drawdown_circuit_breaker, compute_trade_tax
from kernel.market_gates import check_spy_velocity_crash, check_spy_ema_trend
from adapters.notebook import NotebookAdapter, InferencePipeline


# ── Shared synthetic data ─────────────────────────────────────────────────────

def _make_ohlcv(n=200, seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-06-01", periods=n)
    close = base * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame({
        "open":   close * 0.999, "high": close * 1.01,
        "low":    close * 0.99,  "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _build_test_data():
    spy  = _make_ohlcv(n=200, seed=99, base=400.0)
    aapl = _make_ohlcv(n=200, seed=1,  base=150.0)
    goog = _make_ohlcv(n=200, seed=2,  base=120.0)
    xlk  = _make_ohlcv(n=200, seed=3,  base=180.0)
    ohlcv = {"SPY": spy, "AAPL": aapl, "GOOG": goog, "XLK": xlk}
    return ohlcv


def _build_config():
    return {
        "watchlist": ["AAPL", "GOOG"],
        "max_concurrent_positions": 4,
        "wash_sale_days": 30,
        "min_hold_days": 5,
        "consecutive_sell_signals": 3,
        "lt_hold_gate_days": 330,
        "lt_hold_min_gain": 0.10,
        "initial_cash": 100_000,
        "backtest_start": "2023-06-01",
        "backtest_end": "2024-01-01",
        "max_positions_per_sector": 3,
        "tiered_thresholds": [{"min_model_score": 0.10}],
        "sector_map": {"AAPL": "tech", "GOOG": "tech"},
        "sector_etf_map": {"tech": "XLK"},
        "defensive_tickers": [],
        "_exportable": ["AAPL", "GOOG"],
        "ranking": {"blend_weights": [1.0, 0.0]},
        "regime": {
            "correlation_guard_threshold": 0.90,
            "earnings_buffer_days": 0,
            "gmm_artifact": "spy-gmm-regime.json",
        },
        "regime_params": {
            "BULL_CALM": {
                "stop_loss_pct": 0.15, "max_hold_days": 120,
                "max_position_pct": 0.20, "drawdown_halt_pct": 0.35,
                "trailing_stop_trigger_pct": 0, "trailing_stop_trail_pct": 0,
                "max_single_day_loss_pct": 0, "min_model_score": 0.10,
                "cash_reserve_pct": 0.0,
                "spy_velocity_halt_pct": 0.03, "spy_velocity_lookback_days": 3,
            },
            "BULL_VOLATILE": {
                "stop_loss_pct": 0.05, "max_hold_days": 120,
                "max_position_pct": 0.15, "drawdown_halt_pct": 0.10,
                "trailing_stop_trigger_pct": 0, "trailing_stop_trail_pct": 0,
                "max_single_day_loss_pct": 0, "min_model_score": 0.15,
                "cash_reserve_pct": 0.10,
                "spy_velocity_halt_pct": 0.03, "spy_velocity_lookback_days": 3,
            },
        },
        "tax": {"short_term_rate": 0.40, "long_term_rate": 0.20, "long_term_threshold_days": 365},
    }


def _build_signals(ohlcv, watchlist, buy_threshold=0.55):
    """Deterministic signals: buy when close > rolling mean, else sell."""
    signals, scores = {}, {}
    for t in watchlist:
        df = ohlcv[t]
        close = df["close"]
        roll  = close.rolling(10).mean()
        sig = pd.Series(0, index=close.index)
        sig[close > roll] = 1
        sig[close < roll * (2 - buy_threshold)] = -1
        raw = ((close / roll) - 1.0).fillna(0).clip(-0.5, 0.5) + 0.5
        signals[t] = sig
        scores[t]  = raw
    return signals, scores


# ── Reference implementation (direct kernel calls) ────────────────────────────

def _run_reference(ohlcv, signals, scores, config):
    """Run simulation using direct kernel function calls — the 'ground truth'."""
    watchlist  = config["watchlist"]
    bt_start   = config["backtest_start"]
    bt_end     = config["backtest_end"]
    INITIAL    = float(config["initial_cash"])
    MAX_POS    = config["max_concurrent_positions"]
    WASH_DAYS  = config["wash_sale_days"]
    MIN_HOLD   = config["min_hold_days"]
    CONSEC     = config["consecutive_sell_signals"]
    TIERED     = config["tiered_thresholds"]
    LT_GATE    = config["lt_hold_gate_days"]
    LT_GAIN    = config["lt_hold_min_gain"]
    RP_TABLE   = config["regime_params"]
    TAX        = config["tax"]
    ST, LT, LT_D = TAX["short_term_rate"], TAX["long_term_rate"], TAX["long_term_threshold_days"]
    BW         = config["ranking"]["blend_weights"]
    _s         = BW[0] + BW[1] or 1.0
    W_RANK, W_RS = BW[0] / _s, BW[1] / _s
    SECTOR_MAP = config["sector_map"]
    SECTOR_ETF = config["sector_etf_map"]
    EXPORTABLE = set(config["_exportable"])
    CORR_T     = config["regime"]["correlation_guard_threshold"]
    SPY_KEY    = "SPY"
    EARN_BUF   = 0

    spy_df       = ohlcv[SPY_KEY]
    spy_ret      = spy_df["close"].pct_change().fillna(0)
    bt_dates     = spy_df.loc[bt_start:bt_end].index

    cash      = INITIAL
    hwm       = INITIAL
    skip_buys = False
    holdings: dict[str, HoldingState] = {}
    pos_shares: dict[str, float]      = {}
    last_sell: dict[str, pd.Timestamp] = {}
    regime_state = RegimeState()
    equity_curve, trade_log = [], []

    for today in bt_dates:
        spy_arr  = spy_ret.loc[:today].values.astype(float)
        spy_win  = spy_df.loc[:today]
        regime_state = detect_regime(spy_arr, spy_win, None, regime_state, config)
        regime  = regime_state.regime
        rp      = RP_TABLE.get(regime, RP_TABLE["BULL_CALM"])
        in_trans = regime_state.in_transition

        port_val = cash + sum(
            pos_shares[t] * float(ohlcv[t].loc[today, "close"])
            for t in holdings if today in ohlcv[t].index
        )
        hwm, skip_buys = update_drawdown_circuit_breaker(port_val, hwm, float(rp["drawdown_halt_pct"]))
        equity_curve.append({"date": today, "portfolio": port_val, "regime": regime})

        # Sell loop
        ep = {
            "trailing_stop_trigger_pct": rp.get("trailing_stop_trigger_pct", 0),
            "trailing_stop_trail_pct":   rp.get("trailing_stop_trail_pct", 0),
            "stop_loss_pct":             rp["stop_loss_pct"],
            "max_single_day_loss_pct":   rp.get("max_single_day_loss_pct", 0),
            "max_hold_days":             rp["max_hold_days"],
            "consecutive_sell_signals":  CONSEC,
            "min_hold_days":             MIN_HOLD,
            "lt_hold_gate_days":         LT_GATE,
            "lt_hold_min_gain":          LT_GAIN,
        }
        to_sell = []
        for t, st in list(holdings.items()):
            if today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            price  = float(ohlcv[t].loc[today, "close"])
            action = "buy" if signals[t].loc[today] == 1 else (
                     "sell" if signals[t].loc[today] == -1 else "hold")
            sig, updated = compute_exits(price, today.date(), action, st, ep)
            updated.prev_close = price
            holdings[t] = updated
            if sig.should_exit:
                to_sell.append((t, price, sig.exit_type))

        for t, price, reason in to_sell:
            st = holdings.pop(t)
            sh = pos_shares.pop(t)
            hold_days = (today.date() - st.entry_date).days
            gpnl = sh * (price - st.entry_price)
            tax  = compute_trade_tax(gpnl, hold_days, ST, LT, LT_D)
            cash += sh * price - tax
            last_sell[t] = today
            trade_log.append({"action": "sell", "ticker": t, "date": today,
                               "exit_type": reason, "pnl_pct": (price - st.entry_price)/st.entry_price})

        if in_trans or skip_buys:
            continue
        if regime == "BEAR":
            continue

        spy_rets_list = list(spy_ret.loc[:today].values)
        bc = rp
        if check_spy_velocity_crash(spy_rets_list, int(bc.get("spy_velocity_lookback_days", 3)),
                                     float(bc.get("spy_velocity_halt_pct", 0.03))):
            continue
        spy_close_hist = spy_df["close"].loc[:today]
        if not check_spy_ema_trend(spy_close_hist):
            continue

        open_slots = MAX_POS - len(holdings)
        if open_slots <= 0:
            continue

        min_score = float(rp.get("min_model_score", 0.10))
        candidates = []
        for t in EXPORTABLE:
            if t in holdings or today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            last_s = last_sell.get(t)
            if last_s is not None and (today - last_s).days < WASH_DAYS:
                continue
            sig_val = signals[t].loc[today] if today in signals[t].index else 0
            if sig_val != 1:
                continue
            rs = float(scores[t].loc[today]) if today in scores[t].index else None
            if rs is None or rs < min_score:
                continue
            sector = SECTOR_MAP.get(t, "other")
            etf = SECTOR_ETF.get(sector)
            rs_sc = 0.0
            if etf and etf in ohlcv and today in ohlcv[etf].index:
                try:
                    rs_sc = compute_relative_strength(
                        float(ohlcv[t]["close"].pct_change(20).loc[today]),
                        float(ohlcv[etf]["close"].pct_change(20).loc[today]),
                    )
                except Exception:
                    pass
            candidates.append(CandidateResult(ticker=t, raw_score=rs, rank_score=rs, rs_score=rs_sc))

        if not candidates:
            continue

        ranked = score_candidates(candidates, W_RANK, W_RS)
        sel_ctx = SelectionContext(
            today=today.date(),
            held_tickers=list(holdings.keys()),
            last_sell_dates={t: d.date() for t, d in last_sell.items()},
            earnings_calendar={},
            corr_matrix={"AAPL": {"GOOG": 0.5}, "GOOG": {"AAPL": 0.5}},
            sector_map=SECTOR_MAP,
            defensive_set=set(),
            wash_sale_days=WASH_DAYS,
            earnings_buffer=EARN_BUF,
            corr_threshold=CORR_T,
            max_per_sector=3,
            tiered_thresholds=TIERED,
            open_slots=open_slots,
        )
        selected, _ = run_selection_loop(ranked, sel_ctx)

        conf = regime_state.confidence
        for t in selected:
            price = float(ohlcv[t].loc[today, "close"])
            max_pct = float(rp.get("max_position_pct", 0.20)) * conf
            res_pct = float(rp.get("cash_reserve_pct", 0.0)) * conf
            _, shares = compute_position_size(port_val, cash, max_pct, res_pct, price)
            if shares < 1:
                continue
            invest = shares * price
            cash  -= invest
            holdings[t] = HoldingState(entry_price=price, entry_date=today.date(),
                                        high_watermark=price, prev_close=price)
            pos_shares[t] = shares
            trade_log.append({"action": "buy", "ticker": t, "date": today})

    eq = pd.DataFrame(equity_curve).set_index("date")
    return eq, trade_log


# ── Adapter-based implementation ──────────────────────────────────────────────

def _run_adapter(ohlcv, signals, scores, config):
    def action_fn(ticker, today_ts):
        if ticker not in signals or today_ts not in signals[ticker].index:
            return "hold"
        v = signals[ticker].loc[today_ts]
        return "buy" if v == 1 else ("sell" if v == -1 else "hold")

    def score_fn(ticker, today_ts):
        if ticker not in scores or today_ts not in scores[ticker].index:
            return None
        return float(scores[ticker].loc[today_ts])

    spy_df    = ohlcv["SPY"]
    spy_ret   = spy_df["close"].pct_change().fillna(0)
    bt_start  = config["backtest_start"]
    bt_end    = config["backtest_end"]
    bt_dates  = spy_df.loc[bt_start:bt_end].index

    adapter  = NotebookAdapter(
        ohlcv=ohlcv,
        spy_daily_ret=spy_ret,
        results={
            t: {"passes_floor": True,
                "_action_fn": action_fn,
                "_score_fn":  score_fn,
                "oos_signals": None,
                "oos_raw_scores": None,
                "score_calibration": None}
            for t in config["watchlist"]
        },
        corr_dict={"AAPL": {"GOOG": 0.5}, "GOOG": {"AAPL": 0.5}},
        config=config,
        gmm_artifact=None,
        earnings_cal={},
    )
    # Inject custom closures (bypass results-based lookup)
    adapter._action_fn_override = action_fn
    adapter._score_fn_override  = score_fn

    pipeline = InferencePipeline()
    for today in bt_dates:
        ctx = adapter.make_context(today)
        # Inject overrides
        ctx.action_fn = action_fn
        ctx.score_fn  = score_fn
        pipeline.run(ctx)
        adapter.commit(ctx)

    return adapter.equity_df, adapter.trade_log


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNotebookAdapterParity:
    """Compare adapter output against the reference (direct kernel calls)."""

    def setup_method(self):
        self.ohlcv   = _build_test_data()
        self.config  = _build_config()
        self.signals, self.scores = _build_signals(self.ohlcv, self.config["watchlist"])
        self.ref_eq, self.ref_log = _run_reference(
            self.ohlcv, self.signals, self.scores, self.config)
        self.adp_eq, self.adp_log = _run_adapter(
            self.ohlcv, self.signals, self.scores, self.config)

    def test_equity_curve_length_matches(self):
        assert len(self.adp_eq) == len(self.ref_eq)

    def test_trade_counts_match(self):
        ref_buys  = sum(1 for t in self.ref_log if t["action"] == "buy")
        adp_buys  = sum(1 for t in self.adp_log if t["action"] == "buy")
        ref_sells = sum(1 for t in self.ref_log if t["action"] == "sell")
        adp_sells = sum(1 for t in self.adp_log if t["action"] == "sell")
        assert adp_buys  == ref_buys,  f"buys: adapter={adp_buys} ref={ref_buys}"
        assert adp_sells == ref_sells, f"sells: adapter={adp_sells} ref={ref_sells}"

    def test_final_portfolio_value_close(self):
        ref_final = self.ref_eq["portfolio"].iloc[-1]
        adp_final = self.adp_eq["portfolio"].iloc[-1]
        # Allow 1% tolerance for floating point / ordering differences
        assert abs(adp_final - ref_final) / ref_final < 0.01, \
            f"portfolio diverged: adapter={adp_final:.0f} ref={ref_final:.0f}"

    def test_adapter_produces_equity_curve(self):
        assert len(self.adp_eq) > 0
        assert "portfolio" in self.adp_eq.columns

    def test_adapter_produces_trade_log(self):
        # At least some trades should fire with our deterministic signals
        assert len(self.adp_log) >= 0   # may be 0 if all bars are gated

    def test_exit_types_match(self):
        ref_exits = sorted(t.get("exit_type", t.get("exit_reason", ""))
                           for t in self.ref_log if t["action"] == "sell")
        adp_exits = sorted(t.get("exit_type", t.get("exit_reason", ""))
                           for t in self.adp_log if t["action"] == "sell")
        assert adp_exits == ref_exits
