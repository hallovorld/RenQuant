"""SignalJob — regime detection, account state, parallel candidate scoring.

Reads from ctx: ohlcv, df_spy, gmm_artifact, corr_matrix, earnings_cal, models.
Populates ctx: regime, confidence, in_transition, transition_countdown,
               regime_params, spy_price, spy_above_ema50, spy_vel_ok,
               account_value, cash_avail, positions_cache, pending_orders,
               held, circuit_open, state, entry_dates, sell_streaks,
               last_sell_dates, position_hwm, candidates.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import numpy as np

log = logging.getLogger("pipeline.signals")

from ..context import PipelineContext
from ..pipeline import Job
from ..task import run_tasks


class SignalJob(Job):
    """Detect regime, load account/live state, score all non-held candidates."""

    def run(self, ctx: PipelineContext) -> None:
        _detect_regime(ctx)
        _load_account_state(ctx)
        _score_candidates(ctx)


# ── Regime detection ──────────────────────────────────────────────────────────

def _detect_regime(ctx: PipelineContext) -> None:
    from kernel.regime import detect_regime, RegimeState  # noqa: PLC0415

    config    = ctx.config
    df_spy    = ctx.df_spy
    spy_close = df_spy["close"].astype(float)
    spy_rets  = spy_close.pct_change().dropna().values

    rs = detect_regime(spy_rets, df_spy, ctx.gmm_artifact, RegimeState(), config)
    ctx.regime       = rs.regime
    ctx.confidence   = rs.confidence
    ctx.in_transition = rs.in_transition

    # Resolve regime-specific params
    regime_params_map = config.get("regime_params", {})
    current_rp  = regime_params_map.get(ctx.regime, regime_params_map.get("BULL_CALM", {}))
    pos_sizing  = config.get("position_sizing", {})
    ctx.regime_params = {
        "max_position_pct":    float(current_rp.get("max_position_pct",     pos_sizing.get("max_position_pct",     0.15))),
        "cash_reserve_pct":    float(current_rp.get("cash_reserve_pct",     pos_sizing.get("cash_reserve_pct",     0.00))),
        "stop_loss_pct":       float(current_rp.get("stop_loss_pct",        config.get("risk", {}).get("stop_loss_pct",   0.0))),
        "max_single_day_loss_pct": float(current_rp.get("max_single_day_loss_pct", 0.0)),
        "spy_velocity_halt_pct":   float(current_rp.get("spy_velocity_halt_pct",   0.03)),
        "spy_velocity_lookback_days": int(current_rp.get("spy_velocity_lookback_days", 3)),
        "trailing_stop_trigger_pct": float(current_rp.get("trailing_stop_trigger_pct", 0.0)),
        "trailing_stop_trail_pct":   float(current_rp.get("trailing_stop_trail_pct",   0.0)),
        "max_hold_days":       int(current_rp.get("max_hold_days",          500)),
        "min_model_score":     float(current_rp.get("min_model_score",      config.get("min_model_score", 0.0))),
    }

    # SPY market gates
    spy_vel_days = ctx.regime_params["spy_velocity_lookback_days"]
    spy_vel_halt = ctx.regime_params["spy_velocity_halt_pct"]
    ctx.spy_price = float(spy_close.iloc[-1])
    spy_ema50     = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
    ctx.spy_above_ema50 = ctx.spy_price >= spy_ema50

    spy_vel_ret = 0.0
    if len(spy_close) > spy_vel_days:
        spy_vel_ret = float(spy_close.iloc[-1] / spy_close.iloc[-1 - spy_vel_days] - 1)
    ctx.spy_vel_ok = spy_vel_ret >= -spy_vel_halt

    log.info("Regime: %s  conf=%.0f%%  CUSUM=%s  EMA50=%s  VelCrash=%s",
             ctx.regime, ctx.confidence * 100,
             "TRIGGERED" if ctx.in_transition else "clear",
             "CLEAR" if ctx.spy_above_ema50 else "BLOCKING",
             "CLEAR" if ctx.spy_vel_ok else "BLOCKING")


# ── Account + live state ──────────────────────────────────────────────────────

def _load_account_state(ctx: PipelineContext) -> None:
    broker       = ctx.broker
    config       = ctx.config
    strategy_dir = ctx.strategy_dir
    risk_cfg     = config.get("risk", {})

    # Broker account
    ctx.account_value = broker.get_account_value()
    try:
        ctx.cash_avail = broker.get_cash()
    except Exception as exc:
        log.warning("get_cash() failed, using account equity: %s", exc)
        ctx.cash_avail = ctx.account_value

    try:
        all_pos = broker.get_all_positions()
    except Exception as exc:
        log.warning("get_all_positions() failed: %s", exc)
        all_pos = []
    ctx.positions_cache = {p["symbol"]: p for p in all_pos}

    # Load persisted live state
    state_file = strategy_dir / "live_state.json"
    ctx.state  = json.loads(state_file.read_text()) if state_file.exists() else {}
    ctx.entry_dates      = ctx.state.setdefault("entry_dates", {})
    ctx.sell_streaks     = ctx.state.setdefault("sell_streaks", {})
    ctx.last_sell_dates  = ctx.state.setdefault("last_sell_dates", {})
    ctx.position_hwm     = ctx.state.setdefault("position_hwm", {})

    # Persist current regime (sell-only runs retain last regime)
    ctx.state["regime"] = ctx.regime

    # Transition countdown
    regime_cfg = config.get("regime", {})
    trans_bars = int(regime_cfg.get("transition_uncertainty_bars", 3))
    countdown  = int(ctx.state.get("transition_countdown", 0))
    if ctx.in_transition and countdown == 0:
        countdown = trans_bars
        log.info("CUSUM changepoint — transition countdown reset to %d bars", trans_bars)
    elif countdown > 0:
        countdown -= 1
    ctx.state["transition_countdown"] = countdown
    ctx.transition_countdown = countdown

    regime_confidence = 0.5 if countdown > 0 else ctx.confidence
    ctx.state["regime_confidence"] = round(regime_confidence, 4)
    ctx.confidence = regime_confidence  # use dampened confidence for sizing

    # Drawdown circuit breaker
    drawdown_halt_pct = float(risk_cfg.get("portfolio_drawdown_halt_pct", 0.0))
    hwm = float(ctx.state.get("high_water_mark", ctx.account_value))
    hwm = max(hwm, ctx.account_value)
    ctx.state["high_water_mark"] = hwm
    drawdown = (hwm - ctx.account_value) / hwm if hwm > 0 else 0.0
    ctx.circuit_open = drawdown_halt_pct > 0 and drawdown >= drawdown_halt_pct
    log.info("Account: equity=$%.0f  cash=$%.0f  drawdown=%.1f%%  circuit=%s",
             ctx.account_value, ctx.cash_avail, drawdown * 100,
             "OPEN" if ctx.circuit_open else "CLEAR")

    # Held positions from watchlist
    watchlist = config["watchlist"]
    ctx.held  = [s for s in watchlist if ctx.positions_cache.get(s, {}).get("qty", 0.0) > 0]

    # Broker order reconciliation (fill history → patch entry_dates / last_sell_dates)
    _reconcile_broker_history(ctx)

    # Pending orders
    try:
        ctx.pending_orders = broker.get_open_orders()
    except Exception as exc:
        log.warning("get_open_orders() failed: %s", exc)
        ctx.pending_orders = set()


def _reconcile_broker_history(ctx: PipelineContext) -> None:
    """Patch entry_dates / last_sell_dates from Alpaca fill history."""
    import datetime as _dt
    sixty_days_ago = (date.today() - _dt.timedelta(days=60)).isoformat()
    try:
        filled = ctx.broker.get_filled_orders(after=sixty_days_ago)
        last_buy:  dict[str, str] = {}
        last_sell: dict[str, str] = {}
        for o in filled:
            sym      = o["symbol"]
            fill_day = (o.get("filled_at") or "")[:10]
            if not fill_day:
                continue
            if o["action"] == "BUY":
                if sym not in last_buy or fill_day > last_buy[sym]:
                    last_buy[sym] = fill_day
            else:
                if sym not in last_sell or fill_day > last_sell[sym]:
                    last_sell[sym] = fill_day
        for sym, buy_day in last_buy.items():
            if sym not in ctx.entry_dates and ctx.positions_cache.get(sym, {}).get("qty", 0.0) > 0:
                ctx.entry_dates[sym] = buy_day
                log.info("Reconcile: %s entry_date=%s (Alpaca history)", sym, buy_day)
        for sym, sell_day in last_sell.items():
            if ctx.last_sell_dates.get(sym, "") < sell_day:
                ctx.last_sell_dates[sym] = sell_day
                log.info("Reconcile: %s last_sell=%s (Alpaca history)", sym, sell_day)
    except Exception as exc:
        log.warning("Broker reconciliation failed (non-fatal): %s", exc)


# ── Parallel candidate scoring ─────────────────────────────────────────────────

def _score_candidates(ctx: PipelineContext) -> None:
    """Score all non-held, non-blocked symbols in parallel; populate ctx.candidates."""
    from kernel.selection import CandidateResult, compute_relative_strength  # noqa: PLC0415
    from kernel.models import score_artifact                                  # noqa: PLC0415
    from kernel.indicators import build_feature_frame                         # noqa: PLC0415

    config        = ctx.config
    indicator_spec = config.get("indicator_spec", {})
    regime_cfg    = config.get("regime", {})
    earnings_win  = int(regime_cfg.get("earnings_buffer_days", 3))
    wash_days     = int(config.get("wash_sale_days", 30))
    sector_map    = config.get("sector_map", {})
    sector_etf_map = config.get("sector_etf_map", {})
    defensive_set = set(config.get("defensive_tickers", ["GLD", "TLT", "XLV", "XLU"]))

    is_bear      = ctx.regime == "BEAR"
    scan_universe = list(defensive_set) if is_bear else config["watchlist"]

    min_score     = ctx.regime_params.get("min_model_score", 0.0)
    df_spy        = ctx.df_spy
    spy_close     = df_spy["close"].astype(float)
    today         = ctx.today

    def _score_one(symbol: str) -> CandidateResult | None:
        if symbol in ctx.held:
            return None
        if symbol not in ctx.ohlcv or symbol not in ctx.models:
            return None

        # Wash-sale pre-filter
        if wash_days > 0 and symbol in ctx.last_sell_dates:
            try:
                days_since = (today - date.fromisoformat(ctx.last_sell_dates[symbol])).days
                if days_since < wash_days:
                    return None
            except ValueError:
                pass

        # Earnings filter
        if earnings_win > 0 and symbol in ctx.earnings_cal:
            for ds in ctx.earnings_cal.get(symbol, []):
                try:
                    earn_date = date.fromisoformat(ds)
                    if abs((today - earn_date).days) <= earnings_win:
                        return None
                except ValueError:
                    pass

        # Feature frame
        rel = build_feature_frame(ctx.ohlcv[symbol], df_spy, indicator_spec)
        if rel is None or rel.empty:
            return None

        row = rel.iloc[-1].copy()
        row["position_flag"] = 0

        try:
            score_eval = score_artifact(ctx.models[symbol], row)
        except Exception as exc:
            log.warning("score_artifact failed for %s: %s", symbol, exc)
            return None

        if score_eval.signal != "buy":
            return None
        if score_eval.rank_score < min_score:
            return None

        # Relative-strength vs sector ETF (20-day)
        rs_score = 0.0
        sector   = sector_map.get(symbol, "other")
        etf      = sector_etf_map.get(sector)
        if etf and etf in ctx.ohlcv and len(ctx.ohlcv[etf]) >= 21:
            try:
                stock_r = float(ctx.ohlcv[symbol]["close"].iloc[-1] / ctx.ohlcv[symbol]["close"].iloc[-21] - 1)
                etf_r   = float(ctx.ohlcv[etf]["close"].iloc[-1]   / ctx.ohlcv[etf]["close"].iloc[-21]   - 1)
                rs_score = compute_relative_strength(stock_r, etf_r)
            except Exception:
                pass

        return CandidateResult(
            ticker=symbol,
            raw_score=score_eval.raw_score,
            rank_score=score_eval.rank_score,
            rs_score=rs_score,
            detail=score_eval.signal,
        )

    tasks = [(sym, lambda s=sym: _score_one(s)) for sym in scan_universe]
    results = run_tasks(tasks, max_workers=8)

    candidates = []
    for r in results:
        if r.error:
            log.warning("Scoring error for %s: %s", r.name, r.error)
            continue
        if r.result is not None:
            candidates.append(r.result)
            log.info("  %-6s  BUY  raw=%+.4f  calibrated=%+.4f  rs=%+.4f → CANDIDATE",
                     r.result.ticker, r.result.raw_score, r.result.rank_score, r.result.rs_score)

    # Rank candidates
    if len(candidates) > 1:
        ranking_cfg = config.get("ranking", {})
        bw = ranking_cfg.get("blend_weights", [0.5, 0.5])
        total = float(bw[0]) + float(bw[1])
        w_rank = float(bw[0]) / total if total > 0 else 0.5
        w_rs   = float(bw[1]) / total if total > 0 else 0.5
        from kernel.selection import score_candidates  # noqa: PLC0415
        candidates = score_candidates(candidates, w_rank, w_rs)

    ctx.candidates = candidates
    log.info("SignalJob: %d buy candidates after scan", len(candidates))
