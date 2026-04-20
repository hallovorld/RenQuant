"""ExecutionJob — sell phase, buy gate checks, selection loop, order placement.

Reads from ctx: everything populated by DataJob and SignalJob.
Writes ctx.state back to live_state.json.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

log = logging.getLogger("pipeline.execution")

from ..context import PipelineContext
from ..pipeline import Job


class ExecutionJob(Job):
    """Execute the sell phase, then the buy phase."""

    def run(self, ctx: PipelineContext) -> None:
        _sell_phase(ctx)

        # Persist state after sells; return early if sell-only
        _save_state(ctx)
        if ctx.sell_only:
            log.info("sell_only mode — buy phase skipped")
            return

        _buy_phase(ctx)
        _save_state(ctx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_held(symbol: str, ctx: PipelineContext) -> int:
    entry_str = ctx.entry_dates.get(symbol)
    if not entry_str:
        return 0
    try:
        return (ctx.today - date.fromisoformat(entry_str)).days
    except ValueError:
        return 0


def _current_price(symbol: str, ctx: PipelineContext) -> tuple[float, str]:
    """Return (price, source) — prefer live Alpaca market value over OHLCV close."""
    pos = ctx.positions_cache.get(symbol, {})
    qty = float(pos.get("qty", 0))
    mkt = float(pos.get("market_value", 0))
    if qty > 0 and mkt > 0:
        return mkt / qty, "Alpaca"
    df   = ctx.ohlcv.get(symbol)
    if df is not None and not df.empty:
        last_date = str(df.index[-1])[:10]
        return float(df["close"].iloc[-1]), f"OHLCV {last_date}"
    return 0.0, "unknown"


def _log_trade(ctx: PipelineContext, record: dict) -> None:
    strategy_name = ctx.config.get("model_name", "renquant_103")
    # strategy_dir = backtesting/renquant_103/ → go up 2 levels to repo root
    repo_root = ctx.strategy_dir.parent.parent
    log_dir   = repo_root / "live" / "logs" / strategy_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    entries = json.loads(log_file.read_text()) if log_file.exists() else []
    entries.append(record)
    log_file.write_text(json.dumps(entries, indent=2, default=str))


def _save_state(ctx: PipelineContext) -> None:
    state_file = ctx.strategy_dir / "live_state.json"
    state_file.write_text(json.dumps(ctx.state, indent=2))


def _on_sell(symbol: str, ctx: PipelineContext) -> None:
    """Update all per-symbol state dicts after a successful sell."""
    ctx.entry_dates.pop(symbol, None)
    ctx.sell_streaks.pop(symbol, None)
    ctx.position_hwm.pop(symbol, None)
    ctx.last_sell_dates[symbol] = ctx.today_str
    if symbol in ctx.held:
        ctx.held.remove(symbol)


# ── Sell phase ────────────────────────────────────────────────────────────────

def _sell_phase(ctx: PipelineContext) -> None:
    from kernel.exits import HoldingState, compute_exits  # noqa: PLC0415
    from kernel.indicators import build_feature_frame     # noqa: PLC0415
    from kernel.models import score_artifact              # noqa: PLC0415

    config        = ctx.config
    indicator_spec = config.get("indicator_spec", {})
    exit_params    = {**ctx.regime_params,
                      "consecutive_sell_signals": int(config.get("consecutive_sell_signals", 3)),
                      "min_hold_days":            int(config.get("min_hold_days", 0))}

    log.info("SELL PHASE (%d held: %s)", len(ctx.held), ctx.held or "none")

    for symbol in list(ctx.held):
        if symbol not in ctx.ohlcv:
            continue

        price, price_src = _current_price(symbol, ctx)
        pos_data = ctx.positions_cache.get(symbol, {})
        avg_cost = float(pos_data.get("avg_entry_price", 0.0))
        qty      = float(pos_data.get("qty", 0.0))

        # Update per-position HWM
        prev_hwm = float(ctx.position_hwm.get(symbol, price))
        ctx.position_hwm[symbol] = max(prev_hwm, price)

        # Build HoldingState from persisted data
        entry_str = ctx.entry_dates.get(symbol, ctx.today_str)
        try:
            entry_dt = date.fromisoformat(entry_str)
        except ValueError:
            entry_dt = ctx.today

        days = (ctx.today - entry_dt).days
        unrealized = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
        peak_gain  = (ctx.position_hwm[symbol] - avg_cost) / avg_cost if avg_cost > 0 else 0.0

        log.info("  %s  $%.2f [%s]  held=%dd  entry=$%.2f  P&L %+.1f%%  HWM $%.2f  peak %+.1f%%",
                 symbol, price, price_src, days, avg_cost, unrealized * 100,
                 ctx.position_hwm[symbol], peak_gain * 100)

        # Model signal — needed for model-sell exit check
        model_action = "hold"
        if symbol in ctx.models:
            try:
                df_spy = ctx.df_spy
                rel = build_feature_frame(ctx.ohlcv[symbol], df_spy, indicator_spec)
                if rel is not None and not rel.empty:
                    row = rel.iloc[-1].copy()
                    row["position_flag"] = 1
                    ev = score_artifact(ctx.models[symbol], row)
                    model_action = ev.signal
            except Exception as exc:
                log.warning("Feature/model failed for %s sell eval: %s", symbol, exc)

        # Build state for kernel.exits.compute_exits
        prev_close_ser = ctx.ohlcv[symbol]["close"]
        prev_close = float(prev_close_ser.iloc[-2]) if len(prev_close_ser) >= 2 else None

        state = HoldingState(
            entry_price=avg_cost,
            entry_date=entry_dt,
            high_watermark=float(ctx.position_hwm[symbol]),
            sell_streak=int(ctx.sell_streaks.get(symbol, 0)),
            prev_close=prev_close,
        )

        exit_sig, state = compute_exits(
            current_price=price,
            today=ctx.today,
            model_action=model_action,
            state=state,
            params=exit_params,
        )

        # Persist updated sell streak (regardless of exit outcome)
        ctx.sell_streaks[symbol] = state.sell_streak

        if not exit_sig.should_exit:
            log.info("    HOLD  [%s]", exit_sig.reason or "no exit triggered")
            continue

        # Execute sell
        realized_pl = (price - avg_cost) * qty
        try:
            result = ctx.broker.place_order(symbol, "SELL", abs(qty))
        except Exception as exc:
            log.error("SELL order FAILED [%s] %s — %s", exit_sig.exit_type, symbol, exc)
            continue

        log.info("    → SELL [%s]  %s  %.0f shares @ $%.2f  P&L $%+.2f (%+.1f%%)",
                 exit_sig.exit_type, symbol, abs(qty), price, realized_pl, unrealized * 100)

        _log_trade(ctx, {
            "timestamp": datetime.now().isoformat(),
            "symbol":    symbol,
            "signal":    exit_sig.exit_type,
            "sell_price": round(price, 4),
            "avg_cost":   round(avg_cost, 4),
            "qty":        qty,
            "realized_pl": round(realized_pl, 2),
            "reason":    exit_sig.reason,
            "order":     result,
        })
        _on_sell(symbol, ctx)


# ── Buy phase ─────────────────────────────────────────────────────────────────

def _buy_phase(ctx: PipelineContext) -> None:
    from kernel.sizing import compute_position_size    # noqa: PLC0415
    from kernel.selection import SelectionContext, run_selection_loop  # noqa: PLC0415

    config         = ctx.config
    regime_cfg     = config.get("regime", {})
    max_positions  = config.get("max_concurrent_positions", 3)
    sector_map     = config.get("sector_map", {})
    defensive_set  = set(config.get("defensive_tickers", ["GLD", "TLT", "XLV", "XLU"]))
    max_per_sector = int(config.get("max_positions_per_sector", 0))
    wash_days      = int(config.get("wash_sale_days", 0))
    corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.7))
    tiered_thresholds = config.get("tiered_thresholds", [])

    open_slots = max_positions - len(ctx.held)
    log.info("BUY PHASE  (slots %d/%d open)", open_slots, max_positions)

    if open_slots <= 0:
        log.info("  All %d slots filled — no scanning", max_positions)
        return

    if ctx.circuit_open:
        log.info("  Drawdown circuit breaker OPEN — halting buys")
        return

    if ctx.transition_countdown > 0:
        log.info("  Transition uncertainty window: %d bars remaining — no buys",
                 ctx.transition_countdown)
        return

    is_bear = ctx.regime == "BEAR"

    # ── BEAR branch ───────────────────────────────────────────────────────────
    override_pct = None
    if is_bear:
        defensive_held = [s for s in ctx.held if s in defensive_set]
        if defensive_held:
            log.info("  BEAR — defensive already held (%s) — no new buys", defensive_held)
            return
        open_slots  = min(open_slots, 1)
        override_pct = 0.15   # bypass reserve calc in BEAR; defensives use 15%
        log.info("  BEAR — scanning defensives only")
    elif not ctx.spy_vel_ok:
        log.info("  SPY velocity crash BLOCKING — no buys")
        return
    elif not ctx.spy_above_ema50:
        log.info("  SPY EMA50 gate BLOCKING — no buys")
        return

    if not ctx.candidates:
        log.info("  No buy candidates — no buys placed")
        return

    # Apply blend weights from ranked candidates
    ranking_cfg = config.get("ranking", {})
    bw    = ranking_cfg.get("blend_weights", [0.5, 0.5])
    total = float(bw[0]) + float(bw[1])
    w_rank = float(bw[0]) / total if total > 0 else 0.5
    w_rs   = float(bw[1]) / total if total > 0 else 0.5

    log.info("RANKED CANDIDATES (%.0f%% score + %.0f%% RS):", w_rank * 100, w_rs * 100)
    for i, c in enumerate(ctx.candidates, 1):
        log.info("  #%d  %-6s  raw=%+.4f  calibrated=%+.4f  rs=%+.4f",
                 i, c.ticker, c.raw_score, c.rank_score, c.rs_score)

    # Convert last_sell_dates strings → date objects for selection loop
    last_sell_d = {}
    for sym, d_str in ctx.last_sell_dates.items():
        try:
            last_sell_d[sym] = date.fromisoformat(d_str)
        except ValueError:
            last_sell_d[sym] = None

    sel_ctx = SelectionContext(
        today              = ctx.today,
        held_tickers       = list(ctx.held),
        last_sell_dates    = last_sell_d,
        earnings_calendar  = ctx.earnings_cal,
        corr_matrix        = ctx.corr_matrix or None,
        sector_map         = sector_map,
        defensive_set      = defensive_set,
        wash_sale_days     = wash_days,
        earnings_buffer    = int(regime_cfg.get("earnings_buffer_days", 3)),
        corr_threshold     = corr_threshold,
        max_per_sector     = max_per_sector,
        tiered_thresholds  = tiered_thresholds,
        open_slots         = open_slots,
    )

    selected, blocks = run_selection_loop(ctx.candidates, sel_ctx)
    log.info("SELECTION DONE: %d selected, blocks=%s", len(selected), blocks)

    # ── Place buy orders ───────────────────────────────────────────────────────
    max_pos_pct   = ctx.regime_params["max_position_pct"] * ctx.confidence
    cash_res_pct  = ctx.regime_params["cash_reserve_pct"] * ctx.confidence

    for slot_num, symbol in enumerate(selected, 1):
        # Duplicate-order guard
        if symbol in ctx.pending_orders:
            log.info("  %-6s  SKIP  [pending order already in Alpaca]", symbol)
            continue

        # Refresh account value for this slot
        acct_val  = ctx.broker.get_account_value()
        price     = float(ctx.ohlcv[symbol]["close"].iloc[-1])

        if override_pct is not None:
            _, shares = compute_position_size(
                acct_val, ctx.cash_avail, max_pos_pct, cash_res_pct, price,
                override_pct=override_pct,
            )
        else:
            _, shares = compute_position_size(
                acct_val, ctx.cash_avail, max_pos_pct, cash_res_pct, price,
            )

        if shares <= 0:
            log.info("  %-6s  SKIP  [insufficient cash: price=$%.2f]", symbol, price)
            continue

        invest = shares * price
        try:
            result = ctx.broker.place_order(symbol, "BUY", shares)
        except Exception as exc:
            log.error("BUY order FAILED %s — %s", symbol, exc)
            continue

        c = next((c for c in ctx.candidates if c.ticker == symbol), None)
        log.info("  %-6s  BUY  slot=%d  %d shares @ $%.2f  invest=$%.0f",
                 symbol, slot_num, shares, price, invest)

        _log_trade(ctx, {
            "timestamp":       datetime.now().isoformat(),
            "symbol":          symbol,
            "signal":          "buy",
            "slot":            slot_num,
            "raw_model_score":  c.raw_score if c else 0.0,
            "rank_model_score": c.rank_score if c else 0.0,
            "rs_score":         c.rs_score if c else 0.0,
            "shares": shares, "price": price, "invest": invest, "order": result,
        })

        ctx.entry_dates[symbol]    = ctx.today_str
        ctx.sell_streaks.pop(symbol, None)
        ctx.last_sell_dates.pop(symbol, None)
        ctx.held.append(symbol)
        ctx.cash_avail -= invest

    log.info("END  |  positions held: %d/%d  |  %s",
             len(ctx.held), max_positions, ctx.held or "none")
