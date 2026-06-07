#!/usr/bin/env python
"""IGV short-plan monitor — cron entrypoint.

Wires: fetch IGV market -> igv_short_state.step() -> persist -> ntfy alert ->
(only when armed) live option order via live.options_executor.

SAFETY GATES (all must hold for a LIVE order to be placed):
  1. config "mode" == "live"
  2. env IGV_LIVE_ARMED == "1"
  3. no kill-switch:  env IGV_KILL != "1"  AND  no file live/state/IGV_KILL
Any gate off => dry-run: state advances and alerts fire, but NO order is sent.
Per CLAUDE.md §4.1 this OVERRIDES the paper-cron mandate ONLY when the operator
has explicitly armed it; default posture is no live order.

Config: live/igv_short_plan.json  (contracts is REQUIRED — fails closed if unset)
State:  live/state/igv_short_state.json
Run:    python scripts/igv_short_monitor.py [--once]   (cron calls --once)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.igv_short_state import (  # noqa: E402
    Action, Bar, Market, PlanConfig, PlanState, step,
    ENTER, CLOSE_HALF, CLOSE_MOST, CLOSE_ALL, VOID,
)
from live import options_executor as ox  # noqa: E402
from live.alerts import AlertEvent, post_ntfy_alert, stable_alert_key  # noqa: E402

log = logging.getLogger("igv-short-monitor")

CONFIG_PATH = REPO / "live" / "igv_short_plan.json"
STATE_PATH = REPO / "live" / "state" / "igv_short_state.json"
KILL_FILE = REPO / "live" / "state" / "IGV_KILL"
UNDERLYING = "IGV"


# ── config / state ──────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"missing config {CONFIG_PATH} (see doc/ops/igv-short-automation.md)")
    cfg = json.loads(CONFIG_PATH.read_text())
    if not isinstance(cfg.get("contracts"), int) or cfg["contracts"] < 1:
        raise SystemExit("config.contracts must be a positive int (fails closed — no default)")
    if not isinstance(cfg.get("max_debit"), (int, float)) or cfg["max_debit"] <= 0:
        raise SystemExit("config.max_debit (max $ per spread, e.g. 3.50) is required")
    return cfg


def plan_config(cfg: dict) -> PlanConfig:
    z = cfg.get("zones", {})
    base = PlanConfig()
    return replace(
        base,
        reject_zone=tuple(z.get("reject_zone", base.reject_zone)),
        breakdown_level=z.get("breakdown_level", base.breakdown_level),
        failed_bounce_zone=tuple(z.get("failed_bounce_zone", base.failed_bounce_zone)),
        standby_level=z.get("standby_level", base.standby_level),
        void_level=z.get("void_level", base.void_level),
        tp_half_at=z.get("tp_half_at", base.tp_half_at),
        tp_most_at=z.get("tp_most_at", base.tp_most_at),
        sl_half_at=z.get("sl_half_at", base.sl_half_at),
        sl_exit_close_at=z.get("sl_exit_close_at", base.sl_exit_close_at),
    )


def load_state() -> PlanState:
    if STATE_PATH.exists():
        return PlanState.from_dict(json.loads(STATE_PATH.read_text()))
    return PlanState()


def save_state(state: PlanState, extra: dict | None = None) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    if extra:
        payload["_position"] = extra
    STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))


# ── market data ─────────────────────────────────────────────────────────────
def get_market() -> Market:
    from alpaca.data.historical import StockHistoricalDataClient  # noqa: PLC0415
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest  # noqa: PLC0415
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: PLC0415

    key = os.environ.get("ALPACA_API_KEY", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not sec:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    dc = StockHistoricalDataClient(key, sec)

    now = datetime.now(timezone.utc)
    hb = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=UNDERLYING, timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=now - timedelta(days=5),
    )).data.get(UNDERLYING, [])
    hourly = [Bar(high=float(b.high), low=float(b.low), close=float(b.close)) for b in hb]

    db = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=UNDERLYING, timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=now - timedelta(days=7),
    )).data.get(UNDERLYING, [])
    daily_close = float(db[-1].close) if db else None

    price = float(dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=UNDERLYING))[UNDERLYING].price)
    return Market(price=price, hourly_bars=hourly, daily_close=daily_close)


def spread_net_mid(legs: "ox.SpreadLegs", paper: bool) -> float | None:
    """Net mid of (long - short) put. None if quotes unavailable -> fail closed."""
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient  # noqa: PLC0415
        from alpaca.data.requests import OptionLatestQuoteRequest  # noqa: PLC0415
        dc = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        q = dc.get_option_latest_quote(OptionLatestQuoteRequest(
            symbol_or_symbols=[legs.long_put_occ, legs.short_put_occ]))
        lq, sq = q.get(legs.long_put_occ), q.get(legs.short_put_occ)
        if not lq or not sq:
            return None
        long_mid = (float(lq.bid_price) + float(lq.ask_price)) / 2
        short_mid = (float(sq.bid_price) + float(sq.ask_price)) / 2
        return long_mid - short_mid
    except Exception as exc:  # noqa: BLE001
        log.warning("spread quote unavailable: %s", exc)
        return None


# ── alerts ──────────────────────────────────────────────────────────────────
def alert(action: Action, mkt: Market, *, armed: bool, order_note: str) -> None:
    title = f"IGV {action.kind} @ {mkt.price:.2f}"
    body = f"{action.reason}\nmode={'LIVE-ARMED' if armed else 'dry-run'} | {order_note}"
    ev = AlertEvent(
        taxonomy="igv_short_plan",
        title=title,
        body=body,
        key=stable_alert_key("igv", action.kind, action.reason[:40]),
        priority="high",
        force=True,
    )
    url = os.environ.get("RENQUANT_NTFY_URL", "")
    if url:
        post_ntfy_alert(url, ev)
    log.warning("ALERT %s | %s", title, body.replace("\n", " | "))


# ── execution ───────────────────────────────────────────────────────────────
def execute(action: Action, cfg: dict, state: PlanState, paper: bool, armed: bool) -> dict | None:
    """Place the order for an action when armed; return position dict to persist."""
    plan_id = cfg.get("plan_id", "igv-short")
    pos = (load_state().to_dict().get("_position") if STATE_PATH.exists() else None) or {}

    if action.kind == ENTER:
        legs = ox.resolve_put_spread(UNDERLYING, cfg["long_strike"], cfg["short_strike"],
                                     dte_min=cfg.get("dte_min", 0), dte_max=cfg.get("dte_max", 7),
                                     paper=paper)
        mid = spread_net_mid(legs, paper)
        debit = min(cfg["max_debit"], (mid + cfg.get("slippage", 0.10)) if mid else cfg["max_debit"])
        contracts = int(cfg["contracts"])
        if armed:
            ox.open_put_spread(legs, contracts, debit, plan_id=plan_id, paper=paper)
        state.contracts = contracts
        return {"legs": {"long": legs.long_put_occ, "short": legs.short_put_occ,
                         "expiry": legs.expiry.isoformat(),
                         "long_strike": legs.long_strike, "short_strike": legs.short_strike},
                "contracts": contracts, "entry_debit_limit": round(debit, 2)}

    # close actions
    if not pos or "legs" not in pos:
        log.warning("close action %s but no recorded position; alert only", action.kind)
        return pos or None
    legs = ox.SpreadLegs(long_put_occ=pos["legs"]["long"], short_put_occ=pos["legs"]["short"],
                         expiry=__import__("datetime").date.fromisoformat(pos["legs"]["expiry"]),
                         long_strike=pos["legs"]["long_strike"], short_strike=pos["legs"]["short_strike"])
    cur = int(pos.get("contracts", state.contracts) or 0)
    qty = cur if action.fraction and action.fraction >= 1.0 else max(1, cur // 2)
    mid = spread_net_mid(legs, paper)
    credit = max(0.05, (mid - cfg.get("slippage", 0.10)) if mid else 0.05)
    tag = {CLOSE_HALF: "half", CLOSE_MOST: "most", CLOSE_ALL: "exit"}.get(action.kind, "close")
    if armed and qty >= 1:
        ox.close_put_spread(legs, qty, credit, plan_id=plan_id, tag=f"{tag}-{cur}", paper=paper)
    pos["contracts"] = max(0, cur - qty)
    return pos


# ── main ────────────────────────────────────────────────────────────────────
def run_once() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if os.environ.get("IGV_KILL") == "1" or KILL_FILE.exists():
        log.warning("KILL-SWITCH active — exiting without action")
        return 0
    # Market-hours guard: don't act on stale overnight prices. The post-close
    # stop run (daily close >= 101.5) is done by a separate cron entry that
    # sets IGV_IGNORE_HOURS=1 shortly after the close.
    if os.environ.get("IGV_IGNORE_HOURS") != "1":
        try:
            from alpaca.trading.client import TradingClient  # noqa: PLC0415
            tc = TradingClient(os.environ.get("ALPACA_API_KEY", ""),
                               os.environ.get("ALPACA_SECRET_KEY", ""), paper=True)
            if not tc.get_clock().is_open:
                log.info("market closed — skipping (IGV_IGNORE_HOURS=1 to force)")
                return 0
        except Exception as exc:  # noqa: BLE001
            log.warning("clock check failed (%s) — proceeding", exc)
    cfg = load_config()
    mode_live = cfg.get("mode") == "live"
    armed = mode_live and os.environ.get("IGV_LIVE_ARMED") == "1"
    paper = not armed  # never live unless fully armed
    pcfg = plan_config(cfg)
    state = load_state()
    if state.state in ("VOIDED", "CLOSED"):
        log.info("plan terminal (%s) — nothing to do", state.state)
        return 0
    mkt = get_market()
    new_state, actions = step(state, mkt, pcfg)
    extra = (load_state().to_dict().get("_position") if STATE_PATH.exists() else None)
    for a in actions:
        order_note = "no order (dry-run)" if not armed else "LIVE order submitted"
        try:
            if a.kind in (ENTER, CLOSE_HALF, CLOSE_MOST, CLOSE_ALL):
                extra = execute(a, cfg, new_state, paper, armed) or extra
            elif a.kind == VOID:
                order_note = "plan voided (no position)"
        except Exception as exc:  # noqa: BLE001 — never let an order error skip the alert
            order_note = f"ORDER ERROR: {exc}"
            log.exception("order execution failed for %s", a.kind)
        alert(a, mkt, armed=armed, order_note=order_note)
    save_state(new_state, extra)
    log.info("state %s -> %s (%d actions, armed=%s)", state.state, new_state.state, len(actions), armed)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_once())
