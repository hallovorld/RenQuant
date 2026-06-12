"""P0.1 stream watchdog — read-only intraday risk eyes (NO order authority).

Design: renquant-orchestrator
doc/research/2026-06-12-intraday-trading-roadmap.md §4 P0.1 + P0.4 + P0.5:
subscribe trades for held names + SPY over the Alpaca websocket (IEX
feed, free tier), alert on adverse moves bigger than the polled pass
could catch (NVTS post-mortem: −12% fit between two 12-minute polls),
append every tick decision to an event log (the DRPH-intraday corpus
seed), and heartbeat so the dead-man switch (P0.5) can tell "quiet
market" from "dead process".

HARD INVARIANT: this module never constructs a TradingClient and never
submits, cancels, or modifies orders. It observes, logs, and alerts.

Acceptance gate (roadmap): 5 sessions of clean logs before anything
downstream consumes it.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("live.stream_watchdog")

DEFAULT_ALERT_DROP_PCT = 0.05      # alert when a held name falls 5% from session anchor
DEFAULT_SPY_ALERT_DROP_PCT = 0.03  # market-wide move threshold
HEARTBEAT_INTERVAL_SEC = 30
DATA_ROOT = Path.home() / "renquant-data" / "watchdog"


@dataclass
class TickState:
    anchor: float                 # session anchor price (first trade seen or prev close)
    last: float
    last_ts: float
    alerted_levels: set = field(default_factory=set)


class WatchdogCore:
    """Pure decision core — no network, fully unit-testable.

    The stream adapter feeds ``on_trade``; the core decides when a move
    crosses an alert level and what to persist. Alert levels are
    one-shot per (symbol, level) per session: a name sitting at −6%
    does not re-alert every tick (level set = {1×, 2×, 3× threshold}).
    """

    def __init__(self, *, held: set[str], alert_drop_pct: float = DEFAULT_ALERT_DROP_PCT,
                 spy_drop_pct: float = DEFAULT_SPY_ALERT_DROP_PCT,
                 data_root: Path = DATA_ROOT,
                 clock=time.time):
        self.held = set(held)
        self.alert_drop_pct = float(alert_drop_pct)
        self.spy_drop_pct = float(spy_drop_pct)
        self.state: dict[str, TickState] = {}
        self.clock = clock
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._event_log = self.data_root / f"events_{dt.date.today().isoformat()}.jsonl"
        self._heartbeat_file = self.data_root / "heartbeat"
        self._last_heartbeat = 0.0

    # ── event log (P0.4 seed: append-only, replayable) ──────────────────

    def _log_event(self, kind: str, payload: dict) -> None:
        rec = {"ts": self.clock(), "kind": kind, **payload}
        with self._event_log.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ── heartbeat (P0.5 dead-man input) ─────────────────────────────────

    def heartbeat(self) -> None:
        now = self.clock()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
            self._heartbeat_file.write_text(str(now))
            self._last_heartbeat = now

    # ── trade handling ───────────────────────────────────────────────────

    def on_trade(self, symbol: str, price: float, ts: float | None = None) -> list[dict]:
        """Process one trade; returns alert dicts raised by this tick."""
        if price <= 0:
            return []
        ts = ts if ts is not None else self.clock()
        st = self.state.get(symbol)
        if st is None:
            st = self.state[symbol] = TickState(anchor=price, last=price, last_ts=ts)
            self._log_event("anchor", {"symbol": symbol, "price": price})
            return []
        st.last, st.last_ts = price, ts
        threshold = self.spy_drop_pct if symbol == "SPY" else self.alert_drop_pct
        drop = (st.anchor - price) / st.anchor
        alerts = []
        level = int(drop / threshold)
        if level >= 1 and level not in st.alerted_levels:
            st.alerted_levels.add(level)
            alert = {
                "symbol": symbol,
                "drop_pct": round(drop * 100, 2),
                "level": level,
                "anchor": st.anchor,
                "price": price,
                "held": symbol in self.held,
            }
            alerts.append(alert)
            self._log_event("alert", alert)
        self.heartbeat()
        return alerts

    def staleness_seconds(self, symbol: str) -> float | None:
        st = self.state.get(symbol)
        return None if st is None else self.clock() - st.last_ts


def _post_alert(alert: dict) -> None:
    from live.alerts import AlertEvent, post_ntfy_alert, stable_alert_key  # noqa: PLC0415

    sym, lvl = alert["symbol"], alert["level"]
    post_ntfy_alert(AlertEvent(
        taxonomy="watchdog.intraday_drop",
        title=f"WATCHDOG {sym} −{alert['drop_pct']}%",
        body=(f"{sym} at {alert['price']} vs session anchor {alert['anchor']} "
              f"(level {lvl}; held={alert['held']}). Read-only watchdog — "
              f"check broker-side stops."),
        key=stable_alert_key("watchdog", sym, lvl, dt.date.today()),
        priority="high" if alert["held"] else "default",
    ))


def run(symbols: set[str] | None = None) -> int:
    """Daemon entry — connect the IEX stream and pump trades into the core.

    Reconnects with capped exponential backoff + jitter; every reconnect
    is logged to the event log (sequence gaps are visible in replay).
    """
    import random  # noqa: PLC0415

    from alpaca.data.live import StockDataStream  # noqa: PLC0415

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret:
        raise SystemExit("ALPACA_API_KEY/SECRET required (read-only data scope)")

    if symbols is None:
        symbols = _held_symbols() | {"SPY"}
    core = WatchdogCore(held=symbols - {"SPY"})
    log.info("watchdog starting: %d symbols (%s)", len(symbols), sorted(symbols))

    async def _on_trade(t):
        for alert in core.on_trade(str(t.symbol), float(t.price)):
            try:
                _post_alert(alert)
            except Exception as exc:  # noqa: BLE001 — alerting must not kill the watchdog
                log.warning("ntfy post failed: %s", exc)

    backoff = 1.0
    while True:
        stream = StockDataStream(api_key, secret)
        stream.subscribe_trades(_on_trade, *sorted(symbols))
        try:
            core._log_event("connect", {"symbols": sorted(symbols)})
            stream.run()   # blocks until disconnect
            backoff = 1.0
        except KeyboardInterrupt:
            core._log_event("shutdown", {})
            return 0
        except Exception as exc:  # noqa: BLE001
            core._log_event("disconnect", {"error": str(exc)[:200]})
            sleep_s = min(backoff, 60.0) * (1 + random.random())
            log.warning("stream dropped (%s) — reconnect in %.1fs", exc, sleep_s)
            time.sleep(sleep_s)
            backoff *= 2


def _held_symbols() -> set[str]:
    """Held tickers from the live state file (read-only; no broker call)."""
    state_file = (Path(__file__).resolve().parent.parent
                  / "backtesting" / "renquant_104" / "live_state.alpaca.json")
    try:
        state = json.loads(state_file.read_text())
        return set((state.get("entry_dates") or {}).keys())
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read live state (%s) — watching SPY only", exc)
        return set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(run())
