#!/usr/bin/env python3
"""P0.5 dead-man switch — stale watchdog heartbeat ⇒ TRADING_OFF.

Design: renquant-orchestrator
doc/research/2026-06-12-intraday-trading-roadmap.md §4 P0.5, reusing the
G2 mechanism (live/agent_breaker.py TRADING_OFF_FLAG: presence disables
ALL order submission; deleting the file is the operator's re-enable act).

Semantics, deliberately narrow:
  * The watchdog heartbeat (P0.1, #323) is PROCESS liveness — a timer
    thread beats every 30s regardless of trade cadence or stream state,
    so a quiet market or a half-day early close can NOT trip this switch;
    only a dead/hung watchdog process can.
  * We trip only during regular trading hours (NYSE calendar via
    kernel.exits._is_nyse_trading_day + America/New_York wall clock —
    DST-proof by construction). Outside RTH a stopped watchdog is
    legitimate, and a flag would only ambush the next session.
  * We NEVER delete TRADING_OFF — not even when the heartbeat recovers.
    Re-enabling trading is the operator's explicit act (G2 contract).
    On recovery we alert that the flag is still present.

Run from launchd every 5 minutes (template in deploy/).
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

log = logging.getLogger("deadman")

HEARTBEAT_FILE = Path.home() / "renquant-data" / "watchdog" / "heartbeat"
TRADING_OFF = REPO / "TRADING_OFF"
STALE_AFTER_SEC = 180
NY = ZoneInfo("America/New_York")
DEADMAN_MARKER = "deadman_check"


def in_rth(now_utc: dt.datetime) -> bool:
    """Regular trading hours: NYSE trading day, 09:30–16:00 New York wall
    clock. Early closes (13:00) are intentionally NOT special-cased: the
    heartbeat is process-liveness, so an early close cannot fake a dead
    process — at worst we keep checking a healthy daemon for three extra
    hours."""
    ny = now_utc.astimezone(NY)
    from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415

    if not _is_nyse_trading_day(ny.date()):
        return False
    minutes = ny.hour * 60 + ny.minute
    return (9 * 60 + 30) <= minutes < 16 * 60


def heartbeat_age(now_ts: float, hb_file: Path = HEARTBEAT_FILE) -> float | None:
    """Seconds since the last beat; None when the file is missing/garbled
    (treated as stale by the caller — absence of evidence of life IS the
    dead-man condition)."""
    try:
        return now_ts - float(hb_file.read_text().strip())
    except (OSError, ValueError):
        return None


def decide(*, age: float | None, rth: bool, flag_exists: bool,
           stale_after: float = STALE_AFTER_SEC) -> str:
    """Pure decision: 'trip' | 'remind' | 'ok' | 'skip'."""
    if not rth:
        return "skip"
    stale = age is None or age > stale_after
    if stale:
        return "ok_flag_present" if flag_exists else "trip"
    return "remind" if flag_exists else "ok"


def _alert(title: str, body: str, key_parts: tuple, priority: str = "urgent") -> None:
    try:
        from live.alerts import AlertEvent, post_ntfy_alert, stable_alert_key  # noqa: PLC0415

        post_ntfy_alert(AlertEvent(
            taxonomy="watchdog.deadman",
            title=title, body=body,
            key=stable_alert_key("deadman", *key_parts),
            priority=priority,
        ))
    except Exception as exc:  # noqa: BLE001 — alert failure must not stop the trip
        log.warning("ntfy failed: %s", exc)


def main() -> int:
    now = time.time()
    age = heartbeat_age(now)
    rth = in_rth(dt.datetime.now(dt.timezone.utc))
    verdict = decide(age=age, rth=rth, flag_exists=TRADING_OFF.exists())
    age_str = "missing" if age is None else f"{age:.0f}s"
    log.info("deadman: age=%s rth=%s verdict=%s", age_str, rth, verdict)

    if verdict == "trip":
        TRADING_OFF.write_text(
            f"{DEADMAN_MARKER}: watchdog heartbeat {age_str} stale at "
            f"{dt.datetime.now(dt.timezone.utc).isoformat()} during RTH — "
            f"all order submission disabled (G2). Investigate the watchdog "
            f"(launchctl list | grep stream-watchdog; "
            f"~/renquant-data/watchdog/stderr.log), then DELETE this file "
            f"to re-enable trading.\n")
        _alert("DEADMAN TRIPPED — TRADING_OFF created",
               f"Watchdog heartbeat {age_str} stale during RTH. All order "
               f"submission disabled until you delete TRADING_OFF.",
               ("trip", dt.date.today()))
        return 1
    if verdict == "remind":
        _alert("TRADING_OFF still present (heartbeat healthy)",
               "The watchdog recovered but TRADING_OFF remains — trading "
               "stays disabled until you delete the file (operator act by "
               "design).",
               ("remind", dt.date.today()), priority="high")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
