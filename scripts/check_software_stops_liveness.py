#!/usr/bin/env python
"""Software-stop registry liveness watchdog — S-FRAC stage 3 (sprint D2).

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
§3.4 (alert-to-recovery SLA). The sell-only loop
(scripts/intraday_sell_104.sh, 12-minute launchd cadence) stamps
``last_evaluated_at`` into the registry file on every pass. This checker
alarms when armed software stops exist but the loop has NOT evaluated
them within the staleness budget during a market session — the
"machine dead, market open" row of the design's failure-mode table
(§3.3), which is the one genuine regression of a loop-resident stop vs
a broker-resident GTC stop.

Ops file only — nothing installs it. Run ad hoc or from any existing
scheduler:

    python scripts/check_software_stops_liveness.py --broker alpaca

Exit codes (nagios-ish, consumable by any wrapper):
    0  OK        — no registry / no armed stops / heartbeat fresh /
                   market closed (nothing can be evaluated off-session)
    1  STALE     — armed stops exist and the heartbeat is missing or
                   older than max_staleness_minutes during a session
    2  CORRUPT   — the registry file exists but cannot be read/validated:
                   registered stops are UNKNOWABLE and new fractional
                   entries are already fail-closed by the stage-0 gate

Optional ``--ntfy-topic`` posts the alarm to ntfy.sh (same channel the
live runner uses) so a cron wrapper needs no extra logic.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.software_stops import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    _validate_snapshot,
    compute_staleness,
    registry_path_for,
)

OK, STALE, CORRUPT = 0, 1, 2


def market_session_open(now: datetime.datetime) -> bool:
    """True when the NYSE regular session is plausibly open.

    Uses pandas_market_calendars when available (the live entry points
    already depend on it); otherwise falls back to weekday 09:30-16:00
    America/New_York (fail-open toward CHECKING: a holiday false-positive
    produces a spurious page, never a missed one).
    """
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415
        now_et = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        now_et = now
    try:
        import pandas as pd  # noqa: PLC0415
        import pandas_market_calendars as mcal  # noqa: PLC0415
        sched = mcal.get_calendar("NYSE").schedule(
            now_et.strftime("%Y-%m-%d"), now_et.strftime("%Y-%m-%d"),
        )
        if len(sched) == 0:
            return False
        open_ts = pd.Timestamp(sched.iloc[0]["market_open"])
        close_ts = pd.Timestamp(sched.iloc[0]["market_close"])
        now_ts = pd.Timestamp(now.astimezone(datetime.timezone.utc))
        return bool(open_ts <= now_ts <= close_ts)
    except Exception:
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return (9 * 60 + 30) <= minutes <= (16 * 60)


def check(
    registry_path: Path,
    *,
    now: "datetime.datetime | None" = None,
    force_session: bool = False,
) -> tuple[int, str]:
    """Pure check body (unit-tested): returns (exit_code, message)."""
    now_dt = now or datetime.datetime.now().astimezone()
    if not registry_path.exists():
        return OK, (
            f"OK: no software-stop registry at {registry_path} — the layer "
            "has never armed a stop (flag off or no fractional positions)."
        )
    try:
        snapshot = _validate_snapshot(json.loads(registry_path.read_text()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CORRUPT, (
            f"CORRUPT: software-stop registry {registry_path} unreadable "
            f"({type(exc).__name__}: {exc}). Registered stops are "
            "UNKNOWABLE; new fractional entries are fail-closed by the "
            "stage-0 capability gate. OPERATOR ACTION REQUIRED: inspect / "
            "quarantine the file, verify positions are protected or flat."
        )
    state = compute_staleness(snapshot, now=now_dt)
    n = state["n_stops"]
    age = state["age_minutes"]
    age_str = f"{age:.1f}m" if age is not None else "never"
    if n == 0:
        return OK, (
            f"OK: registry {registry_path} has 0 armed stops "
            f"(heartbeat age: {age_str}) — nothing unprotected."
        )
    if not force_session and not market_session_open(now_dt):
        return OK, (
            f"OK: market session closed — {n} armed stop(s) cannot be "
            f"evaluated off-session by design (heartbeat age: {age_str}). "
            "Overnight gap risk parity is the design's §3.3 analysis."
        )
    if state["stale"]:
        return STALE, (
            f"STALE: {n} ARMED software stop(s) in {registry_path} but the "
            f"sell-only loop has not evaluated the registry for {age_str} "
            f"(budget: {state['max_staleness_minutes']:.0f}m) during a "
            "market session. Positions are UNPROTECTED until the loop "
            "returns — restart the intraday loop or manually flatten/"
            "hedge (design §3.4 SLA: respond within 60 minutes)."
        )
    return OK, (
        f"OK: {n} armed stop(s), heartbeat {age_str} old "
        f"(budget {state['max_staleness_minutes']:.0f}m)."
    )


def _post_ntfy(topic: str, message: str) -> None:
    try:
        from urllib import request  # noqa: PLC0415
        req = request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={"Title": "RenQuant SOFTWARE-STOP watchdog"},
        )
        request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 — alerting is best-effort
        print(f"(ntfy post failed: {exc})", file=sys.stderr)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--registry", default=None,
        help=f"registry file path (default: {DEFAULT_REGISTRY_PATH} "
             "under the repo root, broker-tagged)",
    )
    ap.add_argument(
        "--broker", default="alpaca",
        help="broker tag for the registry filename (default: alpaca)",
    )
    ap.add_argument(
        "--now", default=None,
        help="ISO timestamp override for the current time (tests)",
    )
    ap.add_argument(
        "--force-session", action="store_true",
        help="skip the market-session check (treat as in-session)",
    )
    ap.add_argument(
        "--ntfy-topic", default=None,
        help="post STALE/CORRUPT alarms to this ntfy.sh topic",
    )
    args = ap.parse_args(argv)

    if args.registry:
        path = Path(args.registry)
    else:
        path = registry_path_for(REPO_ROOT / DEFAULT_REGISTRY_PATH, args.broker)
    now = (
        datetime.datetime.fromisoformat(args.now) if args.now else None
    )
    code, message = check(path, now=now, force_session=args.force_session)
    print(message)
    if code != OK and args.ntfy_topic:
        _post_ntfy(args.ntfy_topic, message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
