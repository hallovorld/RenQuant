#!/usr/bin/env python3
"""Earnings-calendar staleness rail + recent-print selection (CLI).

Thin CLI over the single-source-of-truth helpers in
backtesting/renquant_104/adapters/earnings_freshness.py, so shell
wrappers (daily_104.sh preflight, refresh_earnings_calendar.sh,
daily_earnings_surprise_refresh.sh) can share the exact same verdict
the live runner logs.

Usage:
    earnings_calendar_rail.py check --calendar PATH [--min-horizon-days 5]
        exit 0 = fresh; 3 = STALE (last date < today+N); 4 = missing/unreadable
    earnings_calendar_rail.py select-recent --calendar PATH [--lookback-days 7]
        prints tickers with a print in [today-N, today], one per line;
        exit 0 (possibly empty output); 4 = missing/unreadable

Both subcommands accept --today YYYY-MM-DD for tests/replays.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.earnings_freshness import (  # noqa: E402
    DEFAULT_MIN_HORIZON_DAYS,
    assess_earnings_calendar_freshness,
    select_recent_prints,
)

EXIT_OK = 0
EXIT_STALE = 3
EXIT_MISSING = 4


def _load_calendar(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        cal = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cal if isinstance(cal, dict) else None


def _parse_today(value: str | None) -> datetime.date:
    return datetime.date.fromisoformat(value) if value else datetime.date.today()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="staleness rail verdict")
    p_check.add_argument("--calendar", required=True)
    p_check.add_argument("--min-horizon-days", type=int,
                         default=DEFAULT_MIN_HORIZON_DAYS)
    p_check.add_argument("--today", default=None)

    p_sel = sub.add_parser("select-recent",
                           help="tickers with a print in the last N days")
    p_sel.add_argument("--calendar", required=True)
    p_sel.add_argument("--lookback-days", type=int, default=7)
    p_sel.add_argument("--today", default=None)

    args = parser.parse_args(argv)
    today = _parse_today(args.today)
    calendar = _load_calendar(args.calendar)

    if args.cmd == "check":
        verdict = assess_earnings_calendar_freshness(
            calendar, today=today, min_horizon_days=args.min_horizon_days)
        print(f"[{verdict['status'].upper()}] {verdict['message']} "
              f"(calendar={args.calendar})")
        if verdict["status"] == "ok":
            return EXIT_OK
        if verdict["status"] == "stale":
            return EXIT_STALE
        return EXIT_MISSING

    # select-recent
    if calendar is None:
        print(f"[MISSING] calendar unreadable: {args.calendar}", file=sys.stderr)
        return EXIT_MISSING
    for ticker in select_recent_prints(calendar, today=today,
                                       lookback_days=args.lookback_days):
        print(ticker)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
