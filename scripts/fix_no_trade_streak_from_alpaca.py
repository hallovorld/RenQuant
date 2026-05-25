#!/usr/bin/env python
"""Recompute live_state.alpaca.json::monitor_state.no_trade_streak from the
LIVE Alpaca order book (source of truth).

Use case: stateful counter has drifted (2026-05-20 saw counter=32 while LIVE
had fills on 16 of last 25 trading days). Patches the live_state file in
place with the broker-derived value plus provenance fields.

Idempotent. Atomic write (.tmp + rename). Backup created before overwrite.

Usage:
  python scripts/fix_no_trade_streak_from_alpaca.py
  python scripts/fix_no_trade_streak_from_alpaca.py --apply

Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from environment (source .env first).
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-file",
                   default="backtesting/renquant_104/live_state.alpaca.json")
    p.add_argument("--lookback-days", type=int, default=120,
                   help="How far back to query broker fills (calendar days)")
    p.add_argument("--apply", action="store_true",
                   help="Actually write live_state. Default is dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print proposed change without writing. This is the default.")
    return p


def _should_write(args: argparse.Namespace) -> bool:
    return bool(args.apply and not args.dry_run)


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    if args.apply and args.dry_run:
        p.error("--apply and --dry-run conflict")

    if "ALPACA_API_KEY" not in os.environ or "ALPACA_SECRET_KEY" not in os.environ:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not in env. "
              "Run: source .env  first.", file=sys.stderr)
        return 2

    state_path = REPO / args.state_file
    if not state_path.exists():
        print(f"ERROR: state file not found: {state_path}", file=sys.stderr)
        return 2

    state = json.loads(state_path.read_text())
    mon = state.get("monitor_state", {}) or {}
    counter_streak = int(mon.get("no_trade_streak", 0))
    print(f"current state file: {state_path}")
    print(f"  no_trade_streak (counter) = {counter_streak}")
    print(f"  last_activity_date        = {mon.get('last_activity_date')}")
    print(f"  first_trade_date          = {mon.get('first_trade_date')}")

    # Query LIVE Alpaca
    from alpaca.trading.client import TradingClient  # noqa: PLC0415
    from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415
    from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
    from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415

    client = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=False,
    )
    today = dt.date.today()
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.lookback_days)
    print(f"\nQuerying LIVE Alpaca for fills since {since.date().isoformat()} ...")

    all_orders = []
    until_cursor: "dt.datetime | None" = None
    for _page in range(10):
        params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500, direction="desc", after=since,
        )
        if until_cursor is not None:
            params.until = until_cursor
        page = client.get_orders(filter=params)
        if not page:
            break
        all_orders.extend(page)
        if len(page) < 500:
            break
        try:
            oldest = min((o.submitted_at for o in page if o.submitted_at), default=None)
        except Exception:
            oldest = None
        if oldest is None:
            break
        until_cursor = oldest

    fill_dates: set[dt.date] = set()
    for o in all_orders:
        status = str(o.status).lower()
        if "filled" not in status:
            continue
        when = o.filled_at or o.submitted_at
        if when:
            fill_dates.add(when.date())

    if not fill_dates:
        broker_streak = args.lookback_days
        most_recent: "dt.date | None" = None
    else:
        most_recent = max(fill_dates)
        broker_streak = 0
        d = most_recent + dt.timedelta(days=1)
        while d <= today:
            if _is_nyse_trading_day(d):
                broker_streak += 1
            d += dt.timedelta(days=1)

    print(f"  fills found in window: {len(fill_dates)} distinct fill dates")
    print(f"  most_recent_fill_date: {most_recent.isoformat() if most_recent else 'NONE'}")
    print(f"  broker_streak (trading days since last fill): {broker_streak}")
    print()
    if broker_streak == counter_streak:
        print("✓ No divergence — counter already matches broker. Nothing to do.")
        return 0

    print(f"⚠ DIVERGENCE: counter={counter_streak}  vs  broker={broker_streak}")
    print("  Proposed write:")
    print(f"    monitor_state.no_trade_streak       : {counter_streak} → {broker_streak}")
    print(f"    monitor_state.no_trade_streak_source: <unset> → 'broker_filled_orders'")
    print(f"    monitor_state.last_fill_date        : <unset> → "
          f"{most_recent.isoformat() if most_recent else 'null'}")
    print(f"    monitor_state.last_activity_date    : {mon.get('last_activity_date')} → "
          f"{most_recent.isoformat() if most_recent else mon.get('last_activity_date')}")

    if not _should_write(args):
        print("\n[dry-run] no write performed; pass --apply to update state")
        return 0

    # Backup + atomic write
    backup = state_path.with_suffix(state_path.suffix + ".bak_no_trade_streak_repair")
    shutil.copy2(state_path, backup)
    print(f"\nBackup saved: {backup}")

    mon["no_trade_streak"] = broker_streak
    mon["no_trade_streak_source"] = "broker_filled_orders"
    mon["last_fill_date"] = most_recent.isoformat() if most_recent else None
    if most_recent is not None:
        mon["last_activity_date"] = most_recent.isoformat()
    state["monitor_state"] = mon

    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)
    print(f"Wrote: {state_path}")
    print(f"✓ no_trade_streak now: {broker_streak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
