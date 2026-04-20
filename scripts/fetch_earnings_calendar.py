#!/usr/bin/env python3
"""Fetch earnings calendar for a watchlist and save to a strategy artifact.

Usage:
    python scripts/fetch_earnings_calendar.py --strategy renquant_103
    python scripts/fetch_earnings_calendar.py --strategy renquant_103 --lookahead 60

Output:
    backtesting/{strategy}/earnings-calendar.json
    {
      "AAPL": ["2026-01-30", "2026-04-30", ...],
      ...
    }
"""
import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent


def load_watchlist(strategy: str) -> list[str]:
    cfg_path = ROOT / "backtesting" / strategy / "strategy_config.json"
    with cfg_path.open() as f:
        return json.load(f)["watchlist"]


def fetch_earnings_dates(ticker: str, lookahead_days: int) -> list[str]:
    """Return upcoming + recent earnings dates for a ticker (best-effort)."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        dates = []

        # yfinance calendar dict may have 'Earnings Date' key
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            for d in raw:
                try:
                    ds = str(d)[:10]
                    dates.append(ds)
                except Exception:
                    pass

        # Also pull from earnings_dates for historical
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                cutoff = date.today() + timedelta(days=lookahead_days)
                for idx in ed.index:
                    try:
                        ds = str(idx)[:10]
                        d  = date.fromisoformat(ds)
                        if d >= date.today() - timedelta(days=30) and d <= cutoff:
                            dates.append(ds)
                    except Exception:
                        pass
        except Exception:
            pass

        return sorted(set(dates))
    except Exception as e:
        print(f"  WARNING: {ticker} earnings fetch failed: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Fetch earnings calendar")
    parser.add_argument("--strategy",  required=True)
    parser.add_argument("--lookahead", type=int, default=90,
                        help="Days ahead to fetch earnings (default: 90)")
    args = parser.parse_args()

    watchlist = load_watchlist(args.strategy)
    strategy_dir = ROOT / "backtesting" / args.strategy
    artifacts_dir = strategy_dir / "artifacts"
    out_dir = artifacts_dir if artifacts_dir.exists() else strategy_dir
    out_path = out_dir / "earnings-calendar.json"

    print(f"Fetching earnings calendar for {len(watchlist)} symbols → {out_path}")
    calendar = {}
    for ticker in watchlist:
        print(f"  {ticker}...", end=" ", flush=True)
        dates = fetch_earnings_dates(ticker, args.lookahead)
        calendar[ticker] = dates
        print(f"{len(dates)} dates")
        time.sleep(0.3)   # polite rate-limit

    out_path.write_text(json.dumps(calendar, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
