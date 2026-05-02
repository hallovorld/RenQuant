#!/usr/bin/env python
"""Populate the insider-trades cache for the watchlist.

Executive-only Form 4 transactions (open-market P/S codes). Respects
SEC EDGAR's rate limit (8 req/sec self-imposed). Run time for a fresh
38-ticker full fetch: ~10-15 minutes.

Usage::

    python scripts/fetch_insider_trades.py --strategy renquant_104
    python scripts/fetch_insider_trades.py --strategy renquant_104 --no-cache
    python scripts/fetch_insider_trades.py --strategy renquant_104 --max-filings 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-insider-trades")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--no-cache", action="store_true",
                   help="Always refetch, don't reuse cached rows.")
    p.add_argument("--max-filings", type=int, default=200,
                   help="Max Form 4 filings to pull per ticker (recent first).")
    p.add_argument("--total-budget-sec", type=float, default=None,
                   help="Total fetch budget. Default uses fetch_insider_trades_watchlist's "
                        "240s (daily-refresh sized). For initial cold backfill on a 100+ "
                        "ticker watchlist, pass a larger value (e.g. 5400 = 90 min).")
    p.add_argument("--per-ticker-sec", type=float, default=None,
                   help="Per-ticker hard timeout. Default 45s.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))

    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    watchlist = config["watchlist"]

    from kernel.insider_trades import fetch_insider_trades_watchlist  # noqa: PLC0415

    kwargs: dict = {}
    if args.total_budget_sec is not None:
        kwargs["total_budget_sec"] = args.total_budget_sec
    if args.per_ticker_sec is not None:
        kwargs["per_ticker_sec"] = args.per_ticker_sec

    budget_str = (f"{args.total_budget_sec:.0f}s"
                  if args.total_budget_sec is not None else "default 240s")
    log.info("Fetching executive Form 4 insider trades for %d tickers "
             "(cache=%s, max_filings=%d, budget=%s)",
             len(watchlist), not args.no_cache, args.max_filings, budget_str)

    out = fetch_insider_trades_watchlist(
        watchlist, cache=not args.no_cache, max_filings=args.max_filings,
        **kwargs,
    )
    non_empty = sum(1 for df in out.values() if not df.empty)
    total_tx  = sum(len(df) for df in out.values())
    log.info("Done — %d/%d tickers with insider rows, %d transactions total",
             non_empty, len(watchlist), total_tx)


if __name__ == "__main__":
    main()
