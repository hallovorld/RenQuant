#!/usr/bin/env python
"""Populate earnings_surprise cache for the watchlist.

Usage::

    python scripts/fetch_earnings_surprise.py --strategy renquant_104
    python scripts/fetch_earnings_surprise.py --strategy renquant_104 --no-cache
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
log = logging.getLogger("fetch-earnings-surprise")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--no-cache", action="store_true",
                   help="Always refetch, don't reuse cached rows.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))

    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    watchlist = config["watchlist"]

    from kernel.earnings_surprise import fetch_earnings_surprise_watchlist  # noqa: PLC0415

    log.info("Fetching earnings surprises for %d tickers (cache=%s)",
             len(watchlist), not args.no_cache)
    out = fetch_earnings_surprise_watchlist(watchlist, cache=not args.no_cache)
    non_empty = sum(1 for df in out.values() if not df.empty)
    log.info("Done — %d/%d tickers have surprise history", non_empty, len(watchlist))


if __name__ == "__main__":
    main()
