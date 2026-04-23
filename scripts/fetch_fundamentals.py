#!/usr/bin/env python
"""Populate data/fundamentals/{SYMBOL}.parquet for a strategy's watchlist.

Reads the strategy's watchlist from `strategy_config.json` and fetches
one snapshot per ticker via OpenBB. Subsequent runs append new rows —
old rows are kept so the panel can access historical fundamentals once
the cache has accumulated enough snapshots.

Usage::

    python scripts/fetch_fundamentals.py
    python scripts/fetch_fundamentals.py --strategy renquant_104
    python scripts/fetch_fundamentals.py --no-cache   # force refetch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-fundamentals")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--no-cache", action="store_true",
                   help="Always refetch, don't reuse cached rows.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    cfg = json.loads((strategy_dir / "strategy_config.json").read_text())

    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.fundamentals import fetch_fundamentals_watchlist  # noqa: PLC0415

    watchlist = list(cfg.get("watchlist", []))
    if not watchlist:
        log.error("No watchlist in %s — nothing to do", strategy_dir / "strategy_config.json")
        sys.exit(1)

    log.info("Fetching fundamentals for %d tickers (cache=%s)",
             len(watchlist), not args.no_cache)
    out = fetch_fundamentals_watchlist(watchlist, cache=not args.no_cache)

    filled = sum(
        1 for f in out.values() if f and any(v == v for v in f.values())  # any non-NaN
    )
    log.info("Done — %d tickers with at least one factor column", filled)


if __name__ == "__main__":
    main()
