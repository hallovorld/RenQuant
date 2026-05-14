#!/usr/bin/env python
"""Fetch shortable + easy_to_borrow flags from Alpaca live API for the production watchlist.

Writes ``data/alpaca_borrow_status.json``. Consumed by:
- ``kernel.pipeline.task_short_candidates`` — filters out non-shortable
  tickers from the short candidate pool.
- ``adapters.sim.SimAdapter._charge_daily_borrow`` — picks ETB vs HTB
  borrow rate per ticker.

Schedule: weekly (Saturday 06:00 PT) — Alpaca's shortable list shifts
slowly. Live-runner is fail-open: a stale file is fine.

Usage:
    python scripts/fetch_alpaca_borrow_status.py
    python scripts/fetch_alpaca_borrow_status.py --watchlist-config strategy_config.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_watchlist(config_name: str) -> list[str]:
    p = REPO_ROOT / "backtesting" / "renquant_104" / config_name
    cfg = json.loads(p.read_text())
    wl = cfg.get("watchlist")
    if not wl:
        raise ValueError(f"{config_name} has no 'watchlist' field")
    return wl


def fetch_status(tickers: list[str]) -> dict:
    from alpaca.trading.client import TradingClient
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY must be set")
    # Live API has the asset metadata for both live & paper accounts
    client = TradingClient(key, secret, paper=False)
    results = {}
    n_fail = 0
    for t in tickers:
        try:
            a = client.get_asset(t)
            results[t] = {
                "shortable": bool(a.shortable),
                "easy_to_borrow": bool(a.easy_to_borrow),
                "marginable": bool(a.marginable),
                "tradable": bool(a.tradable),
            }
        except Exception as exc:
            results[t] = {"error": str(exc)[:120]}
            n_fail += 1
        time.sleep(0.02)  # polite rate-limit
    return results, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--watchlist-config",
        default="strategy_config.sim_baseline_ext.json",
        help="Side config in backtesting/renquant_104/ whose 'watchlist' to fetch.",
    )
    ap.add_argument(
        "--out",
        default="data/alpaca_borrow_status.json",
        help="Output JSON path (overwrites).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("fetch-alpaca-borrow")

    wl = load_watchlist(args.watchlist_config)
    log.info("Fetching shortable+ETB for %d tickers (source: %s)",
             len(wl), args.watchlist_config)
    results, n_fail = fetch_status(wl)
    n_short = sum(1 for v in results.values() if v.get("shortable"))
    n_etb = sum(1 for v in results.values() if v.get("easy_to_borrow"))
    log.info("Result: %d/%d shortable, %d/%d ETB, %d failures",
             n_short, len(wl), n_etb, len(wl), n_fail)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fetched_at": datetime.datetime.now().isoformat(),
        "source": "alpaca live /v2/assets",
        "watchlist_config": args.watchlist_config,
        "n_tickers": len(wl),
        "n_shortable": n_short,
        "n_easy_to_borrow": n_etb,
        "n_failures": n_fail,
        "note": (
            "easy_to_borrow=True → ~0%-0.5%/yr borrow cost at Alpaca retail. "
            "HTB (shortable & !easy_to_borrow) → 1-10%/yr. "
            "Sim defaults: borrow_rate_etb=0.005, borrow_rate_htb=0.05."
        ),
        "results": results,
    }, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
