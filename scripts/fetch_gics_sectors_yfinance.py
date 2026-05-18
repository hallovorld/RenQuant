#!/usr/bin/env python3
"""Fetch GICS sector + industry classification per ticker via yfinance.

Roadmap task #4 (2026-05-18). Unblocks wl200 v2 selection (proper
sector caps replacing the 50-ticker "Other" bucket from the static
SECTOR_FALLBACK in select_wl200_quality_first.py) AND P0 #4 LightGBM
with categorical feature.

Output: data/ticker_sectors.json

  {
    "AAPL": {"sector": "Technology", "industry": "Consumer Electronics",
              "as_of": "2026-05-18"},
    ...
  }

Reference: GICS = Global Industry Classification Standard (MSCI/S&P 2000).
yfinance returns the Yahoo-classified sector/industry which matches GICS
top-level taxonomy: Technology / Health Care / Financials / Consumer
Cyclical / Consumer Defensive / Industrials / Energy / Utilities /
Communication Services / Real Estate / Basic Materials.

Rate considerations: yfinance scrapes Yahoo Finance HTML; ~1s/ticker.
No API key needed. Run sequentially, occasional 404s for delisted
tickers (skip gracefully).
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_P = REPO / "data" / "ticker_sectors.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_gics_sectors")


def _load_target_tickers() -> list[str]:
    """Union of wl200 candidates + current watchlist + every OHLCV-cached ticker
    that has alpha158 panel coverage."""
    targets: set[str] = set()
    # wl200 candidates
    wl200_p = REPO / "data" / "wl200_quality_first.json"
    if wl200_p.exists():
        targets.update(json.loads(wl200_p.read_text())["watchlist"])
    # Current wl103
    cfg = json.loads((REPO / "backtesting" / "renquant_104"
                      / "strategy_config.json").read_text())
    targets.update(cfg.get("watchlist", []))
    # Panel coverage
    import pandas as pd
    panel = pd.read_parquet(
        REPO / "data" / "alpha158_291_fundamental_dataset.parquet",
        columns=["ticker"])
    targets.update(panel["ticker"].unique())
    return sorted(targets)


def _fetch_one(symbol: str) -> dict | None:
    """yfinance.Ticker.info['sector'] + ['industry'] for one symbol."""
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info
    except Exception as exc:
        log.warning("  %s: yfinance failed — %s", symbol, exc)
        return None
    sector = info.get("sector")
    industry = info.get("industry")
    if not sector:
        # Some ETFs return None; skip
        return None
    return {
        "sector": str(sector),
        "industry": str(industry) if industry else None,
        "as_of": str(date.today()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT_P))
    p.add_argument("--symbols", nargs="*", default=None,
                   help="override target list")
    p.add_argument("--refresh", action="store_true",
                   help="overwrite existing entries instead of skipping")
    args = p.parse_args()

    out_p = Path(args.out)
    existing: dict = json.loads(out_p.read_text()) if out_p.exists() else {}
    log.info("existing cache: %d tickers", len(existing))

    targets = args.symbols if args.symbols else _load_target_tickers()
    log.info("target tickers: %d", len(targets))

    n_skip = n_fetch = n_fail = 0
    for i, sym in enumerate(targets):
        if sym in existing and not args.refresh:
            n_skip += 1
            continue
        rec = _fetch_one(sym)
        if rec is None:
            n_fail += 1
            continue
        existing[sym] = rec
        n_fetch += 1
        if (i + 1) % 25 == 0:
            log.info("  %d/%d  fetch=%d skip=%d fail=%d  latest=%s sec=%s",
                     i + 1, len(targets), n_fetch, n_skip, n_fail,
                     sym, rec["sector"])

    out_p.write_text(json.dumps(existing, indent=2, sort_keys=True))
    log.info("DONE. %d total / %d new / %d failed → %s",
             len(existing), n_fetch, n_fail, out_p)

    # Print sector breakdown
    from collections import Counter
    sec_count = Counter(r["sector"] for r in existing.values())
    log.info("Sector breakdown:")
    for s, n in sorted(sec_count.items(), key=lambda x: -x[1]):
        log.info("  %-30s %d", s, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
