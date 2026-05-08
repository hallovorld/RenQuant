#!/usr/bin/env python
"""Bulk-fetch daily OHLCV for Russell 2000 tickers from yfinance.

Saves each ticker to data/ohlcv/{TICKER}/1d.parquet matching existing format.
Skips tickers already on disk. Uses threadpool for parallelism.

Usage:
    python scripts/fetch_russell2000_ohlcv.py
    python scripts/fetch_russell2000_ohlcv.py --tickers BE,CRDO,STRL  # specific
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("r2k-fetch")

REPO = Path(__file__).resolve().parent.parent
OHLCV_DIR = REPO / "data" / "ohlcv"


def fetch_ticker(ticker: str, start: str = "2014-01-01") -> tuple[str, int, str]:
    """Fetch one ticker, save to data/ohlcv/{ticker}/1d.parquet. Returns (ticker, n_bars, status)."""
    out_dir  = OHLCV_DIR / ticker
    out_path = out_dir / "1d.parquet"
    if out_path.exists():
        return ticker, 0, "exists"

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False,
                         threads=False)
        if df is None or df.empty:
            return ticker, 0, "empty"
        # yf returns multi-index columns; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0).str.lower()
        else:
            df.columns = df.columns.str.lower()
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})
        # De-duplicate columns (some yfinance responses have duplicate adj_close)
        df = df.loc[:, ~df.columns.duplicated()]
        # Need at least open/high/low/close/volume
        required = {"open","high","low","close","volume"}
        if not required.issubset(set(df.columns)):
            return ticker, 0, f"missing_cols:{sorted(set(df.columns))}"
        # Save
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        return ticker, len(df), "ok"
    except Exception as e:
        return ticker, 0, f"error:{str(e)[:60]}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tickers-file", default="/tmp/russell2000_tickers.json")
    p.add_argument("--tickers", help="Comma-separated tickers (override file)")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help=">0 to limit for testing")
    args = p.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = json.loads(Path(args.tickers_file).read_text())

    # Filter to alpha-only, not already cached
    tickers = [t for t in tickers if t and t.isalpha()]
    todo = [t for t in tickers
            if not (OHLCV_DIR / t / "1d.parquet").exists()]
    if args.limit:
        todo = todo[:args.limit]
    log.info("Total: %d  already cached: %d  to fetch: %d",
             len(tickers), len(tickers) - len(todo), len(todo))
    if not todo:
        log.info("Nothing to fetch.")
        return

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0; n_fail = 0; n_empty = 0
    statuses: dict[str, list[str]] = {"ok": [], "empty": [], "error": []}

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(fetch_ticker, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, n_bars, status = fut.result()
            if status == "ok":
                n_ok += 1
                statuses["ok"].append(ticker)
            elif status == "empty":
                n_empty += 1
                statuses["empty"].append(ticker)
            else:
                n_fail += 1
                statuses["error"].append(f"{ticker}:{status}")
            if i % 50 == 0:
                log.info("Progress: %d/%d  ok=%d fail=%d empty=%d",
                         i, len(todo), n_ok, n_fail, n_empty)

    log.info("Done. ok=%d fail=%d empty=%d", n_ok, n_fail, n_empty)
    if n_fail < 20:
        log.info("Errors: %s", statuses["error"])
    else:
        log.info("First 20 errors: %s", statuses["error"][:20])

    # Save summary
    out = REPO / "data" / "russell2000_fetch_summary.json"
    out.write_text(json.dumps({
        "n_total": len(tickers), "n_ok": n_ok, "n_empty": n_empty, "n_fail": n_fail,
        "ok_tickers": statuses["ok"],
        "empty_tickers": statuses["empty"],
        "errors": statuses["error"],
    }, indent=2))
    log.info("Summary: %s", out)


if __name__ == "__main__":
    main()
