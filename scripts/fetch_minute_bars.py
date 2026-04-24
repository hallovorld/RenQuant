#!/usr/bin/env python
"""Populate data/intraday/{SYMBOL}/10min.parquet from Alpaca IEX.

Mirrors fetch_hourly_bars.py but at 10-minute resolution. Adds ~6×
the row count per session (39 ten-minute bars vs 7 hourly bars) → the
panel training set grows proportionally, which is the unlock for
revisiting the transformer backend (shelved at 47k rows; target >
200k).

Usage::

    python scripts/fetch_minute_bars.py                      # 2yr lookback
    python scripts/fetch_minute_bars.py --lookback-days 365  # 1 year
    python scripts/fetch_minute_bars.py --symbols AAPL NVDA
    python scripts/fetch_minute_bars.py --refetch
    python scripts/fetch_minute_bars.py --dry-run

Alpaca IEX free tier handles 10-min bars via the same StockBarsRequest;
only difference vs the hourly script is timeframe="10Min" and smaller
chunk windows (IEX returns fewer total rows per request at 10-min).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-minute-bars")


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--lookback-days", type=int, default=730,
                   help="Total historical window (default 730 = 2yr)")
    p.add_argument("--chunk-days", type=int, default=30,
                   help="Per-request window (default 30d; 10-min bars are "
                        "~6x denser than hourly so smaller chunks keep "
                        "responses under Alpaca's page limit)")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--refetch", action="store_true")
    p.add_argument("--batch-size", type=int, default=5,
                   help="Symbols per Alpaca request (default 5 — smaller "
                        "than hourly because each symbol returns more rows)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    _load_env()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    cfg = json.loads((strategy_dir / "strategy_config.json").read_text())

    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.data     import fetch_intraday_bars  # noqa: PLC0415
    from kernel.intraday import MinuteBarStore        # noqa: PLC0415

    watchlist = list(args.symbols or cfg.get("watchlist", []))
    watchlist += list((cfg.get("sector_etf_map") or {}).values())
    benchmark = cfg.get("benchmark", "SPY")
    if benchmark not in watchlist:
        watchlist.append(benchmark)
    seen: set[str] = set()
    symbols = [s for s in watchlist if not (s in seen or seen.add(s))]

    if not symbols:
        log.error("No watchlist in %s", strategy_dir / "strategy_config.json")
        sys.exit(1)

    cache_dir = REPO_ROOT / "data" / "intraday"
    store = MinuteBarStore(data_dir=cache_dir)

    end   = dt.datetime.utcnow()
    start = end - dt.timedelta(days=args.lookback_days)
    chunks: list[tuple[dt.datetime, dt.datetime]] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=args.chunk_days), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end

    log.info("Fetching 10-min bars: %d symbols × %d chunks (%s → %s)",
             len(symbols), len(chunks), start.date(), end.date())
    if args.dry_run:
        log.info("[dry-run] symbols=%s", symbols)
        return

    # Skip tickers already fully covered — Alpaca requests are the slow
    # part, cheap I/O check first.
    skip_covered: set[str] = set()
    if not args.refetch:
        for sym in symbols:
            existing = store.load(sym)
            if existing is not None and not existing.empty:
                last = existing.index.max()
                if last.tz is not None:
                    last = last.tz_convert("UTC").tz_localize(None)
                # Within 2 trading days of target end — considered covered
                if (end.replace(tzinfo=None) - last).days <= 2:
                    skip_covered.add(sym)
        if skip_covered:
            log.info("skipping fully-cached symbols: %s",
                      sorted(skip_covered))

    fetch_list = [s for s in symbols if s not in skip_covered]
    total_rows = 0
    batches = [fetch_list[i:i + args.batch_size]
               for i in range(0, len(fetch_list), args.batch_size)]

    for bi, batch in enumerate(batches, 1):
        log.info("batch %d/%d — %s", bi, len(batches), batch)
        for ci, (c_start, c_end) in enumerate(chunks, 1):
            try:
                results = fetch_intraday_bars(
                    batch, timeframe="10Min",
                    start=c_start, end=c_end,
                    limit=10_000,
                )
            except Exception as exc:
                log.warning("  chunk %d/%d [%s→%s] failed: %s",
                            ci, len(chunks), c_start.date(), c_end.date(), exc)
                continue

            for sym, df in results.items():
                if df is None or df.empty:
                    continue
                if not args.refetch:
                    existing = store.load(sym)
                    if existing is not None and not existing.empty:
                        df = df[~df.index.isin(existing.index)]
                        if df.empty:
                            continue
                store.save(df, sym)
                total_rows += len(df)
            time.sleep(0.3)   # conservative on Alpaca free tier

    log.info("Done — wrote %d total rows across %d symbols",
             total_rows, len(fetch_list))


if __name__ == "__main__":
    main()
