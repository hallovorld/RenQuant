#!/usr/bin/env python
"""Populate data/intraday/{SYMBOL}/1h.parquet from Alpaca IEX (Plan G step 3).

Reads a strategy's watchlist from `strategy_config.json` and fetches
hourly bars in 90-day chunks (Alpaca rejects windows too far in the past
for the IEX free tier; 90d is a safe default). Cached rows are kept —
reruns only add missing sessions unless `--refetch` is passed.

Usage::

    python scripts/fetch_hourly_bars.py                      # 2yr lookback
    python scripts/fetch_hourly_bars.py --lookback-days 1825 # ~5 years
    python scripts/fetch_hourly_bars.py --symbols AAPL NVDA  # subset
    python scripts/fetch_hourly_bars.py --refetch            # ignore cache
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
log = logging.getLogger("fetch-hourly-bars")


def _load_env() -> None:
    """Populate os.environ from .env at repo root (if present)."""
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
                   help="Total historical window in days (default 730 = 2yr)")
    p.add_argument("--chunk-days", type=int, default=90,
                   help="Per-request window size (default 90d)")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Subset of tickers (default: whole watchlist)")
    p.add_argument("--refetch", action="store_true",
                   help="Refetch all windows — still dedups by timestamp")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Symbols per Alpaca request (default 10)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, skip network + writes")
    args = p.parse_args()

    _load_env()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    cfg = json.loads((strategy_dir / "strategy_config.json").read_text())

    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.data     import fetch_intraday_bars  # noqa: PLC0415
    from kernel.intraday import HourlyBarStore        # noqa: PLC0415

    watchlist = list(args.symbols or cfg.get("watchlist", []))
    # Include sector ETFs so RS factors see the same hourly data
    watchlist += list((cfg.get("sector_etf_map") or {}).values())
    benchmark = cfg.get("benchmark", "SPY")
    if benchmark not in watchlist:
        watchlist.append(benchmark)
    # Dedup while preserving order
    seen: set[str] = set()
    symbols = [s for s in watchlist if not (s in seen or seen.add(s))]

    if not symbols:
        log.error("No watchlist in %s — nothing to do",
                  strategy_dir / "strategy_config.json")
        sys.exit(1)

    cache_dir = REPO_ROOT / "data" / "intraday"
    store = HourlyBarStore(data_dir=cache_dir)

    end   = dt.datetime.utcnow()
    start = end - dt.timedelta(days=args.lookback_days)
    chunks: list[tuple[dt.datetime, dt.datetime]] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=args.chunk_days), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end

    log.info("Fetching hourly bars: %d symbols × %d chunks (%s → %s)",
             len(symbols), len(chunks), start.date(), end.date())
    if args.dry_run:
        log.info("[dry-run] symbols=%s", symbols)
        return

    total_rows = 0
    batches = [symbols[i:i+args.batch_size]
               for i in range(0, len(symbols), args.batch_size)]
    for bi, batch in enumerate(batches, 1):
        log.info("batch %d/%d — %s", bi, len(batches), batch)
        for ci, (c_start, c_end) in enumerate(chunks, 1):
            try:
                results = fetch_intraday_bars(
                    batch, timeframe="1Hour",
                    start=c_start, end=c_end,
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
            # Alpaca free tier: stay conservative on request rate.
            time.sleep(0.25)

    log.info("Done — wrote %d total rows across %d symbols", total_rows, len(symbols))


if __name__ == "__main__":
    main()
