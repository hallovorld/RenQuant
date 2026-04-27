#!/usr/bin/env python
"""Fetch OHLCV for macro symbols (VIX/HYG/UUP/...) → MacroFactorStore.

Phase 1E (2026-04-26) of macro_factor_frame implementation.
See doc/components/macro-factor-frame-design.md.

Usage:
    python scripts/fetch_macro_factors.py
    python scripts/fetch_macro_factors.py --symbols VXX HYG UUP
    python scripts/fetch_macro_factors.py --period 5y
    python scripts/fetch_macro_factors.py --strategy renquant_104

Defaults to fetching all 11 symbols in DEFAULT_MACRO_SYMBOLS over 10y
of daily history to data/macro/{SYMBOL}.parquet.

Idempotent — safe to re-run. MacroFactorStore.save() dedupes on date
index, so re-running just appends new bars.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-macro")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104",
                   help="Strategy whose config to read for symbol list (default: renquant_104). "
                        "Reads panel_ltr.macro.symbols if --symbols not given.")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Explicit symbols to fetch (overrides config).")
    p.add_argument("--period", default="10y",
                   help="yfinance period string (default: 10y).")
    p.add_argument("--cache-dir", default=None,
                   help="Override cache dir (default: data/macro under repo root).")
    p.add_argument("--retry-delay-sec", type=int, default=5,
                   help="Seconds between retries on yfinance failure (default: 5).")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries per symbol (default: 3).")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy

    sys.path.insert(0, str(strategy_dir))
    from kernel.macro import (   # noqa: PLC0415
        MacroFactorStore,
        DEFAULT_MACRO_SYMBOLS,
    )

    # Resolve symbol list: CLI > config > default
    symbols = args.symbols
    if symbols is None:
        cfg_path = strategy_dir / "strategy_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            symbols = (cfg.get("panel_ltr", {})
                          .get("macro", {})
                          .get("symbols"))
    if symbols is None:
        symbols = list(DEFAULT_MACRO_SYMBOLS)

    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "data" / "macro"
    store = MacroFactorStore(data_dir=cache_dir)

    log.info("fetch_macro_factors: %d symbols → %s", len(symbols), cache_dir)

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        return 1

    n_ok = n_fail = 0
    for sym in symbols:
        for attempt in range(args.max_retries):
            try:
                t0 = time.time()
                ticker = yf.Ticker(sym)
                df = ticker.history(period=args.period, interval="1d", auto_adjust=False)
                if df is None or df.empty:
                    raise ValueError(f"yfinance returned empty for {sym}")
                # Normalize column names to lowercase
                df.columns = [c.lower() for c in df.columns]
                # Keep just OHLCV (drop dividends / stock splits if present)
                keep = [c for c in ["open", "high", "low", "close", "volume"]
                        if c in df.columns]
                if not keep:
                    raise ValueError(f"yfinance result missing OHLCV cols for {sym}: cols={list(df.columns)}")
                # Strip timezone if present (parquet doesn't roundtrip tz nicely)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                store.save(df[keep], sym)
                elapsed = time.time() - t0
                log.info("✓ %s: %d rows fetched + saved (%.1fs)",
                         sym, len(df), elapsed)
                n_ok += 1
                break
            except Exception as exc:
                log.warning("attempt %d/%d %s failed: %s",
                            attempt + 1, args.max_retries, sym, exc)
                if attempt < args.max_retries - 1:
                    time.sleep(args.retry_delay_sec)
        else:
            log.error("✗ %s: all retries exhausted — skipping", sym)
            n_fail += 1

    log.info("DONE: %d ok, %d failed", n_ok, n_fail)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
