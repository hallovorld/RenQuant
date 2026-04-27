#!/usr/bin/env python
"""Fetch FRED macro series and cache to data/fred/*.parquet.

Tier 2 macro expansion (2026-04-27). Per
`doc/research/macro-data-expansion-plan-2026-04-27.md`.

Setup
-----
1. Get a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Either set `RENQUANT_FRED_API_KEY=your_key` in env, or write the key
   verbatim to `~/.fred_api_key` (file with just the key on one line).
3. Run: `python scripts/fetch_fred_macro.py [--series ID1 ID2 …] [--years 10]`

Defaults to fetching all DEFAULT_FRED_SERIES (22 series) over 10y.

Output
------
- `data/fred/<SERIES_ID>.parquet` — per-series cache, atomically written
- INFO log line per series, summary at end
- Exit code 0 if any series succeeded, 1 if all failed (e.g. bad key)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-fred")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", nargs="+", default=None,
                   help="Explicit series IDs (overrides DEFAULT_FRED_SERIES).")
    p.add_argument("--years", type=int, default=10,
                   help="Years of history to fetch (default: 10).")
    p.add_argument("--cache-dir", default=None,
                   help="Override cache dir (default: data/fred under repo root).")
    p.add_argument("--retry-delay-sec", type=int, default=5,
                   help="Seconds between retries on FRED API failure.")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Max retries per series (default: 2).")
    args = p.parse_args()

    from kernel.fred_macro import (   # noqa: E402
        DEFAULT_FRED_SERIES,
        FredMacroStore,
        _resolve_api_key,
    )
    import pandas as pd                # noqa: E402

    api_key = _resolve_api_key()
    if not api_key:
        log.error(
            "FRED API key not found. Set RENQUANT_FRED_API_KEY env var "
            "or write the key to ~/.fred_api_key. Free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else (REPO_ROOT / "data" / "fred")
    store = FredMacroStore(cache_dir=cache_dir, api_key=api_key)
    log.info("FredMacroStore: cache_dir=%s", store.cache_dir)

    if args.series:
        series_ids = args.series
    else:
        series_ids = [tup[0] for tup in DEFAULT_FRED_SERIES]
    log.info("Fetching %d series, %d years history", len(series_ids), args.years)

    end_ts   = pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=args.years * 365 + 30)

    n_ok = 0
    n_fail = 0
    for series_id in series_ids:
        for attempt in range(args.max_retries + 1):
            try:
                t0 = time.monotonic()
                df = store.fetch(
                    series_id,
                    observation_start=start_ts,
                    observation_end=end_ts,
                )
                if df.empty:
                    log.warning("∅ %s: empty result", series_id)
                    n_fail += 1
                    break
                store.save(series_id, df)
                n_ok += 1
                log.info("✓ %s: %d rows fetched + saved (%.1fs)",
                         series_id, len(df), time.monotonic() - t0)
                break
            except Exception as exc:
                if attempt < args.max_retries:
                    log.warning("✗ %s attempt %d/%d failed (%s) — retrying in %ds",
                                series_id, attempt + 1, args.max_retries + 1,
                                type(exc).__name__, args.retry_delay_sec)
                    time.sleep(args.retry_delay_sec)
                else:
                    log.error("✗ %s FINAL fail: %s: %s", series_id, type(exc).__name__, exc)
                    n_fail += 1

    log.info("DONE: %d ok, %d failed", n_ok, n_fail)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
