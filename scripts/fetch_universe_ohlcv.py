#!/usr/bin/env python
"""Fetch OHLCV for any tickers in the universe file that aren't cached.

Reads ``scripts/watchlist_universe.json``, diffs against ``data/ohlcv/``,
and pulls 5y daily bars for the missing ones via yfinance. Writes to
the standard parquet cache layout (``data/ohlcv/<TICKER>/1d.parquet``).

Production safety: read-only on existing data; only writes new
parquet files for previously-uncached tickers. Existing cached
tickers are NOT touched (use ``fetch_ohlcv_incremental`` for those).

Usage::

    python scripts/fetch_universe_ohlcv.py
    python scripts/fetch_universe_ohlcv.py --workers 8 --period 5y
    python scripts/fetch_universe_ohlcv.py --dry-run    # show what would fetch

Designed to run in background — completion writes a summary file
``data/audit/universe_fetch_<timestamp>.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fetch-universe")


def _cache_path(ticker: str, cache_root: Path) -> Path:
    return cache_root / ticker / "1d.parquet"


def _missing_tickers(universe: list[str], cache_root: Path) -> list[str]:
    """Tickers in the universe with no parquet cache file yet.

    Filters bad inputs (empty strings, sentinel '-' from the iShares CSV,
    tickers with whitespace) so we don't waste yfinance calls on garbage.
    """
    cached = {p.name for p in cache_root.iterdir() if p.is_dir()}
    out = []
    for t in universe:
        if not t or not isinstance(t, str):
            continue
        t = t.strip().upper()
        if not t or t == "-" or " " in t or "/" in t:
            continue
        # iShares uses BRK.A → BRK-A; yfinance also uses BRK-A. Already
        # mapped in build_universe.py.
        if t not in cached:
            out.append(t)
    return sorted(set(out))


def _fetch_one(ticker: str, period: str, cache_root: Path) -> tuple[str, bool, str]:
    """Fetch one ticker and write to cache. Returns (ticker, ok, note)."""
    try:
        import yfinance as yf  # noqa: PLC0415
        import pandas as pd    # noqa: PLC0415
        df = yf.download(
            ticker, period=period, auto_adjust=False,
            progress=False, threads=False,
        )
        if df is None or df.empty:
            return (ticker, False, "empty")
        # yfinance multi-level cols when single ticker — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Standardize column names lower-case
        df.columns = [str(c).lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return (ticker, False, f"missing cols: {required - set(df.columns)}")
        if len(df) < 252:
            return (ticker, False, f"history too short ({len(df)} rows)")
        out_path = _cache_path(ticker, cache_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        return (ticker, True, f"{len(df)} rows")
    except Exception as exc:
        return (ticker, False, f"error: {type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", default=str(
        REPO_ROOT / "scripts" / "watchlist_universe.json",
    ))
    p.add_argument("--cache-root", default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--period", default="5y",
                   help="yfinance period (default 5y)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel fetch workers (default 8). yfinance "
                        "throttles aggressively above ~10.")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of tickers to fetch (testing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be fetched without doing it.")
    args = p.parse_args()

    universe_path = Path(args.universe)
    cache_root    = Path(args.cache_root)
    if not universe_path.exists():
        log.error("Universe file missing: %s", universe_path)
        return 1
    if not cache_root.exists():
        log.error("Cache root missing: %s", cache_root)
        return 1

    universe = json.loads(universe_path.read_text())
    missing = _missing_tickers(universe, cache_root)
    if args.limit:
        missing = missing[: args.limit]

    log.info("Universe size: %d", len(universe))
    log.info("Already cached: %d", len(universe) - len(missing))
    log.info("To fetch: %d", len(missing))
    if missing[:10]:
        log.info("  Sample missing: %s", missing[:10])

    if args.dry_run:
        return 0
    if not missing:
        log.info("Nothing to fetch.")
        return 0

    started_at = _dt.datetime.now(_dt.timezone.utc)
    results: dict[str, dict] = {}
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_fetch_one, t, args.period, cache_root): t
            for t in missing
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            try:
                _, ok, note = fut.result()
            except Exception as exc:
                ok, note = False, f"future_exc: {exc}"
            results[ticker] = {"ok": ok, "note": note}
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(missing):
                log.info("  progress %d/%d  ok=%d fail=%d", i, len(missing), n_ok, n_fail)

    finished_at = _dt.datetime.now(_dt.timezone.utc)

    summary = {
        "started_utc":    started_at.isoformat(),
        "finished_utc":   finished_at.isoformat(),
        "wall_seconds":   (finished_at - started_at).total_seconds(),
        "universe_size":  len(universe),
        "n_attempted":    len(missing),
        "n_ok":           n_ok,
        "n_fail":         n_fail,
        "fail_examples":  {t: r for t, r in list(results.items())[:20]
                            if not r.get("ok")},
    }
    out_path = (REPO_ROOT / "data" / "audit"
                / f"universe_fetch_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Done. ok=%d fail=%d  report=%s", n_ok, n_fail, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
