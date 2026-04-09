#!/usr/bin/env python
"""Export LEAN-compatible daily data for ALL symbols in a strategy watchlist.

Reads cached parquet from data/ohlcv/{SYMBOL}/1d.parquet and writes:
  - backtesting/data/equity/usa/daily/{symbol}.zip   (price CSV)
  - backtesting/data/equity/usa/map_files/{symbol}.csv
  - backtesting/data/equity/usa/factor_files/{symbol}.csv

Usage::

    python scripts/export_lean_watchlist.py --strategy renquant_102
    python scripts/export_lean_watchlist.py --strategy renquant_102 --symbols CRM UNH SHOP
    python scripts/export_lean_watchlist.py --strategy renquant_102 --force   # re-export all

Requires: pandas, pyarrow (available in the renquant conda environment).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_lean_data import export_symbol

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_watchlist(strategy_name: str) -> list[str]:
    """Read watchlist + benchmark from a strategy config."""
    config_path = REPO_ROOT / "backtesting" / strategy_name / "strategy_config.json"
    if not config_path.exists():
        print(f"ERROR: strategy config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    symbols = list(config.get("watchlist", []))
    benchmark = config.get("benchmark", "SPY")
    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)
    # Also include stock_symbol for single-stock strategies
    stock_symbol = config.get("stock_symbol")
    if stock_symbol and stock_symbol not in symbols:
        symbols.append(stock_symbol)
    return symbols


def already_exported(symbol: str) -> bool:
    """Check if LEAN daily zip already exists for this symbol."""
    zip_path = REPO_ROOT / "backtesting" / "data" / "equity" / "usa" / "daily" / f"{symbol.lower()}.zip"
    return zip_path.exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LEAN data for all symbols in a strategy watchlist"
    )
    parser.add_argument("--strategy", required=True, help="Strategy directory name (e.g. renquant_102)")
    parser.add_argument("--symbols", nargs="*", help="Export only these symbols (default: full watchlist)")
    parser.add_argument("--force", action="store_true", help="Re-export even if LEAN data already exists")
    args = parser.parse_args()

    all_symbols = get_watchlist(args.strategy)
    symbols = args.symbols if args.symbols else all_symbols
    # Validate requested symbols are in watchlist
    unknown = set(symbols) - set(all_symbols)
    if unknown:
        print(f"WARNING: {unknown} not in strategy watchlist, exporting anyway")

    exported = 0
    skipped = 0
    failed = 0

    for symbol in sorted(symbols):
        if not args.force and already_exported(symbol):
            print(f"  {symbol:6s} — already exists, skipping (use --force to re-export)")
            skipped += 1
            continue

        parquet_path = REPO_ROOT / "data" / "ohlcv" / symbol.upper() / "1d.parquet"
        if not parquet_path.exists():
            print(f"  {symbol:6s} — NO parquet cache at {parquet_path}")
            print(f"           Run: python -c \"import common; common.fetch_ohlcv('{symbol}')\"")
            failed += 1
            continue

        try:
            daily_zip, map_file, factor_file = export_symbol(symbol)
            print(f"  {symbol:6s} — exported → {daily_zip.name}")
            exported += 1
        except Exception as e:
            print(f"  {symbol:6s} — FAILED: {e}")
            failed += 1

    print(f"\nDone: {exported} exported, {skipped} already existed, {failed} failed")
    print(f"Total LEAN data files: {len(list((REPO_ROOT / 'backtesting' / 'data' / 'equity' / 'usa' / 'daily').glob('*.zip')))}")

    if failed > 0:
        print("\nTo fetch missing parquet caches, run the notebook or:")
        print("  conda activate renquant")
        print("  python -c \"import common; common.fetch_ohlcv('SYMBOL')\"")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
