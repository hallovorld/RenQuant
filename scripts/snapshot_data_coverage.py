#!/usr/bin/env python
"""Snapshot data coverage to data/coverage/{date}.json + console summary.

L3 of the systematic data-coverage plan (2026-05-04). Run daily via cron:

    python scripts/snapshot_data_coverage.py --strategy renquant_104

Snapshots can be diffed across days to detect:
  * coverage drops (data deletion / schema break)
  * stale OHLCV (cron failed to refresh)
  * new ticker added without intraday backfill

The script is read-only — never mutates cache files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("snapshot-coverage")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-config-name", default="strategy_config.json")
    p.add_argument(
        "--out-dir", default=str(REPO_ROOT / "data" / "coverage"),
        help="Snapshot output directory.",
    )
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))

    cfg_path = strategy_dir / args.strategy_config_name
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 2
    cfg = json.loads(cfg_path.read_text())
    watchlist = list(cfg.get("watchlist", []))

    from kernel.data_coverage import compute_coverage, coverage_summary

    log.info("Computing coverage for %d tickers (%s)…",
             len(watchlist), args.strategy_config_name)
    coverage = compute_coverage(watchlist, REPO_ROOT)
    summary = coverage_summary(coverage)

    today = _dt.date.today().isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}.json"
    payload = {
        "date":          today,
        "strategy":      args.strategy,
        "config_name":   args.strategy_config_name,
        "n_tickers":     len(watchlist),
        "summary":       summary,
        "per_ticker":    {t: c.to_dict() for t, c in coverage.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print()
    print("=" * 64)
    print(f"  DATA COVERAGE SNAPSHOT — {today}")
    print(f"  strategy {args.strategy} ({args.strategy_config_name})")
    print(f"  watchlist size = {len(watchlist)}")
    print("=" * 64)
    fmt = "{:<25}{:>10}{:>10}"
    print(fmt.format("source", "n_have", "pct"))
    print("-" * 45)
    for src_label, n_key, pct_key in [
        ("OHLCV daily",        "ohlcv_daily_n",       "ohlcv_daily_pct"),
        ("Hourly bars",        "hourly_n",            "hourly_pct"),
        ("Minute (10m) bars",  "minute_n",            "minute_pct"),
        ("Fundamentals",       "fundamentals_n",      "fundamentals_pct"),
        ("Earnings surprise",  "earnings_surprise_n", "earnings_surprise_pct"),
        ("Insider trades",     "insider_n",           "insider_pct"),
    ]:
        print(fmt.format(src_label, summary[n_key],
                         f"{summary[pct_key]:.1%}"))

    # Surface gaps loud
    gap_threshold = 0.50
    gaps = [k for k in
            ("ohlcv_daily_pct", "hourly_pct", "minute_pct",
             "fundamentals_pct", "earnings_surprise_pct", "insider_pct")
            if summary[k] < gap_threshold]
    if gaps:
        print()
        print(f"⚠️  Coverage < {gap_threshold:.0%} on: {gaps}")
        print("   Backfill or document why this gap is acceptable.")

    print()
    print(f"Snapshot saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
