#!/usr/bin/env python
"""Phase 1 of Transformer data prep: inventory the OHLCV cache + classify
tickers into training tiers based on history length + liquidity.

Reads `data/ohlcv/<TICKER>/1d.parquet` for every ticker. Computes:
  - first / last date with valid close
  - number of trading days
  - mean daily $-volume (close × volume) over the available window
  - any obvious data-quality red flags (NaN gaps > 5d, monotonicity breaks)

Writes `data/transformer_universe_inventory.json` with structured tiers.

Tier-A — production-quality:  ≥10y history AND ADV ≥ $50M
Tier-B — training-quality:     ≥5y  history AND ADV ≥ $10M
Tier-C — skip                   anything else

Usage:
    python scripts/transformer_universe_inventory.py
    python scripts/transformer_universe_inventory.py --min-years-A 10 --min-adv-A 50e6

Per CLAUDE.md §5.6 — every datum that goes into a Transformer must pass
integrity checks. This script is the FIRST gate. Subsequent phases
(integrity audit, label construction) build on this output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-inventory")


def _summarize_ticker(t: str, p: Path) -> dict | None:
    """Read a ticker's parquet + return a summary dict.
    Returns None if the file is missing or unreadable."""
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["close", "volume"])
    except Exception as exc:
        log.warning("  %s: read failed — %s", t, exc)
        return None
    if df.empty or "close" not in df.columns:
        return None
    df = df.dropna(subset=["close"])
    if df.empty:
        return None
    n_rows  = int(len(df))
    first_d = df.index.min()
    last_d  = df.index.max()
    years   = float((last_d - first_d).days / 365.25)
    avg_adv = float((df["close"] * df["volume"]).mean()) if "volume" in df.columns else float("nan")
    # Quick data-quality flags
    gap_max_days = int(df.index.to_series().diff().dt.days.max() or 0)
    monotone_dates = bool(df.index.is_monotonic_increasing)
    return {
        "ticker": t,
        "first_date": first_d.date().isoformat(),
        "last_date":  last_d.date().isoformat(),
        "years":      round(years, 2),
        "n_rows":     n_rows,
        "avg_adv":    avg_adv,
        "gap_max_days": gap_max_days,
        "monotone_dates": monotone_dates,
    }


def _classify(s: dict, min_years_A: float, min_adv_A: float,
              min_years_B: float, min_adv_B: float) -> str:
    """Return "A", "B", or "C" classification for the ticker summary."""
    if s["years"] < min_years_B:
        return "C"
    if not s["monotone_dates"]:
        return "C"
    if s["gap_max_days"] > 30:   # > 1 month gap → broken
        return "C"
    adv_ok_A = (s["avg_adv"] is not None
                and not (isinstance(s["avg_adv"], float) and np.isnan(s["avg_adv"]))
                and s["avg_adv"] >= min_adv_A)
    adv_ok_B = (s["avg_adv"] is not None
                and not (isinstance(s["avg_adv"], float) and np.isnan(s["avg_adv"]))
                and s["avg_adv"] >= min_adv_B)
    if s["years"] >= min_years_A and adv_ok_A:
        return "A"
    if s["years"] >= min_years_B and adv_ok_B:
        return "B"
    return "C"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ohlcv-dir",   default=str(REPO_ROOT / "data" / "ohlcv"))
    p.add_argument("--output",      default=str(REPO_ROOT / "data" / "transformer_universe_inventory.json"))
    p.add_argument("--min-years-A", type=float, default=10.0)
    p.add_argument("--min-adv-A",   type=float, default=50e6,
                    help="Tier-A min avg daily $-volume")
    p.add_argument("--min-years-B", type=float, default=5.0)
    p.add_argument("--min-adv-B",   type=float, default=10e6)
    args = p.parse_args()

    ohlcv_dir = Path(args.ohlcv_dir)
    if not ohlcv_dir.exists():
        log.error("OHLCV dir not found: %s", ohlcv_dir)
        sys.exit(1)

    tickers = sorted([d.name for d in ohlcv_dir.iterdir()
                       if d.is_dir() and (d / "1d.parquet").exists()])
    log.info("Inventorying %d tickers from %s", len(tickers), ohlcv_dir)

    summaries: list[dict] = []
    for i, t in enumerate(tickers):
        if i % 100 == 0:
            log.info("  ... %d/%d", i, len(tickers))
        s = _summarize_ticker(t, ohlcv_dir / t / "1d.parquet")
        if s is None:
            continue
        s["tier"] = _classify(
            s,
            min_years_A=args.min_years_A, min_adv_A=args.min_adv_A,
            min_years_B=args.min_years_B, min_adv_B=args.min_adv_B,
        )
        summaries.append(s)

    by_tier = {"A": [], "B": [], "C": []}
    for s in summaries:
        by_tier[s["tier"]].append(s)

    total_rows = {tier: sum(s["n_rows"] for s in lst)
                   for tier, lst in by_tier.items()}

    summary_doc = {
        "kind":           "transformer_universe_inventory",
        "generated_utc":  pd.Timestamp.now(tz="UTC").isoformat(),
        "ohlcv_dir":      str(ohlcv_dir),
        "min_years_A":    args.min_years_A,
        "min_adv_A":      args.min_adv_A,
        "min_years_B":    args.min_years_B,
        "min_adv_B":      args.min_adv_B,
        "tier_counts": {
            "A": len(by_tier["A"]),
            "B": len(by_tier["B"]),
            "C": len(by_tier["C"]),
        },
        "panel_rows_per_tier": total_rows,
        "tier_A_tickers": [s["ticker"] for s in by_tier["A"]],
        "tier_B_tickers": [s["ticker"] for s in by_tier["B"]],
        "tier_A_summaries": by_tier["A"],
        "tier_B_summaries": by_tier["B"],
        # Tier-C kept as count only (skipped tickers don't need detail)
        "n_tier_C": len(by_tier["C"]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary_doc, f, indent=2, default=str)
    log.info("══ inventory written %s ══", out_path)
    log.info("Tier-A: %d tickers, %d panel rows", summary_doc["tier_counts"]["A"], total_rows["A"])
    log.info("Tier-B: %d tickers, %d panel rows", summary_doc["tier_counts"]["B"], total_rows["B"])
    log.info("Tier-C: %d tickers (skipped)",      summary_doc["tier_counts"]["C"])
    log.info("Tier-A+B combined: %d panel rows",  total_rows["A"] + total_rows["B"])


if __name__ == "__main__":
    main()
