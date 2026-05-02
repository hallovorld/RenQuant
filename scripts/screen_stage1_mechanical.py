#!/usr/bin/env python
"""Track D Stage 1: mechanical screening of the candidate universe.

Filters Russell 1000 (or any input ticker list) to a tractable set
that meets liquidity/age/ETF/sector requirements before running the
expensive Stage 2 (KS distributional) and Stage 3 (greedy IC-additive)
admission stages.

Usage::

    python scripts/screen_stage1_mechanical.py
    python scripts/screen_stage1_mechanical.py --universe scripts/watchlist_universe.json
    python scripts/screen_stage1_mechanical.py --adv-min 5000000 --age-min 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("screen-stage1")


# Approximate ETF + closed-end-fund tickers commonly found in Russell 1000
# constituents that should be excluded for our equity-rank model.
_KNOWN_ETF_LIKE: set[str] = {
    "GLD", "SLV", "TLT", "USO", "UUP", "FXE", "DBA",
    "XLE", "XLF", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "XLB", "XRT", "XME", "XBI",
    "QQQ", "SPY", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
    "HYG", "LQD", "AGG", "BND", "IYR", "VNQ",
    "ARKK", "ARKQ", "ARKW", "TQQQ", "SQQQ",
    "BRK.A", "BRK.B",  # holding company, separate spec
}


def _load_ticker_universe(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        tickers = [t for t in raw if isinstance(t, str) and t and t != "-"]
    elif isinstance(raw, dict) and "tickers" in raw:
        tickers = list(raw["tickers"])
    else:
        raise ValueError(f"Unrecognised universe format in {path}")
    return sorted(set(tickers))


def _load_sector_map(cfg_path: Path) -> dict[str, str]:
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("sector_map", {}) or {}


def _per_ticker_stats(ticker: str, ohlcv_root: Path) -> "dict | None":
    """Compute liquidity + age stats from cached daily OHLCV.

    Returns dict with `adv_60d_usd`, `age_years`, `last_price`,
    `n_bars`, `latest_date`, or None if no OHLCV cached.
    """
    p = ohlcv_root / ticker / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        log.warning("read_parquet(%s) failed — %s", ticker, exc)
        return None
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return None

    close = df["close"].astype(float).replace(0, np.nan).dropna()
    volume = df["volume"].astype(float).replace(0, np.nan)
    if len(close) < 60:
        return None

    # ADV: trailing 60 trading days mean of close × volume (USD)
    dollar_volume = (close * volume).dropna()
    adv_60d = float(dollar_volume.tail(60).mean()) if len(dollar_volume) >= 60 else float("nan")

    age_days = (df.index.max() - df.index.min()).days if len(df.index) > 1 else 0
    age_years = age_days / 365.25

    return {
        "adv_60d_usd":   adv_60d,
        "age_years":     age_years,
        "last_price":    float(close.iloc[-1]),
        "n_bars":        int(len(close)),
        "latest_date":   df.index.max().date().isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe",
                   default=str(REPO_ROOT / "scripts" / "watchlist_universe.json"),
                   help="Path to candidate-universe JSON list.")
    p.add_argument("--ohlcv-root",
                   default=str(REPO_ROOT / "data" / "ohlcv"),
                   help="OHLCV cache root (per-ticker subdirs with 1d.parquet).")
    p.add_argument("--out",
                   default=str(REPO_ROOT / "scripts" / "screen_stage1_results.json"),
                   help="Output JSON path with admit / reject lists.")
    p.add_argument("--sector-config",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json"),
                   help="Path to a config JSON whose `sector_map` is used for ticker→sector lookup. "
                        "Use the IWB-extended map (built via build_sector_map.py) to lift "
                        "coverage from production's ~106 to Russell-1000-wide ~1000.")
    p.add_argument("--adv-min", type=float, default=10_000_000.0,
                   help="Minimum trailing-60d ADV (USD).")
    p.add_argument("--age-min", type=float, default=3.0,
                   help="Minimum years of OHLCV history.")
    p.add_argument("--price-min", type=float, default=5.0,
                   help="Minimum latest close price.")
    p.add_argument("--require-sector", action="store_true", default=True,
                   help="Reject tickers without a sector map entry.")
    p.add_argument("--allow-missing-sector", action="store_true", default=False,
                   help="Override --require-sector: keep tickers without a sector "
                        "mapping. Use to size the unconstrained universe.")
    args = p.parse_args()

    universe = _load_ticker_universe(Path(args.universe))
    sector_map = _load_sector_map(Path(args.sector_config))
    log.info("Stage 1 inputs: %d candidate tickers, %d sector_map entries (from %s)",
             len(universe), len(sector_map), Path(args.sector_config).name)
    log.info("Thresholds: ADV>=%.0f USD, age>=%.1f yrs, price>=%.2f",
             args.adv_min, args.age_min, args.price_min)

    ohlcv_root = Path(args.ohlcv_root)
    admitted: list[dict] = []
    rejected: list[dict] = []

    for tk in universe:
        # Hard filter: ETF-like / closed-end-fund
        if tk in _KNOWN_ETF_LIKE:
            rejected.append({"ticker": tk, "reason": "etf_like"})
            continue

        stats = _per_ticker_stats(tk, ohlcv_root)
        if stats is None:
            rejected.append({"ticker": tk, "reason": "no_ohlcv"})
            continue

        # Sector map
        sector = sector_map.get(tk)
        if args.require_sector and not args.allow_missing_sector and not sector:
            rejected.append({"ticker": tk, "reason": "no_sector_mapping",
                             **stats})
            continue

        # Liquidity + age + price
        if not np.isfinite(stats["adv_60d_usd"]) or stats["adv_60d_usd"] < args.adv_min:
            rejected.append({"ticker": tk, "reason": "below_adv_floor",
                             "sector": sector, **stats})
            continue
        if stats["age_years"] < args.age_min:
            rejected.append({"ticker": tk, "reason": "below_age_floor",
                             "sector": sector, **stats})
            continue
        if stats["last_price"] < args.price_min:
            rejected.append({"ticker": tk, "reason": "below_price_floor",
                             "sector": sector, **stats})
            continue

        admitted.append({"ticker": tk, "sector": sector, **stats})

    log.info("Stage 1 result: %d admitted / %d rejected (input=%d)",
             len(admitted), len(rejected), len(universe))

    # Sector breakdown of admitted
    if admitted:
        sec_counts: dict[str, int] = {}
        for row in admitted:
            sec_counts[row["sector"]] = sec_counts.get(row["sector"], 0) + 1
        log.info("Admitted by sector:")
        for sec, cnt in sorted(sec_counts.items(), key=lambda kv: -kv[1]):
            log.info("  %-20s %d", sec, cnt)

    # Reject reason breakdown
    reason_counts: dict[str, int] = {}
    for row in rejected:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    log.info("Reject reasons:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        log.info("  %-25s %d", reason, cnt)

    out = {
        "kind": "track_d_stage1_mechanical",
        "universe_size":    len(universe),
        "admitted_count":   len(admitted),
        "rejected_count":   len(rejected),
        "thresholds": {
            "adv_min_usd": args.adv_min,
            "age_min_yrs": args.age_min,
            "price_min":   args.price_min,
        },
        "admitted": admitted,
        "rejected": rejected,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
