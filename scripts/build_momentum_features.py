#!/usr/bin/env python3
"""Build momentum features (Jegadeesh-Titman 12-1 + 52w-high distance + sector
momentum) and merge into the alpha158+fund panel.

Per 2026-05-18 model-regime-mismatch finding: current model is mean-reversion
oriented (alpha158 features), losing to tech rally. Add explicit momentum
features so XGB / PatchTST can pivot to trend-following.

Features added:
  • mom_12_1     — Jegadeesh-Titman 12m-minus-1m return (skip-1 momentum, classic)
  • mom_3m       — 3-month return (medium-term)
  • dist_52w_high — (close / 52w_high) - 1, ∈ [-1, 0]; near 0 = momentum, near -1 = laggard
  • sector_mom_30d — sector-relative 30-day return (using sector_map)
  • abs_vol_30d  — 30-day realized vol (un-normalized; helps trend confidence)

References:
  - Jegadeesh-Titman 1993 JF "Returns to buying winners and selling losers"
  - Asness-Moskowitz-Pedersen 2013 JF "Value and Momentum Everywhere"
  - Moskowitz-Grinblatt 1999 JF "Do Industries Explain Momentum?"
  - George-Hwang 2004 JF "The 52-Week High and Momentum Investing"

Output: data/alpha158_291_fundamental_dataset_mom.parquet (panel + 5 cols)
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
log = logging.getLogger("momentum-features")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _compute_per_ticker_momentum(g: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum features for a single ticker's price history."""
    g = g.sort_values("date").reset_index(drop=True).copy()
    # Need raw price for momentum — alpha158 cols are normalized. Pull from OHLCV.
    return g  # placeholder; actual computation below


def build_momentum(panel: pd.DataFrame, ohlcv_dir: Path,
                    sector_map: dict[str, str]) -> pd.DataFrame:
    """Compute and attach momentum features to panel.

    Strategy: load raw OHLCV per ticker → compute momentum features at each
    panel date → left-join back to panel.
    """
    momentum_rows = []
    skipped = 0
    for tkr, _ in panel.groupby("ticker"):
        p = ohlcv_dir / tkr / "1d.parquet"
        if not p.exists():
            skipped += 1
            continue
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = df["close"]
        # mom_12_1: return from 12 months ago to 1 month ago (skip-1 momentum)
        # Daily approximation: 252 → 21 days (12m → 1m). r_12_1 = close[t-21]/close[t-252] - 1
        mom_12_1 = (close.shift(21) / close.shift(252)) - 1.0
        # mom_3m: 3-month return = close[t] / close[t-63] - 1
        mom_3m = (close / close.shift(63)) - 1.0
        # 52w high distance: (close[t] / max(close, 252 window)) - 1, ∈ [-1, 0]
        rolling_52w_high = close.rolling(252, min_periods=60).max()
        dist_52w_high = (close / rolling_52w_high) - 1.0
        # abs_vol_30d: 30-day realized log-vol (raw, NOT normalized)
        log_ret = np.log(close / close.shift(1))
        abs_vol_30d = log_ret.rolling(30, min_periods=15).std() * np.sqrt(252)
        # Build per-date rows
        mom_df = pd.DataFrame({
            "ticker": tkr,
            "date": df.index,
            "mom_12_1": mom_12_1.values,
            "mom_3m":   mom_3m.values,
            "dist_52w_high": dist_52w_high.values,
            "abs_vol_30d":   abs_vol_30d.values,
        })
        momentum_rows.append(mom_df)

    log.info("  skipped %d tickers (no OHLCV)", skipped)
    mom_panel = pd.concat(momentum_rows, ignore_index=True)
    mom_panel["date"] = pd.to_datetime(mom_panel["date"])

    # Now sector_mom_30d: cross-sectional within each sector
    log.info("Computing sector_mom_30d (cross-sectional within sector per date)...")
    mom_panel["sector"] = mom_panel["ticker"].map(sector_map).fillna("Other")
    # 30-day return per ticker
    # Already computed: mom_3m is close[t]/close[t-63]-1; we need a 30-day version
    # Reuse from per-ticker close — actually let me compute properly below
    sector_30d = []
    for tkr, g in mom_panel.groupby("ticker"):
        p = ohlcv_dir / tkr / "1d.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = df["close"]
        ret_30d = (close / close.shift(30)) - 1.0
        sector_30d.append(pd.DataFrame({
            "ticker": tkr,
            "date": df.index,
            "ret_30d": ret_30d.values,
        }))
    ret_30d_panel = pd.concat(sector_30d, ignore_index=True)
    ret_30d_panel["date"] = pd.to_datetime(ret_30d_panel["date"])
    mom_panel = mom_panel.merge(ret_30d_panel, on=["ticker", "date"], how="left")

    # For each (date, sector), compute mean 30d return; then ticker's deviation
    sector_means = mom_panel.groupby(["date", "sector"])["ret_30d"].transform("mean")
    mom_panel["sector_mom_30d"] = mom_panel["ret_30d"] - sector_means
    mom_panel = mom_panel.drop(columns=["sector", "ret_30d"])

    # Drop NaNs (early dates without enough history)
    n_before = len(mom_panel)
    mom_panel = mom_panel.dropna(subset=["mom_12_1", "mom_3m", "dist_52w_high",
                                          "abs_vol_30d", "sector_mom_30d"])
    log.info("  dropped %d rows with NaN momentum (insufficient history)",
             n_before - len(mom_panel))

    return mom_panel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    ap.add_argument("--ohlcv-dir", default="data/ohlcv")
    ap.add_argument("--sector-map", default="data/ticker_sectors.json")
    ap.add_argument("--out", default="data/alpha158_291_fundamental_dataset_mom.parquet")
    args = ap.parse_args()

    log.info("Loading panel: %s", args.panel)
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("  rows=%d  cols=%d  tickers=%d", len(panel), len(panel.columns),
             panel["ticker"].nunique())

    log.info("Loading sector map: %s", args.sector_map)
    sec_data = json.load(open(args.sector_map))
    sector_map = {t: r["sector"] for t, r in sec_data.items()}
    log.info("  sectors: %d tickers mapped", len(sector_map))

    log.info("Building momentum features...")
    mom = build_momentum(panel, Path(args.ohlcv_dir), sector_map)
    log.info("  momentum panel: %d rows × %d cols", len(mom), len(mom.columns))

    log.info("Merging into source panel...")
    merged = panel.merge(mom, on=["ticker", "date"], how="left")
    # Fillna: pre-history rows get NaN → fill with cross-sectional median per date
    mom_cols = ["mom_12_1", "mom_3m", "dist_52w_high", "abs_vol_30d", "sector_mom_30d"]
    for c in mom_cols:
        nan_pct = merged[c].isna().mean() * 100
        cs_med = merged.groupby("date")[c].transform("median")
        merged[c] = merged[c].fillna(cs_med).fillna(0.0)
        log.info("  %s: NaN pre=%.1f%%  post-fill: 0.0%%", c, nan_pct)

    log.info("Saving → %s", args.out)
    merged.to_parquet(args.out, index=False)
    sz = Path(args.out).stat().st_size / 1e6
    log.info("Done: rows=%d cols=%d size=%.1f MB", len(merged), len(merged.columns), sz)

    # Quick sanity: cross-sectional std of momentum features (should be non-trivial)
    print("\n=== Momentum feature x-section sanity ===")
    sample_date = merged["date"].max()
    sub = merged[merged["date"] == sample_date]
    for c in mom_cols:
        v = sub[c]
        print(f"  {c:20s}: mean={v.mean():+.4f} std={v.std():.4f} min={v.min():+.4f} max={v.max():+.4f}")


if __name__ == "__main__":
    main()
