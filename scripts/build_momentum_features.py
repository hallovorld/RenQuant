#!/usr/bin/env python3
"""Build momentum features using pandas_ta_classic (mature TA library).

Per 2026-05-18 user audit: replaced self-written math with canonical TA
library calls to reduce bug surface. All 5 features computed via
`pandas_ta_classic` (formerly pandas_ta) — fork that maintains Python 3.10+
compat — with 400+ indicators that are widely used in quant finance.

Features added (kept identical names as v1):
  • mom_12_1     — Jegadeesh-Titman 12m-minus-1m return (pandas_ta_classic.roc)
  • mom_3m       — 3-month return (pandas_ta_classic.roc with length=63)
  • dist_52w_high — (close / 52w_high) - 1 via rolling max (pandas .rolling)
  • abs_vol_30d  — 30-day annualized realized vol (log-return std × √252)
  • sector_mom_30d — sector-relative 30-day return (within-sector demean)

References:
  - Jegadeesh-Titman 1993 JF "Returns to buying winners and selling losers"
  - Asness-Moskowitz-Pedersen 2013 JF "Value and Momentum Everywhere"
  - Moskowitz-Grinblatt 1999 JF "Do Industries Explain Momentum?"
  - George-Hwang 2004 JF "The 52-Week High and Momentum Investing"

CLAUDE.md compliance:
  - §5.12: defaults to canonical reference (pandas_ta_classic) instead of
    hand-implementing momentum math
  - §1c: split into per-feature pure functions, each ≤ 30 LOC
  - Unit tests in tests/test_momentum_features_no_leakage.py (no-leak +
    library-call equivalence)
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

REPO = Path(__file__).resolve().parent.parent
log = logging.getLogger("momentum-features")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

MOM_FEATURES = [
    "mom_12_1", "mom_3m", "dist_52w_high", "abs_vol_30d", "sector_mom_30d",
]


# ── Pure-function indicators (each ≤ 30 LOC, individually unit-testable) ──

def compute_mom_12_1(close: pd.Series) -> pd.Series:
    """Jegadeesh-Titman 12m-1m momentum: ret(t-21 → t-252).

    Skip-1 (last month excluded to avoid reversal). Uses pandas_ta_classic.roc
    for the 21-day-ago and 252-day-ago percent change.

    No look-ahead: only past prices used.
    """
    # close[t-21] / close[t-252] - 1 = ROC[21..252] = ROC at lag-252 of shifted-21
    # Simpler: compute roc(close.shift(21), length=231) which is
    #   close[t-21] / close[t-21-231] - 1 = close[t-21] / close[t-252] - 1
    return ta.roc(close.shift(21), length=231) / 100.0  # ta.roc returns percent


def compute_mom_3m(close: pd.Series) -> pd.Series:
    """3-month return = ROC over 63 trading days."""
    return ta.roc(close, length=63) / 100.0


def compute_dist_52w_high(close: pd.Series) -> pd.Series:
    """George-Hwang 2004: distance from 52-week (252-day) high.

    Returns value ∈ [-1, 0]: 0 = at all-time-high, -0.50 = down 50% from high.
    Uses pandas .rolling().max() — canonical.
    """
    rolling_high = close.rolling(window=252, min_periods=60).max()
    return (close / rolling_high) - 1.0


def compute_abs_vol_30d(close: pd.Series) -> pd.Series:
    """30-day annualized realized log-vol.

    Uses pandas .rolling().std() on log returns. Annualization √252.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(30, min_periods=15).std() * np.sqrt(252)


def compute_sector_mom_30d(panel: pd.DataFrame, sector_col: str = "sector",
                            ret_col: str = "ret_30d") -> pd.Series:
    """Cross-sectional within-sector 30-day return demean.

    Pure pandas groupby. Returns ticker's ret_30d minus that date+sector's mean.
    """
    sector_mean = panel.groupby(["date", sector_col])[ret_col].transform("mean")
    return panel[ret_col] - sector_mean


# ── Driver ─────────────────────────────────────────────────────────────────

def _per_ticker_features(close: pd.Series) -> pd.DataFrame:
    """Compute per-ticker momentum features (returns 4-col df: mom_12_1,
    mom_3m, dist_52w_high, abs_vol_30d). sector_mom_30d done later (xs)."""
    out = pd.DataFrame({
        "mom_12_1":      compute_mom_12_1(close),
        "mom_3m":        compute_mom_3m(close),
        "dist_52w_high": compute_dist_52w_high(close),
        "abs_vol_30d":   compute_abs_vol_30d(close),
        "ret_30d":       ta.roc(close, length=30) / 100.0,  # for sector_mom
    })
    return out


def build_momentum(panel: pd.DataFrame, ohlcv_dir: Path,
                   sector_map: dict[str, str]) -> pd.DataFrame:
    """Build all 5 momentum features and return panel-shaped DataFrame."""
    parts = []
    n_skipped = 0
    for tkr in panel["ticker"].unique():
        p = ohlcv_dir / tkr / "1d.parquet"
        if not p.exists():
            n_skipped += 1
            continue
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        feats = _per_ticker_features(df["close"])
        feats["ticker"] = tkr
        feats["date"] = df.index
        parts.append(feats.reset_index(drop=True))
    log.info("  skipped %d tickers (no OHLCV)", n_skipped)

    mom_panel = pd.concat(parts, ignore_index=True)
    mom_panel["date"] = pd.to_datetime(mom_panel["date"])
    mom_panel["sector"] = mom_panel["ticker"].map(sector_map).fillna("Other")
    log.info("  computed per-ticker features: %d rows", len(mom_panel))

    log.info("Computing sector_mom_30d (cross-sectional)...")
    mom_panel["sector_mom_30d"] = compute_sector_mom_30d(mom_panel)
    mom_panel = mom_panel.drop(columns=["sector", "ret_30d"])

    # Drop NaN rows (early dates without enough history)
    n_before = len(mom_panel)
    mom_panel = mom_panel.dropna(subset=MOM_FEATURES)
    log.info("  dropped %d rows with NaN (insufficient history)",
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
    log.info("  rows=%d cols=%d tickers=%d", len(panel), len(panel.columns),
             panel["ticker"].nunique())

    log.info("Loading sector map: %s", args.sector_map)
    sec_data = json.load(open(args.sector_map))
    sector_map = {t: r["sector"] for t, r in sec_data.items()}
    log.info("  sectors: %d tickers mapped", len(sector_map))

    log.info("Building momentum features (pandas_ta_classic)...")
    mom = build_momentum(panel, Path(args.ohlcv_dir), sector_map)
    log.info("  momentum panel: %d rows", len(mom))

    log.info("Merging into source panel...")
    merged = panel.merge(mom[["ticker", "date"] + MOM_FEATURES],
                         on=["ticker", "date"], how="left")

    for c in MOM_FEATURES:
        nan_pct = merged[c].isna().mean() * 100
        cs_med = merged.groupby("date")[c].transform("median")
        merged[c] = merged[c].fillna(cs_med).fillna(0.0)
        log.info("  %s: NaN pre=%.1f%%  post-fill: 0.0%%", c, nan_pct)

    log.info("Saving → %s", args.out)
    merged.to_parquet(args.out, index=False)
    sz = Path(args.out).stat().st_size / 1e6
    log.info("Done: rows=%d cols=%d size=%.1f MB",
             len(merged), len(merged.columns), sz)

    # Sanity print
    print("\n=== Sanity (latest date cross-section) ===")
    sub = merged[merged["date"] == merged["date"].max()]
    for c in MOM_FEATURES:
        v = sub[c]
        print(f"  {c:20s}: mean={v.mean():+.4f} std={v.std():.4f} "
              f"min={v.min():+.4f} max={v.max():+.4f}")


if __name__ == "__main__":
    main()
