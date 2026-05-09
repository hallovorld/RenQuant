#!/usr/bin/env python
"""Merge alpha158 panel + 5 SEC fund features → production training panel.

This script reproduces what was previously done in an ad-hoc REPL session
(commit 569a9b1 introduced the artifact, no committed builder existed).
Per CLAUDE.md §5.7 we need this in the cron pipeline so the production
panel-ltr.alpha158_fund.json gets retrained on fresh data daily.

Inputs:
    data/alpha158_qlib_dataset.parquet      (148 alpha158 features +
                                              fwd_5/20/60d_excess + meta)
    data/sec_fundamentals_daily.parquet     (5 fund cols: earnings_yield,
                                              book_to_price, gross_profitability,
                                              roe, asset_growth)

Output:
    data/alpha158_291_fundamental_dataset.parquet  (left-join of the two,
                                                     163 features total)

Usage:
    python scripts/build_alpha158_fund_panel.py
"""
from __future__ import annotations
import logging, time
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build-alpha158-fund-panel")

REPO = Path(__file__).resolve().parent.parent
FUND_COLS = ["earnings_yield", "book_to_price", "gross_profitability",
             "roe", "asset_growth"]


def main():
    alpha_p = REPO / "data" / "alpha158_qlib_dataset.parquet"
    fund_p  = REPO / "data" / "sec_fundamentals_daily.parquet"
    out_p   = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"

    log.info("Loading alpha158 panel: %s", alpha_p.name)
    alpha = pd.read_parquet(alpha_p)
    alpha["date"] = pd.to_datetime(alpha["date"])
    log.info("  rows=%d tickers=%d cols=%d  dates %s..%s",
             len(alpha), alpha["ticker"].nunique(), len(alpha.columns),
             alpha["date"].min().date(), alpha["date"].max().date())

    log.info("Loading fund daily panel: %s", fund_p.name)
    fund = pd.read_parquet(fund_p)
    fund["date"] = pd.to_datetime(fund["date"])
    keep = ["ticker", "date"] + [c for c in FUND_COLS if c in fund.columns]
    fund = fund[keep]
    log.info("  rows=%d tickers=%d cols=%s",
             len(fund), fund["ticker"].nunique(),
             [c for c in keep if c not in ("ticker","date")])

    # Left-join — every alpha158 row keeps its labels, fund cols may be NaN
    # where the ticker has no SEC coverage on that date.
    log.info("Merging on (ticker, date)...")
    t0 = time.time()
    merged = alpha.merge(fund, on=["ticker", "date"], how="left")
    log.info("  merged %d rows in %.1fs", len(merged), time.time()-t0)

    # Sanity check: row count must match alpha158 base (left-join invariant)
    if len(merged) != len(alpha):
        raise RuntimeError(
            f"Merge changed row count: {len(alpha)} → {len(merged)}. "
            f"Likely duplicate (ticker,date) pairs in fund panel."
        )

    # Fill missing fund values with cross-sectional median per date.
    # Fund features are slow-changing → median impute is reasonable for
    # tickers without SEC data (foreign listings, recent IPOs).
    log.info("Cross-sectional median imputation for fund cols...")
    for c in FUND_COLS:
        if c not in merged.columns:
            log.warning("  fund col missing: %s — filling with 0", c)
            merged[c] = 0.0
            continue
        nan_pct_pre = merged[c].isna().mean() * 100
        # Per-date median (exclude NaNs from the median calc itself)
        med = merged.groupby("date")[c].transform("median")
        merged[c] = merged[c].fillna(med)
        # Final fallback for dates where ALL tickers had NaN: zero
        nan_pct_post = merged[c].isna().mean() * 100
        merged[c] = merged[c].fillna(0.0)
        log.info("  %-25s NaN%% before=%.1f%%  after_median=%.1f%%  final_zero_filled",
                 c, nan_pct_pre, nan_pct_post)

    # Verify final shape matches expected schema
    expected_cols = len(alpha.columns) + len(FUND_COLS)
    if len(merged.columns) != expected_cols:
        log.warning("Column count %d (expected %d) — extra cols: %s",
                    len(merged.columns), expected_cols,
                    set(merged.columns) - set(alpha.columns) - set(FUND_COLS))

    log.info("Writing → %s", out_p.name)
    merged.to_parquet(out_p, index=False)
    log.info("Done: rows=%d cols=%d tickers=%d size=%.1fMB",
             len(merged), len(merged.columns), merged["ticker"].nunique(),
             out_p.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
