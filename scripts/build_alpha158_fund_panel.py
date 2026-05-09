#!/usr/bin/env python
"""Merge alpha158 panel + 5 SEC fund + 3 PEAD features → production training panel.

Per CLAUDE.md §5.7 we need this in the cron pipeline so the production
panel-ltr.alpha158_fund.json gets retrained on fresh data daily.

Inputs:
    data/alpha158_qlib_dataset.parquet      (148 alpha158 features +
                                              fwd_5/20/60d_excess + meta)
    data/sec_fundamentals_daily.parquet     (5 fund cols: earnings_yield,
                                              book_to_price, gross_profitability,
                                              roe, asset_growth)
    data/earnings_surprise/{ticker}.parquet (per-ticker quarterly EPS
                                              actual + estimate + surprise%)

Output:
    data/alpha158_291_fundamental_dataset.parquet  (left-join + PEAD,
                                                     **166 features total**)

PEAD features (3, added 2026-05-08 after E47 paired sanity passed):
    days_since_earnings   capped at 60d (Bernard-Thomas drift window)
    pead_signal           = surprise_pct × max(0, 1 − days_since/60)
                            (linear decay over 60d)
    pead_quintile_rank    = cross-sectional rank of most-recent surprise_pct

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
PEAD_COLS = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
PEAD_DECAY_DAYS = 60   # Bernard-Thomas 1989 drift window


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

    # ── Add PEAD features (E47 promotion 2026-05-08) ──────────────────────
    log.info("Computing PEAD features (Bernard-Thomas 1989, 60d decay window)...")
    t0 = time.time()
    merged = _add_pead_features(merged)
    log.info("  PEAD features added in %.1fs", time.time()-t0)

    # Verify final shape matches expected schema (alpha158 + fund + pead)
    expected_cols = len(alpha.columns) + len(FUND_COLS) + len(PEAD_COLS)
    if len(merged.columns) != expected_cols:
        log.warning("Column count %d (expected %d) — extra cols: %s",
                    len(merged.columns), expected_cols,
                    set(merged.columns) - set(alpha.columns) - set(FUND_COLS) - set(PEAD_COLS))

    log.info("Writing → %s", out_p.name)
    merged.to_parquet(out_p, index=False)
    log.info("Done: rows=%d cols=%d tickers=%d size=%.1fMB",
             len(merged), len(merged.columns), merged["ticker"].nunique(),
             out_p.stat().st_size / 1e6)


def _add_pead_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach 3 PEAD features per (ticker, date) using earnings_surprise/."""
    earn_dir = REPO / "data" / "earnings_surprise"
    n_with_earn = 0
    out_blocks = []
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ep = earn_dir / f"{tkr}.parquet"
        if not ep.exists():
            for c in PEAD_COLS + ["pead_surprise"]:
                g[c] = np.nan
            out_blocks.append(g)
            continue
        n_with_earn += 1
        earn = pd.read_parquet(ep).reset_index()
        earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
        earn = earn.sort_values("earnings_date").reset_index(drop=True)
        # For each panel date, find most-recent prior earnings_date
        g_dates = g["date"].values
        e_dates = earn["earnings_date"].values
        e_surps = earn["surprise_pct"].values
        idxs = np.searchsorted(e_dates, g_dates, side="right") - 1
        days_since = np.full(len(g), np.nan)
        surprise   = np.full(len(g), np.nan)
        valid = idxs >= 0
        diff = (g_dates[valid] - e_dates[idxs[valid]]).astype('timedelta64[D]').astype(int)
        days_since[valid] = diff
        surprise[valid]   = e_surps[idxs[valid]]
        # Beyond 60d window → no signal
        days_since = np.where(days_since > PEAD_DECAY_DAYS, np.nan, days_since)
        surprise   = np.where(np.isnan(days_since), np.nan, surprise)
        decay = np.where(np.isnan(days_since), 0.0,
                          np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))
        signal = surprise * decay
        g["days_since_earnings"] = days_since
        g["pead_signal"]         = signal
        g["pead_surprise"]       = surprise   # used to derive quintile_rank below
        out_blocks.append(g)

    log.info("  PEAD coverage: %d/%d tickers had earnings data",
             n_with_earn, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)

    # Cross-sectional quintile rank of pead_surprise per date
    out["pead_quintile_rank"] = (
        out.groupby("date")["pead_surprise"].rank(pct=True, na_option="keep")
    )

    # Cross-sectional median imputation per date for inference; fall back to zero
    for c in PEAD_COLS + ["pead_surprise"]:
        med = out.groupby("date")[c].transform("median")
        out[c] = out[c].fillna(med).fillna(0.0)

    # Drop the helper col — only keep the 3 final PEAD features
    return out.drop(columns=["pead_surprise"])


if __name__ == "__main__":
    main()
