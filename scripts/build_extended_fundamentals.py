#!/usr/bin/env python
"""Compute extended fundamental features beyond the 5-feature baseline.

Source: existing data/sec_fundamentals_daily.parquet which has the raw
quarterly values (NetIncomeLoss, GrossProfit, Revenues, Assets,
StockholdersEquity, CommonStockSharesOutstanding) forward-filled.

Plus we have the 5 derived ratios (earnings_yield, book_to_price,
gross_profitability, roe, asset_growth) already in the panel.

This script adds 8 NEW derived features:
  asset_turnover     = Revenue / Assets
  profit_margin      = NetIncome / Revenue
  return_on_assets   = NetIncome / Assets
  debt_to_assets     = (Assets - Equity) / Assets
  rev_growth_yoy     = Revenue.pct_change(periods=4) per quarter
  ni_growth_yoy      = NetIncome.pct_change(periods=4)
  equity_growth      = Equity.pct_change(periods=4)
  size_log_mktcap    = log(market_cap)

But: the existing daily fundamental file only has the 5 derived ratios
(no raw NetIncome etc). So we re-derive from scratch using the SEC fetch
script's intermediate output, OR re-pull SEC frames.

Quick path: re-run SEC frames API for needed concepts and compute extended.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
import numpy as np, pandas as pd, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ext-fund")

REPO = Path(__file__).resolve().parent.parent
SEC_HEADERS = {"User-Agent": "RenQuant renhao.overflow@gmail.com"}
FRAMES_BASE  = "https://data.sec.gov/api/xbrl/frames"
SLEEP = 0.12


def fetch_frame(concept, taxonomy, unit, period, retries=3):
    url = f"{FRAMES_BASE}/{taxonomy}/{concept}/{unit}/{period}.json"
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 404: return None
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data: return None
            df = pd.DataFrame(data)
            df["concept"] = concept
            df["period"]  = period
            return df
        except requests.exceptions.Timeout:
            if attempt < retries: time.sleep(5*attempt)
            else: return None
        except Exception:
            return None


def build_quarterly_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot SEC frames to quarterly rows with actual filing availability."""
    raw = raw.copy()
    raw["end"] = pd.to_datetime(raw["end"])
    if "filed" in raw.columns:
        raw["filed"] = pd.to_datetime(raw["filed"], errors="coerce")

    pivot_rows = []
    for (ticker, end_date), grp in raw.groupby(["ticker", "end"]):
        row = {"ticker": ticker, "end": end_date}
        filed_dates = (
            pd.to_datetime(grp.get("filed"), errors="coerce").dropna()
            if "filed" in grp.columns else pd.Series(dtype="datetime64[ns]")
        )
        for _, r in grp.iterrows():
            row[r["concept"]] = r["val"]
        row["available_date"] = (
            filed_dates.max()
            if not filed_dates.empty
            else end_date + pd.Timedelta(days=45)
        )
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows).sort_values(["ticker", "end"]).reset_index(drop=True)


def main():
    # Need raw quarterly data, not the daily-aggregated derivatives.
    # Re-fetch the frames we already have to keep code self-contained:
    log.info("Re-fetching raw SEC quarterly data needed for extended features...")
    # We need NetIncomeLoss, GrossProfit, Revenues, Assets, StockholdersEquity,
    # CommonStockSharesOutstanding (already cached in sec_fundamentals_daily but
    # we lost the per-quarter granularity in forward-fill).
    # Fastest: re-run fetch_sec_fundamentals concept by concept and join raw quarterly.

    # Quick alternative: derive ratios directly from the daily aggregated panel
    # using sequential changes per ticker.
    log.info("Loading daily fund panel...")
    fund = pd.read_parquet(REPO / "data" / "sec_fundamentals_daily.parquet")
    fund["date"] = pd.to_datetime(fund["date"])

    # The daily fund panel has: earnings_yield, book_to_price, gross_profitability,
    # roe, asset_growth (already derived ratios — values change at quarter boundaries).
    # We can't derive asset_turnover etc directly from these because we lost
    # raw NetIncome/Revenue. So we need to re-fetch raw values OR use what we have.

    # Honest approach: re-fetch raw concepts and compute extended ratios.
    # This will take ~10 min for 7 concepts × 60 quarters = 420 API calls.
    log.info("Re-fetching raw quarterly concepts for extended features...")

    CONCEPTS = [
        ("NetIncomeLoss",                  "us-gaap", "USD",    "duration"),
        ("GrossProfit",                    "us-gaap", "USD",    "duration"),
        ("Revenues",                       "us-gaap", "USD",    "duration"),
        ("Assets",                         "us-gaap", "USD",    "instant"),
        ("StockholdersEquity",             "us-gaap", "USD",    "instant"),
        ("CommonStockSharesOutstanding",   "us-gaap", "shares", "instant"),
        ("Liabilities",                    "us-gaap", "USD",    "instant"),
    ]

    all_rows = []
    n_done = 0
    n_total = len(CONCEPTS) * (2026 - 2010 + 1) * 4
    for concept, taxonomy, unit, period_type in CONCEPTS:
        suffix = "I" if period_type == "instant" else ""
        for year in range(2010, 2027):
            for q in range(1, 5):
                period = f"CY{year}Q{q}{suffix}"
                df = fetch_frame(concept, taxonomy, unit, period)
                if df is not None: all_rows.append(df)
                n_done += 1
                if n_done % 50 == 0:
                    log.info("  %d / %d", n_done, n_total)
                time.sleep(SLEEP)

    if not all_rows:
        log.error("No data fetched"); return

    raw = pd.concat(all_rows, ignore_index=True)
    log.info("Raw rows: %d, companies: %d", len(raw), raw["cik"].nunique())

    # Build ticker→CIK map
    log.info("Fetching ticker→CIK map...")
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=30)
    cik_map = {int(v["cik_str"]): v["ticker"]
               for v in r.json().values()}
    raw["ticker"] = raw["cik"].map(cik_map)
    raw = raw.dropna(subset=["ticker"])

    # Pivot to quarterly per (ticker, end) panel
    log.info("Building quarterly panel...")
    quarterly = build_quarterly_panel(raw)

    # Derived per-ticker time-series features
    log.info("Computing extended derived features per ticker...")
    out_rows = []
    for ticker, grp in quarterly.groupby("ticker"):
        g = grp.sort_values("end").copy()
        # Ratios available immediately when both legs exist
        g["asset_turnover"]   = g.get("Revenues",       np.nan) / (g.get("Assets",       np.nan) + 1e-9)
        g["profit_margin"]    = g.get("NetIncomeLoss",  np.nan) / (g.get("Revenues",     np.nan) + 1e-9)
        g["return_on_assets"] = g.get("NetIncomeLoss",  np.nan) / (g.get("Assets",       np.nan) + 1e-9)
        g["debt_to_assets"]   = 1 - g.get("StockholdersEquity", np.nan) / (g.get("Assets", np.nan) + 1e-9)
        g["rev_growth_yoy"]   = g.get("Revenues",       np.nan).pct_change(periods=4)
        g["ni_growth_yoy"]    = g.get("NetIncomeLoss",  np.nan).pct_change(periods=4)
        g["equity_growth"]    = g.get("StockholdersEquity", np.nan).pct_change(periods=4)
        out_rows.append(g)
    quarterly_ext = pd.concat(out_rows, ignore_index=True)

    # Forward-fill to daily aligned to existing alpha158 dates
    log.info("Forward-filling to daily...")
    alpha = pd.read_parquet(REPO / "data" / "alpha158_qlib_dataset.parquet",
                             columns=["date"]).drop_duplicates()
    daily_idx = pd.DatetimeIndex(sorted(alpha["date"].unique()))

    EXT_COLS = ["asset_turnover","profit_margin","return_on_assets","debt_to_assets",
                "rev_growth_yoy","ni_growth_yoy","equity_growth"]
    out = []
    for ticker, g in quarterly_ext.groupby("ticker"):
        g = g.sort_values("available_date")
        daily = pd.DataFrame(index=daily_idx)
        for c in EXT_COLS: daily[c] = np.nan
        for _, row in g.iterrows():
            avail = row["available_date"]
            mask = daily.index >= avail
            for c in EXT_COLS:
                v = row.get(c)
                if pd.notna(v): daily.loc[mask, c] = v
        daily["ticker"] = ticker
        out.append(daily.reset_index().rename(columns={"index":"date"}))

    if not out:
        log.error("no output"); return
    daily_ext = pd.concat(out, ignore_index=True)

    # Robust z-score on each feature using train period stats
    log.info("Robust z-scoring extended features...")
    train_end = pd.Timestamp("2022-11-01")  # matches WF train cutoff
    for c in EXT_COLS:
        col_train = daily_ext.loc[daily_ext["date"] < train_end, c].dropna()
        med = float(col_train.median()) if len(col_train) else 0.0
        mad = float((col_train - med).abs().median()) if len(col_train) else 1.0
        daily_ext[c] = ((daily_ext[c] - med) / max(mad*1.4826, 1e-9)).clip(-3,3)
    daily_ext[EXT_COLS] = daily_ext[EXT_COLS].fillna(0.0)

    out_path = REPO / "data" / "sec_fundamentals_extended.parquet"
    daily_ext.to_parquet(out_path, index=False)
    log.info("Written %s: %d rows × %d cols, %d tickers (extended features: %s)",
             out_path, len(daily_ext), len(daily_ext.columns),
             daily_ext["ticker"].nunique(), EXT_COLS)


if __name__ == "__main__":
    main()
