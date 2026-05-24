#!/usr/bin/env python
"""Fetch point-in-time fundamental features from SEC EDGAR for all 816 tickers.

Uses the SEC EDGAR XBRL frames API to bulk-fetch all companies' financial
data in ~200 requests (one per concept per quarter), then aligns to a daily
panel using each filing's accession date for proper look-ahead-free joins.

Concepts fetched:
  NetIncomeLoss          (income stmt, quarterly duration)
  GrossProfit            (income stmt, quarterly duration)
  Revenues               (income stmt, quarterly duration)
  Assets                 (balance sheet, quarterly instant)
  StockholdersEquity     (balance sheet, quarterly instant)

Derived features (all cross-sectionally z-scored per date):
  earnings_yield         = TTM NetIncome / market_cap
  gross_profitability    = TTM GrossProfit / Assets       (Novy-Marx 2013)
  book_to_price          = StockholdersEquity / market_cap
  roe                    = TTM NetIncome / StockholdersEquity
  asset_growth           = QoQ Assets growth rate

Output: data/sec_fundamentals_daily.parquet
  - indexed (ticker, date), daily frequency
  - forward-filled within each fiscal quarter (values available after filing)
  - NaN for dates before first filing or when no data available

Usage:
    python scripts/fetch_sec_fundamentals.py
    python scripts/fetch_sec_fundamentals.py --start-year 2010 --end-year 2026
    python scripts/fetch_sec_fundamentals.py --dry-run

References:
    SEC EDGAR frames API: https://data.sec.gov/api/xbrl/frames/
    Novy-Marx (2013) gross profitability: RFS, 26(1), pp.44-79.
    Gu, Kelly, Xiu (2020) feature importance ranking: RFS, 33(5).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sec-fundamentals")

SEC_HEADERS = {"User-Agent": "RenQuant renhao.overflow@gmail.com"}
FRAMES_BASE  = "https://data.sec.gov/api/xbrl/frames"
SLEEP_BETWEEN = 0.12   # ~8 req/sec, SEC asks for ≤10

# ── Concepts to fetch ─────────────────────────────────────────────────────
# (concept, taxonomy, unit, period_type)
# period_type: "duration" = income stmt (quarterly period)
#              "instant"  = balance sheet (point in time)
CONCEPTS = [
    ("NetIncomeLoss",                  "us-gaap", "USD",    "duration"),
    ("GrossProfit",                    "us-gaap", "USD",    "duration"),
    ("Revenues",                       "us-gaap", "USD",    "duration"),
    ("Assets",                         "us-gaap", "USD",    "instant"),
    ("StockholdersEquity",             "us-gaap", "USD",    "instant"),
    ("CommonStockSharesOutstanding",   "us-gaap", "shares", "instant"),  # for market-cap ratios
]


def quarter_periods(start_year: int, end_year: int) -> list[tuple[str, str]]:
    """Return (period_label, period_type) for all Q1-Q4 of each year."""
    periods = []
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            periods.append((f"CY{year}Q{q}", "duration"))
            periods.append((f"CY{year}Q{q}I", "instant"))
    return periods


def fetch_frame(concept: str, taxonomy: str, unit: str, period: str,
                max_retries: int = 3, backoff: float = 5.0) -> pd.DataFrame | None:
    """Fetch all companies' values for one concept × one period.

    Retries up to max_retries times on timeout or 5xx errors, with
    exponential backoff. 404 (concept not filed for that period) returns None
    immediately — not an error.
    """
    url = f"{FRAMES_BASE}/{taxonomy}/{concept}/{unit}/{period}.json"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 404:
                return None          # period genuinely has no data — not an error
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                return None
            df = pd.DataFrame(data)
            df["concept"] = concept
            df["period"]  = period
            return df
        except requests.exceptions.Timeout:
            wait = backoff * attempt
            if attempt < max_retries:
                log.warning("  timeout %s/%s — retry %d/%d in %.0fs",
                            concept, period, attempt, max_retries, wait)
                time.sleep(wait)
            else:
                log.warning("  timeout %s/%s — giving up after %d retries",
                            concept, period, max_retries)
                return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                log.warning("  HTTP %d %s/%s — skipping",
                            e.response.status_code, concept, period)
                return None          # 4xx: data issue, don't retry
            wait = backoff * attempt
            if attempt < max_retries:
                log.warning("  HTTP error %s/%s — retry %d/%d in %.0fs",
                            concept, period, attempt, max_retries, wait)
                time.sleep(wait)
            else:
                return None
        except Exception as e:
            log.warning("  fetch_frame failed %s/%s: %s", concept, period, e)
            return None


def build_ticker_cik_map(universe: list[str]) -> dict[str, int]:
    """Download SEC ticker→CIK mapping, return subset for our universe."""
    log.info("Fetching SEC ticker→CIK map...")
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    full_map = {v["ticker"]: int(v["cik_str"]) for v in raw.values()}
    result   = {t: full_map[t] for t in universe if t in full_map}
    missing  = [t for t in universe if t not in full_map]
    log.info("  CIK found: %d / %d  (missing: %d — %s)",
             len(result), len(universe), len(missing), missing[:8])
    return result


def fetch_all_concepts(start_year: int, end_year: int, dry_run: bool = False) -> pd.DataFrame:
    """Fetch all concept × period combos; return raw long DataFrame."""
    all_rows = []
    total = len(CONCEPTS) * (end_year - start_year + 1) * 4 * 2  # approx
    done  = 0

    for concept, taxonomy, unit, period_type in CONCEPTS:
        suffix = "I" if period_type == "instant" else ""
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                period = f"CY{year}Q{q}{suffix}"
                if dry_run:
                    log.info("  [dry-run] would fetch %s / %s", concept, period)
                    done += 1
                    continue

                df = fetch_frame(concept, taxonomy, unit, period)
                if df is not None:
                    all_rows.append(df)
                done += 1
                if done % 20 == 0:
                    log.info("  %d / ~%d requests done", done, total)
                time.sleep(SLEEP_BETWEEN)

    if dry_run or not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def build_quarterly_panel(raw: pd.DataFrame, cik_to_ticker: dict[int, str]) -> pd.DataFrame:
    """Convert raw frames data to quarterly (ticker × period_end) panel."""
    raw["ticker"] = raw["cik"].map(cik_to_ticker)
    raw = raw.dropna(subset=["ticker"])

    # Parse end date
    raw["end"] = pd.to_datetime(raw["end"])
    if "filed" in raw.columns:
        raw["filed"] = pd.to_datetime(raw["filed"], errors="coerce")

    # For duration concepts: use the reported quarter end
    # For instant: same
    # Pivot to wide
    rows = []
    for (ticker, end_date), grp in raw.groupby(["ticker", "end"]):
        row = {"ticker": ticker, "end": end_date}
        filed_dates = (
            pd.to_datetime(grp.get("filed"), errors="coerce").dropna()
            if "filed" in grp.columns else pd.Series(dtype="datetime64[ns]")
        )
        for _, r in grp.iterrows():
            row[r["concept"]] = r["val"]
        if not filed_dates.empty:
            # PIT contract: a derived row that combines multiple concepts is
            # available only after the last contributing SEC filing arrived.
            row["available_date"] = filed_dates.max()
        else:
            # Conservative fallback for old/raw fixtures without SEC `filed`.
            row["available_date"] = end_date + pd.Timedelta(days=45)
        rows.append(row)

    panel = pd.DataFrame(rows)
    return panel.sort_values(["ticker", "end"]).reset_index(drop=True)


def forward_fill_to_daily(quarterly: pd.DataFrame,
                          daily_index: pd.DatetimeIndex,
                          tickers: list[str]) -> pd.DataFrame:
    """Expand quarterly panel to daily, forward-filling after available_date."""
    feat_cols = ["NetIncomeLoss", "GrossProfit", "Revenues", "Assets",
                 "StockholdersEquity", "CommonStockSharesOutstanding"]
    feat_cols = [c for c in feat_cols if c in quarterly.columns]

    all_daily = []
    for ticker in tickers:
        tq = quarterly[quarterly["ticker"] == ticker].sort_values("available_date")
        if tq.empty:
            continue

        # Build time series: on each available_date, update the values
        daily = pd.DataFrame(index=daily_index)
        for c in feat_cols:
            daily[c] = np.nan

        for _, row in tq.iterrows():
            avail = row["available_date"]
            mask  = daily.index >= avail
            for c in feat_cols:
                if pd.notna(row.get(c)):
                    daily.loc[mask, c] = row[c]

        daily["ticker"] = ticker
        all_daily.append(daily.reset_index().rename(columns={"index": "date"}))

    if not all_daily:
        return pd.DataFrame()
    return pd.concat(all_daily, ignore_index=True)


def compute_derived_features(daily: pd.DataFrame,
                              ohlcv_dir: Path) -> pd.DataFrame:
    """Add market-cap-normalized features and derived ratios.

    Point-in-time: uses daily close price from OHLCV for market cap.
    """
    derived_rows = []
    for ticker, grp in daily.groupby("ticker"):
        ohlcv_path = ohlcv_dir / ticker / "1d.parquet"
        if not ohlcv_path.exists():
            continue
        price_df = pd.read_parquet(ohlcv_path)[["close"]].rename(columns={"close": "price"})
        price_df.index = pd.to_datetime(price_df.index)

        # Merge with price on date
        g = grp.set_index("date")
        g.index = pd.to_datetime(g.index)
        merged = g.join(price_df, how="left")

        # TTM Net Income (sum of last 4 quarters is already in the quarterly
        # value since we use the period-end value). For simplicity use as-is.
        ni     = merged.get("NetIncomeLoss",                pd.Series(np.nan, index=merged.index))
        gp     = merged.get("GrossProfit",                 pd.Series(np.nan, index=merged.index))
        ast    = merged.get("Assets",                      pd.Series(np.nan, index=merged.index))
        eq     = merged.get("StockholdersEquity",          pd.Series(np.nan, index=merged.index))
        px     = merged.get("price",                       pd.Series(np.nan, index=merged.index))
        shares = merged.get("CommonStockSharesOutstanding",pd.Series(np.nan, index=merged.index))

        # Market cap = shares × price (both from daily data, point-in-time)
        mktcap = shares * px

        result = pd.DataFrame(index=merged.index)
        result["ticker"] = ticker

        with np.errstate(invalid="ignore", divide="ignore"):
            result["earnings_yield"]      = ni  / (mktcap + 1e-9)   # E/P, Gu et al. top feature
            result["book_to_price"]       = eq  / (mktcap + 1e-9)   # B/M, Fama-French
            result["gross_profitability"] = gp  / (ast    + 1e-9)   # Novy-Marx (2013)
            result["roe"]                 = ni  / (eq     + 1e-9)   # profitability
            # 2026-05-09 BUG #5 FIX: pct_change(periods=4) on a daily forward-
            # filled series computed change over 4 DAYS (not 4 quarters). On
            # a ffill'd daily series, consecutive days have identical values
            # → 93.9% zero asset_growth in the panel → XGB feature gain = 0.
            # Per Cooper-Gulen-Schill 2008 "Asset Growth and the Cross-Section
            # of Stock Returns" the correct horizon is 1 year YoY. We use
            # periods=252 (≈ 252 trading days per year) on the daily ffill'd
            # series, which is mathematically equivalent to comparing
            # current quarter's filing to the same-period 1y ago.
            # AUDIT 2026-05-10 BUG #5b — M&A / spin-off events (BKR 2017,
            # VTRS 2020, VICI 2017, PRMB SPAC, MSTR BTC accumulation) produce
            # inf and >>500% YoY values that poison the training matrix. Per
            # §5.13.12 (range-bound at train site) + §5.13.11 (explicit inf
            # guard). Floor: pct_change ≥ -1 always. Ceiling: 5.0 = 500% YoY
            # ≈ 99.7th pct of legit growth distribution.
            result["asset_growth"]        = ast.pct_change(periods=252).clip(-0.99, 5.0)  # YoY clipped (Cooper-Gulen-Schill 2008 + BUG #5b)

        derived_rows.append(result.reset_index().rename(columns={"index": "date"}))

    if not derived_rows:
        return pd.DataFrame()
    return pd.concat(derived_rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year",   type=int, default=2026)
    p.add_argument("--universe",   default=str(REPO_ROOT / "scripts" / "watchlist_universe.json"))
    p.add_argument("--output",     default=str(REPO_ROOT / "data" / "sec_fundamentals_daily.parquet"))
    p.add_argument("--dry-run",    action="store_true")
    args = p.parse_args()

    # Load universe
    universe = [t for t in json.loads(Path(args.universe).read_text()) if t and t != "-"]
    log.info("Universe: %d tickers (start=%d end=%d)", len(universe), args.start_year, args.end_year)

    # CIK map
    ticker_cik = build_ticker_cik_map(universe)
    cik_ticker = {v: k for k, v in ticker_cik.items()}

    # Fetch all frames
    log.info("Fetching SEC EDGAR frames (~200 API calls)...")
    raw = fetch_all_concepts(args.start_year, args.end_year, dry_run=args.dry_run)

    if args.dry_run:
        log.info("Dry run complete.")
        return

    if raw.empty:
        log.error("No data fetched. Check network / SEC EDGAR status.")
        return

    log.info("Raw rows: %d  companies in data: %d", len(raw), raw["cik"].nunique())

    # Build quarterly panel
    log.info("Building quarterly panel...")
    quarterly = build_quarterly_panel(raw, cik_ticker)
    log.info("Quarterly panel: %d rows, %d tickers", len(quarterly), quarterly["ticker"].nunique())

    # Forward fill to daily
    log.info("Forward-filling to daily frequency...")
    # Use dates from the alpha158 dataset
    alpha158 = pd.read_parquet("data/alpha158_816_dataset.parquet",
                               columns=["date"]).drop_duplicates()
    daily_index = pd.DatetimeIndex(sorted(alpha158["date"].unique()))

    daily = forward_fill_to_daily(quarterly, daily_index, universe)
    log.info("Daily panel: %d rows, %d tickers", len(daily), daily["ticker"].nunique())

    # Compute derived features with price
    log.info("Computing derived features (market-cap normalized)...")
    ohlcv_dir = REPO_ROOT / "data" / "ohlcv"
    features = compute_derived_features(daily, ohlcv_dir)
    log.info("Features panel: %d rows, %d tickers", len(features), features["ticker"].nunique())

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    log.info("Written: %s", out)
    log.info("Coverage: %d / %d tickers have at least one non-NaN fundamental feature",
             (features.groupby("ticker")[["earnings_yield","roe"]].any().any(axis=1).sum()),
             len(universe))


if __name__ == "__main__":
    main()
