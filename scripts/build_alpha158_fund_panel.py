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
SUE_COLS  = ["sue_signal", "surprise_momentum", "surprise_streak"]
# Sentiment columns (added 2026-05-18 after regime-stratified IC eval; survivors only).
# Per 2026-05-18 verdict doc: sentiment_pos_share + mean_sentiment + n_articles
# clear shuffle-noise in HIGH_SPIKED / HIGH_NORMAL / MED_CALM regimes.
# Drop sentiment_dispersion (ts-30 placebo eats it) and sentiment_neg_share (NULL).
SENT_COLS = ["sentiment_pos_share", "mean_sentiment", "n_articles_log"]
PEAD_DECAY_DAYS = 60   # Bernard-Thomas 1989 drift window
SUE_WINDOW = 4         # Foster-Olsen-Shevlin 1984 — 4 prior quarters for std denom


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truncate-to-sec-max", action="store_true",
                    help="2026-05-18: when alpha158 panel extends beyond "
                         "SEC max date (SEC has natural 7d publication lag), "
                         "truncate alpha158 rows to SEC max instead of "
                         "hard-failing on the BUG #2 guard. Trains use "
                         "cutoffs ≤ 2024-02 anyway so recent rows are "
                         "label-irrelevant; this just lets the rebuild "
                         "succeed for fixing historic data quality "
                         "(e.g. asset_growth regeneration).")
    args = ap.parse_args()

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

    # ── Invariant guard: SEC date coverage ≥ panel date coverage ─────────
    # 2026-05-09 BUG #2 root: panel max date 2026-02-11 vs sec max 2026-02-10
    # → that one extra day got fund all-zero (cross-sectional median of
    # all-NaN reduces to 0). Hard-fail to surface the SEC-fetch lag
    # immediately rather than silently zero-filling.
    panel_max = alpha["date"].max()
    sec_max   = fund["date"].max()
    if panel_max > sec_max:
        if args.truncate_to_sec_max:
            n_drop = int((alpha["date"] > sec_max).sum())
            alpha = alpha[alpha["date"] <= sec_max].reset_index(drop=True)
            log.warning(
                "BUG #2 guard bypassed by --truncate-to-sec-max: dropped "
                "%d alpha158 rows (dates > %s) to align with SEC max. "
                "Panel rebuild proceeds with truncated date range %s..%s. "
                "Subsequent retraining cutoffs ≤ %s should be unaffected.",
                n_drop, sec_max.date(),
                alpha["date"].min().date(), alpha["date"].max().date(),
                sec_max.date(),
            )
        else:
            raise RuntimeError(
                f"BUG #2 guard: alpha158 panel max date {panel_max.date()} > "
                f"sec_fundamentals_daily max {sec_max.date()}. "
                f"Refresh sec_fundamentals_daily before rebuilding panel — "
                f"otherwise the {(panel_max - sec_max).days} unmatched day(s) "
                f"will get fund features silently zero-filled (since "
                f"cross-sectional median over all-NaN candidates is itself "
                f"NaN, falling back to fillna(0)). Override: "
                f"--truncate-to-sec-max (intended for historic data fixes)."
            )

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

    # ── Add SUE features (E49 promotion 2026-05-09) ───────────────────────
    log.info("Computing SUE features (Foster-Olsen-Shevlin 1984, 4Q std)...")
    t0 = time.time()
    merged = _add_sue_features(merged)
    log.info("  SUE features added in %.1fs", time.time()-t0)

    # ── Add Sentiment features (2026-05-18, regime-conditional SCREEN) ─────
    log.info("Merging sentiment features (Tetlock 2007, Garcia 2013)...")
    t0 = time.time()
    merged = _add_sentiment_features(merged)
    log.info("  sentiment features added in %.1fs", time.time()-t0)

    # Verify final shape matches expected schema (alpha158 + fund + pead + sue + sent)
    expected_cols = (len(alpha.columns) + len(FUND_COLS) + len(PEAD_COLS)
                     + len(SUE_COLS) + len(SENT_COLS))
    if len(merged.columns) != expected_cols:
        log.warning("Column count %d (expected %d) — extra cols: %s",
                    len(merged.columns), expected_cols,
                    set(merged.columns) - set(alpha.columns) - set(FUND_COLS)
                    - set(PEAD_COLS) - set(SUE_COLS) - set(SENT_COLS))

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


def _add_sue_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach 3 SUE-class features per (ticker, date) using earnings_surprise/.

    Foster-Olsen-Shevlin 1984 SUE:
        SUE_t = surprise_t / σ(surprise_(t-1)..(t-4))
    plus surprise_momentum (QoQ change in surprise%) and surprise_streak
    (signed consecutive-same-direction count). All three carry only
    within the 60d post-earnings Bernard-Thomas window.
    """
    earn_dir = REPO / "data" / "earnings_surprise"
    n_with_data = 0
    out_blocks = []
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ep = earn_dir / f"{tkr}.parquet"
        if not ep.exists():
            for c in SUE_COLS:
                g[c] = np.nan
            out_blocks.append(g); continue
        n_with_data += 1
        earn = pd.read_parquet(ep).reset_index()
        earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
        earn = earn.sort_values("earnings_date").reset_index(drop=True)

        s = earn["surprise_pct"].astype(float)
        # SUE per-event: rolling std of prior 4Q surprises (exclude current)
        rolling_std = s.shift(1).rolling(SUE_WINDOW, min_periods=2).std()
        sue_per_event = (s / (rolling_std + 1e-6)).clip(-5, 5)
        momentum_per_event = s.diff()
        # Signed consecutive-direction streak
        sign = np.sign(s).fillna(0).astype(int)
        streak = np.zeros(len(s), dtype=int)
        for i in range(len(s)):
            if i == 0 or sign.iloc[i] == 0 or sign.iloc[i] != sign.iloc[i-1]:
                streak[i] = sign.iloc[i]
            else:
                streak[i] = streak[i-1] + sign.iloc[i]

        # Forward-fill to daily panel — only carry within 60d window
        g_dates = g["date"].values
        e_dates = earn["earnings_date"].values
        idxs = np.searchsorted(e_dates, g_dates, side="right") - 1
        days_since = np.full(len(g), np.nan)
        sue = np.full(len(g), np.nan)
        mom = np.full(len(g), np.nan)
        strk = np.full(len(g), np.nan)
        valid = idxs >= 0
        diff = (g_dates[valid] - e_dates[idxs[valid]]).astype('timedelta64[D]').astype(int)
        days_since[valid] = diff
        sue[valid] = sue_per_event.iloc[idxs[valid]].values
        mom[valid] = momentum_per_event.iloc[idxs[valid]].values
        strk[valid] = streak[idxs[valid]]
        # Decay over 60d
        out_of_window = (days_since > PEAD_DECAY_DAYS) | np.isnan(days_since)
        decay = np.where(out_of_window, 0.0,
                          np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))
        g["sue_signal"]        = np.where(out_of_window, 0.0, sue * decay)
        g["surprise_momentum"] = np.where(out_of_window, 0.0, mom * decay)
        g["surprise_streak"]   = np.where(out_of_window, 0.0, strk * decay)
        out_blocks.append(g)

    log.info("  SUE coverage: %d/%d tickers had earnings data",
             n_with_data, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)

    # Cross-sectional median imputation per date for inference; final 0
    for c in SUE_COLS:
        med = out.groupby("date")[c].transform("median")
        out[c] = out[c].fillna(med).fillna(0.0)
    return out


def _add_sentiment_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach sentiment columns from data/news_sentiment_alpaca/{ticker}.parquet.

    Added 2026-05-18 after regime-stratified IC eval revealed HIGH_SPIKED
    winner (sentiment_pos_share × fwd_5d IC=+0.054, mean_sentiment IC=+0.045).
    Pre-regime-stratification pooled IC looked NULL — see 2026-05-18
    sentiment verdict doc.

    Three features (regime-conditional integration, deploy via
    `regime_params.<R>.sentiment.enabled` overlay at inference time):
      sentiment_pos_share — fraction of articles scored > +0.2 (HIGH_SPIKED
                             IC +0.054, cleanest signal)
      mean_sentiment      — average per-article signed score in [-1, +1]
                             (HIGH_SPIKED IC +0.045)
      n_articles_log      — log(1 + n_articles), proxy for news flow
                             intensity (HIGH_SPIKED fwd_60d IC +0.023)

    For tickers without sentiment data (pre-2020-01 dates, or non-watchlist
    tickers): NaN initially → cross-sectional median fill → final 0.
    Refs: Tetlock 2007, Garcia 2013, Da-Engelberg-Gao 2011, Araci 2019.
    """
    sent_dir = REPO / "data" / "news_sentiment_alpaca"
    if not sent_dir.exists() or not any(sent_dir.glob("*.parquet")):
        log.warning("  no sentiment data at %s — filling SENT_COLS with 0",
                    sent_dir)
        for c in SENT_COLS:
            panel[c] = 0.0
        return panel

    parts = []
    n_with_sent = 0
    for f in sent_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        # Source schema: symbol, date, mean_sentiment, sentiment_dispersion,
        # n_articles, sentiment_pos_share, sentiment_neg_share
        df = df.rename(columns={"symbol": "ticker"})
        df["date"] = pd.to_datetime(df["date"])
        # Derive n_articles_log (compress heavy right tail; max is 42)
        df["n_articles_log"] = np.log1p(df["n_articles"].astype(float))
        keep = ["ticker", "date"] + SENT_COLS
        parts.append(df[keep])
        n_with_sent += 1
    sent = pd.concat(parts, ignore_index=True)
    log.info("  sentiment coverage: %d/%d tickers  rows=%d  dates=[%s, %s]",
             n_with_sent, panel["ticker"].nunique(), len(sent),
             sent["date"].min().date(), sent["date"].max().date())

    merged = panel.merge(sent, on=["ticker", "date"], how="left")
    if len(merged) != len(panel):
        raise RuntimeError(
            f"sentiment merge changed row count: {len(panel)} → {len(merged)}. "
            f"Check duplicate (ticker, date) pairs in sentiment parquets.")

    # Cross-sectional median imputation per date for inference; final 0
    # (Important: 2020-01 onwards has sentiment; pre-2020 dates get 0
    # which is fine — the model learns sentiment effects mostly on
    # 2020+ training data where coverage is real.)
    for c in SENT_COLS:
        nan_pct_pre = merged[c].isna().mean() * 100
        med = merged.groupby("date")[c].transform("median")
        merged[c] = merged[c].fillna(med).fillna(0.0)
        log.info("  %-25s NaN%% pre=%.1f%%  post_median+zero=%.1f%%",
                 c, nan_pct_pre, merged[c].isna().mean() * 100)

    return merged


if __name__ == "__main__":
    main()
