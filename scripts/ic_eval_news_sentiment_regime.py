#!/usr/bin/env python3
"""Regime-stratified IC eval for news sentiment (PRIME DIRECTIVE compliant).

Pooled-mean IC eval (ic_eval_news_sentiment.py) showed sentiment NULL
at +0.006. But news sentiment is theoretically regime-conditional:
  • Garcia 2013 *JF* "Sentiment During Recessions" — effect amplifies
    in BEAR / SPIKED vol regimes
  • Tetlock 2007 *JF* — high-attention periods (≈high-vol) 3-5×
    signal strength
  • Da-Engelberg-Gao 2011 *JF* "In Search of Attention" — search-
    interest predicts returns more in volatile periods

Pooled-mean would average across regimes and bury the signal —
exactly the failure mode CLAUDE.md PRIME DIRECTIVE warns about.

This script:
  1. Joins panel × sentiment (same as ic_eval_news_sentiment.py)
  2. Joins SPY-regime per date (TREND × VOL → 9 regimes)
  3. Computes Spearman IC per (regime × feature × label)
  4. Reports IC matrix + per-regime ts-30 placebo for the strongest
     regime (so we can verify the regime-specific signal is real,
     not still endogeneity)

Output: artifacts/ic_eval_news_sentiment_regime.json
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ic_eval_regime")


def _load_panel(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p, columns=[
        "ticker", "date", "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_sentiment(d: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))]
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"symbol": "ticker"})
    return out


def _compute_regimes(spy_path: Path,
                     trend_window: int = 60,
                     vol_window: int = 20,
                     vol_hist_window: int = 252) -> pd.DataFrame:
    """SPY-based per-date regime label, same scheme as eval_regime_stratified.py."""
    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    spy["ret"] = np.log(spy["close"] / spy["close"].shift(1))
    spy["roll_mean"] = spy["ret"].rolling(trend_window, min_periods=trend_window // 2).mean()
    spy["roll_vol"]  = spy["ret"].rolling(trend_window, min_periods=trend_window // 2).std(ddof=1)
    spy["sharpe60"]  = (spy["roll_mean"] / spy["roll_vol"]) * math.sqrt(252.0)
    spy["vol20"]     = spy["ret"].rolling(vol_window, min_periods=vol_window // 2).std(ddof=1) * math.sqrt(252.0)

    def _pct_rank(s):
        if len(s) < 30 or pd.isna(s.iloc[-1]):
            return np.nan
        last = s.iloc[-1]
        return float((s < last).sum()) / float(len(s) - 1)

    spy["vol_pct"] = spy["vol20"].rolling(vol_hist_window, min_periods=30).apply(
        _pct_rank, raw=False)
    spy["trend_label"] = pd.cut(
        spy["sharpe60"],
        bins=[-np.inf, 0.5, 1.5, np.inf],
        labels=["LOW", "MED", "HIGH"])
    spy["vol_label"] = pd.cut(
        spy["vol_pct"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["CALM", "NORMAL", "SPIKED"])
    spy["regime"] = spy["trend_label"].astype(str) + "_" + spy["vol_label"].astype(str)
    out = spy[["regime"]].copy()
    out.index.name = "date"
    return out.reset_index()


def _xs_ic(merged: pd.DataFrame, feat: str, label: str) -> tuple[float, int]:
    from scipy.stats import spearmanr
    ics = []
    for d, g in merged.groupby("date"):
        if len(g) < 5:
            continue
        x = g[feat].values
        y = g[label].values
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        r, _ = spearmanr(x, y)
        if not np.isnan(r):
            ics.append(r)
    if not ics:
        return float("nan"), 0
    return float(np.mean(ics)), len(ics)


def _ts_minus_30_ic(panel: pd.DataFrame, sent: pd.DataFrame,
                    regimes: pd.DataFrame, feat: str, label: str,
                    regime_filter: str | None = None) -> tuple[float, int]:
    """Time-shift placebo: shift sentiment +30 days into future
    (correlate news at date X with returns at X-29 to X-90).
    If raw IC was real predictive power, ts-30 should collapse.
    """
    s = sent.copy()
    s["date"] = s["date"] - pd.Timedelta(days=-30)  # FORWARD shift
    merged = panel.merge(s[["ticker", "date", feat]], on=["ticker", "date"], how="inner")
    if regime_filter is not None:
        merged = merged.merge(regimes, on="date", how="left")
        merged = merged[merged["regime"] == regime_filter]
    return _xs_ic(merged, feat, label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--sent-dir", default="data/news_sentiment_alpaca")
    p.add_argument("--spy", default="data/ohlcv/SPY/1d.parquet")
    p.add_argument("--features", nargs="*", default=[
        "mean_sentiment", "sentiment_dispersion", "n_articles",
        "sentiment_pos_share", "sentiment_neg_share"])
    p.add_argument("--labels", nargs="*", default=[
        "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--out", default="artifacts/ic_eval_news_sentiment_regime.json")
    args = p.parse_args()

    log.info("loading panel + sentiment + SPY ...")
    panel = _load_panel(Path(args.panel))
    sent = _load_sentiment(Path(args.sent_dir))
    regimes = _compute_regimes(Path(args.spy))
    log.info("panel %s rows  sent %s rows  regimes %s days",
             f"{len(panel):,}", f"{len(sent):,}", f"{len(regimes):,}")

    merged = panel.merge(sent, on=["ticker", "date"], how="inner")
    merged = merged.merge(regimes, on="date", how="left")
    log.info("merged %s rows  with regime labels: %s",
             f"{len(merged):,}", merged["regime"].notna().sum())

    regime_counts = merged["regime"].value_counts().sort_index()
    log.info("\nRegime coverage (rows):")
    for r, n in regime_counts.items():
        log.info("  %-20s %s", r, f"{n:,}")

    results: dict = {
        "merged_rows": int(len(merged)),
        "regimes_seen": list(regime_counts.index),
        "regime_row_counts": {str(k): int(v) for k, v in regime_counts.items()},
        "by_regime": {},
    }

    log.info("\n=== regime-stratified IC ===")
    for regime in sorted(regime_counts.index):
        sub = merged[merged["regime"] == regime]
        if len(sub) < 100:
            log.info("[%s] n=%d too few, skip", regime, len(sub))
            continue
        log.info("\n[%s] n=%s rows  dates=%s", regime, f"{len(sub):,}",
                 sub["date"].nunique())
        block: dict = {}
        for feat in args.features:
            if feat not in sub.columns:
                continue
            row = {}
            for lab in args.labels:
                ic, n = _xs_ic(sub, feat, lab)
                ts30, _ = _ts_minus_30_ic(panel, sent, regimes, feat, lab,
                                          regime_filter=regime)
                row[lab] = {"ic": ic, "n_dates": n, "ts-30_placebo": ts30,
                            "ic_net_of_placebo": ic - ts30}
                log.info("  %-22s × %-18s  IC=%+.4f  n=%4d  ts-30=%+.4f  net=%+.4f",
                         feat, lab, ic, n, ts30, ic - ts30)
            block[feat] = row
        results["by_regime"][regime] = block

    # Pooled for reference
    log.info("\n=== pooled (for reference) ===")
    pooled: dict = {}
    for feat in args.features:
        if feat not in merged.columns:
            continue
        block = {}
        for lab in args.labels:
            ic, n = _xs_ic(merged, feat, lab)
            block[lab] = {"ic": ic, "n_dates": n}
        pooled[feat] = block
        log.info("  %s: %s", feat,
                 " ".join(f"{lab}={pooled[feat][lab]['ic']:+.4f}"
                          for lab in args.labels))
    results["pooled"] = pooled

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2, default=str))
    log.info("\nwrote %s", out_p)


if __name__ == "__main__":
    main()
