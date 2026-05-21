#!/usr/bin/env python
"""Evaluate prod XGB panel-LTR vs shadow HF PatchTST seed44 on real market data.

Loads both production-equivalent scorers, runs them on the most recent OOS
window of the canonical panel, and reports:

  (1) Pure agreement: per-day Spearman correlation between prod-rank and
      shadow-rank — does the shadow model produce similar rankings?
  (2) Top-N pick overlap (N=10, 30): how often do the top picks agree?
  (3) OOS IC vs fwd_60d_excess: which model has better signal on observed
      forward returns within the shadow's val window (2025-02-06 → 2026-02-10)
      (caveat: prod XGB was trained 2026-05-18 and may have these dates in
      training; treat as a *score-quality bound*, not unbiased OOS).
  (4) Regime stratification: split dates by SPY return regime (BULL_CALM /
      BULL_VOLATILE / BEAR / CHOPPY) and report per-regime IC.

Output JSON to artifacts/prod_vs_shadow_eval_<YYYY-MM-DD>.json + console summary.
"""
from __future__ import annotations
import argparse
import faulthandler
faulthandler.enable()
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))

# 2026-05-20: single-threaded torch — concurrent BG training (3-way 5cut×5seed
# PatchTST eval) already saturates OMP; running eval at OMP=10 caused a deadlock
# in modeling_patchtst.py:1159 forward pass.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eval_prod_vs_shadow")


def detect_regime_from_spy(panel: pd.DataFrame) -> pd.Series:
    """Coarse SPY-return-based regime per date. Loads SPY closes from
    yfinance since the alpha158 panel only has normalized features.

    BULL_STRONG: 60d ret > +10% AND vol < 18%
    BULL_CALM:   60d ret in (0, +10%] AND vol < 15%
    BULL_VOLATILE: 60d ret > 0 AND vol >= 15%
    BEAR:        60d ret < -8%
    CHOPPY:      otherwise
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        log.warning("yfinance not installed — regime stratification skipped")
        return pd.Series(dtype=str)
    start = panel["date"].min() - pd.Timedelta(days=120)
    end = panel["date"].max() + pd.Timedelta(days=2)
    try:
        spy_df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:
        log.warning("yfinance SPY fetch failed (%s) — regime stratification skipped", exc)
        return pd.Series(dtype=str)
    if spy_df.empty:
        return pd.Series(dtype=str)
    # Handle multiindex columns (yfinance returns (Field, Ticker))
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [c[0] for c in spy_df.columns]
    spy = pd.DataFrame({"close": spy_df["Close"]})
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy["ret_60d"] = spy["close"].pct_change(60)
    spy["ret_1d"] = spy["close"].pct_change()
    spy["vol_60d_ann"] = spy["ret_1d"].rolling(60).std() * np.sqrt(252)

    def label(r):
        if pd.isna(r["ret_60d"]) or pd.isna(r["vol_60d_ann"]):
            return "UNKNOWN"
        if r["ret_60d"] < -0.08:
            return "BEAR"
        if r["ret_60d"] > 0.10 and r["vol_60d_ann"] < 0.18:
            return "BULL_STRONG"
        if r["ret_60d"] > 0 and r["vol_60d_ann"] < 0.15:
            return "BULL_CALM"
        if r["ret_60d"] > 0:
            return "BULL_VOLATILE"
        return "CHOPPY"

    return spy.apply(label, axis=1)


def load_prod_xgb_scorer(art_path: Path):
    import xgboost as xgb  # noqa: PLC0415
    import torch as _t; _t.set_num_threads(1)
    art = json.loads(art_path.read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))
    return {
        "kind": "xgb",
        "booster": booster,
        "feature_cols": art["feature_cols"],
        "label_col": art["label_col"],
        "lookahead_days": art["lookahead_days"],
        "feature_means": np.asarray(art["feature_means"], dtype=np.float64),
        "feature_stds": np.asarray(art["feature_stds"], dtype=np.float64),
        "trained_date": art.get("trained_date"),
    }


def score_xgb(prod, frame: pd.DataFrame) -> pd.Series:
    import xgboost as xgb  # noqa: PLC0415
    X = frame[prod["feature_cols"]].fillna(0.0).values.astype(np.float64)
    # PROD artifact uses internal preprocessing (panel-LTR XGBoost). Norm is
    # applied by the kernel scorer path. To match, replicate: standardize via
    # stored means/stds + clip ±5.
    Xn = ((X - prod["feature_means"]) / np.where(prod["feature_stds"] > 0, prod["feature_stds"], 1.0)).clip(-5, 5)
    d = xgb.DMatrix(Xn, feature_names=prod["feature_cols"])
    return pd.Series(prod["booster"].predict(d), index=frame.index, name="prod_score")


def load_shadow_patchtst(art_path: Path):
    from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415
    s = HFPatchTSTPanelScorer.load(art_path)
    return {
        "kind": "hf_patchtst",
        "scorer": s,
        "feature_cols": s.feature_cols,
        "seq_len": s.seq_len,
    }


def score_shadow_for_date(shadow, panel_history: pd.DataFrame, target_tickers: list) -> pd.Series:
    return shadow["scorer"].score_with_history(panel_history, target_tickers)


def evaluate_one_date(prod, shadow, panel: pd.DataFrame, today: pd.Timestamp,
                       label_col: str, lookback_days: int) -> dict:
    today_frame = panel[panel["date"] == today].reset_index(drop=True)
    if today_frame.empty:
        return {}
    # Hand the shadow ≥seq_len days of history per ticker
    hist_window = panel[(panel["date"] <= today) &
                        (panel["date"] >= today - pd.Timedelta(days=lookback_days))]
    target_tickers = today_frame["ticker"].tolist()

    prod_scores = score_xgb(prod, today_frame)
    today_frame["prod_score"] = prod_scores.values

    shadow_scores = score_shadow_for_date(shadow, hist_window, target_tickers)
    sh_map = shadow_scores.to_dict()
    today_frame["shadow_score"] = today_frame["ticker"].map(sh_map)

    common = today_frame.dropna(subset=["prod_score", "shadow_score", label_col])
    if len(common) < 5:
        return {"date": str(today.date()), "n_common": int(len(common))}

    # Pure agreement: Spearman rank correlation between prod and shadow scores
    from scipy.stats import spearmanr  # noqa: PLC0415
    rho_pair, _ = spearmanr(common["prod_score"], common["shadow_score"])

    # IC vs observed label
    prod_ic, _ = spearmanr(common["prod_score"], common[label_col])
    shadow_ic, _ = spearmanr(common["shadow_score"], common[label_col])

    # Top-N overlap
    def top_n(s, n):
        return set(common.nlargest(n, s)["ticker"].tolist())
    top10_prod = top_n("prod_score", 10)
    top10_shadow = top_n("shadow_score", 10)
    top30_prod = top_n("prod_score", 30)
    top30_shadow = top_n("shadow_score", 30)
    ov10 = len(top10_prod & top10_shadow)
    ov30 = len(top30_prod & top30_shadow)

    # Bottom-10 (would-be-shorts) overlap
    bot10_prod = set(common.nsmallest(10, "prod_score")["ticker"].tolist())
    bot10_shadow = set(common.nsmallest(10, "shadow_score")["ticker"].tolist())
    ov10_bot = len(bot10_prod & bot10_shadow)

    # Realization: avg fwd label of each model's top-10 vs full-universe mean
    def avg_fwd(tickers, col):
        x = common[common["ticker"].isin(tickers)][col]
        return float(x.mean()) if len(x) else float("nan")

    universe_mean_fwd = float(common[label_col].mean())
    prod_top10_fwd    = avg_fwd(top10_prod,   label_col)
    shadow_top10_fwd  = avg_fwd(top10_shadow, label_col)
    prod_top30_fwd    = avg_fwd(top30_prod,   label_col)
    shadow_top30_fwd  = avg_fwd(top30_shadow, label_col)
    prod_bot10_fwd    = avg_fwd(bot10_prod,   label_col)
    shadow_bot10_fwd  = avg_fwd(bot10_shadow, label_col)

    return {
        "date": str(today.date()),
        "n_common": int(len(common)),
        "rho_pair": float(rho_pair),
        "prod_ic": float(prod_ic),
        "shadow_ic": float(shadow_ic),
        "ov10": int(ov10),
        "ov30": int(ov30),
        "ov10_bot": int(ov10_bot),
        "prod_top10": sorted(top10_prod),
        "shadow_top10": sorted(top10_shadow),
        # alignment-with-market realization (label = fwd_60d_excess vs SPY)
        "universe_mean_fwd":    universe_mean_fwd,
        "prod_top10_fwd":       prod_top10_fwd,
        "shadow_top10_fwd":     shadow_top10_fwd,
        "prod_top30_fwd":       prod_top30_fwd,
        "shadow_top30_fwd":     shadow_top30_fwd,
        "prod_bot10_fwd":       prod_bot10_fwd,
        "shadow_bot10_fwd":     shadow_bot10_fwd,
        "prod_top10_alpha":     prod_top10_fwd   - universe_mean_fwd,
        "shadow_top10_alpha":   shadow_top10_fwd - universe_mean_fwd,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--prod-artifact",
                   default="backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json")
    p.add_argument("--shadow-artifact",
                   default="artifacts/patchtst_shadow/canonical_5seed_mps/seed_44/hf_patchtst_all_seed44_model.pt")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--start", default="2025-02-06",
                   help="OOS window start (default: shadow seed44 val start)")
    p.add_argument("--end",   default="2026-02-10",
                   help="OOS window end (default: shadow seed44 val end)")
    p.add_argument("--max-dates", type=int, default=None,
                   help="Cap to N dates (sample every k for speed); default ALL")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: artifacts/prod_vs_shadow_eval_YYYY-MM-DD.json)")
    args = p.parse_args()

    log.info("Loading panel %s", args.panel)
    panel = pd.read_parquet(REPO / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: %s rows, %d tickers, %s → %s",
             f"{len(panel):,}", panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    log.info("Loading PROD XGB artifact: %s", args.prod_artifact)
    prod = load_prod_xgb_scorer(REPO / args.prod_artifact)
    log.info("PROD: kind=%s n_feat=%d label=%s trained=%s",
             prod["kind"], len(prod["feature_cols"]),
             prod["label_col"], prod["trained_date"])

    log.info("Loading SHADOW HF PatchTST: %s", args.shadow_artifact)
    shadow = load_shadow_patchtst(REPO / args.shadow_artifact)
    log.info("SHADOW: kind=%s seq_len=%d n_feat=%d",
             shadow["kind"], shadow["seq_len"], len(shadow["feature_cols"]))

    # Verify feature_cols overlap
    common_cols = set(prod["feature_cols"]) & set(shadow["feature_cols"])
    log.info("Feature overlap: %d/%d (prod) %d/%d (shadow)",
             len(common_cols), len(prod["feature_cols"]),
             len(common_cols), len(shadow["feature_cols"]))

    # Regime labels
    regimes = detect_regime_from_spy(panel)
    log.info("Regime label dist: %s",
             regimes.value_counts().to_dict() if len(regimes) else "N/A")

    # OOS window
    start = pd.Timestamp(args.start)
    end   = pd.Timestamp(args.end)
    panel_oos = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    eval_dates = sorted(panel_oos["date"].unique())
    if args.max_dates and len(eval_dates) > args.max_dates:
        step = len(eval_dates) // args.max_dates
        eval_dates = eval_dates[::step][:args.max_dates]
    log.info("Evaluating %d dates from %s → %s",
             len(eval_dates), eval_dates[0].date() if eval_dates else "?",
             eval_dates[-1].date() if eval_dates else "?")

    lookback = (shadow["seq_len"] + 5) * 2  # calendar days; some weekends/holidays

    results = []
    for i, today in enumerate(eval_dates):
        log.info("date[%d/%d] %s ...", i+1, len(eval_dates), pd.Timestamp(today).date())
        r = evaluate_one_date(prod, shadow, panel, today, args.label, lookback)
        if r and "rho_pair" in r:
            r["regime"] = str(regimes.get(today, "UNKNOWN"))
            results.append(r)
        if (i + 1) % 1 == 0:
            log.info("Progress: %d/%d dates evaluated", i + 1, len(eval_dates))

    if not results:
        log.error("No dates evaluated successfully")
        sys.exit(1)

    df = pd.DataFrame(results)
    log.info("=" * 70)
    log.info("SUMMARY: %d dates", len(df))
    log.info("=" * 70)
    log.info("Score agreement (Spearman ρ prod↔shadow):")
    log.info("  mean=%+.3f  std=%.3f  median=%+.3f  pos_days=%d/%d",
             df["rho_pair"].mean(), df["rho_pair"].std(),
             df["rho_pair"].median(),
             int((df["rho_pair"] > 0).sum()), len(df))
    log.info("Top-10 overlap (out of 10): mean=%.2f median=%.0f  pct_zero=%.0f%%",
             df["ov10"].mean(), df["ov10"].median(),
             100.0 * (df["ov10"] == 0).mean())
    log.info("Top-30 overlap (out of 30): mean=%.2f median=%.0f",
             df["ov30"].mean(), df["ov30"].median())
    log.info("Bot-10 (short-side) overlap (out of 10): mean=%.2f median=%.0f",
             df["ov10_bot"].mean(), df["ov10_bot"].median())
    log.info("")
    log.info("PROD XGB IC vs %s:", args.label)
    log.info("  mean=%+.4f  std=%.4f  median=%+.4f  pos_days=%d/%d",
             df["prod_ic"].mean(), df["prod_ic"].std(),
             df["prod_ic"].median(),
             int((df["prod_ic"] > 0).sum()), len(df))
    log.info("SHADOW PatchTST IC vs %s:", args.label)
    log.info("  mean=%+.4f  std=%.4f  median=%+.4f  pos_days=%d/%d",
             df["shadow_ic"].mean(), df["shadow_ic"].std(),
             df["shadow_ic"].median(),
             int((df["shadow_ic"] > 0).sum()), len(df))
    log.info("")
    log.info("MARKET-ALIGNMENT: realized fwd-60d excess of top picks")
    log.info("  universe-mean    = %+.4f", df["universe_mean_fwd"].mean())
    log.info("  PROD   top-10    = %+.4f  (alpha vs universe = %+.4f)",
             df["prod_top10_fwd"].mean(), df["prod_top10_alpha"].mean())
    log.info("  SHADOW top-10    = %+.4f  (alpha vs universe = %+.4f)",
             df["shadow_top10_fwd"].mean(), df["shadow_top10_alpha"].mean())
    log.info("  PROD   top-30    = %+.4f", df["prod_top30_fwd"].mean())
    log.info("  SHADOW top-30    = %+.4f", df["shadow_top30_fwd"].mean())
    log.info("  PROD   bot-10    = %+.4f  (would-be-shorts)", df["prod_bot10_fwd"].mean())
    log.info("  SHADOW bot-10    = %+.4f  (would-be-shorts)", df["shadow_bot10_fwd"].mean())
    log.info("  PROD   long-short spread (top10-bot10)   = %+.4f",
             (df["prod_top10_fwd"] - df["prod_bot10_fwd"]).mean())
    log.info("  SHADOW long-short spread (top10-bot10)   = %+.4f",
             (df["shadow_top10_fwd"] - df["shadow_bot10_fwd"]).mean())
    log.info("")
    log.info("Per-regime IC + realization:")
    for regime, grp in df.groupby("regime"):
        log.info("  %-14s n=%3d  prod_ic=%+.4f  shadow_ic=%+.4f  rho_pair=%+.3f  ov10=%.1f  "
                 "prod_top10_α=%+.4f  shadow_top10_α=%+.4f",
                 regime, len(grp),
                 grp["prod_ic"].mean(), grp["shadow_ic"].mean(),
                 grp["rho_pair"].mean(), grp["ov10"].mean(),
                 grp["prod_top10_alpha"].mean(), grp["shadow_top10_alpha"].mean())

    out_path = REPO / (args.out or f"artifacts/prod_vs_shadow_eval_{datetime.now(timezone.utc).date()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "args": vars(args),
        "n_dates": len(df),
        "prod_meta": {k: v for k, v in prod.items() if k not in {"booster", "feature_means", "feature_stds"}},
        "shadow_meta": {k: v for k, v in shadow.items() if k != "scorer"},
        "agreement": {
            "rho_pair_mean":     float(df["rho_pair"].mean()),
            "rho_pair_median":   float(df["rho_pair"].median()),
            "ov10_mean":         float(df["ov10"].mean()),
            "ov30_mean":         float(df["ov30"].mean()),
            "ov10_bot_mean":     float(df["ov10_bot"].mean()),
        },
        "ic": {
            "prod_mean":   float(df["prod_ic"].mean()),
            "prod_std":    float(df["prod_ic"].std()),
            "shadow_mean": float(df["shadow_ic"].mean()),
            "shadow_std":  float(df["shadow_ic"].std()),
        },
        "per_regime": {
            r: {
                "n": int(len(g)),
                "prod_ic_mean":   float(g["prod_ic"].mean()),
                "shadow_ic_mean": float(g["shadow_ic"].mean()),
                "rho_pair_mean":  float(g["rho_pair"].mean()),
                "ov10_mean":      float(g["ov10"].mean()),
            }
            for r, g in df.groupby("regime")
        },
        "per_date": df.to_dict(orient="records"),
    }
    # Drop heavy per-date arrays from sidecar JSON to keep file small
    for r in summary["per_date"]:
        r.pop("prod_top10", None)
        r.pop("shadow_top10", None)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary JSON → %s", out_path)


if __name__ == "__main__":
    main()
