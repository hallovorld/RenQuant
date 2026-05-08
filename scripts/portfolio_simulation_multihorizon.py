#!/usr/bin/env python
"""D2 — Multi-horizon ensemble portfolio simulation.

Per E42: ensembling fwd_5d + fwd_20d + fwd_60d cross-sectional rank
predictions gave +0.074 mean IC vs single-horizon fwd_60d +0.067. Δ
+0.007 IC was below the +0.010 promotion threshold but the
failed-experiments-log entry said "worth testing in portfolio sim"
because IC and Sharpe don't always move together — execution timing
might lift Sharpe even at flat IC.

This script mirrors scripts/portfolio_simulation.py exactly except
the prediction step:

  Per cut:
    Train 3 separate XGB rank:pairwise models (fwd_5d, fwd_20d, fwd_60d)
    For each test bar:
      pred_h = each model's prediction
      rank_h = cross-sectional percentile of pred_h on that date
    Ensemble = mean of (rank_5, rank_20, rank_60)

Then top-decile long-only OR top-vs-bottom long-short, REBAL_DAYS
turnover with TC, exactly the same scoring/cost/sharpe protocol
as the single-horizon baseline.

Output: data/portfolio_sim_multihorizon.json
"""
from __future__ import annotations
import logging, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from portfolio_simulation import (
    CUTS, REBAL_DAYS, HOLD_DAYS, DECILE, TC_BPS,
    daily_returns_from_close, simulate_portfolio, metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("portfolio-sim-mh")

LABELS = ["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}
N_ROUNDS = 100


def train_one_horizon(panel, feat_cols, label, cut):
    """Returns DataFrame[date, ticker, pred] on test period for ONE horizon."""
    tr_s, tr_e, te_s, te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[label])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)]
    if len(tr) < 1000 or len(te) < 100:
        return pd.DataFrame()
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[label].clip(-5, 5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    si = np.argsort(tr["date"].values)
    Xs, ys, ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _, gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    out = te[["date","ticker"]].copy()
    out["pred"] = booster.predict(xgb.DMatrix(Xte_n))
    return out


def ensemble_predict(panel, feat_cols, cut):
    """Train 3 horizon models on cut, return DataFrame[date,ticker,pred]
    where pred is the mean cross-sectional rank percentile across horizons."""
    log.info("  Training %d horizons in cut...", len(LABELS))
    per_h = {}
    for h in LABELS:
        t0 = time.time()
        df = train_one_horizon(panel, feat_cols, h, cut)
        if df.empty:
            log.warning("  %s — empty (insufficient data)", h)
            continue
        df = df.rename(columns={"pred": f"pred_{h}"})
        per_h[h] = df
        log.info("    %-15s trained in %.1fs (%d test rows)", h, time.time()-t0, len(df))

    if len(per_h) < 3:
        log.warning("  fewer than 3 horizons; skipping ensemble")
        return pd.DataFrame()

    merged = per_h[LABELS[0]][["date","ticker"]].copy()
    for h in LABELS:
        merged = merged.merge(per_h[h], on=["date","ticker"], how="inner")

    # Cross-sectional rank-percentile per horizon per date, then mean
    for h in LABELS:
        merged[f"rank_{h}"] = merged.groupby("date")[f"pred_{h}"].rank(pct=True)
    merged["pred"] = merged[[f"rank_{h}" for h in LABELS]].mean(axis=1)
    return merged[["date","ticker","pred"]]


def main():
    log.info("Loading panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    ohlcv_dir = REPO / "data" / "ohlcv"
    spy_ret = daily_returns_from_close(ohlcv_dir, "SPY")

    cut_metrics = []
    all_lo = []; all_ls = []

    for i, cut in enumerate(CUTS, 1):
        tr_s, tr_e, te_s, te_e = cut
        log.info("Cut %d/%d: train=[%s..%s] test=[%s..%s]", i, len(CUTS), tr_s, tr_e, te_s, te_e)
        preds = ensemble_predict(panel, feat_cols, cut)
        if preds.empty:
            log.warning("Cut %d empty — skipping", i)
            cut_metrics.append({"cut": cut, "error": "empty"})
            continue

        log.info("  Simulating LO + LS portfolios...")
        lo = simulate_portfolio(preds, ohlcv_dir, mode="long_only")
        ls = simulate_portfolio(preds, ohlcv_dir, mode="long_short")

        if not lo.empty:
            r = lo.set_index("date")["ret"]
            m_lo = metrics(r); m_lo["mode"] = "long_only"; m_lo["cut"] = i
            cut_metrics.append(m_lo)
            log.info("  LO  Sharpe=%+.2f  AnnRet=%+.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                     m_lo["sharpe"], m_lo["ann_return"]*100, m_lo["ann_vol"]*100, m_lo["max_dd"]*100)
            all_lo.append(r)
        if not ls.empty:
            r = ls.set_index("date")["ret"]
            m_ls = metrics(r); m_ls["mode"] = "long_short"; m_ls["cut"] = i
            cut_metrics.append(m_ls)
            log.info("  LS  Sharpe=%+.2f  AnnRet=%+.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                     m_ls["sharpe"], m_ls["ann_return"]*100, m_ls["ann_vol"]*100, m_ls["max_dd"]*100)
            all_ls.append(r)

    log.info("\n══ AGGREGATE ACROSS ALL CUTS ══")
    for mode, returns_list in [("long_only", all_lo), ("long_short", all_ls)]:
        if not returns_list: continue
        all_r = pd.concat(returns_list).sort_index()
        # Dedupe overlapping cuts at boundaries
        all_r = all_r[~all_r.index.duplicated(keep="first")]
        m = metrics(all_r)
        log.info("  %s  Sharpe=%+.2f  AnnRet=%+.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                 mode.upper(), m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)

    log.info("\n══ vs PRODUCTION BASELINE (E41) ══")
    log.info("  baseline LO Sharpe=1.06 AnnRet=34.4%% MaxDD=-42.3%%")
    log.info("  baseline LS Sharpe=1.04")

    out = REPO / "data" / "portfolio_sim_multihorizon.json"
    out.write_text(json.dumps({
        "cut_metrics": cut_metrics,
        "label": "multi_horizon_ensemble (fwd_5d + fwd_20d + fwd_60d, mean of rank percentiles)",
    }, indent=2, default=str))
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
