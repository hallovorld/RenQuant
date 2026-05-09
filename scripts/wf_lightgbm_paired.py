#!/usr/bin/env python
"""C3 — LightGBM retest on alpha158+5fund+3PEAD (166-feat panel).

E27-era LightGBM rejection was on the 21-feat production panel
(IC dropped 60%). Now that the panel is 166-feat (alpha158 + fund +
PEAD), retest on the same protocol. Per CLAUDE.md §5.12 use the
canonical microsoft/LightGBM library (not a hand-rolled variant).

Compares LightGBM-rank (lambdarank objective) and LightGBM-regression
(MSE) to XGBoost rank:pairwise baseline, on the same 7-cut WF
protocol.

Reference: Ke et al. 2017 NeurIPS LightGBM (15k+ citations).
Configuration mirrors XGBoost: max_depth=5, learning_rate=0.05,
n_estimators=100, subsample=0.7, colsample_bytree=0.7.

Output: data/wf_lightgbm_paired.json
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-lightgbm-paired")

CUTS = [
    ("2016-01-01","2018-12-31","2019-02-01","2019-12-31"),
    ("2017-01-01","2019-12-31","2020-02-01","2020-12-31"),
    ("2018-01-01","2020-12-31","2021-02-01","2021-12-31"),
    ("2019-01-01","2021-12-31","2022-02-01","2022-12-31"),
    ("2020-01-01","2022-12-31","2023-02-01","2023-12-31"),
    ("2021-01-01","2023-12-31","2024-02-01","2024-12-31"),
    ("2022-01-01","2024-12-31","2025-02-01","2025-12-31"),
]
LABEL = "fwd_60d_excess"


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def wf_xgb(panel, feat_cols, cut):
    """Production XGB rank:pairwise baseline."""
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100: return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    si = np.argsort(tr["date"].values)
    Xs,ys,ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _,gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(
        {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
         "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42},
        dtr, num_boost_round=100,
    )
    return cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)


def wf_lgb_rank(panel, feat_cols, cut):
    """LightGBM lambdarank — direct analog of XGB rank:pairwise."""
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100: return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    # lambdarank needs INT label (relevance score). Discretize float fwd_60d
    # to 5-bucket per-date rank (0..4).
    si = np.argsort(tr["date"].values)
    tr_sorted = tr.iloc[si].copy()
    Xs = Xtr_n[si]
    ytr_int = (tr_sorted.groupby("date")[LABEL].rank(pct=True) * 4.99).astype(int).clip(0, 4).values
    _, gsz = np.unique(tr_sorted["date"].values, return_counts=True)
    dataset = lgb.Dataset(Xs, label=ytr_int, group=gsz)
    booster = lgb.train(
        {"objective":"lambdarank","metric":"ndcg","learning_rate":0.05,
         "num_leaves":31,"min_data_in_leaf":50,
         "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42,
         "num_threads":10},
        dataset, num_boost_round=100,
    )
    return cs_ic(booster.predict(Xte_n), yte, te["date"].values)


def wf_lgb_reg(panel, feat_cols, cut):
    """LightGBM regression (MSE) — same training protocol as XGB."""
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100: return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    booster = lgb.train(
        {"objective":"regression","metric":"l2","learning_rate":0.05,
         "num_leaves":31,"min_data_in_leaf":50,
         "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42,
         "num_threads":10},
        lgb.Dataset(Xtr_n, label=ytr), num_boost_round=100,
    )
    return cs_ic(booster.predict(Xte_n), yte, te["date"].values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--out",   default="data/wf_lightgbm_paired.json")
    a = p.parse_args()

    log.info("Loading 166-feat panel...")
    panel = pd.read_parquet(a.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("rows=%d features=%d tickers=%d",
             len(panel), len(feat_cols), panel["ticker"].nunique())

    results = {"xgb": [], "lgb_rank": [], "lgb_reg": []}
    t0 = time.time()
    for ci, cut in enumerate(CUTS, 1):
        log.info("Cut %d/%d", ci, len(CUTS))
        for name, fn in [("xgb", wf_xgb), ("lgb_rank", wf_lgb_rank), ("lgb_reg", wf_lgb_reg)]:
            t = time.time()
            ic = fn(panel, feat_cols, cut)
            results[name].append(ic)
            log.info("  %-9s IC=%+.4f  (%.1fs)", name, ic, time.time()-t)

    log.info("\n══ AGGREGATE (7-cut, %.0fs total) ══", time.time()-t0)
    log.info("%-10s %8s %8s %5s %s",
             "model", "mean", "std", "pos", "per-cut")
    agg = {}
    for name, ics in results.items():
        valid = [x for x in ics if not np.isnan(x)]
        m = float(np.mean(valid)); s = float(np.std(valid))
        pos = sum(1 for x in valid if x>0)
        agg[name] = {"mean": m, "std": s, "pos": pos, "per_cut": valid}
        log.info("%-10s %+8.4f %8.4f %3d/%d  [%s]",
                 name, m, s, pos, len(valid),
                 ", ".join(f"{x:+.3f}" for x in valid))

    log.info("\n══ vs XGBoost baseline ══")
    for name in ("lgb_rank", "lgb_reg"):
        delta = agg[name]["mean"] - agg["xgb"]["mean"]
        log.info("  %s − xgb = %+.4f  → %s",
                 name, delta,
                 "PROMOTE-CANDIDATE" if delta > 0.005 else
                 "marginal" if delta > 0 else "NO-GO (worse)")

    Path(a.out).write_text(json.dumps(agg, indent=2))
    log.info("Saved → %s", a.out)


if __name__ == "__main__":
    main()
