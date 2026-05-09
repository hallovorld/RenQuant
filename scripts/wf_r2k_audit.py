#!/usr/bin/env python
"""B1 R2K NO-GO audit (CLAUDE.md §2b) — distinguish bug from signal weakness.

The R2K result IC=+0.018 (vs prod +0.067) is a 75% drop. Before
accepting the NO-GO, run focused differential tests:

  (a) Alpha158-only — drop the 5 fund features. If IC recovers,
      the fund-feature 53%-zero-variance problem is the issue.
  (b) Liquid subset — top 300 by row count (proxy for liquidity).
      If IC recovers, small-cap survivorship/data-quality is the issue.
  (c) Compare against TYPED MATCHING WL — random 1640-ticker sample
      from PROD universe wouldn't be a thing (we only have 291).
      Skip.

Results saved to data/wf_r2k_audit.json.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("r2k-audit")

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
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def wf_xgb(panel, feat_cols, cut):
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100:
        return np.nan
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
    booster = xgb.train(PARAMS, dtr, num_boost_round=100)
    return cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)


def run_battery(panel, feat_cols, label):
    log.info("=== %s ===", label)
    log.info("  rows=%d  features=%d  tickers=%d",
             len(panel), len(feat_cols), panel["ticker"].nunique())
    t0 = time.time()
    ics = []
    for i, cut in enumerate(CUTS, 1):
        ic = wf_xgb(panel, feat_cols, cut)
        ics.append(ic)
        log.info("  cut %d  IC=%+.4f", i, ic)
    valid = [x for x in ics if not np.isnan(x)]
    mean = np.mean(valid); std = np.std(valid); pos = sum(1 for x in valid if x>0)
    log.info("  AGGREGATE  mean=%+.4f std=%.4f pos=%d/%d  (%.0fs)",
             mean, std, pos, len(valid), time.time()-t0)
    return {"label": label, "mean": mean, "std": std, "pos": f"{pos}/{len(valid)}",
            "per_cut": [float(x) if not np.isnan(x) else None for x in ics]}


def main():
    log.info("Loading R2K panel...")
    panel = pd.read_parquet("data/alpha158_r2k_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    fund_cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
    feat_all  = [c for c in panel.columns if c not in excl]
    feat_alpha = [c for c in feat_all if c not in fund_cols]
    log.info("All features: %d  Alpha158-only: %d", len(feat_all), len(feat_alpha))

    results = []

    # (a) alpha158-only on full R2K
    results.append(run_battery(panel, feat_alpha, "R2K_alpha158_only"))

    # (b) full features on top-quality subset (most rows = longest history = most liquid)
    rc = panel.groupby("ticker").size()
    top_300 = rc.sort_values(ascending=False).head(300).index
    sub300 = panel[panel["ticker"].isin(top_300)]
    log.info("\nTop-300 subset: %d/%d tickers, rows=%d",
             len(top_300), panel["ticker"].nunique(), len(sub300))
    results.append(run_battery(sub300, feat_all, "R2K_top300_full_features"))

    # (c) BONUS: top-300 + alpha158-only (combine both fixes)
    results.append(run_battery(sub300, feat_alpha, "R2K_top300_alpha158_only"))

    log.info("\n══ AUDIT SUMMARY ══")
    log.info("  Production baseline (291, alpha158+fund):  XGB mean IC = +0.067")
    log.info("  R2K full (1640, alpha158+fund):            XGB mean IC = +0.018  (initial NO-GO)")
    for r in results:
        log.info("  %s:  XGB mean IC = %+.4f  (pos=%s)",
                 r["label"], r["mean"], r["pos"])

    out = Path("data/wf_r2k_audit.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
