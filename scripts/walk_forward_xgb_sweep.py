#!/usr/bin/env python
"""XGB hyperparameter sweep around current best (d=5, eta=0.05) on R1K + fund.

Grid (3^3 = 27 combos × 7 cuts = ~14 min total at 5s/cut):
  max_depth        ∈ {3, 5, 7}
  eta              ∈ {0.03, 0.05, 0.10}
  min_child_weight ∈ {10, 50, 100}
  num_rounds       = 100 (fixed; eta lower → more rounds equivalent)

Plus check num_rounds independently for the best (d, eta, mcw) triplet.
"""
from __future__ import annotations
import json, logging
from itertools import product
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("xgb-sweep")

CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]
LABEL = "fwd_60d_excess"


def cs_rank_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"],g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def run_wf_one(panel, feat_cols, params, n_rounds):
    ics = []
    for cut in CUTS:
        tr_s,tr_e,te_s,te_e = cut
        tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
        te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
        if len(tr)<1000 or len(te)<100: ics.append(np.nan); continue

        Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
        ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
        Xte = te[feat_cols].fillna(0).values.astype(np.float64)
        yte = te[LABEL].values

        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
        Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)

        sort_idx = np.argsort(tr["date"].values)
        Xs, ys, ds = Xtr_n[sort_idx], ytr[sort_idx], tr["date"].values[sort_idx]
        _, gsz = np.unique(ds, return_counts=True)
        dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
        booster = xgb.train(params, dtr, num_boost_round=n_rounds)
        ic = cs_rank_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)
        ics.append(ic)
    return ics


def main():
    log.info("Loading R1K + fund panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("Panel: %d rows × %d features, %d tickers",
             len(panel), len(feat_cols), panel["ticker"].nunique())

    base_params = {"objective":"rank:pairwise","subsample":0.7,"colsample_bytree":0.7,
                   "nthread":8,"verbosity":0}

    grid = list(product(
        [3, 5, 7],            # max_depth
        [0.03, 0.05, 0.10],   # eta
        [10, 50, 100],        # min_child_weight
    ))
    log.info("Grid: %d combos × 7 cuts", len(grid))

    results = []
    for i, (depth, eta, mcw) in enumerate(grid, 1):
        params = {**base_params, "max_depth": depth, "eta": eta, "min_child_weight": mcw}
        ics = run_wf_one(panel, feat_cols, params, n_rounds=100)
        clean = [x for x in ics if not np.isnan(x)]
        if not clean: continue
        mean_ic = float(np.mean(clean))
        std_ic  = float(np.std(clean))
        n_pos   = sum(1 for x in clean if x > 0)
        results.append({"depth":depth,"eta":eta,"mcw":mcw,"n_rounds":100,
                        "mean":mean_ic,"std":std_ic,"pos":n_pos,"per_cut":ics})
        log.info("[%d/%d] d=%d eta=%.2f mcw=%d  mean=%+.4f std=%.4f pos=%d/7",
                 i, len(grid), depth, eta, mcw, mean_ic, std_ic, n_pos)

    log.info("\n══ TOP 10 by mean IC ══")
    sorted_results = sorted(results, key=lambda r: -r["mean"])
    for r in sorted_results[:10]:
        log.info("  d=%d eta=%.2f mcw=%-3d  mean=%+.4f  std=%.4f  pos=%d/7  IR=%.2f",
                 r["depth"], r["eta"], r["mcw"], r["mean"], r["std"], r["pos"],
                 r["mean"]/max(r["std"],1e-9))

    log.info("\n══ TOP 10 by IR (mean/std) ══")
    sorted_ir = sorted(results, key=lambda r: -(r["mean"]/max(r["std"],1e-9)))
    for r in sorted_ir[:10]:
        log.info("  d=%d eta=%.2f mcw=%-3d  mean=%+.4f  std=%.4f  pos=%d/7  IR=%.2f",
                 r["depth"], r["eta"], r["mcw"], r["mean"], r["std"], r["pos"],
                 r["mean"]/max(r["std"],1e-9))

    json.dump(results, open("data/xgb_sweep_results.json","w"), indent=2)
    log.info("Saved: data/xgb_sweep_results.json")


if __name__ == "__main__":
    main()
