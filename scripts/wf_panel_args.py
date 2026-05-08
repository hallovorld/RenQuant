#!/usr/bin/env python
"""Parameterized 7-cut walk-forward IC eval. Pass --panel + --label.

Mirrors scripts/walk_forward_panel.py logic exactly (same cuts, same
xgb params) but reads the panel/label from CLI so we can A/B different
feature sets against the same baseline (panel-ltr.alpha158_fund.json
production: alpha158 + 5-fund, fwd_60d, mean IC +0.066 std 0.072).

Usage:
    python scripts/wf_panel_args.py \
        --panel data/alpha158_291_fund_ext_dataset.parquet \
        --label fwd_60d_excess \
        --out data/wf_fund_ext.json
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-panel-args")

CUTS = [
    ("2016-01-01","2018-12-31","2019-02-01","2019-12-31"),
    ("2017-01-01","2019-12-31","2020-02-01","2020-12-31"),
    ("2018-01-01","2020-12-31","2021-02-01","2021-12-31"),
    ("2019-01-01","2021-12-31","2022-02-01","2022-12-31"),
    ("2020-01-01","2022-12-31","2023-02-01","2023-12-31"),
    ("2021-01-01","2023-12-31","2024-02-01","2024-12-31"),
    ("2022-01-01","2024-12-31","2025-02-01","2025-12-31"),
]
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}


def cs_rank_ic(pred, actual, dates):
    df = pd.DataFrame({"p":pred,"y":actual,"date":dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return (float(np.mean(ics)) if ics else np.nan,
            float(np.median(ics)) if ics else np.nan, len(ics))


def evaluate_cut(panel, feat_cols, label, cut):
    tr_s, tr_e, te_s, te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[label])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[label])
    if len(tr)<1000 or len(te)<100:
        return {"cut":cut,"error":f"insufficient train={len(tr)} test={len(te)}"}
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[label].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[label].values; te_d = te["date"].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)

    out = {"cut":cut,"train_size":len(tr),"test_size":len(te)}
    p_ols = LinearRegression().fit(Xtr_n, ytr).predict(Xte_n)
    out["ols"] = dict(zip(("ic_mean","ic_median","n_dates"), cs_rank_ic(p_ols, yte, te_d)))
    p_rdg = Ridge(alpha=1.0, solver="lsqr").fit(Xtr_n, ytr).predict(Xte_n)
    out["ridge"] = dict(zip(("ic_mean","ic_median","n_dates"), cs_rank_ic(p_rdg, yte, te_d)))
    si = np.argsort(tr["date"].values)
    Xs, ys, ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _, gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=100)
    p_xgb = booster.predict(xgb.DMatrix(Xte_n))
    out["xgb"] = dict(zip(("ic_mean","ic_median","n_dates"), cs_rank_ic(p_xgb, yte, te_d)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--out",   required=True)
    a = p.parse_args()

    log.info("Panel: %s  label=%s", a.panel, a.label)
    panel = pd.read_parquet(a.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("rows=%d features=%d tickers=%d dates %s→%s",
             len(panel), len(feat_cols), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    all_r = []
    t0 = time.time()
    for i, cut in enumerate(CUTS, 1):
        log.info("Cut %d/%d  train=[%s..%s] test=[%s..%s]", i, len(CUTS), *cut)
        r = evaluate_cut(panel, feat_cols, a.label, cut)
        all_r.append(r)
        if "error" in r:
            log.warning("  %s", r["error"]); continue
        for m in ("ols","ridge","xgb"):
            log.info("  %-5s ic_mean=%+.4f median=%+.4f n=%d",
                     m, r[m]["ic_mean"], r[m]["ic_median"], r[m]["n_dates"])

    elapsed = time.time() - t0
    agg = {}
    for m in ("ols","ridge","xgb"):
        ics = [r[m]["ic_mean"] for r in all_r if m in r and not np.isnan(r[m]["ic_mean"])]
        if not ics: continue
        agg[m] = {"mean": float(np.mean(ics)), "std": float(np.std(ics)),
                  "n_pos": sum(1 for x in ics if x>0), "n": len(ics),
                  "per_cut": ics}
    log.info("\n══ AGGREGATE (7-cut, %.0fs) ══", elapsed)
    for m, s in agg.items():
        log.info("%-5s mean=%+.4f std=%.4f pos=%d/%d",
                 m, s["mean"], s["std"], s["n_pos"], s["n"])

    Path(a.out).write_text(json.dumps({"panel":a.panel,"label":a.label,
                                        "elapsed_sec":elapsed,
                                        "per_cut":all_r,"aggregate":agg}, indent=2, default=str))
    log.info("Saved → %s", a.out)


if __name__ == "__main__":
    main()
