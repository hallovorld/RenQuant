#!/usr/bin/env python
"""§5.2 sanity battery on a candidate panel, paired against baseline.

For each panel, runs:
  1. A/A (3 seeds): IC reproducibility
  2. Label-shuffle (per-date, 1 seed): pure-noise floor
  3. Time-shift +60d (1 seed): regime-persistence artifact

Reports per-panel real signal = mean_IC − shuffled_IC (after E40
analysis: shuffled IC = stock-type-identification artifact, not noise).

Promotion test (paired): is regime real_signal > baseline real_signal?

Usage:
    python scripts/wf_sanity_paired.py
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sanity-paired")

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
    ics = [spearmanr(g["p"],g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def run_wf(panel, feat_cols, *, shift_days=0, shuffle=False, seed=42):
    params = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,
              "min_child_weight":50,"subsample":0.7,"colsample_bytree":0.7,
              "nthread":10,"verbosity":0,"seed":seed}
    rng = np.random.default_rng(seed)
    p = panel
    if shift_days:
        p = panel.copy().sort_values(["ticker","date"]).reset_index(drop=True)
        p[LABEL] = p.groupby("ticker")[LABEL].shift(-shift_days)
        p = p.dropna(subset=[LABEL])
    ics = []
    for cut in CUTS:
        tr_s,tr_e,te_s,te_e = cut
        tr = p[(p["date"]>=tr_s)&(p["date"]<=tr_e)].dropna(subset=[LABEL])
        te = p[(p["date"]>=te_s)&(p["date"]<=te_e)].dropna(subset=[LABEL])
        if len(tr)<1000 or len(te)<100:
            ics.append(np.nan); continue
        Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
        ytr = tr[LABEL].clip(-5,5).values.astype(np.float64).copy()
        if shuffle:
            tr_dates = tr["date"].values
            for d in np.unique(tr_dates):
                idx = np.where(tr_dates == d)[0]
                ytr[idx] = rng.permutation(ytr[idx])
        Xte = te[feat_cols].fillna(0).values.astype(np.float64)
        yte = te[LABEL].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
        Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
        si = np.argsort(tr["date"].values)
        Xs, ys, ds = Xtr_n[si], ytr[si], tr["date"].values[si]
        _, gsz = np.unique(ds, return_counts=True)
        dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
        booster = xgb.train(params, dtr, num_boost_round=100)
        ics.append(cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values))
    return ics


def battery(panel_path, label):
    log.info("══ %s ══", label)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("  rows=%d features=%d", len(panel), len(feat_cols))

    out = {}
    # 1. A/A (3 seeds)
    aa = []
    for s in (42, 43, 44):
        t0 = time.time()
        ics = run_wf(panel, feat_cols, seed=s)
        mean = np.nanmean(ics)
        aa.append(mean)
        log.info("  A/A seed=%d  mean=%+.4f  (%.1fs)", s, mean, time.time()-t0)
    out["aa_mean"]   = float(np.mean(aa))
    out["aa_std"]    = float(np.std(aa))
    out["aa_seeds"]  = aa

    # 2. Label-shuffle (1 seed for speed)
    t0 = time.time()
    sh = run_wf(panel, feat_cols, shuffle=True, seed=42)
    out["shuffle_ic"] = float(np.nanmean(sh))
    log.info("  shuffle    mean=%+.4f  (%.1fs)  per-cut=[%s]",
             out["shuffle_ic"], time.time()-t0,
             ", ".join(f"{x:+.3f}" if not np.isnan(x) else "NA" for x in sh))

    # 3. Time-shift +60d
    t0 = time.time()
    ts = run_wf(panel, feat_cols, shift_days=60, seed=42)
    out["timeshift_ic"] = float(np.nanmean(ts))
    log.info("  shift+60d  mean=%+.4f  (%.1fs)  per-cut=[%s]",
             out["timeshift_ic"], time.time()-t0,
             ", ".join(f"{x:+.3f}" if not np.isnan(x) else "NA" for x in ts))

    out["real_signal"] = out["aa_mean"] - out["shuffle_ic"]
    log.info("  REAL SIGNAL = aa(%+.4f) − shuffle(%+.4f) = %+.4f",
             out["aa_mean"], out["shuffle_ic"], out["real_signal"])
    return out


def main():
    base = battery("data/alpha158_291_fundamental_dataset.parquet",
                   "BASELINE alpha158+5fund")
    rgm  = battery("data/alpha158_291_fund_regime_dataset.parquet",
                   "CANDIDATE alpha158+5fund+3regime")

    log.info("\n══ PAIRED VERDICT ══")
    log.info("  metric              baseline    regime      Δ")
    log.info("  A/A mean IC         %+.4f     %+.4f    %+.4f",
             base["aa_mean"], rgm["aa_mean"], rgm["aa_mean"]-base["aa_mean"])
    log.info("  A/A std (3 seeds)   %.4f      %.4f     %+.4f",
             base["aa_std"],  rgm["aa_std"],  rgm["aa_std"]-base["aa_std"])
    log.info("  shuffle IC          %+.4f     %+.4f    %+.4f",
             base["shuffle_ic"], rgm["shuffle_ic"], rgm["shuffle_ic"]-base["shuffle_ic"])
    log.info("  time-shift +60d IC  %+.4f     %+.4f    %+.4f",
             base["timeshift_ic"], rgm["timeshift_ic"], rgm["timeshift_ic"]-base["timeshift_ic"])
    log.info("  REAL SIGNAL         %+.4f     %+.4f    %+.4f",
             base["real_signal"], rgm["real_signal"], rgm["real_signal"]-base["real_signal"])
    if rgm["real_signal"] > base["real_signal"]:
        log.info("  ✓ regime adds %+.4f real signal",
                 rgm["real_signal"]-base["real_signal"])
    else:
        log.info("  ✗ regime does NOT add real signal")

    Path("data/sanity_paired_baseline_vs_regime.json").write_text(
        json.dumps({"baseline":base, "regime":rgm}, indent=2))
    log.info("Saved → data/sanity_paired_baseline_vs_regime.json")


if __name__ == "__main__":
    main()
