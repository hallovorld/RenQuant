#!/usr/bin/env python
"""Compare BEST baseline (fwd_60d, XGB d=5 e=0.05) WITH vs WITHOUT regime features.

Paired test on identical 7 cuts.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-regime")

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
PARAMS = {"objective": "rank:pairwise", "eta": 0.05, "max_depth": 5,
          "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7,
          "nthread": 8, "verbosity": 0}
N_ROUNDS = 100


def cs_rank_ic(pred, actual, dates):
    df = pd.DataFrame({"p": pred, "y": actual, "date": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("date") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def evaluate_cut(panel, feat_cols, cut):
    tr_start, tr_end, te_start, te_end = cut
    tr = panel[(panel["date"] >= tr_start) & (panel["date"] <= tr_end)].dropna(subset=[LABEL])
    te = panel[(panel["date"] >= te_start) & (panel["date"] <= te_end)].dropna(subset=[LABEL])
    if len(tr) < 1000 or len(te) < 100:
        return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    te_dates = te["date"].values

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Xtr_n = ((Xtr - mu) / sd).clip(-5, 5)
    Xte_n = ((Xte - mu) / sd).clip(-5, 5)

    sort_idx = np.argsort(tr["date"].values)
    Xs, ys, ds = Xtr_n[sort_idx], ytr[sort_idx], tr["date"].values[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    dte = xgb.DMatrix(Xte_n)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    return cs_rank_ic(booster.predict(dte), yte, te_dates)


def main():
    log.info("Loading panel WITH regime...")
    panel = pd.read_parquet("data/alpha158_291_fund_regime_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    all_cols    = [c for c in panel.columns if c not in excl]
    regime_cols = [c for c in all_cols if c.startswith("regime_p_")]
    base_cols   = [c for c in all_cols if c not in regime_cols]
    log.info("Total features: %d (base=%d, regime=%d)", len(all_cols), len(base_cols), len(regime_cols))

    log.info("\nRunning paired comparison: BASE vs BASE+REGIME")
    log.info("Model: XGB d=5 eta=0.05, label=%s", LABEL)
    log.info("%-3s %-25s %-10s %-10s %-10s", "Cut", "Period", "BASE", "+REGIME", "Δ")

    base_ics, regime_ics = [], []
    for i, cut in enumerate(CUTS, 1):
        ic_base   = evaluate_cut(panel, base_cols,   cut)
        ic_regime = evaluate_cut(panel, all_cols,    cut)
        delta = ic_regime - ic_base
        base_ics.append(ic_base); regime_ics.append(ic_regime)
        log.info("%-3d %-25s %+.4f    %+.4f    %+.4f",
                 i, f"{cut[2][:7]} → {cut[3][:7]}", ic_base, ic_regime, delta)

    log.info("")
    log.info("BASE     mean=%+.4f  std=%.4f", np.mean(base_ics), np.std(base_ics))
    log.info("+REGIME  mean=%+.4f  std=%.4f", np.mean(regime_ics), np.std(regime_ics))
    log.info("Δ mean=%+.4f  win_rate=%d/7",
             np.mean(regime_ics) - np.mean(base_ics),
             sum(1 for b, r in zip(base_ics, regime_ics) if r > b))


if __name__ == "__main__":
    main()
