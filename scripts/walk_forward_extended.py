#!/usr/bin/env python
"""Extended walk-forward IC sweep — multiple labels, models, hyperparameters.

Runs 7-cut walk-forward across all combinations of:
  - Labels: fwd_5d, fwd_20d, fwd_60d
  - Models: OLS, Ridge, XGB(depth=3,5,7) × eta∈{0.03,0.05,0.1}
  - Feature sets: alpha158, alpha158+fund, alpha158+fund_extended

Output: data/walk_forward_extended_results.json with per-cut + aggregate IC.
Also prints summary table sorted by mean IC.
"""
from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-ext")

REPO = Path(__file__).resolve().parent.parent

# Same 7 cuts as walk_forward_panel.py
CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]


def cs_rank_ic(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> float:
    df = pd.DataFrame({"p": pred, "y": actual, "date": dates})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["p"], g["y"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def evaluate_one(panel: pd.DataFrame, feat_cols: list[str], label: str,
                 cut: tuple, model_spec: dict) -> float:
    tr_start, tr_end, te_start, te_end = cut
    train = panel[(panel["date"] >= tr_start) & (panel["date"] <= tr_end)].dropna(subset=[label])
    test  = panel[(panel["date"] >= te_start) & (panel["date"] <= te_end)].dropna(subset=[label])
    if len(train) < 1000 or len(test) < 100:
        return np.nan

    X_tr = train[feat_cols].fillna(0).values.astype(np.float64)
    y_tr = train[label].clip(-5, 5).values.astype(np.float64)
    X_te = test[feat_cols].fillna(0).values.astype(np.float64)
    y_te = test[label].values
    te_dates = test["date"].values

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    X_tr_n = ((X_tr - mu) / sd).clip(-5, 5)
    X_te_n = ((X_te - mu) / sd).clip(-5, 5)

    kind = model_spec["kind"]
    if kind == "ols":
        m = LinearRegression().fit(X_tr_n, y_tr)
        return cs_rank_ic(m.predict(X_te_n), y_te, te_dates)
    if kind == "ridge":
        m = Ridge(alpha=model_spec.get("alpha", 1.0), solver="lsqr").fit(X_tr_n, y_tr)
        return cs_rank_ic(m.predict(X_te_n), y_te, te_dates)
    if kind == "xgb":
        # Need group sizes per date for rank:pairwise
        train_dates = train["date"].values
        sort_idx = np.argsort(train_dates)
        Xs = X_tr_n[sort_idx]
        ys = y_tr[sort_idx]
        ds = train_dates[sort_idx]
        _, gsz = np.unique(ds, return_counts=True)
        dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
        dte = xgb.DMatrix(X_te_n)
        params = {"objective": "rank:pairwise",
                  "eta": model_spec["eta"],
                  "max_depth": model_spec["max_depth"],
                  "min_child_weight": model_spec.get("min_child_weight", 50),
                  "subsample": 0.7, "colsample_bytree": 0.7,
                  "nthread": 8, "verbosity": 0}
        n_rounds = model_spec.get("n_rounds", 100)
        booster = xgb.train(params, dtr, num_boost_round=n_rounds)
        return cs_rank_ic(booster.predict(dte), y_te, te_dates)
    raise ValueError(f"unknown kind: {kind}")


def add_extended_fundamentals(panel: pd.DataFrame, fund_raw: pd.DataFrame) -> pd.DataFrame:
    """Add 5 derived ratios from raw SEC fundamentals.

    Inputs (already in panel from prior merge — but we re-derive from raw):
      NetIncomeLoss, GrossProfit, Revenues, Assets, StockholdersEquity

    Derived (point-in-time per ticker, then merged to daily):
      asset_turnover  = Revenue / Assets
      profit_margin   = NetIncome / Revenue
      operating_lev   = GrossProfit / Revenue
      debt_to_assets  = (Assets - StockholdersEquity) / Assets   # = Liabilities/Assets
      revenue_growth  = Revenue.pct_change(periods=4)            # YoY
    """
    # Raw fundamentals are in /tmp/sec_fundamental_raw_panel via prior pipeline,
    # but easier: re-fetch from SEC artifacts. For now, use the daily aggregated
    # fundamentals dataset and derive ratios.
    # (Skip if fund_raw doesn't have required cols)
    return panel  # no-op for now; derived features will come from a separate prep


def main():
    log.info("Loading 291-ticker + fundamentals panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("Panel: %d rows × %d features, %d tickers",
             len(panel), len(feat_cols), panel["ticker"].nunique())

    # Define experiment grid
    labels = ["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]
    models = [
        {"name": "OLS",        "kind": "ols"},
        {"name": "Ridge_a1",   "kind": "ridge", "alpha": 1.0},
        {"name": "Ridge_a10",  "kind": "ridge", "alpha": 10.0},
        {"name": "XGB_d3_e03", "kind": "xgb", "max_depth": 3, "eta": 0.03, "n_rounds": 200},
        {"name": "XGB_d5_e05", "kind": "xgb", "max_depth": 5, "eta": 0.05, "n_rounds": 100},
        {"name": "XGB_d7_e10", "kind": "xgb", "max_depth": 7, "eta": 0.10, "n_rounds": 50},
    ]

    all_results = []
    for label, model in product(labels, models):
        per_cut = []
        for i, cut in enumerate(CUTS, 1):
            ic = evaluate_one(panel, feat_cols, label, cut, model)
            per_cut.append(ic)
        per_cut_clean = [x for x in per_cut if not np.isnan(x)]
        if not per_cut_clean:
            continue
        result = {
            "label": label,
            "model": model["name"],
            "model_spec": model,
            "per_cut": per_cut,
            "ic_mean": float(np.mean(per_cut_clean)),
            "ic_std":  float(np.std(per_cut_clean)),
            "ic_min":  float(min(per_cut_clean)),
            "ic_max":  float(max(per_cut_clean)),
            "n_pos":   sum(1 for x in per_cut_clean if x > 0),
            "n_cuts":  len(per_cut_clean),
        }
        all_results.append(result)
        log.info("%-12s %-15s  mean=%+.4f std=%.4f  pos=%d/%d  cuts=[%s]",
                 label.replace("_excess",""), model["name"],
                 result["ic_mean"], result["ic_std"],
                 result["n_pos"], result["n_cuts"],
                 ", ".join(f"{x:+.3f}" if not np.isnan(x) else "  NA " for x in per_cut))

    # Sort by mean IC, print top results
    log.info("\n══ TOP 10 by mean IC ══")
    sorted_results = sorted(all_results, key=lambda r: -r["ic_mean"])
    for r in sorted_results[:10]:
        log.info("  %-15s %-12s  mean=%+.4f  std=%.4f  pos=%d/%d",
                 r["label"].replace("_excess",""), r["model"],
                 r["ic_mean"], r["ic_std"], r["n_pos"], r["n_cuts"])

    out = REPO / "data" / "walk_forward_extended_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()
