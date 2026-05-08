#!/usr/bin/env python
"""Multi-horizon ensemble: combine fwd_5d, fwd_20d, fwd_60d predictions.

Per DeMiguel et al. (2009): equal-weight or 1/IC-weighted ensemble often
beats single best in OOS due to noise diversification.

Tests:
  - Each individual horizon (XGB d=5 e=0.05)
  - Equal-weight ensemble of all 3
  - 1/std-weighted (more weight to stable predictions)
  - Inverse-IC weighted (less weight to high-variance predictions)
"""
from __future__ import annotations
import logging, json
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("multi-horizon")

CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":8,"verbosity":0,"seed":42}
N_ROUNDS = 100


def cs_rank_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"],g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def train_predict(panel, feat_cols, label, cut):
    tr_s, tr_e, te_s, te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[label])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[label])
    if len(tr)<1000 or len(te)<100: return None, None
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[label].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    sort_idx = np.argsort(tr["date"].values)
    Xs, ys, ds = Xtr_n[sort_idx], ytr[sort_idx], tr["date"].values[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    return booster.predict(xgb.DMatrix(Xte_n)), te


def main():
    log.info("Loading R1K + fund panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    LABELS = ["fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"]

    # Per-cut: train each horizon, predict, compute IC, then ensemble
    log.info("Running 7-cut WF for each horizon + ensembles...")
    cut_records = []
    horizon_ics = {h: [] for h in LABELS}
    eq_ens_ics = []
    inv_std_ens_ics = []
    inv_var_ens_ics = []

    for i, cut in enumerate(CUTS, 1):
        log.info("\nCut %d", i)
        preds_per_horizon = {}  # h -> dict {(date,ticker): pred}
        ics_per_horizon = {}
        te_label_60d = None

        for h in LABELS:
            preds, te = train_predict(panel, feat_cols, h, cut)
            if preds is None: continue
            te = te.copy(); te["pred"] = preds
            ic = cs_rank_ic(preds, te[h].values, te["date"].values)
            horizon_ics[h].append(ic)
            ics_per_horizon[h] = ic
            preds_per_horizon[h] = te[["date","ticker","pred", h]].rename(columns={"pred": f"pred_{h}"})
            log.info("  %-15s  IC=%+.4f", h, ic)

        if len(preds_per_horizon) < 3: continue

        # Build ensemble: rank-normalize each horizon's pred per date, then average
        # Use the longest horizon's test labels (fwd_60d) as the evaluation target
        merged = preds_per_horizon[LABELS[0]][["date","ticker"]].copy()
        for h in LABELS:
            merged = merged.merge(preds_per_horizon[h][["date","ticker",f"pred_{h}"]],
                                   on=["date","ticker"])
        merged = merged.merge(panel[["date","ticker","fwd_60d_excess"]],
                               on=["date","ticker"]).dropna(subset=["fwd_60d_excess"])

        # Cross-sectional rank per date for each horizon's prediction
        for h in LABELS:
            merged[f"rank_{h}"] = merged.groupby("date")[f"pred_{h}"].rank(pct=True)

        # Equal-weight ensemble of ranks
        merged["eq_ens"] = merged[[f"rank_{h}" for h in LABELS]].mean(axis=1)
        eq_ic = cs_rank_ic(merged["eq_ens"].values, merged["fwd_60d_excess"].values,
                            merged["date"].values)
        eq_ens_ics.append(eq_ic)

        # 1/std weighted (per-horizon std from training data — use this cut's full panel)
        weights_inv_std = []
        for h in LABELS:
            train_dates = panel[(panel["date"] >= cut[0]) & (panel["date"] <= cut[1])]
            label_std_per_date = train_dates.groupby("date")[h].std().mean()
            weights_inv_std.append(1.0 / max(label_std_per_date, 1e-6))
        weights_inv_std = np.array(weights_inv_std)
        weights_inv_std /= weights_inv_std.sum()
        merged["inv_std_ens"] = sum(w * merged[f"rank_{h}"]
                                     for w, h in zip(weights_inv_std, LABELS))
        inv_std_ic = cs_rank_ic(merged["inv_std_ens"].values, merged["fwd_60d_excess"].values,
                                  merged["date"].values)
        inv_std_ens_ics.append(inv_std_ic)

        # Inverse variance weight (higher IC variance → lower weight)
        # Use the per-cut historical std of IC (proxy: prior cuts' std)
        if i >= 2:
            weights_inv_var = []
            for h in LABELS:
                hist_std = np.std(horizon_ics[h][:-1]) if len(horizon_ics[h]) > 1 else 0.05
                weights_inv_var.append(1.0 / max(hist_std**2, 1e-4))
            weights_inv_var = np.array(weights_inv_var)
            weights_inv_var /= weights_inv_var.sum()
            merged["inv_var_ens"] = sum(w * merged[f"rank_{h}"]
                                         for w, h in zip(weights_inv_var, LABELS))
            inv_var_ic = cs_rank_ic(merged["inv_var_ens"].values, merged["fwd_60d_excess"].values,
                                      merged["date"].values)
            inv_var_ens_ics.append(inv_var_ic)
        else:
            inv_var_ens_ics.append(np.nan)

        log.info("  EQ ensemble    IC=%+.4f", eq_ic)
        log.info("  1/std ensemble IC=%+.4f  (weights=%s)", inv_std_ic,
                 {h: round(w,2) for h,w in zip(LABELS, weights_inv_std)})
        log.info("  inv_var ens    IC=%+.4f", inv_var_ens_ics[-1])

    log.info("\n══ AGGREGATE (7-cut mean ± std, evaluated on fwd_60d test) ══")
    for h in LABELS:
        clean = [x for x in horizon_ics[h] if not np.isnan(x)]
        log.info("%-20s mean=%+.4f std=%.4f pos=%d/%d",
                 h, np.mean(clean), np.std(clean), sum(1 for x in clean if x>0), len(clean))
    for name, ics in [("EQ ensemble", eq_ens_ics),
                       ("1/std ensemble", inv_std_ens_ics),
                       ("inv_var ensemble", inv_var_ens_ics)]:
        clean = [x for x in ics if not np.isnan(x)]
        if clean:
            log.info("%-20s mean=%+.4f std=%.4f pos=%d/%d",
                     name, np.mean(clean), np.std(clean), sum(1 for x in clean if x>0), len(clean))
    log.info("\nBaseline (fwd_60d alone): mean=+0.066 std=0.072")


if __name__ == "__main__":
    main()
