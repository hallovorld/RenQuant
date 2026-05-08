#!/usr/bin/env python
"""Train final production model: R1K + alpha158 + 5-fund + XGB d=5 e=0.05 fwd_60d.

Trained on ALL train data up through latest valid label date (current_date - 60d).
Saves artifact in PanelScorer format (kind=panel_ltr_xgboost) so existing
inference pipeline can load it.

Output: data/panel-ltr-prod-alpha158-fund-fwd60d.json
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train-prod")

REPO = Path(__file__).resolve().parent.parent
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":8,"verbosity":0,"seed":42}
N_ROUNDS = 100
LABEL = "fwd_60d_excess"


def main():
    log.info("Loading R1K + 5-fund panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    # Use ALL data with valid labels (everything except embargo + most recent 60d)
    train = panel.dropna(subset=[LABEL])
    log.info("Train rows: %d (panel total: %d), tickers: %d, dates: %s → %s",
             len(train), len(panel), train["ticker"].nunique(),
             train["date"].min().date(), train["date"].max().date())

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-5,5).values.astype(np.float64)

    # Save normalization stats so inference can match
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Xtr_n = ((Xtr - mu) / sd).clip(-5, 5)

    sort_idx = np.argsort(train["date"].values)
    Xs, ys, ds = Xtr_n[sort_idx], ytr[sort_idx], train["date"].values[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)

    log.info("Training XGB rank:pairwise (params=%s)...", PARAMS)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)

    # Compute training IC for sanity
    train_pred = booster.predict(xgb.DMatrix(Xtr_n))
    from scipy.stats import spearmanr
    train_check = train.copy()
    train_check["pred"] = train_pred
    train_ics = []
    for _, g in train_check.groupby("date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["pred"], g[LABEL])
        if not np.isnan(ic): train_ics.append(ic)
    train_ic_mean = float(np.mean(train_ics))
    log.info("In-sample train IC: %+.4f (sanity check, not OOS)", train_ic_mean)

    # Save artifact
    raw_json = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    artifact = {
        "version": 2,
        "kind": "panel_ltr_xgboost",
        "trained_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "feature_cols": feat_cols,
        "feature_means": mu.tolist(),
        "feature_stds":  sd.tolist(),
        "params": PARAMS,
        "best_iter": N_ROUNDS,
        "booster_raw_json": raw_json,
        "panel_shape": list(train.shape),
        "label_col": LABEL,
        "lookahead_days": 60,
        "training_notes": (
            "alpha158 + SEC fund (5 features) on R1K 291 tickers, fwd_60d label. "
            "WF baseline IC=+0.066 std=0.072 6/7 cuts positive. "
            "Sanity-adjusted real signal ~+0.041 (after stock-type residual subtracted). "
            "Portfolio sim: Long-only top decile Sharpe=1.06, MaxDD=-42%."
        ),
        "config_fingerprint_fields": ["feature_cols", "params", "label_col"],
    }
    fp = hashlib.sha256(json.dumps({k: artifact[k] for k in artifact["config_fingerprint_fields"]},
                                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    artifact["config_fingerprint"] = fp

    out = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
    out.write_text(json.dumps(artifact))
    log.info("Saved artifact: %s  (size=%.1f MB)", out, out.stat().st_size / 1e6)
    log.info("Fingerprint: %s", fp)
    log.info("Feature cols (n=%d): %s ... %s", len(feat_cols), feat_cols[:3], feat_cols[-3:])


if __name__ == "__main__":
    main()
