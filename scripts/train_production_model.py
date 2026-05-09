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
    log.info("Loading R1K + 5-fund panel (already normalized: alpha158=zscore, fund=robust-zscore)...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    # Use ALL data with valid labels (everything except embargo + most recent 60d)
    train = panel.dropna(subset=[LABEL])
    log.info("Train rows: %d (panel total: %d), tickers: %d, dates: %s → %s",
             len(train), len(panel), train["ticker"].nunique(),
             train["date"].min().date(), train["date"].max().date())

    # Train directly on the already-normalized panel — no extra z-score
    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-5,5).values.astype(np.float64)
    Xtr_n = Xtr  # no additional normalization

    # Build the FULL inference normalization chain stored in artifact:
    # For each feature, (mean, std) such that (raw - mean) / std = normalized value
    # alpha158 cols: from build_alpha158_qlib panel z-score (data/alpha158_qlib_dataset.stats.json)
    # fund cols: from robust z-score recomputed on train period
    ps = json.loads(Path("data/alpha158_qlib_dataset.stats.json").read_text())
    alpha_norm = dict(zip(ps["feature_cols"], zip(ps["feature_means"], ps["feature_stds"])))

    fund_cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
    fund_raw = pd.read_parquet("data/sec_fundamentals_daily.parquet")
    fund_raw["date"] = pd.to_datetime(fund_raw["date"])
    train_dates = set(train["date"])
    fund_train_raw = fund_raw[fund_raw["date"].isin(train_dates)
                               & fund_raw["ticker"].isin(set(train["ticker"]))]
    log.info("Fund train rows for robust z-score recompute: %d", len(fund_train_raw))
    fund_norm = {}
    for c in fund_cols:
        col = fund_train_raw[c].dropna()
        med = float(col.median()) if len(col) else 0.0
        mad = float((col - med).abs().median()) if len(col) else 1.0
        scale = max(mad * 1.4826, 1e-9)
        fund_norm[c] = (med, scale)

    feat_means, feat_stds, feat_norm_kind = [], [], []
    for c in feat_cols:
        if c in alpha_norm:
            m, s = alpha_norm[c]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("global_z")
        elif c in fund_norm:
            m, s = fund_norm[c]
            feat_means.append(m); feat_stds.append(s); feat_norm_kind.append("robust_z")
        else:
            feat_means.append(0.0); feat_stds.append(1.0); feat_norm_kind.append("identity")
    log.info("Normalization chain: %d global_z, %d robust_z, %d identity",
             feat_norm_kind.count("global_z"), feat_norm_kind.count("robust_z"),
             feat_norm_kind.count("identity"))
    mu = np.array(feat_means)
    sd = np.array(feat_stds)

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
    # Stamp fingerprint_fields as a DICT of (field → value) — matches the
    # legacy 21-feat artifact format that kernel/preflight expects so
    # P-CONFIG-FP and P-WATCHLIST can do meaningful diffs. Including
    # `watchlist` here lets P-WATCHLIST detect drift between training
    # universe and live config.
    fingerprint_fields = {
        "watchlist":          sorted(train["ticker"].unique().tolist()),
        "feature_cols":       feat_cols,
        "objective":          PARAMS.get("objective"),
        "label_col":          LABEL,
        "lookahead_days":     60,
        "asset_embeddings":   False,
        "training_resolution": "daily",
        "hourly_enabled":     False,
        "minute_enabled":     False,
        "params":             PARAMS,
    }
    artifact = {
        "version": 3,   # bump: fingerprint_fields format change (list → dict)
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
            "alpha158 + SEC fund (5) + PEAD (3, E47 promoted 2026-05-08) on R1K "
            "291 tickers, fwd_60d label. PEAD real_signal lift +0.022 over "
            "alpha158+5fund baseline (paired §5.2 sanity passed)."
        ),
        "config_fingerprint_fields": fingerprint_fields,
    }
    fp = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    artifact["config_fingerprint"] = fp

    out = REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
    out.write_text(json.dumps(artifact))
    log.info("Saved artifact: %s  (size=%.1f MB)", out, out.stat().st_size / 1e6)
    log.info("Fingerprint: %s", fp)
    log.info("Feature cols (n=%d): %s ... %s", len(feat_cols), feat_cols[:3], feat_cols[-3:])


if __name__ == "__main__":
    main()
