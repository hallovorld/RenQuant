#!/usr/bin/env python3
"""XGB baseline for a walk-forward cut — fair comparison vs HF PatchTST.

Per CLAUDE.md §5.11 range-finding: 30-min smoke comparing XGB vs HF on
same train/val periods. Cannot use prod XGB pool_ic (different val period
+ different methodology).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.walk_forward_splits import build_default_cuts
from kernel.hmm_regime_labels import (compute_hmm_regime_labels,
                                        per_hmm_regime_ic, bull_regime_ic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--cut", required=True)
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--n-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    panel = pd.read_parquet(args.dataset)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[args.label])

    cut = next(c for c in build_default_cuts() if c.name == args.cut)
    train = panel[panel["date"] < cut.val_start].copy()
    val = panel[(panel["date"] >= cut.val_start) & (panel["date"] < cut.val_end)].copy()

    feat_cols = [c for c in panel.columns
                 if c not in {"date", "ticker", "split_label",
                              "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
                 and panel[c].dtype.kind in "fiub"]

    print(f"cut={args.cut} train={len(train)} val={len(val)} n_feat={len(feat_cols)}")

    # Group queries per day for rank:pairwise
    train_groups = train.groupby("date").size().values
    val_groups = val.groupby("date").size().values

    Xt = train[feat_cols].fillna(0.0).values
    yt = train[args.label].values
    Xv = val[feat_cols].fillna(0.0).values
    yv = val[args.label].values

    # XGB rank:pairwise — same as prod config
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "rmse",   # NDCG default requires non-neg int labels; use rmse for float labels
        "eta": 0.05, "max_depth": 5, "min_child_weight": 50,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "nthread": 8, "verbosity": 0, "seed": args.seed,
    }
    dtrain = xgb.DMatrix(Xt, label=yt)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(Xv, label=yv)
    dval.set_group(val_groups)

    booster = xgb.train(params, dtrain, num_boost_round=args.n_rounds,
                         evals=[(dval, "val")], verbose_eval=False)
    preds = booster.predict(dval)

    # Per-day Spearman IC (pool)
    from scipy.stats import spearmanr
    val = val.reset_index(drop=True).copy()
    val["pred"] = preds
    pool_ic = []
    for d, g in val.groupby("date"):
        if len(g) < 5: continue
        r, _ = spearmanr(g["pred"], g[args.label])
        if not np.isnan(r): pool_ic.append(r)
    pool_mean = float(np.mean(pool_ic))
    print(f"XGB pool_ic = {pool_mean:+.4f}  (n_days={len(pool_ic)})")

    # Per-HMM-regime IC
    hmm = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")
    preds_df = pd.DataFrame({"date": val["date"], "pred": preds, "label": val[args.label]})
    per_regime = per_hmm_regime_ic(preds_df, hmm, min_samples_per_day=5, min_days_per_regime=5)
    print(f"per_regime: {json.dumps({k: round(v, 4) for k, v in per_regime.items()})}")
    bull_ic = bull_regime_ic(per_regime)
    print(f"bull_regime_ic = {bull_ic:+.4f}")

    # MLflow logging (per policy 2026-05-19: all experiments → DB)
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    exp_name = "renquant_104_xgb_baseline"
    exp = mlflow.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(exp_name)
    with mlflow.start_run(experiment_id=exp_id,
                          run_name=f"xgb_{args.cut}_seed{args.seed}"):
        mlflow.log_params({
            "cut": args.cut, "seed": args.seed, "label": args.label,
            "n_rounds": args.n_rounds, "arch": "xgb_rank_pairwise",
            **{f"xgb_{k}": str(v) for k, v in params.items()},
        })
        metrics = {"pool_ic": pool_mean, "bull_regime_ic": float(bull_ic)
                   if not np.isnan(bull_ic) else 0.0}
        for regime, ic in per_regime.items():
            metrics[f"per_regime_ic_{regime}"] = float(ic)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "experiment_phase": "phase0_range_finding",
            "model_kind": "xgb",
            "fair_comparison_to": "hf_patchtst_doe",
        })
        # Dump val preds + log as artifact for cross-comparison with HF
        preds_dump = Path("/tmp") / f"xgb_{args.cut}_seed{args.seed}_val_preds.parquet"
        preds_df.to_parquet(preds_dump, index=False)
        mlflow.log_artifact(str(preds_dump))
    print(f"  → MLflow run logged to {exp_name}")


if __name__ == "__main__":
    main()
