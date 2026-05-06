#!/usr/bin/env python
"""Qlib-pattern XGBoost baseline (MSE + CSZScoreNorm labels).

Production RenQuant XGB uses `rank:pairwise` on 21 features. This script
replicates Qlib's standard recipe instead: `reg:squarederror` (MSE) +
158 alpha158 features + cross-sectionally z-scored labels.

Hypothesis: feature breadth (158 vs 21) + Qlib's preprocessing recipe
matters more than the ranking objective.

Reference: Qlib uses LightGBM as gradient-boosted tree of choice (gbdt.py),
but production RenQuant standardized on XGBoost. We use XGBoost with the
same MSE-on-CSZScoreNorm pattern for apples-to-apples lift measurement.

Usage::

    python scripts/qlib_xgb_baseline.py --label fwd_60d_excess
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("qlib-xgb")


def per_day_ic(preds, labels, dates) -> tuple[float, float]:
    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 5:
            continue
        rho, _ = spearmanr(group["pred"], group["label"])
        if not np.isnan(rho):
            ics.append(rho)
    if not ics:
        return 0.0, 0.0
    return float(np.mean(ics)), float(np.median(ics))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "qlib_xgb.json"))
    # XGB params — start from production-104 panel-LTR config + Qlib LGB defaults
    p.add_argument("--num-boost-round", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.05,
                   help="learning_rate (panel-LTR uses 0.05)")
    p.add_argument("--max-depth", type=int, default=6,
                   help="(panel-LTR uses 6)")
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample-bytree", type=float, default=0.85)
    p.add_argument("--lambda-l2", type=float, default=2.0)
    p.add_argument("--alpha-l1", type=float, default=0.5)
    p.add_argument("--objective", default="reg:squarederror",
                   choices=["reg:squarederror", "rank:pairwise"])
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]

    panel = panel.dropna(subset=[args.label])
    train = panel[panel["split_label"] == "train"]
    val   = panel[panel["split_label"] == "val"]
    test  = panel[panel["split_label"] == "test"]
    log.info("Splits — train: %d  val: %d  test: %d", len(train), len(val), len(test))

    X_train = train[feat_cols].values
    y_train = train[args.label].values
    X_val   = val[feat_cols].values
    y_val   = val[args.label].values
    X_test  = test[feat_cols].values
    y_test  = test[args.label].values

    params = {
        "objective": args.objective,
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.lambda_l2,
        "alpha": args.alpha_l1,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 10,
    }
    log.info("XGB params: %s", params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    if args.objective == "rank:pairwise":
        # rank:pairwise needs `group` info — one group per date in train
        train_groups = train.groupby("date").size().values
        val_groups   = val.groupby("date").size().values
        dtrain.set_group(train_groups)
        dval.set_group(val_groups)

    log.info("Training (num_boost_round=%d, early_stop=%d)…",
             args.num_boost_round, args.early_stopping)
    model = xgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=50,
    )

    preds_train = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
    preds_val   = model.predict(dval,   iteration_range=(0, model.best_iteration + 1))
    preds_test  = model.predict(dtest,  iteration_range=(0, model.best_iteration + 1))

    train_dates = train["date"].astype("int64").values
    val_dates   = val["date"].astype("int64").values
    test_dates  = test["date"].astype("int64").values

    train_mean, train_med = per_day_ic(preds_train, y_train, train_dates)
    val_mean,   val_med   = per_day_ic(preds_val,   y_val,   val_dates)
    test_mean,  test_med  = per_day_ic(preds_test,  y_test,  test_dates)

    log.info("══ Qlib-pattern XGB on alpha158 ══")
    log.info("  label = %s  objective = %s", args.label, args.objective)
    log.info("  best_iter = %d", model.best_iteration)
    log.info("  train  mean=%+.4f  median=%+.4f", train_mean, train_med)
    log.info("  val    mean=%+.4f  median=%+.4f", val_mean,   val_med)
    log.info("  test   mean=%+.4f  median=%+.4f", test_mean,  test_med)

    summary = {
        "model": "qlib_xgb",
        "objective": args.objective,
        "label": args.label,
        "n_features": len(feat_cols),
        "best_iter": model.best_iteration,
        "params": params,
        "test_mean_ic": test_mean,
        "test_median_ic": test_med,
        "val_mean_ic": val_mean,
        "val_median_ic": val_med,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary: %s", args.output)


if __name__ == "__main__":
    main()
