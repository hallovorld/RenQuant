#!/usr/bin/env python
"""Qlib-faithful LightGBM (LGBModel) baseline.

Source: qlib/contrib/model/gbdt.py:LGBModel — read 2026-05-06.

  params = {"objective": "mse", "verbosity": -1}
  lgb.train(params, train_set, num_boost_round=1000,
            valid_sets=[train, valid],
            callbacks=[early_stopping(50), log_evaluation(20)])
  preds = model.predict(X_test)

This is Qlib's strongest single-model baseline on alpha158/Alpha360.
Their published IC: **+0.045** on csi500/10y. Our equivalent: 290/8y.

Adaptation: same Qlib pattern (MSE objective + CSZScoreNorm labels +
no ranking loss). Features = 158 alpha158 vars from
`alpha158_qlib_dataset.parquet`.

Usage::

    python scripts/qlib_lightgbm_baseline.py
    python scripts/qlib_lightgbm_baseline.py --label fwd_60d_excess
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("qlib-lgbm")


def per_day_ic(preds, labels, dates) -> tuple[float, float, int]:
    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 5:
            continue
        rho, _ = spearmanr(group["pred"], group["label"])
        if not np.isnan(rho):
            ics.append(rho)
    if not ics:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.median(ics)), len(ics)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--output",
                   default=str(REPO_ROOT / "artifacts" / "qlib_lightgbm.json"))
    # Qlib LGB defaults (from gbdt.py + qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_*.yaml)
    p.add_argument("--num-boost-round", type=int, default=1000)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--num-leaves", type=int, default=210)
    p.add_argument("--learning-rate", type=float, default=0.0421)
    p.add_argument("--feature-fraction", type=float, default=0.8879)
    p.add_argument("--bagging-fraction", type=float, default=0.8789)
    p.add_argument("--bagging-freq", type=int, default=2)
    p.add_argument("--min-data-in-leaf", type=int, default=210)
    p.add_argument("--lambda-l1", type=float, default=205.6999)
    p.add_argument("--lambda-l2", type=float, default=580.9768)
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    log.info("Features (%d)  Train mass=%d", len(feat_cols),
             (panel["split_label"] == "train").sum())

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
        "objective": "mse", "verbosity": -1,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
    }
    log.info("LGB params (Qlib alpha158 reference): %s", params)

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    evals_result: dict = {}
    log.info("Training (num_boost_round=%d, early_stop=%d)…",
             args.num_boost_round, args.early_stopping)
    model = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    preds_train = model.predict(X_train)
    preds_val   = model.predict(X_val)
    preds_test  = model.predict(X_test)

    train_dates = train["date"].astype("int64").values
    val_dates   = val["date"].astype("int64").values
    test_dates  = test["date"].astype("int64").values

    train_mean, train_med, _ = per_day_ic(preds_train, y_train, train_dates)
    val_mean,   val_med,   _ = per_day_ic(preds_val,   y_val,   val_dates)
    test_mean,  test_med,  _ = per_day_ic(preds_test,  y_test,  test_dates)

    log.info("══ Qlib LGBModel on alpha158 ══")
    log.info("  label = %s", args.label)
    log.info("  best_iter      = %d", model.best_iteration)
    log.info("  train  mean=%+.4f  median=%+.4f", train_mean, train_med)
    log.info("  val    mean=%+.4f  median=%+.4f", val_mean,   val_med)
    log.info("  test   mean=%+.4f  median=%+.4f", test_mean,  test_med)
    log.info("  ────────────────────────────────────")
    log.info("  Qlib benchmark: IC ~ +0.045 on csi500/10y")

    summary = {
        "model": "lightgbm",
        "label": args.label,
        "n_features": len(feat_cols),
        "best_iter": model.best_iteration,
        "params": params,
        "train_mean_ic": train_mean,
        "train_median_ic": train_med,
        "val_mean_ic": val_mean,
        "val_median_ic": val_med,
        "test_mean_ic": test_mean,
        "test_median_ic": test_med,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary written: %s", args.output)


if __name__ == "__main__":
    main()
