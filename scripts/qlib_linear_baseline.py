#!/usr/bin/env python
"""Qlib-faithful LinearModel baseline.

Source: qlib/contrib/model/linear.py:LinearModel — read 2026-05-06.

  model = LinearRegression(fit_intercept=False, copy_X=False)
  model.fit(X, y)      # MSE on (158 features → CS-z label)
  preds = X_test @ coef_

Evaluate via per-day Spearman IC on the test split. This is a faithful
replication of Qlib's published Alpha158 + LinearModel benchmark, on
RenQuant's 290-ticker / 8-year data instead of Qlib's csi500 / 10-year.

Goal: validate IF Qlib's reported +0.045 IC translates to our scale, or
whether watchlist breadth / training window matter materially.

Usage::

    python scripts/qlib_linear_baseline.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, Lasso

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("qlib-linear-baseline")


def per_day_ic(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray
               ) -> tuple[float, float, int]:
    """(mean_ic, median_ic, n_days)."""
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
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--estimator", default="ols",
                   choices=["ols", "ridge", "lasso"])
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Regularization strength (only used for ridge / lasso)")
    p.add_argument("--output",
                   default=str(REPO_ROOT / "artifacts" / "qlib_linear_baseline.json"))
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    log.info("Features (%d): %s ... (truncated)", len(feat_cols), feat_cols[:5])

    panel = panel.dropna(subset=[args.label])
    train = panel[panel["split_label"] == "train"]
    val   = panel[panel["split_label"] == "val"]
    test  = panel[panel["split_label"] == "test"]
    log.info("Train: %d rows  Val: %d  Test: %d", len(train), len(val), len(test))

    X_train = train[feat_cols].values
    y_train = train[args.label].values
    X_val   = val[feat_cols].values
    y_val   = val[args.label].values
    X_test  = test[feat_cols].values
    y_test  = test[args.label].values

    if args.estimator == "ols":
        model = LinearRegression(fit_intercept=False, copy_X=False)
    elif args.estimator == "ridge":
        model = Ridge(alpha=args.alpha, fit_intercept=False, copy_X=False)
    else:
        model = Lasso(alpha=args.alpha, fit_intercept=False, copy_X=False)
    log.info("Fitting %s on %d × %d train matrix …",
             type(model).__name__, X_train.shape[0], X_train.shape[1])
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_val   = model.predict(X_val)
    preds_test  = model.predict(X_test)

    train_dates = train["date"].astype("int64").values
    val_dates   = val["date"].astype("int64").values
    test_dates  = test["date"].astype("int64").values

    train_mean, train_med, train_n = per_day_ic(preds_train, y_train, train_dates)
    val_mean,   val_med,   val_n   = per_day_ic(preds_val,   y_val,   val_dates)
    test_mean,  test_med,  test_n  = per_day_ic(preds_test,  y_test,  test_dates)

    log.info("══ Qlib LinearModel (%s) on alpha158 ══", args.estimator)
    log.info("  label = %s", args.label)
    log.info("  train  mean=%+.4f  median=%+.4f  n_days=%d", train_mean, train_med, train_n)
    log.info("  val    mean=%+.4f  median=%+.4f  n_days=%d", val_mean,   val_med,   val_n)
    log.info("  test   mean=%+.4f  median=%+.4f  n_days=%d", test_mean,  test_med,  test_n)
    log.info("  ────────────────────────────────────")
    log.info("  benchmark: Qlib repo reports IC ~ +0.045 on csi500/10y")

    summary = {
        "estimator": args.estimator,
        "alpha": args.alpha,
        "label": args.label,
        "n_features": len(feat_cols),
        "feat_cols_sample": feat_cols[:10],
        "train_mean_ic": train_mean,
        "train_median_ic": train_med,
        "val_mean_ic": val_mean,
        "val_median_ic": val_med,
        "test_mean_ic": test_mean,
        "test_median_ic": test_med,
        "n_test_days": test_n,
        "qlib_benchmark_ic": 0.045,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary written: %s", args.output)


if __name__ == "__main__":
    main()
