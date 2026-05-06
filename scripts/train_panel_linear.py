#!/usr/bin/env python
"""Train alpha158 + sklearn LinearRegression and save as PanelLinearScorer.

Phase 1 of alpha158+Linear production integration (2026-05-06). Wraps
the +29 pts walk-forward alpha winner as a `panel_linear` kind artifact
loadable via `PanelScorer.load()` — fully compatible with the existing
inference pipeline (calibrator → JointPortfolioQPTask → sell gates).

Reference: `qlib/contrib/model/linear.py:LinearModel.fit` — uses
sklearn `LinearRegression(fit_intercept=False)` on (X, y) where y is
cross-sectionally z-scored forward returns (CSZScoreNorm processor).

Usage::

    python scripts/train_panel_linear.py --label fwd_5d_excess
    python scripts/train_panel_linear.py --label fwd_60d_excess \\
        --output backtesting/renquant_104/artifacts/panel-ltr.alpha158_linear_60d.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.linear_ltr import PanelLinearScorer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-panel-linear")


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
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--estimator", default="ols",
                   choices=["ols", "ridge"])
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularization (only for --estimator=ridge)")
    p.add_argument("--output",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                                / "artifacts" / "panel-ltr.alpha158_linear.json"))
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    log.info("Features: %d  Total rows: %d", len(feat_cols), len(panel))

    panel = panel.dropna(subset=[args.label])
    train = panel[panel["split_label"] == "train"]
    val   = panel[panel["split_label"] == "val"]
    test  = panel[panel["split_label"] == "test"]
    log.info("Splits — train: %d  val: %d  test: %d",
             len(train), len(val), len(test))

    # Load pre-normalization train-only stats from the sidecar saved by
    # build_alpha158_qlib.py. These are the RAW (un-z-scored) means/stds
    # that the dataset builder used. Storing them in the scorer artifact
    # lets `score_raw()` reproduce the same normalization at inference
    # given fresh `compute_alpha158_at()` output.
    stats_path = Path(args.dataset).with_suffix(".stats.json")
    if stats_path.exists():
        log.info("Loading raw feature stats from %s", stats_path)
        sidecar = json.loads(stats_path.read_text())
        # Align order to feat_cols
        stats_cols = sidecar["feature_cols"]
        col_to_idx = {c: i for i, c in enumerate(stats_cols)}
        feature_means_pre = np.array(
            [sidecar["feature_means"][col_to_idx[c]] for c in feat_cols]
        )
        feature_stds_pre = np.array(
            [sidecar["feature_stds"][col_to_idx[c]] for c in feat_cols]
        )
    else:
        log.warning(
            "Stats sidecar not found at %s — saving stats from already-normalized "
            "panel (score_raw will assume input is already-normalized)", stats_path,
        )
        feature_means_pre = train[feat_cols].mean().values
        feature_stds_pre = train[feat_cols].std().values

    X_train = train[feat_cols].values
    y_train = train[args.label].values
    if args.estimator == "ridge":
        model = Ridge(alpha=args.alpha, fit_intercept=False, copy_X=False)
    else:
        model = LinearRegression(fit_intercept=False, copy_X=False)
    log.info("Fitting %s on %d × %d …",
             type(model).__name__, X_train.shape[0], X_train.shape[1])
    model.fit(X_train, y_train)

    scorer = PanelLinearScorer.from_sklearn(
        model, feature_cols=feat_cols,
        feature_means=feature_means_pre,
        feature_stds=feature_stds_pre,
        metadata={
            "trained_date": str(date.today()),
            "label": args.label,
            "estimator": args.estimator,
            "alpha_l2": args.alpha if args.estimator == "ridge" else 0.0,
            "n_train_rows": len(train),
            "panel_shape": {
                "rows": len(panel),
                "tickers": int(panel["ticker"].nunique()),
                "dates": int(panel["date"].nunique()),
            },
            "training_notes": (
                "alpha158 (Qlib-faithful 158 features) + sklearn LinearRegression "
                "MSE on CSZScoreNorm-z-scored labels. Reference: "
                "qlib/contrib/model/linear.py:LinearModel."
            ),
        }
    )

    # Compute diagnostic metrics on the saved scorer (round-trip via score())
    preds_train = scorer.score(train[feat_cols])
    preds_val   = scorer.score(val[feat_cols])   if len(val) else None
    preds_test  = scorer.score(test[feat_cols])  if len(test) else None

    train_dates = train["date"].astype("int64").values
    train_mean_ic, train_med_ic = per_day_ic(
        preds_train.values, y_train, train_dates
    )
    log.info("DIAGNOSTIC IC — train: mean=%+.4f  median=%+.4f",
             train_mean_ic, train_med_ic)
    if preds_val is not None:
        val_mean_ic, val_med_ic = per_day_ic(
            preds_val.values, val[args.label].values,
            val["date"].astype("int64").values,
        )
        log.info("DIAGNOSTIC IC — val:   mean=%+.4f  median=%+.4f",
                 val_mean_ic, val_med_ic)
    else:
        val_mean_ic = val_med_ic = None
    if preds_test is not None:
        test_mean_ic, test_med_ic = per_day_ic(
            preds_test.values, test[args.label].values,
            test["date"].astype("int64").values,
        )
        log.info("DIAGNOSTIC IC — test:  mean=%+.4f  median=%+.4f",
                 test_mean_ic, test_med_ic)
    else:
        test_mean_ic = test_med_ic = None

    # Save with diagnostic metrics in metadata
    scorer.metadata["training_train_ic"] = train_mean_ic
    scorer.metadata["val_mean_ic"] = val_mean_ic
    scorer.metadata["val_median_ic"] = val_med_ic
    scorer.metadata["test_mean_ic"] = test_mean_ic
    scorer.metadata["test_median_ic"] = test_med_ic
    # Mirror "oos_mean_ic" key used by XGB artifacts so PanelScorer-aware
    # downstream (calibrator, regime-conditional logic) reads consistent stats.
    scorer.metadata["oos_mean_ic"] = test_mean_ic

    out_path = Path(args.output)
    scorer.save(out_path)
    log.info("══ Artifact written: %s ══", out_path)
    log.info("  size: %.1f KB", out_path.stat().st_size / 1024)
    log.info("  sample feature_cols: %s", scorer.feature_cols[:5])
    log.info("  coef stats: min=%+.4f  max=%+.4f  ‖coef‖₂=%.4f",
             scorer.coef.min(), scorer.coef.max(), float(np.linalg.norm(scorer.coef)))


if __name__ == "__main__":
    main()
