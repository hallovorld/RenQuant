#!/usr/bin/env python
"""Train XGBoost rank:pairwise on alpha158 features (Qlib + XGB hybrid).

E29 resume condition #1: the linear-on-alpha158 thesis is closed, but
the alpha158 feature space + XGB non-linear interactions might still
work where sklearn LinearRegression failed. This script trains a
PanelLTRModel (XGBoost rank:pairwise) on the same alpha158 features
that fed `panel-ltr.alpha158_linear.json`.

Output: `backtesting/renquant_104/artifacts/panel-ltr.alpha158_xgb.json`
(kind=panel_ltr_xgboost; loadable via PanelScorer.load → existing
inference path; no new dispatcher needed because XGB scorer is the
default).

Usage::

    python scripts/train_panel_alpha158_xgb.py
    python scripts/train_panel_alpha158_xgb.py \\
        --dataset data/alpha158_qlib_dataset.extended_train.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.ltr_model import PanelLTRModel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("train-alpha158-xgb")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--output",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                                / "artifacts" / "panel-ltr.alpha158_xgb.json"))
    p.add_argument("--num-boost-round", type=int, default=400)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--min-child-weight", type=float, default=60.0)
    args = p.parse_args()

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                 "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excluded]
    log.info("Features: %d  Total rows: %d", len(feat_cols), len(panel))

    panel = panel.dropna(subset=[args.label]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    train = panel[panel["split_label"] == "train"].reset_index(drop=True)
    val   = panel[panel["split_label"] == "val"].reset_index(drop=True)
    test  = panel[panel["split_label"] == "test"].reset_index(drop=True)
    log.info("Splits — train: %d  val: %d  test: %d",
             len(train), len(val), len(test))

    # Group sizes (per-date row counts) — required by XGB rank:pairwise
    train_groups = train.groupby("date").size().values.astype(int)
    val_groups = val.groupby("date").size().values.astype(int)

    # Use cross-sectional rank as label (matches Qlib's CSZScoreNorm idiom
    # for ranking — XGB rank:pairwise needs a sortable target).
    train["label"] = (train.groupby("date")[args.label]
                          .rank(pct=True))
    val["label"]   = (val.groupby("date")[args.label]
                          .rank(pct=True))

    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@5",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "reg_lambda": 5.0,
        "reg_alpha": 2.0,
        "seed": 42,
        "nthread": 10,
    }
    model = PanelLTRModel(params=params)
    log.info("Training XGB rank:pairwise on %d × %d features (alpha158)…",
             len(train), len(feat_cols))

    meta = model.train(
        panel=train,
        group_sizes=train_groups,
        feature_cols=feat_cols,
        label_col="label",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_panel=val,
        eval_group_sizes=val_groups,
    )
    log.info("Trained: best_iter=%s  train_ic=%s",
             meta.get("best_iter"), meta.get("train_ic"))

    # Evaluate on test split
    if len(test) > 0:
        test["label"] = test.groupby("date")[args.label].rank(pct=True)
        test_groups = test.groupby("date").size().values.astype(int)
        # Score test rows
        test_scores = model.score(test[feat_cols])
        # Per-day Spearman IC of model_score vs label
        test_with = test.assign(_score=test_scores).copy()
        from scipy.stats import spearmanr
        ics = []
        for _, group in test_with.groupby("date"):
            if len(group) >= 2:
                rho, _ = spearmanr(group["_score"], group["label"])
                if not np.isnan(rho):
                    ics.append(rho)
        if ics:
            log.info("Test mean IC: %+.4f  median IC: %+.4f",
                     float(np.mean(ics)), float(np.median(ics)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "label": args.label,
        "estimator": "xgboost_rank_pairwise",
        "dataset": str(args.dataset),
        "training_train_ic": meta.get("train_ic"),
        "best_iter": meta.get("best_iter"),
        "n_train_rows": len(train),
        "panel_shape": {
            "rows": int(len(panel)),
            "tickers": int(panel["ticker"].nunique()),
            "dates": int(panel["date"].nunique()),
        },
    }
    model.save(out_path, metadata=metadata)
    log.info("Artifact written: %s (size: %.1f KB)",
             out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
