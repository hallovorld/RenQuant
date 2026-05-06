#!/usr/bin/env python
"""Fit Isotonic calibrator on alpha158+Linear scores.

Phase 2 of alpha158+Linear integration (2026-05-06). Skips the production
feature pipeline (which builds 21 features) and uses the alpha158 dataset
directly. Output is `panel-rank-calibration.alpha158_linear.json` —
loadable by `GlobalPanelCalibration.load()` (existing calibrator class).

Reference: `qlib/contrib/data/handler.py:_DEFAULT_LEARN_PROCESSORS` —
CSZScoreNorm on label is already done in alpha158_qlib_dataset.parquet.

Usage::

    python scripts/fit_alpha158_linear_calibrator.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.linear_ltr import PanelLinearScorer  # noqa: E402
from training_panel.global_calibrator import fit_global_calibrator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fit-alpha158-calibrator")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--scorer",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                                / "artifacts" / "panel-ltr.alpha158_linear.json"))
    p.add_argument("--output",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"
                                / "artifacts" / "panel-rank-calibration.alpha158_linear.json"))
    p.add_argument("--method", default="isotonic", choices=["isotonic", "platt"])
    p.add_argument("--lookahead", type=int, default=5,
                   help="Days for fwd return label (matches dataset's fwd_5d label)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Threshold defining 'outperform': "
                        "0.0 → outperform iff ret > 0 (sign-based)")
    args = p.parse_args()

    log.info("Loading dataset: %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    log.info("Loading scorer: %s", args.scorer)
    scorer = PanelLinearScorer.load(args.scorer)

    feat_cols = scorer.feature_cols
    log.info("Features (%d) — sample: %s", len(feat_cols), feat_cols[:5])

    # ── Replace CSZScoreNorm-d labels with RAW returns ──────────────────────
    # alpha158_qlib_dataset.parquet has labels that are cross-sectionally
    # z-scored per date (Qlib LEARN_PROCESSORS). For calibrator fitting we
    # need RAW future return (e.g. 0.025 = 2.5%) so the calibrator's E[r]
    # output has natural units that JointPortfolioQPTask can consume as μ.
    raw_labels_path = REPO_ROOT / "data" / "transformer_dataset_engineered.parquet"
    if raw_labels_path.exists():
        log.info("Loading raw labels from %s", raw_labels_path)
        raw_panel = pd.read_parquet(raw_labels_path,
                                      columns=["ticker", "date",
                                               "fwd_5d_excess",
                                               "fwd_20d_excess",
                                               "fwd_60d_excess"])
        raw_panel["date"] = pd.to_datetime(raw_panel["date"])
        # Drop the CSZScoreNorm'd label columns and merge raw ones
        for c in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"):
            if c in panel.columns:
                panel = panel.drop(columns=[c])
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.merge(raw_panel, on=["ticker", "date"], how="inner")
        log.info("After raw-label merge: %d rows", len(panel))
    else:
        log.warning("Raw labels parquet not found — calibrator E[r] units may "
                    "be in z-score not return")

    # ── Score the entire panel via the loaded scorer ─────────────────────────
    log.info("Scoring %d rows …", len(panel))
    panel["raw_score"] = scorer.score(panel)

    # ── Build per-(ticker, date) score series + fwd return ───────────────────
    # The dataset already has CSZScoreNorm-normalized labels; for
    # calibrator fitting we want the RAW future return so the calibrator
    # learns score → P(actual outperform) and score → E[real return].
    # We re-derive raw fwd_Nd_excess by reading OHLCV and SPY.
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Build panel_scores (ticker → date → raw_score) and future_returns dicts
    # in the format fit_global_calibrator expects.
    log.info("Reorganizing scores into per-ticker series …")
    panel_scores: dict[str, pd.Series] = {}
    future_returns: dict[str, pd.Series] = {}
    label_col = "fwd_5d_excess"  # use raw 5d excess as proxy

    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date")
        scores = pd.Series(group["raw_score"].values, index=group["date"], name=ticker)
        rets   = pd.Series(group[label_col].values,   index=group["date"], name=ticker)
        # Drop rows with NaN (cache tail or early warmup)
        valid = (~scores.isna()) & (~rets.isna())
        if valid.sum() < 50:
            continue
        panel_scores[ticker] = scores[valid]
        future_returns[ticker] = rets[valid]

    log.info("Built scoring data — tickers=%d", len(panel_scores))

    # ── Fit calibrator ──────────────────────────────────────────────────────
    log.info("Fitting %s calibrator (lookahead=%dd, threshold=%.3f) …",
             args.method, args.lookahead, args.threshold)
    calib = fit_global_calibrator(
        panel_scores, future_returns,
        lookahead_days=args.lookahead,
        threshold=args.threshold,
        threshold_mode="absolute",
        method=args.method,
    )

    out_path = Path(args.output)
    calib.save(out_path, metadata={
        "scorer_artifact": str(args.scorer),
        "scorer_kind": "panel_linear",
        "scorer_train_ic": scorer.metadata.get("training_train_ic"),
        "scorer_test_ic":  scorer.metadata.get("test_mean_ic"),
        "method": args.method,
    })
    log.info("══ Calibrator saved: %s ══", out_path)
    log.info("  n_rows=%d  n_tickers=%d  pool_IC=%+.4f  base_rate=%.3f  unique_prob_y=%d",
             calib.metadata["n_rows"], calib.metadata["n_tickers"],
             calib.metadata.get("pool_ic", 0.0) or 0.0,
             calib.metadata.get("prob_base_rate", 0.0),
             calib.metadata.get("n_unique_prob_y", 0))


if __name__ == "__main__":
    main()
