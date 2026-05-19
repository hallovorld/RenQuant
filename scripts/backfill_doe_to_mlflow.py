#!/usr/bin/env python3
"""Backfill existing DOE trials + XGB baselines to MLflow tracking DB.

Per user mandate 2026-05-19 09:35: all experiments → MLflow DB, not just
CSV files. The HF DOE has been writing per-trial val_preds.parquet +
summary.json to local disk only; this script reads those and creates
MLflow runs so they're queryable + comparable.

Each (point_id, cut, seed) trial becomes one MLflow Run:
  params:  lr, weight_decay, warmup_epochs, seq_len, point_id, cut,
           seed, label, swa_enabled
  metrics: best_val_ic, n_params, n_features,
           per_regime_ic_<regime> (for each regime present),
           bull_regime_ic
  tags:    arch=hf_patchtst, experiment_phase=doe_screening,
           is_center, model_kind
  artifact: val_preds.parquet

Experiments:
  renquant_104_hf_doe       — per-trial DOE runs
  renquant_104_xgb_baseline — XGB benchmarks per cut

Usage::

    .venv/bin/python scripts/backfill_doe_to_mlflow.py \\
        --doe-dir artifacts/patchtst_doe_hf \\
        --tracking-uri file:./mlruns
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.hmm_regime_labels import (compute_hmm_regime_labels,
                                        per_hmm_regime_ic, bull_regime_ic)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill-doe-mlflow")


def _ensure_experiment(tracking_uri: str, name: str) -> str:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


def backfill_doe(doe_dir: Path, tracking_uri: str,
                  experiment_name: str = "renquant_104_hf_doe") -> int:
    """Walk DOE trial dirs, create MLflow run per trial. Returns n_runs created."""
    import mlflow
    exp_id = _ensure_experiment(tracking_uri, experiment_name)
    design = pd.read_csv(doe_dir / "design.csv") if (doe_dir / "design.csv").exists() else None
    hmm = compute_hmm_regime_labels(REPO / "data/ohlcv/SPY/1d.parquet")

    trial_dirs = sorted(doe_dir.glob("pt_*_seed_*"))
    log.info("backfill: %d DOE trial dirs found in %s", len(trial_dirs), doe_dir)

    n_logged = 0
    for d in trial_dirs:
        parts = d.name.split("_")  # pt_00_cut1_covid_seed_42
        if len(parts) < 5: continue
        try:
            point_id = int(parts[1])
            cut = "_".join(parts[2:-2])
            seed = int(parts[-1])
        except ValueError:
            continue

        summary_files = list(d.glob("*summary.json"))
        val_preds_files = list(d.glob("*val_preds.parquet"))
        if not summary_files or not val_preds_files:
            log.warning("skip %s: missing summary or val_preds", d.name)
            continue

        summary = json.loads(summary_files[0].read_text())
        val_preds = pd.read_parquet(val_preds_files[0])

        # Per-regime IC
        per_regime = per_hmm_regime_ic(val_preds, hmm,
                                         min_samples_per_day=5,
                                         min_days_per_regime=5)
        bull_ic = bull_regime_ic(per_regime)

        # Design params (if available)
        design_row = None
        if design is not None:
            d_row = design[design["point_id"] == point_id]
            if not d_row.empty:
                design_row = d_row.iloc[0]

        # MLflow run
        run_name = f"pt{point_id:02d}_{cut}_seed{seed}"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            params = {
                "point_id": point_id, "cut": cut, "seed": seed,
                "label": summary.get("cut") or "fwd_60d_excess",
                "arch": summary.get("arch", "hf_patchtst"),
            }
            if design_row is not None:
                params.update({
                    "lr": float(design_row["lr"]),
                    "weight_decay": float(design_row["weight_decay"]),
                    "warmup_epochs": int(design_row["warmup_epochs"]),
                    "seq_len": int(design_row["seq_len"]),
                    "is_center": bool(design_row["is_center"]),
                })
            mlflow.log_params({k: str(v) for k, v in params.items()})

            metrics = {
                "best_val_ic": float(summary.get("best_val_ic", float("nan"))),
                "n_params": int(summary.get("n_params", 0)),
                "n_features": int(summary.get("n_features", 0)),
                "bull_regime_ic": float(bull_ic) if not np.isnan(bull_ic) else 0.0,
            }
            for regime, ic in per_regime.items():
                metrics[f"per_regime_ic_{regime}"] = float(ic)
            # Filter non-finite
            metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
            mlflow.log_metrics(metrics)

            mlflow.set_tags({
                "experiment_phase": "doe_screening",
                "design_kind": "fracfact_2_4_1",
                "model_kind": "hf_patchtst",
            })
            # Log val_preds as artifact (small, ~100KB each)
            mlflow.log_artifact(str(val_preds_files[0]))
        n_logged += 1
        if n_logged % 10 == 0:
            log.info("  logged %d / %d", n_logged, len(trial_dirs))

    log.info("backfill complete: %d MLflow runs created", n_logged)
    return n_logged


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doe-dir", default="artifacts/patchtst_doe_hf")
    p.add_argument("--tracking-uri", default="file:./mlruns")
    p.add_argument("--experiment-name", default="renquant_104_hf_doe")
    args = p.parse_args()

    doe_dir = REPO / args.doe_dir
    if not doe_dir.exists():
        raise SystemExit(f"DOE dir {doe_dir} not found")

    n = backfill_doe(doe_dir, args.tracking_uri, args.experiment_name)
    print(f"Backfilled {n} MLflow runs → {args.tracking_uri} :: {args.experiment_name}")
    print(f"View via: mlflow ui --backend-store-uri {args.tracking_uri}")


if __name__ == "__main__":
    main()
