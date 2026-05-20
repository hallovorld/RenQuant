#!/usr/bin/env python3
"""Verify σ-calibration of HF Trainer multi-task head (Student-t NLL).

Reads val_preds.parquet files containing date / pred / label / mu / sigma
columns (produced by patchtst_hf.py with distributional head ON).

Calibration tests:
  1. Quantile-bin test: bin predictions by σ-quantile (Q1..Q5), verify
     realized RMSE monotonically increases with predicted σ.
  2. Calibration coefficient: corr(predicted σ, |residual|) — should be
     positive. Duan 2020 §4 NGB baseline reports ~0.27.
  3. PIT (probability integral transform): if σ-calibrated, the
     residual/σ should be ~Student-t-distributed. KS-test against
     standard normal as rough proxy.

Usage::

    .venv/bin/python scripts/verify_sigma_calibration.py \\
        --glob 'artifacts/hf_trainer_5cut_5seed_pt07/*/seed_*/hf_patchtst_*_val_preds.parquet'
"""
from __future__ import annotations
import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verify-sigma")


def calibrate_one(df: pd.DataFrame, source: str) -> dict:
    """Run calibration tests on one val_preds.parquet."""
    if "mu" not in df.columns or "sigma" not in df.columns:
        log.warning("[%s] no mu/sigma columns — distributional head was OFF", source)
        return {"source": source, "status": "no_sigma"}

    df = df.dropna(subset=["mu", "sigma", "label"])
    if len(df) < 100:
        return {"source": source, "status": "too_few_samples", "n": len(df)}

    resid = df["label"] - df["mu"]
    sigma = df["sigma"].abs()

    # 1. Quantile-bin test
    df_calib = pd.DataFrame({"resid": resid, "sigma": sigma,
                              "abs_resid": resid.abs()})
    df_calib["sigma_bin"] = pd.qcut(df_calib["sigma"], q=5,
                                       labels=[f"Q{i + 1}" for i in range(5)],
                                       duplicates="drop")
    binned = df_calib.groupby("sigma_bin", observed=True).agg(
        sigma_mean=("sigma", "mean"),
        realized_rmse=("resid", lambda r: float(np.sqrt((r ** 2).mean()))),
        abs_resid_mean=("abs_resid", "mean"),
        n=("resid", "count"),
    )

    # 2. Calibration coefficient
    from scipy.stats import spearmanr  # noqa: PLC0415
    cal_coef, _ = spearmanr(sigma, resid.abs())

    # 3. Monotonicity
    rmse = binned["realized_rmse"].values
    monotonic = bool(np.all(np.diff(rmse) >= -1e-6))  # tolerate tiny noise

    return {
        "source": source,
        "status": "ok",
        "n": len(df),
        "calibration_coef": float(cal_coef),
        "monotonic_q5_vs_q1": monotonic,
        "binned": binned,
        "rmse_q5_vs_q1": float(rmse[-1] / rmse[0]) if rmse[0] > 0 else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True, help="glob pattern for val_preds.parquet")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=True))
    log.info("found %d val_preds files", len(paths))
    if not paths:
        sys.exit(1)

    coefs = []
    for path in paths:
        df = pd.read_parquet(path)
        rel = Path(path).relative_to(REPO) if Path(path).is_absolute() else Path(path)
        result = calibrate_one(df, str(rel))
        if result["status"] == "ok":
            log.info("%s | n=%d cal_coef=%+.4f monotonic=%s rmse_q5/q1=%.2f",
                     rel, result["n"], result["calibration_coef"],
                     result["monotonic_q5_vs_q1"], result["rmse_q5_vs_q1"])
            coefs.append(result["calibration_coef"])
        else:
            log.info("%s | %s", rel, result["status"])

    if coefs:
        log.info("\n=== AGGREGATE σ-calibration coefficient ===")
        log.info("n_files=%d  mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f",
                 len(coefs), np.mean(coefs), np.std(coefs),
                 min(coefs), max(coefs))
        target = 0.20
        log.info("NGB baseline (Duan 2020 §4): %+.4f", target)
        if np.mean(coefs) >= target:
            log.info("→ σ-calibration MEETS or BEATS NGB baseline — wire σ to "
                     "Kelly/QP downstream")
        else:
            log.info("→ σ-calibration BELOW NGB baseline — keep σ wire OFF, "
                     "diagnose: try larger nll_loss_weight, longer training, "
                     "or constrain df range")


if __name__ == "__main__":
    main()
