#!/usr/bin/env python
"""Diagnose Platt calibrator saturation.

Q: why does today's calibrator output collapse to IQR=0.016 on the live
panel scores, when the calibrator's output range is [0.41, 0.69]?

H1: Today's raw XGB scores fall in a near-flat region of the Platt
    sigmoid → all map to similar y.
H2: Calibrator's x-domain doesn't cover today's score range, so most
    inputs clip to a single end.

Approach:
  1. Load calibrator artifact (Platt sigmoid stored as 100 x,y pairs)
  2. Reconstruct sigmoid; compute slope at each x → identify flat regions
  3. Load TODAY's raw panel scores from log/inference + plot/print histogram
  4. Cross-reference: are today's scores in a flat region?

Output: numeric diagnosis + recommendation (re-fit / re-train scorer / widen domain)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diagnose-cal")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrator",
                   default="backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json")
    p.add_argument("--panel",
                   default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--scorer",
                   default="backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json")
    p.add_argument("--probe-date", default="2026-02-10",
                   help="Use this panel date as the 'today' diagnostic")
    args = p.parse_args()

    log.info("Loading calibrator: %s", args.calibrator)
    cal = json.loads((REPO / args.calibrator).read_text())
    prob_x = np.asarray(cal["probability"]["x"], dtype=np.float64)
    prob_y = np.asarray(cal["probability"]["y"], dtype=np.float64)
    er_y   = np.asarray(cal["expected_return"]["y"], dtype=np.float64)

    log.info("Calibrator probability curve: x ∈ [%.4f, %.4f], y ∈ [%.4f, %.4f]",
             prob_x.min(), prob_x.max(), prob_y.min(), prob_y.max())
    log.info("                              x_unique=%d  y_unique=%d  monotonic=%s",
             len(np.unique(prob_x)), len(np.unique(prob_y)),
             bool(np.all(np.diff(prob_y) >= -1e-9)))

    # Slope per knot via finite diff (dy/dx). Flat = slope < threshold.
    dx = np.diff(prob_x)
    dy = np.diff(prob_y)
    slope = np.where(dx > 0, dy / np.where(dx > 0, dx, 1.0), 0.0)
    log.info("Slope quartiles: q25=%.4f q50=%.4f q75=%.4f max=%.4f",
             np.quantile(slope, 0.25), np.quantile(slope, 0.50),
             np.quantile(slope, 0.75), slope.max())
    flat_thresh = 0.10  # dy/dx < 0.10 = ~"5% prob over 0.5 score unit"
    flat_mask = slope < flat_thresh
    n_flat = int(flat_mask.sum())
    log.info("FLAT knots (slope<%.2f): %d/%d", flat_thresh, n_flat, len(slope))
    if n_flat:
        flat_x = prob_x[:-1][flat_mask]
        log.info("Flat-region x ranges: [%.3f, %.3f]  (= %d knots)",
                 float(flat_x.min()), float(flat_x.max()), n_flat)

    # Load scorer + probe date
    log.info("Loading PROD XGB scorer: %s", args.scorer)
    import xgboost as xgb  # noqa: PLC0415
    art = json.loads((REPO / args.scorer).read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))
    feat_cols = art["feature_cols"]
    feat_means = np.asarray(art["feature_means"], dtype=np.float64)
    feat_stds  = np.asarray(art["feature_stds"],  dtype=np.float64)

    log.info("Loading panel + probing date %s", args.probe_date)
    panel = pd.read_parquet(REPO / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    today_frame = panel[panel["date"] == pd.Timestamp(args.probe_date)].copy()
    log.info("Probe date frame: %d rows", len(today_frame))
    if today_frame.empty:
        log.error("No rows for probe date %s", args.probe_date)
        return 1

    # Score
    X = today_frame[feat_cols].fillna(0.0).values.astype(np.float64)
    safe_stds = np.where(feat_stds > 0, feat_stds, 1.0)
    Xn = ((X - feat_means) / safe_stds).clip(-5, 5)
    d = xgb.DMatrix(Xn, feature_names=feat_cols)
    raw_scores = booster.predict(d)
    raw_q = np.quantile(raw_scores, [0.0, 0.25, 0.50, 0.75, 1.0])
    log.info("PROD RAW score on probe date: min=%.4f q25=%.4f q50=%.4f q75=%.4f max=%.4f  std=%.4f",
             *raw_q, float(np.std(raw_scores)))
    log.info("                                IQR=%.4f", raw_q[3] - raw_q[1])

    # Map via calibrator (linear interp on prob curve)
    cal_y = np.interp(raw_scores, prob_x, prob_y,
                       left=prob_y[0], right=prob_y[-1])
    cal_q = np.quantile(cal_y, [0.0, 0.25, 0.50, 0.75, 1.0])
    cal_iqr = float(cal_q[3] - cal_q[1])
    log.info("CALIBRATED PROB on probe date: min=%.4f q25=%.4f q50=%.4f q75=%.4f max=%.4f  std=%.4f",
             *cal_q, float(np.std(cal_y)))
    log.info("                                IQR=%.4f  (threshold 0.05 for saturated)", cal_iqr)

    # Domain coverage: how many of today's raw scores are INSIDE the calibrator's training x range?
    inside = float(np.mean((raw_scores >= prob_x.min()) & (raw_scores <= prob_x.max())))
    log.info("Domain coverage: %.1f%% of raw scores inside calibrator x range [%.3f, %.3f]",
             100*inside, prob_x.min(), prob_x.max())

    # Diagnose
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if cal_iqr < 0.05:
        print(f"✗ SATURATED: calibrator IQR={cal_iqr:.4f} < 0.05 threshold")
    else:
        print(f"✓ HEALTHY: calibrator IQR={cal_iqr:.4f} >= 0.05")

    # Find slope of calibrator AT today's raw score quartiles
    today_slopes = []
    for q in [raw_q[1], raw_q[2], raw_q[3]]:
        idx = int(np.searchsorted(prob_x[:-1], q))
        idx = min(max(idx, 0), len(slope) - 1)
        today_slopes.append((q, slope[idx]))
    print(f"\nSlope of Platt sigmoid at today's score quartiles:")
    for q, sl in today_slopes:
        print(f"  raw_score={q:+.4f}  →  slope={sl:.4f}  ({'FLAT' if sl < 0.10 else 'STEEP'})")

    if raw_q[3] - raw_q[1] < 0.10:
        print(f"\nROOT CAUSE A: raw scorer IQR={raw_q[3]-raw_q[1]:.4f} is itself small.")
        print("   The model is producing nearly identical scores across candidates.")
        print("   → Likely cause: model trained on data with little signal in current regime.")
        print("   → Fix: retrain scorer (esp. with current data) OR widen sigmoid slope.")
    if inside < 0.95:
        print(f"\nROOT CAUSE B: only {100*inside:.0f}% of raw scores are within calibrator's training x range.")
        print(f"   Calibrator x ∈ [{prob_x.min():.3f}, {prob_x.max():.3f}] but today's scores span "
              f"[{raw_q[0]:.3f}, {raw_q[-1]:.3f}].")
        print("   → Calibrator domain mismatch.  Fix: refit calibrator on current scorer's output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
