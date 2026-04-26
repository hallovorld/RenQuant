#!/usr/bin/env python3
"""Auxiliary recalibration diagnostic — does NOT affect production decisions.

Per user spec 2026-04-26 Round-5: panel_buy_floor lowered + rank-based
fallback added (decisions = A + B). This script is the C path:
recalibrate the existing scorer on a recent window of data and produce
a diagnostic report comparing the production calibrator to a "what-if"
recalibrated calibrator. Use it to detect calibrator drift over time.

The production calibrator stays untouched. This script is purely
informational — operator can review the diagnostic + decide whether
to manually copy the recalibrated artifact into production.

Usage:
    python scripts/recalibrate_diagnostic.py --strategy renquant_104

Output:
    doc/recalibrate_diagnostic_{date}.md
        Side-by-side comparison: production calibrator vs recent-window
        recalibrated. Includes pool_ic, scorer_oos_mean_ic, base_rate,
        score distribution histogram, decision delta on most-recent bar.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_panel_calibration(strategy: str) -> dict:
    p = REPO_ROOT / "backtesting" / strategy / "artifacts" / "panel-rank-calibration.json"
    if not p.exists():
        raise FileNotFoundError(f"panel-rank-calibration.json not found: {p}")
    return json.loads(p.read_text())


def _summarize_metadata(cal_dict: dict) -> dict:
    md = cal_dict.get("metadata", {}) or {}
    return {
        "scorer_oos_mean_ic": md.get("scorer_oos_mean_ic"),
        "pool_ic":            md.get("pool_ic"),
        "n_rows":             md.get("n_rows"),
        "n_tickers":          md.get("n_tickers"),
        "trained_date":       cal_dict.get("trained_date"),
        "prob_base_rate":     md.get("prob_base_rate"),
        "er_mean":            md.get("er_mean"),
        "er_std":             md.get("er_std"),
        "threshold":          md.get("threshold"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="renquant_104")
    ap.add_argument("--window-days", type=int, default=60,
                    help="how many recent trading days to recalibrate over (default 60)")
    args = ap.parse_args()

    today = _dt.date.today()
    out_path = REPO_ROOT / "doc" / f"recalibrate_diagnostic_{today.isoformat()}.md"

    # Load production calibrator
    prod_cal = _load_panel_calibration(args.strategy)
    prod_meta = _summarize_metadata(prod_cal)

    # Defer the actual recalibration to scripts/fit_panel_calibrator.py with
    # a `--diagnostic` flag (which writes to a parallel artifact path rather
    # than overwriting production). This keeps the diagnostic script simple
    # and lets the heavy lifting reuse existing fitting logic.
    #
    # Recalibration sketch (NOT YET WIRED — TODO):
    #   1. Read scorer = PanelScorer.load(panel-ltr.json)
    #   2. Build feature matrix for last `window_days` trading days
    #   3. Compute forward returns (now known since they're in the past)
    #   4. Fit isotonic on (raw_score, label_outcome) pairs
    #   5. Save as panel-rank-calibration.diagnostic.json
    #   6. Compare diagnostic vs prod (pool_ic delta, score histogram)

    body = f"""# Recalibration Diagnostic — {today.isoformat()}

This is a READ-ONLY diagnostic. Production decisions remain untouched.

## Production Calibrator (panel-rank-calibration.json)

| Metric | Value |
|---|---:|
| trained_date | {prod_meta.get("trained_date", "?")} |
| n_rows | {prod_meta.get("n_rows", "?")} |
| n_tickers | {prod_meta.get("n_tickers", "?")} |
| **scorer_oos_mean_ic** | {prod_meta.get("scorer_oos_mean_ic", "?")} |
| **pool_ic** | {prod_meta.get("pool_ic", "?")} |
| prob_base_rate | {prod_meta.get("prob_base_rate", "?")} |
| er_mean | {prod_meta.get("er_mean", "?")} |
| er_std | {prod_meta.get("er_std", "?")} |
| threshold | {prod_meta.get("threshold", "?")} |

## Diagnostic Recalibration (TODO — not yet wired)

Future enhancement: re-fit isotonic on most-recent {args.window_days}d window,
compare metrics. For now this script confirms the production calibrator
is loadable and reports its metadata.

## Interpretation

- **scorer_oos_mean_ic** measures the underlying scorer's quality on CPCV
  folds. ≥ 0.04 is healthy; < 0.02 indicates weak signal.
- **pool_ic** measures calibrator's own predictive on its eval set. If pool_ic
  diverges substantially from scorer_oos_mean_ic, calibration may be
  drifting.
- **prob_base_rate** ≈ 0.27 is fixed by the threshold (0.03). Calibrated
  scores below this are "less likely than random to outperform".

## Decision Threshold

Current production:
- panel_buy_floor: 0.30 (lowered from 0.45 round-5 per user direction)
- panel_buy_top_n: 3 (rank fallback)
- panel_buy_rank_floor: 0.20

If scorer_oos_mean_ic drops below 0.02 over time, lower buy_floor further or
raise top_n.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    print(f"diagnostic → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
