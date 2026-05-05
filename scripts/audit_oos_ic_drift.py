#!/usr/bin/env python
"""Walk-forward OOS IC drift audit.

2026-05-04: surfaced as D12 in the comprehensive audit. RenQuant's
production runs daily104 cron weekly retrains, but until now nobody
plotted the per-retrain ``oos_mean_ic`` across time — drift that would
have caught the arm A NaN-leaf collapse / panel-LTR signal degradation
goes undetected.

This script reads ``training_runs`` from ``data/runs.db``, filters to
production-class artifacts (``panel-ltr.json`` / ``panel-ltr.golden.json``
patterns), groups by week, computes:

  * Time series of oos_mean_ic per retrain
  * Rolling 4-week mean + std
  * Drift flags: > 1σ-decline vs trailing 12-week mean

Output: stdout text report + optional CSV.

Usage::

    python scripts/audit_oos_ic_drift.py
    python scripts/audit_oos_ic_drift.py --pattern '%golden%' --out report.csv
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "runs.db"


def load_training_runs(
    db_path: Path,
    pattern: str = "%panel-ltr.json%",
    artifact_type: str = "panel-ltr",
) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """SELECT run_date, run_id, oos_mean_ic, train_ic, n_tickers,
                  n_features, training_window_years, artifact_path
           FROM training_runs
           WHERE artifact_type = ?
             AND artifact_path LIKE ?
           ORDER BY run_date""",
        conn, params=(artifact_type, pattern),
    )
    conn.close()
    if df.empty:
        return df
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df


def compute_drift_signals(
    df: pd.DataFrame,
    window_size: int = 12,
    drift_sigma: float = 1.0,
) -> pd.DataFrame:
    """Add rolling mean / std / drift flag columns."""
    out = df.copy().sort_values("run_date").reset_index(drop=True)
    out["roll_mean"] = out["oos_mean_ic"].rolling(window_size, min_periods=3).mean()
    out["roll_std"] = out["oos_mean_ic"].rolling(window_size, min_periods=3).std(ddof=1)
    # Drift = current value below (rolling mean - drift_sigma × rolling std)
    out["drift_lower"] = out["roll_mean"] - drift_sigma * out["roll_std"]
    out["is_drift"] = (out["oos_mean_ic"] < out["drift_lower"]).fillna(False)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--pattern", default="%panel-ltr.json%",
        help="LIKE pattern for artifact_path (default: production panel-ltr).",
    )
    p.add_argument("--artifact-type", default="panel-ltr")
    p.add_argument("--window-size", type=int, default=12,
                   help="Rolling window for mean/std (in retrain runs).")
    p.add_argument("--drift-sigma", type=float, default=1.0,
                   help="Drift flag triggers when current < mean - σ × std.")
    p.add_argument("--out", default=None, help="CSV output (optional).")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: db not found at {db_path}")
        return 2

    df = load_training_runs(db_path, args.pattern, args.artifact_type)
    if df.empty:
        print(f"No training runs match pattern '{args.pattern}' "
              f"+ artifact_type '{args.artifact_type}'.")
        return 1

    print(f"Loaded {len(df)} runs. Date range: "
          f"{df['run_date'].min().date()} → {df['run_date'].max().date()}")
    print()

    out = compute_drift_signals(df, args.window_size, args.drift_sigma)

    print(f"{'run_date':<22}{'oos_ic':>9}{'roll_mean':>11}{'roll_std':>10}"
          f"{'drift?':>8}{'n_feat':>8}{'n_tickers':>11}")
    print("-" * 80)
    for _, r in out.iterrows():
        flag = "⚠️" if r["is_drift"] else ""
        rm = f"{r['roll_mean']:+.4f}" if pd.notna(r["roll_mean"]) else "—"
        rs = f"{r['roll_std']:.4f}"  if pd.notna(r["roll_std"])  else "—"
        nf = int(r["n_features"]) if pd.notna(r["n_features"]) else 0
        nt = int(r["n_tickers"]) if pd.notna(r["n_tickers"]) else 0
        print(f"{str(r['run_date'])[:19]:<22}{r['oos_mean_ic']:>+9.4f}"
              f"{rm:>11}{rs:>10}{flag:>8}{nf:>8}{nt:>11}")
    print()

    n_drift = int(out["is_drift"].sum())
    if n_drift:
        print(f"⚠️  Drift flagged on {n_drift} retrain(s) — "
              f"OOS IC fell > {args.drift_sigma}σ below rolling-{args.window_size} mean.")
    else:
        print("✓ No drift flags in window.")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nWrote CSV → {args.out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
