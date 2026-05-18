#!/usr/bin/env python3
"""Quick OOS val_IC eval for B4 multi-horizon retest on fresh panel.

Compares fwd_5d / fwd_20d / fwd_60d val_IC on the regenerated panel
(post-A7 fix, 2026-05-18). Decides whether multi-horizon ensemble
warrants the 1-week E42 retest.

References:
  - Bali-Cakici-Whitelaw 2011 JF "Maxing Out: Stocks as Lotteries" —
    multi-horizon return aggregation.
  - Qlib reference: alpha158 dataset uses 5/20/60-day horizons for
    multi-target training; this script measures the simplest 1-of-3
    selection rather than ensemble.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent


def load_booster_predict(art_path: Path, X: np.ndarray) -> np.ndarray:
    import tempfile, os
    art = json.loads(art_path.read_text())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(art["booster_raw_json"])
        bp = f.name
    booster = xgb.Booster()
    booster.load_model(bp)
    os.unlink(bp)
    # Apply normalization (z-score using stored stats)
    feat_cols = art["feature_cols"]
    means = np.array(art.get("feature_means", [0.0] * len(feat_cols)))
    stds  = np.array(art.get("feature_stds",  [1.0] * len(feat_cols)))
    stds = np.where(stds < 1e-8, 1.0, stds)
    X_z = (X - means) / stds
    return booster.predict(xgb.DMatrix(X_z)), feat_cols


def cs_ic_per_date(scores: np.ndarray, y: np.ndarray, dates) -> float:
    df = pd.DataFrame({"s": scores, "y": y, "d": dates})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["s"], g["y"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def main():
    print("Loading regenerated panel (post-A7)...")
    panel = pd.read_parquet(REPO / "data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    val = panel[panel["date"] > pd.Timestamp("2024-02-01")].copy()
    print(f"  val rows: {len(val):,} ({val['date'].min().date()} → {val['date'].max().date()})")

    horizons = [
        ("fwd_60d (baseline)", "fwd_60d_excess_raw",
         REPO / "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json"),
        ("fwd_20d (B4 retest)", "fwd_20d_excess",
         REPO / "backtesting/renquant_104/artifacts/walkforward_b4_fwd20d/2024-02-01/panel-ltr.json"),
    ]
    results = []
    for name, label_col, art_path in horizons:
        if not art_path.exists():
            print(f"  SKIP {name}: artifact missing at {art_path}")
            continue
        if label_col not in val.columns:
            print(f"  SKIP {name}: label {label_col} not in panel")
            continue
        sub = val.dropna(subset=[label_col]).copy()
        art = json.loads(art_path.read_text())
        feat_cols = art["feature_cols"]
        X = sub[feat_cols].fillna(0).values.astype(np.float64)
        y = sub[label_col].values.astype(np.float64)
        scores, _ = load_booster_predict(art_path, X)
        ic = cs_ic_per_date(scores, y, sub["date"].values)
        # Clipped label for direct comparability with NGB proper baseline
        y_c = np.clip(y, -0.5, 0.5)
        ic_c = cs_ic_per_date(scores, y_c, sub["date"].values)
        results.append((name, ic, ic_c, len(sub)))
        print(f"  {name}: val_IC={ic:+.4f}  val_IC(clipped)={ic_c:+.4f}  n_rows={len(sub):,}")

    print()
    print("=" * 70)
    if len(results) >= 2:
        base_name, base_ic, _, _ = results[0]
        for name, ic, ic_c, n in results[1:]:
            delta = ic - base_ic
            sig = "MEANINGFUL LIFT" if delta > +0.003 else ("WORSE" if delta < -0.003 else "WITHIN NOISE")
            print(f"  {name} Δ vs {base_name}: {delta:+.4f}  → {sig}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
