#!/usr/bin/env python3
"""4-way + ensemble head-to-head comparing the renquant_104 panel-LTR
production LightGBM against the Rust transformer artifact, isolating
two production config questions:

  Fix A: training_window_years 5.0 → restrict to hourly-era only (~1.5y)
  Fix B: nan_prone_cols → wire up missingness indicators

Usage:
    python scripts/audit_transformer_vs_lgbm_4way.py \\
        --panel-full /tmp/real_panel.csv \\
        --panel-hourly /tmp/real_panel_hourly_era.csv \\
        --transformer-scores /tmp/rust_real_v5_hourly_era.scores.csv

Reproduces the headline finding: Fix A alone gives +164% IC over
current production, beating the transformer (which only gives +61%
over current production). See doc/experiments/rust-transformer-ic.md
for the full analysis.

The transformer score CSV is optional (skipped gracefully if missing).
Generate it via the Rust score-panel binary first (see README).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.lgbm_ltr import PanelLGBMModel, DEFAULT_PARAMS  # noqa: E402

# 17 cols with +60% NaN-rate divergence between train (2021-04→2025-04)
# and val (2025+) on the full panel — see audit doc.
DROP_COLS_HOURLY_MINUTE = [
    "morning_drift_z", "afternoon_drift_z", "vwap_premium_z",
    "vol_ratio_z", "intraday_realized_vol_z", "overnight_gap_z",
    "m_morning_drift_z", "m_morning_30min_drift_z", "m_afternoon_drift_z",
    "m_closing_30min_drift_z", "m_vwap_premium_z", "m_vol_ratio_z",
    "m_first_hour_vol_pct_z", "m_intraday_realized_vol_z",
    "m_overnight_gap_z", "m_reversal_ratio_z",
    "insider_net_buy_90d_z",
]


def per_date_ic(val_df: pd.DataFrame, preds: np.ndarray) -> tuple[float, float, int]:
    val_df = val_df.copy()
    val_df["pred"] = preds
    ics: list[float] = []
    for _, g in val_df.groupby("date"):
        if g["label"].notna().sum() < 2:
            continue
        rho, _ = spearmanr(g["pred"], g["label"])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)), float(np.median(ics)), len(ics)


def add_missingness_inds(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_is_missing"] = df[c].isna().astype(np.int8)
    return df


def split_80_20(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    n_val = int(len(dates) * 0.2)
    split_date = dates[len(dates) - n_val]
    return (
        df[df["date"] < split_date].copy(),
        df[df["date"] >= split_date].copy(),
    )


def fit_lgbm_predict(train: pd.DataFrame, val: pd.DataFrame, feat: list[str]) -> np.ndarray:
    m = PanelLGBMModel(params=DEFAULT_PARAMS, feature_cols=feat)
    m.train(
        train,
        group_sizes=train.groupby("date").size().values,
        feature_cols=feat,
        label_col="label",
        weight_col=None,
        num_boost_round=300,
    )
    return m.predict(val).values


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-full", default="/tmp/real_panel.csv")
    p.add_argument("--panel-hourly", default="/tmp/real_panel_hourly_era.csv")
    p.add_argument(
        "--transformer-scores", default=None,
        help="Optional CSV with columns date,ticker,score from Rust transformer."
    )
    args = p.parse_args()

    df_full = pd.read_csv(args.panel_full).sort_values(
        ["date", "ticker"]).reset_index(drop=True)
    df_hr = pd.read_csv(args.panel_hourly).sort_values(
        ["date", "ticker"]).reset_index(drop=True)
    print(f"FULL panel: {df_full.shape}, {df_full['date'].nunique()} dates")
    print(f"HOURLY-ERA panel: {df_hr.shape}, {df_hr['date'].nunique()} dates\n")

    rows: list[tuple[str, float, float, int, int]] = []

    # 1: current production
    train, val = split_80_20(df_full)
    feat = [c for c in train.columns if c not in ("date", "ticker", "label")]
    preds = fit_lgbm_predict(train, val, feat)
    rows.append(("1. LGBM FULL panel (current prod)", *per_date_ic(val, preds), len(feat)))
    val1 = val.copy()
    val1["pred_lgbm_full"] = preds

    # 2: Fix A — training window
    train, val = split_80_20(df_hr)
    feat = [c for c in train.columns if c not in ("date", "ticker", "label")]
    preds = fit_lgbm_predict(train, val, feat)
    rows.append(("2. LGBM HOURLY-ERA only (Fix A)", *per_date_ic(val, preds), len(feat)))
    val2 = val.copy()
    val2["pred_lgbm_hr"] = preds

    # 3: Fix B — missingness indicators on full
    df_full_mi = add_missingness_inds(df_full, DROP_COLS_HOURLY_MINUTE)
    train, val = split_80_20(df_full_mi)
    feat = [c for c in train.columns if c not in ("date", "ticker", "label")]
    preds = fit_lgbm_predict(train, val, feat)
    rows.append(("3. LGBM FULL + missingness (Fix B)", *per_date_ic(val, preds), len(feat)))

    # 4: Fix A+B
    df_hr_mi = add_missingness_inds(df_hr, DROP_COLS_HOURLY_MINUTE)
    train, val = split_80_20(df_hr_mi)
    feat = [c for c in train.columns if c not in ("date", "ticker", "label")]
    preds = fit_lgbm_predict(train, val, feat)
    rows.append(("4. LGBM HOURLY-ERA + missingness (A+B)", *per_date_ic(val, preds), len(feat)))

    # 5: Rust transformer score (if provided)
    if args.transformer_scores and Path(args.transformer_scores).exists():
        tfm = pd.read_csv(args.transformer_scores)
        merged = val2.merge(tfm[["date", "ticker", "score"]],
                            on=["date", "ticker"], how="left")
        if merged["score"].notna().any():
            rows.append((
                "5. Rust transformer v5 (HOURLY-ERA)",
                *per_date_ic(merged, merged["score"].values),
                merged["score"].notna().sum(),
            ))
            # Ensemble = 0.5 * LGBM(Fix-A) + 0.5 * transformer
            ens = 0.5 * merged["pred_lgbm_hr"].values + 0.5 * merged["score"].fillna(0).values
            rows.append((
                "6. Ensemble 50/50 (Fix A LGBM + transformer)",
                *per_date_ic(merged, ens),
                len(ens),
            ))
        else:
            print("WARN: transformer scores merged to all-NaN — skipping rows 5-6")
    else:
        print("(skipping transformer rows — pass --transformer-scores to include)\n")

    # ── Print ─────────────────────────────────────────────────────────
    baseline_mean = rows[0][1]
    print("=" * 96)
    print(f"{'Config':<48} {'val_IC':>10} {'median':>10} {'n_dates':>8} {'n_feat':>8} {'Δ%':>8}")
    print("-" * 96)
    for name, mean, med, n, nf in rows:
        delta = 100.0 * mean / baseline_mean - 100.0 if baseline_mean else float("nan")
        print(f"{name:<48} {mean:>+10.4f} {med:>+10.4f} {n:>8} {nf:>8} {delta:>+7.0f}%")
    print()
    best = max(rows, key=lambda r: r[1])
    print(f"Best: {best[0]}  val_IC={best[1]:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
