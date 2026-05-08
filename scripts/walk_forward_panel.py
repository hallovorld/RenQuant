#!/usr/bin/env python
"""Walk-forward IC validation per López de Prado's standard method.

7 cuts × {Linear OLS, Ridge, XGBoost} on 291-ticker alpha158 + fundamentals.
Each cut: train on rolling 3-year window, 21-day embargo, test on 1 year.

Output: per-cut IC (mean and median), aggregate stats (mean ± std across cuts).
This distinguishes:
  - Stable signal (low IC std across cuts)
  - Regime-dependent signal (high IC std, certain cuts only)
  - Pure overfitting (all cuts low/negative)

Usage:
    python scripts/walk_forward_panel.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("walk-forward")

REPO = Path(__file__).resolve().parent.parent

LABEL = "fwd_20d_excess"
EMBARGO_DAYS = 21    # 1 month embargo to prevent fwd_20d label leakage
TRAIN_YEARS = 3
TEST_YEARS = 1

# Cuts: (train_start, train_end, test_start, test_end)
# Each test window is 1 year, train is the prior 3 years (rolling)
CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]


def cs_rank_ic(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> tuple[float, float, int]:
    """Cross-sectional rank IC per date, return (mean, median, n_dates)."""
    df = pd.DataFrame({"p": pred, "y": actual, "date": dates})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["p"], g["y"])
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return np.nan, np.nan, 0
    return float(np.mean(ics)), float(np.median(ics)), len(ics)


def evaluate_cut(panel: pd.DataFrame, feat_cols: list[str],
                 cut: tuple[str, str, str, str]) -> dict:
    """Train on [tr_start, tr_end], test on [te_start, te_end]."""
    tr_start, tr_end, te_start, te_end = cut

    train = panel[(panel["date"] >= tr_start) & (panel["date"] <= tr_end)].dropna(subset=[LABEL])
    test  = panel[(panel["date"] >= te_start) & (panel["date"] <= te_end)].dropna(subset=[LABEL])

    if len(train) < 1000 or len(test) < 100:
        return {"cut": cut, "error": f"insufficient data train={len(train)} test={len(test)}"}

    # Re-normalize features per cut using train-only stats — proper no-leak
    X_tr = train[feat_cols].fillna(0).values.astype(np.float64)
    y_tr = train[LABEL].clip(-5, 5).values.astype(np.float64)
    X_te = test[feat_cols].fillna(0).values.astype(np.float64)
    y_te = test[LABEL].values
    te_dates = test["date"].values

    # Standardize features per cut (train-only stats)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    X_tr_n = ((X_tr - mu) / sd).clip(-5, 5)
    X_te_n = ((X_te - mu) / sd).clip(-5, 5)

    results = {"cut": cut, "train_size": len(train), "test_size": len(test)}

    # ── Linear OLS ──
    m_ols = LinearRegression().fit(X_tr_n, y_tr)
    p_ols = m_ols.predict(X_te_n)
    ic_mean, ic_med, n = cs_rank_ic(p_ols, y_te, te_dates)
    results["ols"] = {"ic_mean": ic_mean, "ic_median": ic_med, "n_dates": n}

    # ── Ridge ──
    m_rdg = Ridge(alpha=1.0, solver="lsqr").fit(X_tr_n, y_tr)
    p_rdg = m_rdg.predict(X_te_n)
    ic_mean, ic_med, n = cs_rank_ic(p_rdg, y_te, te_dates)
    results["ridge"] = {"ic_mean": ic_mean, "ic_median": ic_med, "n_dates": n}

    # ── XGBoost rank:pairwise ──
    # Need group sizes for rank objective: count rows per date
    train_dates = train["date"].values
    train_dates_sorted_idx = np.argsort(train_dates)
    tr_dates_sorted = train_dates[train_dates_sorted_idx]
    X_tr_sorted = X_tr_n[train_dates_sorted_idx]
    y_tr_sorted = y_tr[train_dates_sorted_idx]
    _, group_sizes = np.unique(tr_dates_sorted, return_counts=True)

    dtr = xgb.DMatrix(X_tr_sorted, label=y_tr_sorted)
    dtr.set_group(group_sizes)
    dte = xgb.DMatrix(X_te_n)

    params = {"objective": "rank:pairwise", "eta": 0.05, "max_depth": 5,
              "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7,
              "nthread": 8, "verbosity": 0}
    model = xgb.train(params, dtr, num_boost_round=100)
    p_xgb = model.predict(dte)
    ic_mean, ic_med, n = cs_rank_ic(p_xgb, y_te, te_dates)
    results["xgb"] = {"ic_mean": ic_mean, "ic_median": ic_med, "n_dates": n}

    return results


def main():
    log.info("Loading panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    log.info("Panel: %d rows × %d features, %d tickers, dates %s → %s",
             len(panel), len(feat_cols), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    all_results = []
    for i, cut in enumerate(CUTS, 1):
        log.info("Cut %d/%d: train=[%s, %s] test=[%s, %s]", i, len(CUTS), *cut)
        r = evaluate_cut(panel, feat_cols, cut)
        all_results.append(r)
        if "error" in r:
            log.warning("  %s", r["error"])
            continue
        log.info("  OLS   ic_mean=%+.4f median=%+.4f n=%d",
                 r["ols"]["ic_mean"], r["ols"]["ic_median"], r["ols"]["n_dates"])
        log.info("  Ridge ic_mean=%+.4f median=%+.4f n=%d",
                 r["ridge"]["ic_mean"], r["ridge"]["ic_median"], r["ridge"]["n_dates"])
        log.info("  XGB   ic_mean=%+.4f median=%+.4f n=%d",
                 r["xgb"]["ic_mean"], r["xgb"]["ic_median"], r["xgb"]["n_dates"])

    # Aggregate
    log.info("\n══ Walk-Forward Summary (7-cut, fwd_20d, 291-ticker + fundamentals) ══")
    for model_key in ("ols", "ridge", "xgb"):
        ics = [r[model_key]["ic_mean"] for r in all_results
               if "error" not in r and not np.isnan(r[model_key]["ic_mean"])]
        if not ics:
            continue
        log.info("%-6s  mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f  per-cut=%s",
                 model_key.upper(), np.mean(ics), np.std(ics),
                 min(ics), max(ics),
                 [f"{x:+.4f}" for x in ics])

    out = REPO / "data" / "walk_forward_results.json"
    out.write_text(json.dumps([{**r, "cut": list(r["cut"])} for r in all_results],
                              indent=2, default=str))
    log.info("Detailed: %s", out)


if __name__ == "__main__":
    main()
