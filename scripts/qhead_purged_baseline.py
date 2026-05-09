#!/usr/bin/env python
"""Phase D1 — Purged train/val baseline (Lopez de Prado §7).

Prior baseline E51 had val_ic +0.0294 ± 0.0029 on 80/20 date split.
Methodology bug: training rows in [val_cut - 60d, val_cut] have
fwd_60d_excess_raw labels whose forward window OVERLAPS the val
period. Model "sees" val period via labels at the training boundary.

Per Lopez de Prado AFML §7 (Purged Train/Test Split):
  - Drop training rows with date > (val_cut - h) where h = label horizon
  - Optionally embargo h additional days (we already do via dropna)

This script measures the CLEAN val_ic on properly-purged training data,
across 5 seeds. Compares to E51's 0.0294 to see if the inflated
estimate was due to leakage.

Also runs the §5.2 placebo check on purged data — if placebo IC
drops significantly, the prior 42% persistence ratio was inflated
by leakage at the boundary.

Reference: López de Prado 2018 "Advances in Financial Machine Learning"
Chapter 7, "Cross-Validation in Finance".
"""
from __future__ import annotations
import json, time, sys, logging
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("purged-baseline")

REPO = Path(__file__).resolve().parent.parent
QUANTILES = [0.16, 0.50, 0.84]
LABEL = "fwd_60d_excess_raw"
HORIZON = 60   # trading days


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def fit_xgb_quantile(Xtr, ytr, Xva, yva, seed=42):
    boosters = {}
    for q in QUANTILES:
        m = xgb.XGBRegressor(
            objective="reg:quantileerror", tree_method="hist", n_estimators=200,
            max_depth=5, learning_rate=0.05, min_child_weight=50,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            n_jobs=10, random_state=seed, quantile_alpha=q,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        boosters[q] = m.get_booster()
    return boosters


def predict_q(boosters, X):
    D = xgb.DMatrix(X)
    return {q: boosters[q].predict(D) for q in QUANTILES}


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"

    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    distinct_dates = sorted(panel["date"].unique())

    # Original 80/20 cut
    val_cut_idx = int(len(distinct_dates) * 0.8)
    val_cut = distinct_dates[val_cut_idx]
    log.info("val_cut (80%% date) = %s", val_cut.date())

    # Purged train cutoff: val_cut - HORIZON trading days
    train_cut_idx = max(0, val_cut_idx - HORIZON)
    train_cut = distinct_dates[train_cut_idx]
    log.info("train_cut (purged) = %s  (val_cut - %d trading days)",
             train_cut.date(), HORIZON)

    # Original train/val (NO PURGE — for comparison)
    train_orig = panel[panel["date"] <= val_cut].copy()
    val        = panel[panel["date"] >  val_cut].copy()
    # Purged train (DROP last 60d to prevent label leakage into val)
    train_purged = panel[panel["date"] <= train_cut].copy()

    log.info("Train ORIG (no purge): %d rows, dates %s..%s",
             len(train_orig), train_orig.date.min().date(), train_orig.date.max().date())
    log.info("Train PURGED:          %d rows, dates %s..%s  (-%d rows / %.1f%% dropped)",
             len(train_purged), train_purged.date.min().date(), train_purged.date.max().date(),
             len(train_orig) - len(train_purged),
             100 * (len(train_orig) - len(train_purged)) / len(train_orig))
    log.info("Val:                    %d rows, dates %s..%s",
             len(val), val.date.min().date(), val.date.max().date())

    Xva = val[feat_cols].fillna(0).values.astype(np.float32)
    yva_raw = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates = val["date"].values

    # ─── Original baseline (NO PURGE) — repro of E51 +0.0294 ───
    log.info("\n══ ORIG (no purge) — 5-seed baseline ══")
    Xtr = train_orig[feat_cols].fillna(0).values.astype(np.float32)
    ytr = train_orig[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    orig_ics = []
    for s in [42, 7, 123, 2024, 31415]:
        t0 = time.time()
        boosters = fit_xgb_quantile(Xtr, ytr, Xva, yva_raw, s)
        qva = predict_q(boosters, Xva)
        ic = cs_ic(qva[0.5], yva_raw, val_dates)
        orig_ics.append(ic)
        log.info("  ORIG seed=%-5d val_ic=%+.4f (%.1fs)", s, ic, time.time()-t0)
    log.info("ORIG  mean=%+.4f  std=%.4f  (E51 had +0.0294 ± 0.0029)",
             np.mean(orig_ics), np.std(orig_ics, ddof=1))

    # ─── PURGED baseline — clean val_ic ───
    log.info("\n══ PURGED (-60d horizon) — 5-seed baseline ══")
    Xtr_p = train_purged[feat_cols].fillna(0).values.astype(np.float32)
    ytr_p = train_purged[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    purged_ics = []
    for s in [42, 7, 123, 2024, 31415]:
        t0 = time.time()
        boosters = fit_xgb_quantile(Xtr_p, ytr_p, Xva, yva_raw, s)
        qva = predict_q(boosters, Xva)
        ic = cs_ic(qva[0.5], yva_raw, val_dates)
        purged_ics.append(ic)
        log.info("  PURGED seed=%-5d val_ic=%+.4f (%.1fs)", s, ic, time.time()-t0)
    purged_mean = float(np.mean(purged_ics)); purged_std = float(np.std(purged_ics, ddof=1))
    log.info("PURGED  mean=%+.4f  std=%.4f", purged_mean, purged_std)

    # ─── Compare ───
    orig_mean = float(np.mean(orig_ics)); orig_std = float(np.std(orig_ics, ddof=1))
    delta = purged_mean - orig_mean
    se = float(np.sqrt(orig_std**2 / 5 + purged_std**2 / 5))
    t = delta / se if se > 0 else float("inf")
    log.info("\n══ AUDIT VERDICT ══")
    log.info("  ORIG (with leakage):   %+.4f ± %.4f", orig_mean, orig_std)
    log.info("  PURGED (clean):        %+.4f ± %.4f", purged_mean, purged_std)
    log.info("  Δ(PURGED - ORIG):      %+.4f  (t-stat %+.2f)", delta, t)
    if abs(t) > 2.0 and delta < 0:
        log.info("  ✗ ORIG was inflated by label leakage (PURGED < ORIG significantly).")
    elif abs(t) > 2.0 and delta > 0:
        log.info("  ?? PURGED beat ORIG significantly — unexpected, audit further.")
    else:
        log.info("  ✓ No significant difference — purge has minimal effect on this panel.")

    # ─── Placebo on purged data ───
    log.info("\n══ §5.2 placebo (purged data, seed=42) ══")
    panel_s = panel.sort_values(["ticker", "date"]).copy()
    panel_s["__shift__"] = panel_s.groupby("ticker")[LABEL].shift(-HORIZON)
    panel_s = panel_s.dropna(subset=["__shift__"])
    train_p = panel_s[panel_s["date"] <= train_cut].copy()
    val_p   = panel_s[panel_s["date"] >  val_cut].copy()
    Xtr_pl = train_p[feat_cols].fillna(0).values.astype(np.float32)
    Xva_pl = val_p[feat_cols].fillna(0).values.astype(np.float32)
    ytr_pl = train_p["__shift__"].clip(-0.5, 0.5).values.astype(np.float32)
    yva_pl = val_p["__shift__"].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates_pl = val_p["date"].values
    log.info("  Placebo train=%d val=%d", len(Xtr_pl), len(Xva_pl))
    boosters = fit_xgb_quantile(Xtr_pl, ytr_pl, Xva_pl, yva_pl, seed=42)
    qva_pl = predict_q(boosters, Xva_pl)
    placebo_ic = cs_ic(qva_pl[0.5], yva_pl, val_dates_pl)
    log.info("  Placebo val_ic = %+.4f (E52 had +0.0368 unpurged)", placebo_ic)

    # Cross-eval on purged data: real-trained model predictions vs placebo y
    boosters_real = fit_xgb_quantile(Xtr_p, ytr_p, Xva_pl, yva_pl, seed=42)
    qva_real_on_placebo = predict_q(boosters_real, Xva_pl)
    cross_ic = cs_ic(qva_real_on_placebo[0.5], yva_pl, val_dates_pl)
    real_ic_p = cs_ic(qva_real_on_placebo[0.5], val_p[LABEL].clip(-0.5, 0.5).values, val_dates_pl)
    persist_pct = 100 * cross_ic / real_ic_p if real_ic_p > 0 else float("nan")
    log.info("  Cross-eval (purged real → placebo y): %+.4f", cross_ic)
    log.info("  Persistence ratio = %.0f%% (E52 had 42%%)", persist_pct)


if __name__ == "__main__":
    sys.exit(main())
