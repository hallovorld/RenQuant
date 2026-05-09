#!/usr/bin/env python
"""§5.2 placebo (label-shift +60d) on NGBoost-proper config.

5-seed A/A on Phase D2 NGBoost showed mean val_ic = +0.0354 ± 0.0026,
significantly above XGB-quantile baseline +0.0294 ± 0.0029 (t=+3.43).
But XGB-quantile had 42% regime-persistence component (E53 cross-eval).
Need to verify NGBoost-proper isn't fitting the same persistence.

Run shifted-label (+60 trading days per ticker) and compare placebo IC
to real IC. If placebo ≈ real → NGBoost is fitting persistence, not
new alpha. If placebo ≈ 0 → genuine 60d-specific signal.

Reference: CLAUDE.md §5.2 sanity battery; Lopez de Prado AFML Ch 8.
"""
from __future__ import annotations
import json, time, sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ngb-placebo")

REPO = Path(__file__).resolve().parent.parent
LABEL = "fwd_60d_excess_raw"
HORIZON = 60


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def fit_ngb(Xtr, ytr, Xva, yva, seed=42):
    model = NGBRegressor(
        Dist=Normal, Score=LogScore,
        Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, splitter="best"),
        natural_gradient=True, n_estimators=500, learning_rate=0.1,
        minibatch_frac=0.1, col_sample=1.0, verbose=False,
        random_state=seed, validation_fraction=0.1, early_stopping_rounds=20,
    )
    model.fit(Xtr, ytr, X_val=Xva, Y_val=yva)
    return model


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])

    # Build placebo label by shifting per ticker
    panel_s = panel.sort_values(["ticker", "date"]).copy()
    panel_s["__shift__"] = panel_s.groupby("ticker")[LABEL].shift(-HORIZON)
    panel_s = panel_s.dropna(subset=["__shift__"])

    distinct_dates = sorted(panel_s["date"].unique())
    val_cut_idx = int(len(distinct_dates) * 0.8)
    val_cut = distinct_dates[val_cut_idx]
    train_cut_idx = max(0, val_cut_idx - HORIZON)
    train_cut = distinct_dates[train_cut_idx]

    train = panel_s[panel_s["date"] <= train_cut].copy()
    val   = panel_s[panel_s["date"] >  val_cut].copy()
    log.info("Train PURGED %d (≤ %s) | Val %d (> %s)",
             len(train), train_cut.date(), len(val), val_cut.date())

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    Xva = val[feat_cols].fillna(0).values.astype(np.float64)

    val_dates = val["date"].values
    yva_real = val[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    yva_placebo = val["__shift__"].clip(-0.5, 0.5).values.astype(np.float64)

    # ─── REAL: train on real fwd_60d, eval real ───
    log.info("Train NGBoost on REAL fwd_60d_excess_raw...")
    ytr_real = train[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    t0 = time.time()
    model_real = fit_ngb(Xtr, ytr_real, Xva, yva_real, seed=42)
    mu_real = model_real.pred_dist(Xva).loc
    ic_real = cs_ic(mu_real, yva_real, val_dates)
    log.info("  REAL real_ic=%+.4f (%.1fs)", ic_real, time.time() - t0)

    # ─── PLACEBO: train on shifted label, eval shifted ───
    log.info("Train NGBoost on PLACEBO (+60d shifted)...")
    ytr_placebo = train["__shift__"].clip(-0.5, 0.5).values.astype(np.float64)
    t0 = time.time()
    model_placebo = fit_ngb(Xtr, ytr_placebo, Xva, yva_placebo, seed=42)
    mu_placebo_pred = model_placebo.pred_dist(Xva).loc
    ic_placebo_self = cs_ic(mu_placebo_pred, yva_placebo, val_dates)
    log.info("  PLACEBO self_ic=%+.4f (%.1fs)", ic_placebo_self, time.time() - t0)

    # ─── CROSS-EVAL: real model μ vs placebo y (key test for persistence) ───
    ic_cross = cs_ic(mu_real, yva_placebo, val_dates)
    log.info("  CROSS (real model μ vs placebo y) = %+.4f", ic_cross)

    # Persistence ratio
    ratio = ic_cross / ic_real if ic_real > 0 else float("nan")
    log.info("")
    log.info("=" * 60)
    log.info("PHASE D2 NGB §5.2 PLACEBO VERDICT")
    log.info("=" * 60)
    log.info("  real_ic                 = %+.4f", ic_real)
    log.info("  placebo self_ic         = %+.4f", ic_placebo_self)
    log.info("  CROSS (real μ → placebo y) = %+.4f", ic_cross)
    log.info("  persistence ratio       = %.0f%%", ratio * 100 if not np.isnan(ratio) else 0)
    log.info("")
    log.info("Compare XGB-quantile (E53): real=+0.0509 cross=+0.0216 ratio=42%%")
    log.info("")
    if abs(ic_cross) < 0.010:
        log.info("✓ §5.2 PLACEBO PASS — NGB-proper captures genuine 60d alpha")
    elif ratio < 0.30:
        log.info("≈ partial pass — persistence ratio %.0f%% is acceptable", ratio * 100)
    else:
        log.info("✗ §5.2 PLACEBO FAIL — NGB-proper IC is %.0f%% persistence",
                 ratio * 100)


if __name__ == "__main__":
    main()
