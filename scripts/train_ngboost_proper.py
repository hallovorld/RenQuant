#!/usr/bin/env python
"""Phase D2 — Proper NGBoost training (Duan 2020 large-data config).

Prior NGBoost runs in this codebase were misconfigured ("1h+ didn't
finish" → replaced with XGB-quantile). Per Duan 2020 §4 paragraph on
Year MSD (515,345 samples — similar to our 568k panel):

    "For the Year MSD dataset, being extremely large relative to the
     rest, was fit using a learning rate η of 0.1 ... For the Year MSD
     dataset we use a mini-batch size of 10%, for all other datasets we
     use 100%."

Recommended large-data config (from paper §4):
  - Distribution: Normal (loc, log-scale parameterization)
  - Base learner: DecisionTreeRegressor max_depth=3
  - Score: LogScore (NLL) — default, fastest
  - n_estimators: M chosen by val NLL via early stop
  - learning_rate: 0.1 (large data) vs 0.01 (small)
  - minibatch_frac: 0.1 (large data) vs 1.0 (small)
  - col_sample: 1.0 default

Param/sample math at our scale:
  568,563 train rows × 0.1 minibatch = 56,856 rows per iteration
  Per iter: 2 trees (loc, log-scale) × DecisionTreeRegressor depth=3
  Cost ≈ O(N · log(N) · n_features · p) per tree ≈ feasible

Compare to our XGB-quantile baseline +0.0294 ± 0.0029 (E51).
Hypothesis: NGBoost with proper config + natural gradient should
match or beat XGB-quantile on val_mu_ic, with strictly proper LogScore
(NLL) optimization.

References:
- Duan, Avati, Ding, Thai, Basu, Ng, Schuler 2020. "NGBoost: Natural
  Gradient Boosting for Probabilistic Prediction" ICML 2020.
- ngboost source: github.com/stanfordmlgroup/ngboost
"""
from __future__ import annotations
import json, time, sys, logging, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ngb-proper")

REPO = Path(__file__).resolve().parent.parent
LABEL = "fwd_60d_excess_raw"
HORIZON = 60


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"

    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    distinct_dates = sorted(panel["date"].unique())
    val_cut_idx = int(len(distinct_dates) * 0.8)
    val_cut = distinct_dates[val_cut_idx]
    # Apply purge — drop training rows whose forward window overlaps val
    train_cut_idx = max(0, val_cut_idx - HORIZON)
    train_cut = distinct_dates[train_cut_idx]
    train = panel[panel["date"] <= train_cut].copy()
    val   = panel[panel["date"] >  val_cut].copy()
    log.info("Train PURGED: %d rows (≤ %s) | Val: %d rows (> %s)",
             len(train), train_cut.date(), len(val), val_cut.date())

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)   # ngboost wants float64
    Xva = val[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    val_dates = val["date"].values

    # Paper-recommended large-data config (§4 Year MSD)
    # n_estimators=500 with early_stopping_rounds for safety
    base_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        splitter="best",
    )
    log.info("Config: Normal dist, LogScore, max_depth=3, lr=0.1, minibatch_frac=0.1, n_est=500")

    # 5-seed A/A per CLAUDE.md §5.2 — single-seed +0.0356 was promising;
    # need σ characterization to claim significance vs XGB +0.0294 ± 0.0029.
    val_ics = []
    sigma_calibs = []
    mu_xs_stds = []
    fit_times = []
    for SEED in [42, 7, 123, 2024, 31415]:
        log.info("Fitting NGBoost (seed=%d)...", SEED)
        t0 = time.time()
        model = NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            Base=DecisionTreeRegressor(
                criterion="friedman_mse",
                max_depth=3,
                splitter="best",
            ),
            natural_gradient=True,
            n_estimators=500,
            learning_rate=0.1,
            minibatch_frac=0.1,
            col_sample=1.0,
            verbose=False,        # quiet for 5-seed loop
            random_state=SEED,
            validation_fraction=0.1,
            early_stopping_rounds=20,
        )
        model.fit(Xtr, ytr, X_val=Xva, Y_val=yva)
        ft = time.time() - t0

        dist = model.pred_dist(Xva)
        mu_va = dist.loc
        sigma_va = dist.scale
        v_ic = cs_ic(mu_va, yva, val_dates)
        sc = float(spearmanr(sigma_va, np.abs(yva - mu_va))[0])
        ms = float(pd.DataFrame({"mu": mu_va, "d": val_dates}).groupby("d")["mu"].std().mean())
        log.info("  seed=%-5d val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_iter=%d (%.1fs)",
                 SEED, v_ic, sc, ms, model.best_val_loss_itr or model.n_estimators, ft)
        val_ics.append(v_ic); sigma_calibs.append(sc); mu_xs_stds.append(ms); fit_times.append(ft)

    log.info("=" * 60)
    log.info("NGBoost-proper 5-seed result (Duan 2020 §4 large-data config)")
    log.info("=" * 60)
    log.info("  val μ-IC mean=%+.4f std=%.4f range=[%+.4f, %+.4f]",
             np.mean(val_ics), np.std(val_ics, ddof=1), min(val_ics), max(val_ics))
    log.info("  σ̂ calib mean=%+.3f", np.mean(sigma_calibs))
    log.info("  μ̂ x-sec std mean=%.5f", np.mean(mu_xs_stds))
    log.info("  fit time mean=%.1fs total=%.0fs", np.mean(fit_times), sum(fit_times))
    log.info("")
    log.info("Compare baseline XGB-quantile: mean=+0.0294 std=0.0029  (E51 5-seed A/A)")
    delta = np.mean(val_ics) - 0.0294
    se = np.sqrt(0.0029**2/5 + np.std(val_ics, ddof=1)**2/5)
    t = delta / se if se > 0 else float("inf")
    log.info("Δ(NGB-proper - XGB) = %+.4f  t-stat = %+.2f", delta, t)
    if abs(t) > 2.0 and delta > 0:
        log.info("✓ SIGNIFICANT BEAT — NGBoost-proper > XGB-quantile at 95%%")
    elif delta > 0:
        log.info("? Trend positive but not 2σ significant on n=5")
    else:
        log.info("✗ NGBoost-proper does NOT beat XGB-quantile")


if __name__ == "__main__":
    sys.exit(main())
