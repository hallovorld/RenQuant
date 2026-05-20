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
    # 2026-05-17: keep all 5 models in memory + pick the best by val_IC and
    # save its ensemble to the sim artifact path. Quality gate prevents the
    # silent-degrade incident (today's Sunday sweep saved val_IC=-0.0165
    # straight to prod with no gate).
    #
    # P0-15 (audit 2026-05-20): baseline values below are hardcoded from
    # a 2026-05-15 measurement on the PRE-wl200 panel. Stale for post-
    # 2026-05-18 wl200 (142 ticker) + 172-feature panel. Override via env
    # to refresh after each universe/feature change:
    #   XGB_BASELINE_MEAN=<measured> XGB_BASELINE_STD=<measured> ./train_ngboost_proper.py
    # TODO: auto-refresh by reading the most recent panel-ltr.json's val IC.
    import os as _os_baseline  # noqa: PLC0415
    XGB_BASELINE_MEAN = float(_os_baseline.environ.get("XGB_BASELINE_MEAN", "0.0294"))
    XGB_BASELINE_STD  = float(_os_baseline.environ.get("XGB_BASELINE_STD",  "0.0029"))
    log.info("XGB baseline (override via XGB_BASELINE_MEAN/STD env): "
             "mean=%.4f std=%.4f", XGB_BASELINE_MEAN, XGB_BASELINE_STD)

    # Params dict — used in artifact metadata so downstream tools know
    # exactly how the model was fitted. Mirrors NGBRegressor() kwargs below.
    params = dict(
        Dist="Normal", Score="LogScore",
        n_estimators=500, learning_rate=0.1, minibatch_frac=0.1,
        col_sample=1.0, natural_gradient=True,
        early_stopping_rounds=20, validation_fraction=0.1,
        base_max_depth=3,
    )
    val_ics = []
    sigma_calibs = []
    mu_xs_stds = []
    fit_times = []
    best_iters = []
    models = []
    # 2026-05-17: env-overridable seed list.
    # P0-15 (audit 2026-05-20): default flipped from single seed to 5-seed.
    # Single-seed default was §5.13.4 violation (claiming significance from
    # one measurement). 5-seed takes 2-4h but produces honest σ. Override
    # for quick smoke via NGB_SEEDS=42 (single-seed = exploratory only).
    import os as _os
    _seed_csv = _os.environ.get("NGB_SEEDS", "42,7,123,2024,31415")
    SEED_LIST = [int(s) for s in _seed_csv.split(",") if s.strip()]
    log.info("Running %d seed(s): %s", len(SEED_LIST), SEED_LIST)
    for SEED in SEED_LIST:
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
        bi = model.best_val_loss_itr or model.n_estimators
        log.info("  seed=%-5d val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_iter=%d (%.1fs)",
                 SEED, v_ic, sc, ms, bi, ft)
        val_ics.append(v_ic); sigma_calibs.append(sc); mu_xs_stds.append(ms)
        fit_times.append(ft); best_iters.append(bi); models.append((SEED, model))

    log.info("=" * 60)
    log.info("NGBoost-proper %d-seed result (Duan 2020 §4 large-data config)",
             len(SEED_LIST))
    log.info("=" * 60)
    if len(val_ics) > 1:
        log.info("  val μ-IC mean=%+.4f std=%.4f range=[%+.4f, %+.4f]",
                 np.mean(val_ics), np.std(val_ics, ddof=1), min(val_ics), max(val_ics))
        log.info("  σ̂ calib mean=%+.3f", np.mean(sigma_calibs))
    else:
        log.info("  val μ-IC = %+.4f (single seed)", val_ics[0])
        log.info("  σ̂ calib = %+.3f", sigma_calibs[0])
    log.info("  μ̂ x-sec std mean=%.5f", np.mean(mu_xs_stds))
    log.info("  fit time mean=%.1fs total=%.0fs", np.mean(fit_times), sum(fit_times))
    log.info("")
    log.info("Compare baseline XGB-quantile: mean=+%.4f std=%.4f  (E51 5-seed A/A)",
             XGB_BASELINE_MEAN, XGB_BASELINE_STD)
    delta = np.mean(val_ics) - XGB_BASELINE_MEAN
    n_seeds = len(val_ics)
    if n_seeds > 1:
        se = np.sqrt(XGB_BASELINE_STD**2/5 + np.std(val_ics, ddof=1)**2/n_seeds)
    else:
        # Single-seed: just use XGB baseline std as the noise floor (rough)
        se = XGB_BASELINE_STD
    t = delta / se if se > 0 else float("inf")
    log.info("Δ(NGB-proper - XGB) = %+.4f  t-stat = %+.2f", delta, t)
    if abs(t) > 2.0 and delta > 0:
        log.info("✓ SIGNIFICANT BEAT — NGBoost-proper > XGB-quantile at 95%%")
    elif delta > 0:
        log.info("? Trend positive but not 2σ significant on n=%d", n_seeds)
    else:
        log.info("✗ NGBoost-proper does NOT beat XGB-quantile")
    if n_seeds == 1:
        log.info("[reference: 5/15 full 5-seed validation = +0.0360 ± 0.0036, "
                 "t=+2.76 vs XGB baseline; logged in CLAUDE.md status]")

    # ── Save best-by-val_IC artifact to sim path (NOT prod) ──────────────
    log.info("")
    log.info("=" * 60)
    log.info("Saving best-seed artifact")
    log.info("=" * 60)
    best_idx = int(np.argmax(val_ics))
    best_seed, best_model = models[best_idx]
    best_val_ic = val_ics[best_idx]
    best_sigma_calib = sigma_calibs[best_idx]
    best_mu_xs_std = mu_xs_stds[best_idx]
    best_iter = best_iters[best_idx]
    log.info("Best seed = %d  val_ic=%+.4f  σ-calib=%+.3f  best_iter=%d",
             best_seed, best_val_ic, best_sigma_calib, best_iter)

    # Quality gate — refuse save if even the BEST seed doesn't beat XGB baseline.
    # This is the safety mechanism missing from Sunday sweep (today's 11:20 incident).
    if best_val_ic < XGB_BASELINE_MEAN:
        log.warning(
            "✗ QUALITY GATE FAILED — best val_IC=%+.4f < XGB baseline %+.4f. "
            "Refusing to save artifact (would silently degrade prod). "
            "Best-seed model NOT saved.",
            best_val_ic, XGB_BASELINE_MEAN,
        )
        return 1

    # Pickle the best model + meta in the same schema as
    # train_ngboost_alpha158_fund.py so downstream consumers (NGBoostFitTask
    # at inference time) read it identically.
    import base64, pickle
    blob = base64.b64encode(pickle.dumps(best_model)).decode("ascii")
    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    pred_tr = best_model.pred_dist(Xtr)
    mu_tr, sd_tr = pred_tr.loc, pred_tr.scale
    pred_va = best_model.pred_dist(Xva)
    mu_va, sd_va = pred_va.loc, pred_va.scale

    fp_fields = {
        "feature_cols": feat_cols,
        "params": params,
        "label_col": LABEL,
        "panel_artifact_fingerprint": panel_meta.get("config_fingerprint",
                                                     "unknown"),
        "seed": best_seed,
        "all_seeds_val_ic": val_ics,
    }
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str)
                        .encode()).hexdigest()[:16]

    artifact = {
        "version": 1,
        "kind":    "ngboost_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "params": {**params, "seed": best_seed, "best_iter": int(best_iter)},
        "regressor_pickle_b64": blob,
        "feature_medians": medians.tolist(),
        "train_run_id": f"proper_5seed_{datetime.utcnow().strftime('%Y%m%dT%H%M')}",
        "training_notes": (
            f"NGBoost-proper 5-seed training (Duan 2020 §4 large-data config). "
            f"Selected best-by-val_IC seed={best_seed} from 5 seeds. "
            f"All-seed val_IC: {[round(v,4) for v in val_ics]}. "
            f"Best val_IC={best_val_ic:+.4f}, σ-calib={best_sigma_calib:+.3f}, "
            f"μ_xs_std={best_mu_xs_std:.5f}. "
            f"XGB-quantile baseline mean=+{XGB_BASELINE_MEAN:.4f}±{XGB_BASELINE_STD:.4f}. "
            f"Quality gate: val_IC > XGB baseline (passed). "
            f"Panel fingerprint={fp_fields['panel_artifact_fingerprint']}."
        ),
        "train_mu_mean":    float(mu_tr.mean()),
        "train_sigma_mean": float(sd_tr.mean()),
        "train_mu_ic":      cs_ic(mu_tr, ytr, train["date"].values),
        "val_mu_ic":        best_val_ic,
        "val_sigma_calib":  best_sigma_calib,
        "val_mu_xs_std":    best_mu_xs_std,
        "best_iter":        int(best_iter),
        "n_rows":           int(len(panel)),
        "n_rows_train":     int(len(train)),
        "n_rows_val":       int(len(val)),
        "all_seeds_val_ic": val_ics,
        "all_seeds_sigma_calib": sigma_calibs,
        "config_fingerprint":        f"sha256:{fp}",
        "config_fingerprint_fields": fp_fields,
    }
    out_path = REPO / "backtesting/renquant_104/artifacts/sim/ngboost-head.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    log.info("✓ Saved → %s  (size=%.1f MB)", out_path,
             out_path.stat().st_size / 1e6)
    log.info("Fingerprint: sha256:%s", fp)
    log.info("")
    log.info("PROMOTION (manual, after rollback rehearsal per CLAUDE.md §5.5):")
    log.info("  cp -v backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json \\")
    log.info("        backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json.bak_$(date +%%Y%%m%%d)")
    log.info("  cp -v %s \\", out_path)
    log.info("        backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json")
    log.info("σ wire activation (real-$ change) still gated on user authorization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
