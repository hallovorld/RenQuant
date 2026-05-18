#!/usr/bin/env python3
"""Vol-adjusted label NGB retest (roadmap #5, 2026-05-17).

Hypothesis: `fwd_60d_excess_raw / vol_60d` is a more stable label than
`fwd_60d_excess_raw` because it reduces heteroscedasticity-driven noise.

Reference:
  - Lim et al 2021 ICLR §3.4 (vol-normalization for stable quantile estimation)
  - Qlib `qlib/contrib/data/handler.py` (vol-targeted label transformation)

Method:
  1. Load panel + compute rolling vol_60d per ticker (annualized via √252).
  2. Create label `fwd_60d_excess_volnorm = fwd_60d_excess_raw / vol_60d`.
  3. Train NGB single-seed (seed=42) with SAME hyperparams as the proper
     baseline (Duan 2020 §4 large-data config).
  4. Compare val_IC vs baseline (+0.0352 from 5/17 same-day raw label).
  5. Don't auto-promote — write artifact to sim path for inspection.
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
log = logging.getLogger("ngb-vol-adj")
REPO = Path(__file__).resolve().parent.parent
LABEL_RAW = "fwd_60d_excess_raw"
LABEL_NEW = "fwd_60d_excess_volnorm"
SEED = 42


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    feat_cols  = list(json.loads(art_panel.read_text())["feature_cols"])
    log.info("loading panel %s", panel_path.name)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL_RAW])
    log.info("panel: %d rows × %d tickers", len(panel), panel["ticker"].nunique())

    # ── compute vol_60d per ticker ────────────────────────────────────────
    # Need daily returns to compute rolling vol. Use the panel's
    # daily-frequency rows. Sort within ticker by date, take std of last 60
    # rows of fwd_5d_excess as a noisy daily-return proxy if true daily
    # returns aren't on panel, OR derive from STD60 if alpha158 supplies it.
    if "STD60" in panel.columns:
        log.info("Using alpha158 STD60 as 60-day vol proxy")
        panel["vol_60d"] = panel["STD60"].abs()
    elif "STD20" in panel.columns:
        log.info("Using alpha158 STD20 as 20-day vol proxy (STD60 missing)")
        panel["vol_60d"] = panel["STD20"].abs()
    else:
        log.info("No STD col in panel — computing vol from fwd_5d_excess as proxy")
        panel = panel.sort_values(["ticker", "date"])
        panel["vol_60d"] = (
            panel.groupby("ticker")["fwd_5d_excess"]
            .rolling(60, min_periods=20)
            .std()
            .reset_index(0, drop=True)
            * np.sqrt(252)
        )

    # Avoid division blowup
    panel["vol_60d_safe"] = panel["vol_60d"].clip(lower=0.05)  # floor at 5%
    panel[LABEL_NEW] = panel[LABEL_RAW] / panel["vol_60d_safe"]
    log.info("Label stats RAW:     mean=%+.4f std=%.4f",
             panel[LABEL_RAW].mean(), panel[LABEL_RAW].std())
    log.info("Label stats VOLNORM: mean=%+.4f std=%.4f",
             panel[LABEL_NEW].mean(), panel[LABEL_NEW].std())
    panel = panel.dropna(subset=[LABEL_NEW])
    log.info("Post vol-norm rows: %d", len(panel))

    # ── purged train/val split (same as proper baseline) ─────────────────
    HORIZON = 60
    dates = sorted(panel["date"].unique())
    val_idx = int(len(dates) * 0.8)
    val_cut = dates[val_idx]
    train_cut = dates[max(0, val_idx - HORIZON)]
    train = panel[panel["date"] <= train_cut].copy()
    val   = panel[panel["date"] >  val_cut].copy()
    log.info("Train PURGED %d rows (≤ %s) | Val %d rows (> %s)",
             len(train), train_cut.date(), len(val), val_cut.date())

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    Xva = val[feat_cols].fillna(0).values.astype(np.float64)
    # Clip volnorm label to ±5 (after volnorm, +0.20/0.05=4 is plausible)
    ytr = train[LABEL_NEW].clip(-5, 5).values.astype(np.float64)
    yva = val[LABEL_NEW].clip(-5, 5).values.astype(np.float64)
    val_dates = val["date"].values

    # ── train NGB (same config as 5/17 proper baseline) ──────────────────
    log.info("Fitting NGBoost (seed=%d) on VOLNORM label ...", SEED)
    t0 = time.time()
    model = NGBRegressor(
        Dist=Normal, Score=LogScore,
        Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, splitter="best"),
        natural_gradient=True,
        n_estimators=500, learning_rate=0.1, minibatch_frac=0.1,
        col_sample=1.0, verbose=False,
        random_state=SEED,
        validation_fraction=0.1, early_stopping_rounds=20,
    )
    model.fit(Xtr, ytr, X_val=Xva, Y_val=yva)
    elapsed = time.time() - t0

    dist = model.pred_dist(Xva)
    mu_va, sigma_va = dist.loc, dist.scale
    val_ic = cs_ic(mu_va, yva, val_dates)
    sigma_calib = float(spearmanr(sigma_va, np.abs(yva - mu_va))[0])
    bi = model.best_val_loss_itr or model.n_estimators
    log.info("=" * 60)
    log.info("VOLNORM result (single seed=%d):", SEED)
    log.info("  val_ic=%+.4f σ-calib=%+.3f best_iter=%d (%.1fs)",
             val_ic, sigma_calib, bi, elapsed)
    log.info("  vs baseline RAW (5/17 same-seed): +0.0352")
    delta = val_ic - 0.0352
    log.info("  Δ(VOLNORM - RAW) = %+.4f", delta)
    if delta > +0.003:
        log.info("  ✓ MEANINGFUL LIFT (+0.003+ IC) — investigate full 5-seed retest")
    elif delta > +0.001:
        log.info("  ? marginal lift — within noise; need 5-seed std to call it")
    else:
        log.info("  ✗ NO LIFT (or worse) — keep raw label, this hypothesis fails")
    log.info("")
    log.info("Note: VOLNORM val_ic is a Spearman corr on the VOLNORM scale, not directly")
    log.info("comparable to RAW val_ic dollar-for-dollar. Spearman is rank-invariant, so")
    log.info("a rank-preserving label transform should produce ≈ same IC if there's no")
    log.info("noise-reduction benefit. Lift > 0 means rank order is more stable.")

    # Quality gate: refuse save if val_ic < XGB baseline 0.0294
    if val_ic < 0.0294:
        log.warning("✗ QUALITY GATE FAILED — val_ic=%+.4f < XGB baseline 0.0294. Not saved.",
                    val_ic)
        return 1

    # Save to sim path for inspection (NOT prod — this is exploratory)
    import base64, pickle
    blob = base64.b64encode(pickle.dumps(model)).decode("ascii")
    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    fp_fields = {
        "feature_cols": feat_cols, "label_col": LABEL_NEW,
        "seed": SEED, "n_estimators": 500, "best_iter": int(bi),
    }
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str).encode()).hexdigest()[:16]
    artifact = {
        "version": 1, "kind": "ngboost_head_volnorm",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "regressor_pickle_b64": blob, "feature_medians": medians.tolist(),
        "label_col": LABEL_NEW,
        "val_mu_ic": val_ic, "val_sigma_calib": sigma_calib,
        "best_iter": int(bi),
        "training_notes": (
            f"Vol-adjusted label retest (roadmap #5 / 2026-05-17). "
            f"Label = fwd_60d_excess_raw / vol_60d. Single seed={SEED}. "
            f"Compare to 5/17 raw-label proper baseline val_ic=+0.0352."
        ),
        "config_fingerprint": f"sha256:{fp}",
    }
    out = REPO / "backtesting/renquant_104/artifacts/sim/ngboost-head-volnorm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact))
    log.info("✓ Saved → %s (size %.1f MB, fingerprint %s)",
             out, out.stat().st_size / 1e6, fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
