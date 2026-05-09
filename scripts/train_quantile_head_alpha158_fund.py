#!/usr/bin/env python
"""Replace NGBoost μ/σ head with XGBoost-quantile (multi-threaded).

NGBoost-Normal on 516k×163 is single-threaded → 1h+ wallclock didn't
finish on M2 Pro. Per CLAUDE.md §5.12 prefer canonical OS solutions
that match local-hardware constraints.

Method (Lim et al. 2021 TFT §3, Koenker & Bassett 1978 quantile regression):
    Fit 3 XGBoost regressors with `objective="reg:quantileerror"`:
        q=0.16  → μ − σ      (lower 1σ band)
        q=0.50  → μ          (median, used as μ̂)
        q=0.84  → μ + σ      (upper 1σ band)

Parametric Gaussian recovery (Wakefield 2013 §3.4):
    σ̂ = (q_0.84 − q_0.16) / 2

Multi-threaded (nthread=10) → ~30s on M2 Pro vs NGBoost 1h+.

Output artifact format compatible with existing production pipeline:
    backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json
    kind="ngboost_head"  (downstream ApplyNGBoostTask reads it as-is via
                          a quantile-aware load path added separately —
                          see training_panel/quantile_head.py)
"""
from __future__ import annotations
import json, logging, sys, time, hashlib, base64, pickle
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-quantile-head")

QUANTILES = [0.16, 0.50, 0.84]   # ±1σ + median (Gaussian)
PARAMS = {
    "objective": "reg:quantileerror",
    "tree_method": "hist",
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "n_jobs": 10,
    "random_state": 42,
}


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json"
    LABEL = "fwd_60d_excess"

    log.info("Loading panel + production XGB artifact for fingerprint match...")
    panel_meta = json.loads(art_panel.read_text())
    feat_cols  = list(panel_meta["feature_cols"])
    panel_fp   = panel_meta["config_fingerprint"]
    log.info("XGB-LTR fingerprint=%s n_features=%d", panel_fp, len(feat_cols))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # Time-ordered 80/20 split by distinct dates (no leakage)
    distinct_dates = sorted(panel["date"].unique())
    val_cut_date = distinct_dates[int(len(distinct_dates) * 0.8)]
    train = panel[panel["date"] <= val_cut_date]
    val   = panel[panel["date"] >  val_cut_date]
    log.info("Train ≤ %s (%d rows) | Val > %s (%d rows)",
             val_cut_date, len(train), val_cut_date, len(val))

    Xtr = train[feat_cols].fillna(0).values.astype(np.float32)
    ytr = train[LABEL].clip(-5, 5).values.astype(np.float32)
    Xva = val[feat_cols].fillna(0).values.astype(np.float32)
    yva = val[LABEL].clip(-5, 5).values.astype(np.float32)

    # Median imputation vector (for inference-time NaN handling, matches NGBoost convention)
    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    # ── Fit 3 quantile heads ──────────────────────────────────────────────
    boosters: dict[float, xgb.Booster] = {}
    for q in QUANTILES:
        t0 = time.time()
        params = dict(PARAMS); params["quantile_alpha"] = q
        log.info("Fitting q=%.2f ...", q)
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        boosters[q] = m.get_booster()
        log.info("  q=%.2f done in %.1fs", q, time.time() - t0)

    # ── Eval ──────────────────────────────────────────────────────────────
    def predict_quantiles(X):
        D = xgb.DMatrix(X)
        return {q: boosters[q].predict(D) for q in QUANTILES}

    qtr = predict_quantiles(Xtr)
    qva = predict_quantiles(Xva)
    mu_tr  = qtr[0.5]; sigma_tr = (qtr[0.84] - qtr[0.16]) / 2
    mu_va  = qva[0.5]; sigma_va = (qva[0.84] - qva[0.16]) / 2

    # σ must be non-negative (could be violated if quantile crossing happens —
    # rare with monotone constraints but possible). Floor at 1e-6.
    sigma_tr = np.maximum(sigma_tr, 1e-6)
    sigma_va = np.maximum(sigma_va, 1e-6)

    # μ-IC (cross-sectional, matches NGBoost script)
    from scipy.stats import spearmanr
    def cs_ic(mu, y, dates):
        df = pd.DataFrame({"p": mu, "y": y, "d": dates})
        ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
        ics = [x for x in ics if not np.isnan(x)]
        return float(np.mean(ics)) if ics else float("nan")

    train_ic = cs_ic(mu_tr, ytr, train["date"].values)
    val_ic   = cs_ic(mu_va, yva, val["date"].values)
    log.info("μ-IC train=%+.4f  val=%+.4f", train_ic, val_ic)
    log.info("σ stats train: mean=%.4f median=%.4f", sigma_tr.mean(), np.median(sigma_tr))
    log.info("σ stats val:   mean=%.4f median=%.4f", sigma_va.mean(), np.median(sigma_va))

    # σ should correlate negatively with realized error magnitude (calibration)
    abs_err_tr = np.abs(ytr - mu_tr)
    sigma_calib = float(spearmanr(sigma_tr, abs_err_tr)[0])
    log.info("σ calibration (Spearman σ vs |y-μ|) train=%.4f  (>0 = wider σ where err larger)", sigma_calib)

    # ── Save artifact ─────────────────────────────────────────────────────
    # Pickle the 3 boosters together with metadata so a single load() returns
    # everything needed. ApplyNGBoostTask's load path will dispatch on
    # `kind="quantile_head"` (separate loader added in training_panel/quantile_head.py).
    payload_obj = {
        "quantiles": QUANTILES,
        "boosters_raw": {q: bytes(boosters[q].save_raw(raw_format="json")).decode()
                         for q in QUANTILES},
        "feature_cols": feat_cols,
        "feature_medians": medians.tolist(),
    }
    blob = base64.b64encode(pickle.dumps(payload_obj)).decode("ascii")

    fp_fields = {
        "feature_cols": feat_cols,
        "params": PARAMS,
        "quantiles": QUANTILES,
        "label_col": LABEL,
        "panel_artifact_fingerprint": panel_fp,
    }
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str).encode()).hexdigest()[:16]

    artifact = {
        "version": 1,
        "kind": "quantile_head",       # downstream loader dispatches on this
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "params": PARAMS,
        "quantiles": QUANTILES,
        "regressor_pickle_b64": blob,    # name kept for NGBoost-style API parity
        "feature_medians": medians.tolist(),
        "train_run_id": f"quantile_alpha158_fund_{datetime.utcnow().strftime('%Y%m%dT%H%M')}",
        "training_notes": (
            f"3-quantile XGBoost head (q=0.16/0.50/0.84) on alpha158+5fund 163-feature "
            f"panel (matches panel-ltr.alpha158_fund.json fingerprint={panel_fp}). "
            f"μ̂=q_0.50, σ̂=(q_0.84-q_0.16)/2 (Gaussian parametric recovery). "
            f"Multi-threaded XGBoost replaces NGBoost (single-threaded 1h+ on this scale). "
            f"Train μ-IC={train_ic:+.4f}, Val μ-IC={val_ic:+.4f}, σ-calib={sigma_calib:+.4f}."
        ),
        "train_mu_mean":    float(mu_tr.mean()),
        "train_sigma_mean": float(sigma_tr.mean()),
        "train_mu_ic":      train_ic,
        "val_mu_ic":        val_ic,
        "sigma_calibration": sigma_calib,
        "n_rows":         int(len(panel)),
        "n_rows_train":   int(len(train)),
        "n_rows_val":     int(len(val)),
        "config_fingerprint":        f"sha256:{fp}",
        "config_fingerprint_fields": fp_fields,
    }
    out_path.write_text(json.dumps(artifact))
    log.info("Saved → %s  (size=%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    log.info("Fingerprint: sha256:%s", fp)


if __name__ == "__main__":
    main()
