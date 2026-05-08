#!/usr/bin/env python
"""Train NGBoost μ/σ head on the alpha158 + 5-fund 163-feature space.

The OLD ngboost-head.json was trained against the 21-feature production
panel; after the alpha158+fund promotion (commit ca350c0) it would
fingerprint-mismatch and was disabled. This script retrains on the new
163-feature panel so we can re-enable σ-aware QP + Kelly sizing.

Target: fwd_60d_excess (SPY-residualized 60-day forward return). This
is a proxy for `residual_return_raw` — sector-residualization is dropped
to keep this script standalone (matching scale and downstream consumers
since the head is used for ranking/sizing, not for return forecasting).

Output: backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json
        with feature_cols matching panel-ltr.alpha158_fund.json (163 feats)

Usage:
    python scripts/train_ngboost_alpha158_fund.py
"""
from __future__ import annotations
import json, logging, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-ngboost")


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json"
    LABEL = "fwd_60d_excess"

    log.info("Loading panel + production panel-LTR artifact...")
    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])
    panel_fp  = panel_meta["config_fingerprint"]
    log.info("Panel-LTR fingerprint=%s n_features=%d", panel_fp, len(feat_cols))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # Drop rows missing label
    panel = panel.dropna(subset=[LABEL])
    log.info("After dropna(label): %d rows", len(panel))

    # Same feature normalization as panel-LTR: full panel z-score (matches
    # alpha158_qlib_dataset.stats.json) — but our panel is already z-scored,
    # so we train directly on the panel values. The 5 fund cols use robust
    # z-score; the panel-LTR artifact stores the merged normalization stats.
    means = np.array(panel_meta["feature_means"])
    stds  = np.array(panel_meta["feature_stds"])

    # Time-ordered 80/20 split by distinct dates (matches NGBoostFitTask)
    distinct_dates = sorted(panel["date"].unique())
    val_cut_idx = int(len(distinct_dates) * 0.8)
    val_cut_date = distinct_dates[val_cut_idx]
    log.info("Train ≤ %s  |  Val > %s  (dates: %d train, %d val)",
             val_cut_date, val_cut_date, val_cut_idx, len(distinct_dates) - val_cut_idx)

    train = panel[panel["date"] <= val_cut_date]
    val   = panel[panel["date"] >  val_cut_date]

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)
    Xva = val[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-5, 5).values.astype(np.float64)
    yva = val[LABEL].clip(-5, 5).values.astype(np.float64)

    log.info("Train shape=%s  Val shape=%s", Xtr.shape, Xva.shape)
    log.info("Label stats train: mean=%+.4f std=%.4f", ytr.mean(), ytr.std())
    log.info("Label stats val:   mean=%+.4f std=%.4f", yva.mean(), yva.std())

    # ── Train NGBoost ──────────────────────────────────────────────────────
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    params = {
        "n_estimators":   400,
        "learning_rate":  0.01,
        "minibatch_frac": 1.0,
        "natural_gradient": True,
        "verbose":        False,
        "random_state":   17,
    }
    log.info("Training NGBoost(Normal) — params=%s", params)

    t0 = time.time()
    reg = NGBRegressor(Dist=Normal, **params)
    # Use early_stopping_rounds=25 on the validation set (mirrors PanelNGBoostJob)
    reg.fit(Xtr, ytr, X_val=Xva, Y_val=yva, early_stopping_rounds=25)
    elapsed = time.time() - t0
    best_iter = getattr(reg, "best_val_loss_itr", reg.n_estimators) or reg.n_estimators
    log.info("NGBoost fit done in %.1fs  best_iter=%d", elapsed, best_iter)

    # ── Eval ──────────────────────────────────────────────────────────────
    pred_tr = reg.pred_dist(Xtr)
    pred_va = reg.pred_dist(Xva)
    mu_tr, sd_tr = pred_tr.params["loc"], pred_tr.params["scale"]
    mu_va, sd_va = pred_va.params["loc"], pred_va.params["scale"]

    # μ → cross-sectional rank IC
    from scipy.stats import spearmanr
    def cs_ic(mu, y, dates):
        df = pd.DataFrame({"p": mu, "y": y, "date": dates})
        ics = []
        for _, g in df.groupby("date"):
            if len(g) < 5: continue
            ic, _ = spearmanr(g["p"], g["y"])
            if not np.isnan(ic): ics.append(ic)
        return float(np.mean(ics)) if ics else np.nan

    train_ic = cs_ic(mu_tr, ytr, train["date"].values)
    val_ic   = cs_ic(mu_va, yva, val["date"].values)
    log.info("μ-IC train=%+.4f  val=%+.4f", train_ic, val_ic)
    log.info("σ stats train: mean=%.4f median=%.4f", sd_tr.mean(), np.median(sd_tr))
    log.info("σ stats val:   mean=%.4f median=%.4f", sd_va.mean(), np.median(sd_va))

    # ── Save artifact ─────────────────────────────────────────────────────
    import base64, pickle
    blob = base64.b64encode(pickle.dumps(reg)).decode("ascii")

    # Median imputation vector (for inference-time NaN handling)
    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    fp_fields = {
        "feature_cols": feat_cols,
        "params": params,
        "label_col": LABEL,
        "panel_artifact_fingerprint": panel_fp,
    }
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str).encode()).hexdigest()[:16]

    artifact = {
        "version": 1,
        "kind":    "ngboost_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "params": params,
        "regressor_pickle_b64": blob,
        "feature_medians": medians.tolist(),
        "train_run_id": f"alpha158_fund_{datetime.utcnow().strftime('%Y%m%dT%H%M')}",
        "training_notes": (
            f"NGBoost Normal(μ,σ) head retrained on alpha158+5fund 163-feature "
            f"panel (matches panel-ltr.alpha158_fund.json fingerprint={panel_fp}). "
            f"Label=fwd_60d_excess (SPY-residualized, sector-residualization "
            f"omitted vs production NGBoostFitTask). Train 80% / Val 20% by "
            f"date. Train μ-IC={train_ic:+.4f}, Val μ-IC={val_ic:+.4f}. "
            f"Use to re-enable σ-aware QP + Kelly sizing in strategy_config.json."
        ),
        "train_mu_mean":  float(mu_tr.mean()),
        "train_sigma_mean": float(sd_tr.mean()),
        "train_mu_ic":    train_ic,
        "val_mu_ic":      val_ic,
        "best_iter":      int(best_iter),
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
