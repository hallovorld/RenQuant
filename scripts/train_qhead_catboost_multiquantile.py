#!/usr/bin/env python
"""Track A v2 — CatBoost MultiQuantile head on raw fwd_60d.

XGBoost-quantile fit 3 separate models (one per quantile) → can suffer
"quantile crossing" (q_0.84 < q_0.5 in some regions) and doesn't share
information across quantiles during fitting. CatBoost MultiQuantile
fits all 3 quantiles JOINTLY in a single model with monotonicity
constraints — strictly canonical for parametric Gaussian recovery.

Reference (per CLAUDE.md §5.12):
- Prokhorenkova et al. 2018 NeurIPS "CatBoost: unbiased boosting with
  categorical features" (8.5k+ citations)
- CatBoost MultiQuantileLoss docs (built-in objective):
  https://catboost.ai/en/docs/references/training-parameters/quantile

Output: backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund_catboost.json
"""
from __future__ import annotations
import json, logging, sys, time, hashlib, base64, pickle
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from catboost import CatBoostRegressor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-cb-mq")

QUANTILES = [0.16, 0.50, 0.84]


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund_catboost.json"
    LABEL = "fwd_60d_excess_raw"

    log.info("Loading raw-label panel + production XGB artifact for fingerprint match...")
    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])
    panel_fp  = panel_meta["config_fingerprint"]
    log.info("Panel-LTR fingerprint=%s n_features=%d", panel_fp, len(feat_cols))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])

    distinct_dates = sorted(panel["date"].unique())
    val_cut_date = distinct_dates[int(len(distinct_dates) * 0.8)]
    train = panel[panel["date"] <= val_cut_date]
    val   = panel[panel["date"] >  val_cut_date]
    log.info("Train ≤ %s (%d rows) | Val > %s (%d rows)",
             val_cut_date, len(train), val_cut_date, len(val))

    Xtr = train[feat_cols].fillna(0).values.astype(np.float32)
    Xva = val[feat_cols].fillna(0).values.astype(np.float32)
    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)

    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    # CatBoost MultiQuantile — joint loss for all quantiles in one model
    qstr = ",".join(str(q) for q in QUANTILES)
    log.info("Fitting CatBoost MultiQuantile (alpha=%s)...", qstr)
    t0 = time.time()
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={qstr}",
        iterations=300,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=10,
        verbose=False,
    )
    model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=20)
    log.info("CatBoost fit in %.1fs  best_iter=%d", time.time()-t0, model.tree_count_)

    # Predictions: returns (n_samples, n_quantiles) array
    qtr = model.predict(Xtr)   # shape (n, 3)
    qva = model.predict(Xva)
    mu_tr = qtr[:, 1]; sd_tr = np.maximum((qtr[:, 2] - qtr[:, 0]) / 2.0, 1e-6)
    mu_va = qva[:, 1]; sd_va = np.maximum((qva[:, 2] - qva[:, 0]) / 2.0, 1e-6)

    from scipy.stats import spearmanr
    def cs_ic(mu, y, dates):
        df = pd.DataFrame({"p": mu, "y": y, "d": dates})
        ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
        ics = [x for x in ics if not np.isnan(x)]
        return float(np.mean(ics)) if ics else float("nan")

    train_ic = cs_ic(mu_tr, ytr, train["date"].values)
    val_ic   = cs_ic(mu_va, yva, val["date"].values)
    sigma_calib = float(spearmanr(sd_tr, np.abs(ytr - mu_tr))[0])
    val_df = pd.DataFrame({"mu": mu_va, "d": val["date"].values})
    mu_xs_std_mean = float(val_df.groupby("d")["mu"].std().mean())

    log.info("\n══ EVAL (CatBoost MultiQuantile) ══")
    log.info("  μ-IC train=%+.4f  val=%+.4f  (TARGET ≥ +0.04)", train_ic, val_ic)
    log.info("  σ̂ calibration: train=%+.4f", sigma_calib)
    log.info("  μ̂ x-sec std (val): %.5f  (TARGET ≥ 0.005)", mu_xs_std_mean)
    log.info("  σ̂ stats (val): mean=%.4f median=%.4f", sd_va.mean(), np.median(sd_va))

    # Save artifact
    blob = base64.b64encode(pickle.dumps({"model": model, "quantiles": QUANTILES,
                                            "feature_cols": feat_cols,
                                            "feature_medians": medians.tolist()})).decode("ascii")
    fp_fields = {"feature_cols": feat_cols, "loss": "MultiQuantile",
                 "quantiles": QUANTILES, "label_col": LABEL,
                 "panel_artifact_fingerprint": panel_fp}
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str).encode()).hexdigest()[:16]
    artifact = {
        "version": 1, "kind": "catboost_multiquantile_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "quantiles": QUANTILES,
        "regressor_pickle_b64": blob,
        "feature_medians": medians.tolist(),
        "training_notes": (
            "CatBoost MultiQuantile (Prokhorenkova 2018) on raw fwd_60d_excess "
            f"(±0.5 clipped). Joint quantile fit avoids crossing. "
            f"Val μ-IC={val_ic:+.4f}, σ̂-calib={sigma_calib:+.3f}, "
            f"μ̂ x-sec std={mu_xs_std_mean:.5f}."
        ),
        "train_mu_ic": train_ic, "val_mu_ic": val_ic,
        "sigma_calibration": sigma_calib,
        "mu_xs_std_val": mu_xs_std_mean,
        "best_iter": int(model.tree_count_),
        "config_fingerprint": f"sha256:{fp}",
        "config_fingerprint_fields": fp_fields,
    }
    out_path.write_text(json.dumps(artifact))
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
