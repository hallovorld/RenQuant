#!/usr/bin/env python
"""Track A — Train QuantileHead on RAW (un-z-scored) fwd_60d_excess.

Diagnostic: previous QuantileHead trained on cross-sectionally
z-scored label → val μ-IC = +0.021 (weak). Hypothesis: the label's
per-date zero-mean made q=0.5 collapse to ≈ 0 conditional on X.

This script trains the same 3-quantile architecture on
`fwd_60d_excess_raw` (built by build_raw_fwd60d_label.py).
Target val μ-IC ≥ +0.04 (matches early-LTR contribution).

Output: backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund_rawlabel.json
"""
from __future__ import annotations
import json, logging, sys, time, hashlib, base64, pickle
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-qh-raw")

QUANTILES = [0.16, 0.50, 0.84]
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
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund_rawlabel.json"
    LABEL = "fwd_60d_excess_raw"

    log.info("Loading raw-label panel + production XGB artifact...")
    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])
    panel_fp  = panel_meta["config_fingerprint"]
    log.info("Panel-LTR fingerprint=%s n_features=%d", panel_fp, len(feat_cols))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # Time-ordered 80/20 split
    distinct_dates = sorted(panel["date"].unique())
    val_cut_date = distinct_dates[int(len(distinct_dates) * 0.8)]
    train = panel[panel["date"] <= val_cut_date]
    val   = panel[panel["date"] >  val_cut_date]
    log.info("Train ≤ %s (%d rows) | Val > %s (%d rows)",
             val_cut_date, len(train), val_cut_date, len(val))

    Xtr = train[feat_cols].fillna(0).values.astype(np.float32)
    Xva = val[feat_cols].fillna(0).values.astype(np.float32)
    # Clip raw labels to ±50% return cap (Bernard-Thomas drift practice)
    # so quantile fit isn't dominated by the +29 outlier
    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    log.info("Label clip ±50%%. train: mean=%+.4f std=%.4f",
             ytr.mean(), ytr.std())
    log.info("Label                val: mean=%+.4f std=%.4f",
             yva.mean(), yva.std())

    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    boosters: dict[float, xgb.Booster] = {}
    for q in QUANTILES:
        t0 = time.time()
        params = dict(PARAMS); params["quantile_alpha"] = q
        log.info("Fitting q=%.2f ...", q)
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        boosters[q] = m.get_booster()
        log.info("  q=%.2f done in %.1fs", q, time.time() - t0)

    # Eval
    def predict_q(X):
        D = xgb.DMatrix(X)
        return {q: boosters[q].predict(D) for q in QUANTILES}
    qtr = predict_q(Xtr); qva = predict_q(Xva)
    mu_tr  = qtr[0.5];  sd_tr = np.maximum((qtr[0.84] - qtr[0.16]) / 2.0, 1e-6)
    mu_va  = qva[0.5];  sd_va = np.maximum((qva[0.84] - qva[0.16]) / 2.0, 1e-6)

    # Cross-sectional IC of μ̂
    from scipy.stats import spearmanr
    def cs_ic(mu, y, dates):
        df = pd.DataFrame({"p": mu, "y": y, "d": dates})
        ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
        ics = [x for x in ics if not np.isnan(x)]
        return float(np.mean(ics)) if ics else float("nan")

    train_ic = cs_ic(mu_tr, ytr, train["date"].values)
    val_ic   = cs_ic(mu_va, yva, val["date"].values)

    # σ̂ calibration
    abs_err_tr = np.abs(ytr - mu_tr)
    sigma_calib = float(spearmanr(sd_tr, abs_err_tr)[0])

    # Cross-sectional spread of μ̂ on val (key metric — pre-fix was ~0.0001)
    val_df = pd.DataFrame({"mu": mu_va, "d": val["date"].values})
    mu_xs_std_per_date = val_df.groupby("d")["mu"].std()
    mu_xs_std_mean = float(mu_xs_std_per_date.mean())

    log.info("\n══ EVAL ══")
    log.info("  μ-IC train=%+.4f", train_ic)
    log.info("  μ-IC val  =%+.4f  (TARGET ≥ +0.04 to enable σ-aware path)", val_ic)
    log.info("  σ̂ calibration (Spearman σ̂ vs |y−μ̂|): train=%+.4f", sigma_calib)
    log.info("  μ̂ cross-sectional std (val, per-date mean): %.5f  (TARGET ≥ 0.005)",
             mu_xs_std_mean)
    log.info("  σ̂ stats (val): mean=%.4f median=%.4f", sd_va.mean(), np.median(sd_va))

    # Save artifact (same format as quantile_head)
    blob = base64.b64encode(pickle.dumps({
        "quantiles": QUANTILES,
        "boosters_raw": {q: bytes(boosters[q].save_raw(raw_format="json")).decode()
                         for q in QUANTILES},
        "feature_cols": feat_cols,
        "feature_medians": medians.tolist(),
    })).decode("ascii")
    fp_fields = {"feature_cols": feat_cols, "params": PARAMS,
                 "quantiles": QUANTILES, "label_col": LABEL,
                 "panel_artifact_fingerprint": panel_fp}
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str).encode()).hexdigest()[:16]
    artifact = {
        "version": 2, "kind": "quantile_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols, "params": PARAMS,
        "quantiles": QUANTILES,
        "regressor_pickle_b64": blob,
        "feature_medians": medians.tolist(),
        "training_notes": (
            f"Track A — QuantileHead retrained on RAW (un-z-scored) "
            f"fwd_60d_excess label clipped ±50%. Target ≥+0.04 val μ-IC. "
            f"Achieved val μ-IC={val_ic:+.4f}, σ̂-calibration={sigma_calib:+.3f}, "
            f"μ̂ x-sec std={mu_xs_std_mean:.5f}."
        ),
        "train_mu_ic": train_ic, "val_mu_ic": val_ic,
        "sigma_calibration": sigma_calib,
        "mu_xs_std_val": mu_xs_std_mean,
        "n_rows_train": int(len(train)), "n_rows_val": int(len(val)),
        "config_fingerprint": f"sha256:{fp}",
        "config_fingerprint_fields": fp_fields,
    }
    out_path.write_text(json.dumps(artifact))
    log.info("Saved → %s", out_path)
    log.info("Fingerprint: sha256:%s", fp)


if __name__ == "__main__":
    main()
