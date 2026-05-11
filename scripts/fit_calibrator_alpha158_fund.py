#!/usr/bin/env python
"""Fit panel-rank-calibration directly on the pre-built alpha158+fund panel.

The standard scripts/fit_panel_calibrator.py rebuilds features via
build_inference_matrix → only knows the production 30-feat panel,
not alpha158. So scoring with the new 163-feature alpha158_fund
artifact through that pipeline produced garbage (pool_ic=-0.013,
prob head collapsed to 3 unique y values).

This script bypasses the rebuild: load the pre-built panel parquet,
predict with the panel-LTR XGB artifact directly, then call
fit_global_calibrator with the predictions + actual returns.

Output: backtesting/renquant_104/artifacts/panel-rank-calibration.json
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fit-calib-direct")


def main():
    # 2026-05-11 audit G2: prod artifacts moved to artifacts/prod/.
    # Defaults updated; CLI args added so sim + ablation paths can override.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scorer-artifact", default=None,
        help="Path to panel-LTR XGB JSON. Defaults to artifacts/prod/panel-ltr.alpha158_fund.json. "
             "Relative paths resolve against repo root. Use a sim-only scorer "
             "(trained with cutoff < sim_start) to get a leak-free sim calibrator.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output calibrator path. Defaults to artifacts/prod/panel-rank-calibration.json.",
    )
    p.add_argument(
        "--panel", default=None,
        help="Panel parquet. Defaults to data/alpha158_291_fundamental_dataset.parquet.",
    )
    p.add_argument(
        "--data-start", default=None,
        help="ISO date. Drop scoring dates < this. Used with --data-end for "
             "true OOS calibration (scorer trained ≤T → score (T, T+window)).",
    )
    p.add_argument(
        "--data-end", default=None,
        help="ISO date. Drop scoring dates >= this. Must be ≤ "
             "(sim_start - lookahead_days - safety_buffer) for leak-free sim.",
    )
    args = p.parse_args()

    panel_path = Path(args.panel) if args.panel else REPO / "data" / "alpha158_291_fundamental_dataset.parquet"
    art_path = (
        Path(args.scorer_artifact)
        if args.scorer_artifact and Path(args.scorer_artifact).is_absolute()
        else (REPO / args.scorer_artifact) if args.scorer_artifact
        else REPO / "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json"
    )
    out_path = (
        Path(args.out)
        if args.out and Path(args.out).is_absolute()
        else (REPO / args.out) if args.out
        else REPO / "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json"
    )
    LABEL_60D  = "fwd_60d_excess"

    log.info("Loading panel + panel-LTR artifact...")
    art = json.loads(art_path.read_text())
    feat_cols = art["feature_cols"]
    # 2026-05-11: walkforward-fold artifacts skip config_fingerprint stamping
    # (§5.13.13). Production artifacts have it. Both should work here.
    fingerprint = art.get("config_fingerprint", "<walkforward — no fingerprint>")
    # Round 3 audit (G10): label column from the artifact, not hardcoded.
    # A short-horizon scorer (fwd_5d / fwd_20d) used with the previous
    # hardcoded `fwd_60d_excess` produced a silent label/horizon mismatch.
    label_col = art.get("label_col", LABEL_60D)
    log.info("Artifact fingerprint=%s  features=%d  label_col=%s",
             fingerprint, len(feat_cols), label_col)

    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # 2026-05-11: optional date window filter for OOS sim calibration.
    if args.data_start:
        start = pd.Timestamp(args.data_start)
        before = len(panel)
        panel = panel[panel["date"] >= start]
        log.info("--data-start=%s: filtered %d → %d rows", args.data_start, before, len(panel))
    if args.data_end:
        end = pd.Timestamp(args.data_end)
        before = len(panel)
        panel = panel[panel["date"] < end]
        log.info("--data-end=%s: filtered %d → %d rows", args.data_end, before, len(panel))

    # Score the entire panel — predictions are RAW XGB output (already
    # operating on z-scored features since the panel was z-scored at build time)
    log.info("Scoring %d rows...", len(panel))
    X = panel[feat_cols].fillna(0).values.astype(np.float64)
    panel["panel_score"] = booster.predict(xgb.DMatrix(X))

    # Sanity check IC vs fwd_60d
    from scipy.stats import spearmanr
    valid = panel.dropna(subset=[label_col])
    ics = []
    for _, g in valid.groupby("date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["panel_score"], g[label_col])
        if not np.isnan(ic): ics.append(ic)
    log.info("In-sample fwd_60d cross-sectional IC: mean=%+.4f median=%+.4f n_dates=%d",
             np.mean(ics), np.median(ics), len(ics))

    # Build the dicts the calibrator wants:
    #   panel_scores   = {ticker: series indexed by date → score}
    #   future_returns = {ticker: series indexed by date → fwd_excess_return}
    log.info("Building per-ticker score + return series for calibrator pool...")
    panel_scores = {}
    future_returns = {}
    for tkr, g in panel.groupby("ticker"):
        gs = g.sort_values("date").set_index("date")
        panel_scores[tkr] = gs["panel_score"]
        # Use fwd_60d_excess (label the model was trained on). Calibrator
        # quantizes "outperform = fwd_return >= threshold" so the threshold
        # must be on the same scale as the label.
        if label_col in gs.columns:
            future_returns[tkr] = gs[label_col].dropna()

    log.info("Pool: %d tickers with both score + 60d-fwd returns",
             len(set(panel_scores) & set(future_returns)))

    # Fit calibrator. Use lookahead_days=60 to MATCH the label horizon,
    # threshold_mode=crosssectional so the base rate is ~50% regardless
    # of the bull-skew on 60-day windows (per global_calibrator.py docs).
    from training_panel.global_calibrator import fit_global_calibrator
    log.info("Fitting calibrator (method=isotonic, lookahead=60d, threshold_mode=crosssectional)")
    calib = fit_global_calibrator(
        panel_scores, future_returns,
        lookahead_days=60,
        threshold=0.0,                # ignored when threshold_mode='crosssectional'
        threshold_mode="crosssectional",
        method="isotonic",
        min_rows=1000,
    )

    # Hand-build the artifact since GlobalPanelCalibration.save isn't a one-liner
    log.info("Saving artifact to %s", out_path)
    p_x, p_y = calib.prob_x.tolist(), calib.prob_y.tolist()
    e_x, e_y = calib.er_x.tolist(),   calib.er_y.tolist()

    metadata = dict(calib.metadata)
    # Stamp the source artifact path so we can detect drift later
    metadata["scorer_artifact"] = str(art_path)
    metadata["scorer_artifact_fingerprint"] = fingerprint
    metadata["scorer_oos_mean_ic"] = float(np.mean(ics))
    # 2026-05-11: record OOS window for future audits.
    if args.data_start:
        metadata["data_window_start"] = args.data_start
    if args.data_end:
        metadata["data_window_end"] = args.data_end
    metadata["lookahead_days_used"] = 60

    payload = {
        "version": 1,
        "kind":    "global_panel_calibration",
        "trained_date": pd.Timestamp.utcnow().date().isoformat(),
        "probability":      {"x": p_x, "y": p_y},
        "expected_return":  {"x": e_x, "y": e_y},
        "metadata":         metadata,
    }
    out_path.write_text(json.dumps(payload, default=str))
    log.info("Saved: n_unique_prob_y=%d  pool_ic=%+.4f  per_date_ic=%+.4f  base_rate=%.4f",
             metadata["n_unique_prob_y"],
             metadata["pool_ic"],
             metadata["per_date_ic_mean"],
             metadata.get("prob_base_rate", float("nan")))


if __name__ == "__main__":
    main()
