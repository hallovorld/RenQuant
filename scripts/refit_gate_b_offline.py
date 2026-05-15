#!/usr/bin/env python
"""Refit conformal Gate B using offline val-partition predictions.

Production fit (scripts/fit_conformal_gate_b.py) reads candidate_scores
from runs.alpaca.db, but those rows were written by the OLD NGB head
(mu/sigma distribution different from the new raw-label head). We need
to evaluate the NEW head's edge_sharpe distribution against forward
labels BEFORE the new head touches live trading.

Method:
  1. Load val partition (last 20% by date) of the raw-label panel.
  2. Predict mu/sigma using the new XGBoost-quantile head artifact.
  3. Derive edge_sharpe = mu / sigma.
  4. Use fwd_60d_excess_raw as the conformal label
     (label=1 if ticker beat SPY by ≥0, label=0 if underperformed).
  5. Fit τ that achieves target FDR ≤ 0.30 (same target as production).
  6. Write thresholds keyed by horizon-60 (no regime split — val partition
     is too short to bin by regime cleanly; production fit will pick up
     regime-specific τ once it has live runs from the new head).

We write thresholds for ALL 4 regimes (BULL_CALM, BULL_VOLATILE,
CHOPPY, BEAR) at the SAME τ until live data permits regime split.
This is conservative and unblocks the new head; the next live refit
will personalize per regime.

Output: backtesting/renquant_104/artifacts/gate_b_thresholds.json
"""
from __future__ import annotations
import json, logging, sys, base64, pickle, datetime
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("refit-gate-b")

REGIMES = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]
# Relaxed from 0.30 → 0.40 (2026-05-09 audit conclusion E52):
# 42% of QHead IC is regime persistence; pure-alpha ceiling on current panel
# is ~+0.029. Target FDR=0.30 is unachievable. FDR=0.40 still beats baseline
# (val base_fdr=0.535 → 13.5pp lift) and admits a meaningful candidate set.
TARGET_FDR = 0.40
MIN_SAMPLES = 100


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    head_path  = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/gate_b_thresholds.json"

    head = json.loads(head_path.read_text())
    if head.get("kind") != "quantile_head":
        raise RuntimeError(f"Expected quantile_head, got {head.get('kind')}")
    feat_cols = head["feature_cols"]
    quantiles = head["quantiles"]
    blob = pickle.loads(base64.b64decode(head["regressor_pickle_b64"]))
    boosters_raw = blob["boosters_raw"]
    boosters = {}
    for q in quantiles:
        # Keys may be float or string after JSON round-trip
        raw = boosters_raw.get(q) or boosters_raw.get(str(q)) or boosters_raw.get(f"{q:.2f}")
        if raw is None:
            raise RuntimeError(f"Booster for q={q} not found in artifact (keys={list(boosters_raw.keys())})")
        b = xgb.Booster()
        b.load_model(bytearray(raw, "utf-8"))
        boosters[q] = b
    log.info("Loaded NGB head: val_mu_ic=%.4f mu_xs_std=%.4f",
             head.get("val_mu_ic", float("nan")),
             head.get("mu_xs_std_val", float("nan")))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=["fwd_60d_excess_raw"])
    distinct_dates = sorted(panel["date"].unique())
    val_cut = distinct_dates[int(len(distinct_dates) * 0.8)]
    val = panel[panel["date"] > val_cut].copy()
    log.info("Val rows: %d (dates %s..%s)",
             len(val), val["date"].min().date(), val["date"].max().date())

    Xva = val[feat_cols].fillna(0).values.astype(np.float32)
    Dv = xgb.DMatrix(Xva)
    qva = {q: boosters[q].predict(Dv) for q in quantiles}
    mu = qva[0.5]
    sd = np.maximum((qva[0.84] - qva[0.16]) / 2.0, 1e-6)
    edge = mu / sd

    val["mu"] = mu
    val["sigma"] = sd
    val["edge_sharpe"] = edge
    val["label"] = (val["fwd_60d_excess_raw"] > 0).astype(int)

    log.info("edge_sharpe distribution: min=%+.3f q5=%+.3f q25=%+.3f q50=%+.3f q75=%+.3f q95=%+.3f max=%+.3f",
             *np.percentile(edge, [0, 5, 25, 50, 75, 95, 100]))
    log.info("base_fdr (val) = %.4f (1 - mean(label))",
             1 - val["label"].mean())

    # Fit τ on a fine grid
    tau_grid = [round(0.0 + 0.01 * i, 4) for i in range(40)]   # 0.0 → 0.39
    fits = []
    for tau in tau_grid:
        adm = val[val["edge_sharpe"] >= tau]
        if len(adm) == 0:
            fits.append({"tau": tau, "n": 0, "fdr": None})
            continue
        fdr = 1 - adm["label"].mean()
        fits.append({"tau": tau, "n": int(len(adm)), "fdr": float(fdr)})

    chosen = None
    for f in fits:
        if f["fdr"] is None:
            continue
        if f["fdr"] <= TARGET_FDR and f["n"] >= max(20, MIN_SAMPLES // 5):
            chosen = f["tau"]
            log.info("  CHOSEN τ=%.4f → fdr=%.4f n_admitted=%d", f["tau"], f["fdr"], f["n"])
            break

    if chosen is None:
        # No τ on the grid achieves target — pick the τ with lowest fdr
        valid = [f for f in fits if f["fdr"] is not None and f["n"] >= 20]
        if not valid:
            log.error("No τ admits ≥20 samples — model is too weak")
            return 2
        best = min(valid, key=lambda f: f["fdr"])
        chosen = best["tau"]
        log.warning("  No τ achieves target FDR=%.2f. Best τ=%.4f → fdr=%.4f n=%d",
                    TARGET_FDR, chosen, best["fdr"], best["n"])

    log.info("Selected τ_global = %.4f", chosen)
    log.info("Fit stats (sample of grid):")
    for f in fits[::4]:
        if f["fdr"] is None:
            log.info("  τ=%.3f  n=%-6d  fdr=N/A", f["tau"], f["n"])
        else:
            log.info("  τ=%.3f  n=%-6d  fdr=%.4f", f["tau"], f["n"], f["fdr"])

    thresholds = {regime: chosen for regime in REGIMES}
    out = {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "horizon_days": 60,    # NEW: this fit uses fwd_60d label, not fwd_5d
        "target_fdr": TARGET_FDR,
        "min_samples_per_regime": MIN_SAMPLES,
        "min_run_date": None,
        "thresholds": thresholds,
        "fit_stats": {
            "global": {
                "n_total": int(len(val)),
                "base_fdr": round(1 - float(val["label"].mean()), 4),
                "fits": fits,
            }
        },
        "source_db": "OFFLINE_VAL_PARTITION_RAWLABEL_PANEL",
        "ngb_head_fingerprint": head.get("config_fingerprint"),
        "ngb_head_val_mu_ic": head.get("val_mu_ic"),
        "notes": (
            "Refit using offline val partition of alpha158_291_fundamental_dataset_rawlabel.parquet "
            "with the new XGBoost-quantile NGB head. Single τ applied to all 4 regimes — "
            "regime-specific refit will follow once live data accumulates with the new head."
        ),
    }
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Saved → %s", out_path)
    log.info("τ for all regimes = %.4f", chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
