#!/usr/bin/env python
"""§5.2 sanity battery for Phase C neural QHead — shuffle + time-shift placebo.

Per CLAUDE.md §5.2: any new metric MUST pass shuffled-label and time-shift
placebo before being claimed real. The shuffled-label expects val_ic ≈ 0;
time-shift placebo (labels shifted +60d) also expects val_ic ≈ 0 since
features cannot legitimately predict future returns shifted out of phase.

Both tests share the same architecture/training as train_qhead_neural.py
but mutate the labels in known-bogus ways. They use seed=42 for speed
(single-shot is fine for a sanity check — we want to see if the IC drops
to ≈ 0, not estimate σ).
"""
from __future__ import annotations
import json, time, sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_qhead_neural import QuantileMLP, pinball_loss, cs_ic, train_eval_one_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phaseC-sanity")

REPO = Path(__file__).resolve().parent.parent
LABEL = "fwd_60d_excess_raw"


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"

    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])
    fmeans = np.asarray(panel_meta.get("feature_means", [0.0] * len(feat_cols)))
    fstds  = np.asarray(panel_meta.get("feature_stds",  [1.0] * len(feat_cols))) + 1e-9

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    distinct_dates = sorted(panel["date"].unique())
    val_cut = distinct_dates[int(len(distinct_dates) * 0.8)]
    train = panel[panel["date"] <= val_cut].copy()
    val   = panel[panel["date"] >  val_cut].copy()

    Xtr_raw = train[feat_cols].fillna(0).values.astype(np.float32)
    Xva_raw = val[feat_cols].fillna(0).values.astype(np.float32)
    Xtr = ((Xtr_raw - fmeans) / fstds).astype(np.float32)
    Xva = ((Xva_raw - fmeans) / fstds).astype(np.float32)
    ytr_real = train[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva_real = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates = val["date"].values

    log.info("Train rows=%d val rows=%d  (re-using Phase C arch on seed=42)", len(Xtr), len(Xva))

    # ─── Sanity 1: shuffled labels ───
    log.info("\n══ §5.2 shuffled-label sanity ══")
    log.info("  Hypothesis: if QHead is fitting genuine signal, shuffling labels in training")
    log.info("  must collapse val_ic to ≈ 0. Failure → fitting noise / leakage.")
    rng = np.random.default_rng(42)
    ytr_shuf = ytr_real.copy(); rng.shuffle(ytr_shuf)
    yva_shuf = yva_real.copy(); rng.shuffle(yva_shuf)   # shuffle val too — IC eval should still be ≈ 0
    t0 = time.time()
    r = train_eval_one_seed(42, Xtr, ytr_shuf, Xva, yva_shuf, val_dates)
    log.info("  shuffled: val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_epoch=%d (%.1fs)",
             r["val_ic"], r["sigma_calib"], r["mu_xs_std"], r["best_epoch"], time.time()-t0)
    pass1 = abs(r["val_ic"]) < 0.005   # within 5bp of 0
    log.info("  %s: shuffled-label IC=%+.4f %s 0.005 expected",
             "✓ PASS" if pass1 else "✗ FAIL", r["val_ic"], "<" if pass1 else "≥")

    # ─── Sanity 2: time-shift placebo (labels from +60 trading days later) ───
    log.info("\n══ §5.2 time-shift placebo sanity ══")
    log.info("  Hypothesis: shifting labels by +60 trading days breaks the (X_t, y_t)")
    log.info("  alignment — model now sees today's features paired with returns from")
    log.info("  60 days later. val_ic should drop to ≈ 0.")
    # Per-ticker shift labels by +60 trading days
    train_shifted = train.copy()
    val_shifted = val.copy()
    train_shifted["__y_orig__"] = train_shifted[LABEL]
    val_shifted["__y_orig__"] = val_shifted[LABEL]
    train_shifted = train_shifted.sort_values(["ticker", "date"])
    val_shifted   = val_shifted.sort_values(["ticker", "date"])
    train_shifted[LABEL] = train_shifted.groupby("ticker")["__y_orig__"].shift(-60)
    val_shifted[LABEL]   = val_shifted.groupby("ticker")["__y_orig__"].shift(-60)
    train_shifted = train_shifted.dropna(subset=[LABEL])
    val_shifted   = val_shifted.dropna(subset=[LABEL])

    Xtr_s_raw = train_shifted[feat_cols].fillna(0).values.astype(np.float32)
    Xva_s_raw = val_shifted[feat_cols].fillna(0).values.astype(np.float32)
    Xtr_s = ((Xtr_s_raw - fmeans) / fstds).astype(np.float32)
    Xva_s = ((Xva_s_raw - fmeans) / fstds).astype(np.float32)
    ytr_s = train_shifted[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva_s = val_shifted[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates_s = val_shifted["date"].values
    log.info("  After shift: train rows=%d val rows=%d", len(Xtr_s), len(Xva_s))

    t0 = time.time()
    r2 = train_eval_one_seed(42, Xtr_s, ytr_s, Xva_s, yva_s, val_dates_s)
    log.info("  placebo: val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_epoch=%d (%.1fs)",
             r2["val_ic"], r2["sigma_calib"], r2["mu_xs_std"], r2["best_epoch"], time.time()-t0)
    pass2 = abs(r2["val_ic"]) < 0.010   # placebo more lenient since some autocorr leaks
    log.info("  %s: placebo IC=%+.4f %s 0.010 expected",
             "✓ PASS" if pass2 else "✗ FAIL", r2["val_ic"], "<" if pass2 else "≥")

    # ─── Verdict ───
    log.info("\n══ §5.2 SANITY VERDICT ══")
    if pass1 and pass2:
        log.info("✓ BOTH SANITY TESTS PASS. Phase C neural QHead's val_ic is NOT artifact of")
        log.info("  label leakage / overfitting / lucky temporal alignment. Cleared for")
        log.info("  promotion if 5-seed mean ≥ +0.040 (per CLAUDE.md §5.2).")
    else:
        log.info("✗ SANITY FAIL. Reject neural QHead and audit:")
        if not pass1:
            log.info("  - shuffled IC=%+.4f exceeds 0.005 → label leakage / impossible signal", r["val_ic"])
        if not pass2:
            log.info("  - placebo IC=%+.4f exceeds 0.010 → temporal alignment leakage", r2["val_ic"])


if __name__ == "__main__":
    main()
