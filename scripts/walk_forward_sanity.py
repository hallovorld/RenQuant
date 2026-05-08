#!/usr/bin/env python
"""§5.2 sanity test suite for the production baseline.

Validates that R1K + alpha158 + fund + XGB d=5 fwd_60d IC=+0.066 is real, not artifact.

Tests:
  1. A/A test: same config, 3 different seeds → IC variance must be small
  2. Label shuffle: shuffle labels per training cut, retrain → IC must ≈ 0
  3. Time-shift placebo: shift labels +60d (peek future) → IC must ≈ 0

If any test fails, the +0.066 baseline is invalidated.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sanity")

CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]
LABEL = "fwd_60d_excess"

def cs_rank_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"],g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def run_wf(panel, feat_cols, label_override=None, shift_days=0,
           shuffle_labels=False, seed=42):
    """Run 7-cut WF and return per-cut IC list."""
    label = label_override or LABEL
    params = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,
              "min_child_weight":50,"subsample":0.7,"colsample_bytree":0.7,
              "nthread":8,"verbosity":0,"seed":seed}
    rng = np.random.default_rng(seed)

    # If time-shift requested, build shifted label per ticker
    if shift_days:
        panel = panel.copy()
        panel = panel.sort_values(["ticker","date"]).reset_index(drop=True)
        panel[label] = panel.groupby("ticker")[label].shift(-shift_days)
        panel = panel.dropna(subset=[label])

    ics = []
    for cut in CUTS:
        tr_s,tr_e,te_s,te_e = cut
        tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[label])
        te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[label])
        if len(tr)<1000 or len(te)<100:
            ics.append(np.nan); continue

        Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
        ytr = tr[label].clip(-5,5).values.astype(np.float64).copy()
        if shuffle_labels:
            # Proper per-date shuffle: preserves date-level distribution,
            # randomizes only the cross-sectional ranking within each date.
            # Global shuffle is wrong because cross-sectional order is what we test.
            tr_dates = tr["date"].values
            for date in np.unique(tr_dates):
                idx = np.where(tr_dates == date)[0]
                ytr[idx] = rng.permutation(ytr[idx])

        Xte = te[feat_cols].fillna(0).values.astype(np.float64)
        yte = te[label].values

        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
        Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)

        sort_idx = np.argsort(tr["date"].values)
        Xs, ys, ds = Xtr_n[sort_idx], ytr[sort_idx], tr["date"].values[sort_idx]
        _, gsz = np.unique(ds, return_counts=True)
        dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
        booster = xgb.train(params, dtr, num_boost_round=100)
        ic = cs_rank_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)
        ics.append(ic)
    return ics


def main():
    log.info("Loading R1K + fund baseline...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    # ── 1. A/A test (3 seeds) ──────────────────────────────────────────────
    log.info("\n══ TEST 1: A/A (3 seeds, same config) ══")
    aa_ics = []
    for seed in [42, 43, 44]:
        ics = run_wf(panel, feat_cols, seed=seed)
        mean_ic = np.mean([x for x in ics if not np.isnan(x)])
        aa_ics.append(mean_ic)
        log.info("  seed=%d  mean IC=%+.4f  per-cut=[%s]",
                 seed, mean_ic, ", ".join(f"{x:+.3f}" for x in ics))
    aa_std = np.std(aa_ics)
    log.info("A/A: mean=%+.4f  std=%.4f  (lower std = more reproducible)",
             np.mean(aa_ics), aa_std)
    log.info("VERDICT: %s (threshold: std < 0.005 means reproducible)",
             "✓ PASS" if aa_std < 0.005 else "✗ FAIL — high seed variance")

    # ── 2. Label shuffle ───────────────────────────────────────────────────
    log.info("\n══ TEST 2: Label shuffle (IC must ≈ 0) ══")
    shuffled_ics_per_seed = []
    for seed in [42, 43, 44]:
        ics = run_wf(panel, feat_cols, shuffle_labels=True, seed=seed)
        mean_ic = np.mean([x for x in ics if not np.isnan(x)])
        shuffled_ics_per_seed.append(mean_ic)
        log.info("  seed=%d  shuffled IC=%+.4f", seed, mean_ic)
    sh_mean = np.mean(shuffled_ics_per_seed)
    log.info("Label-shuffle: mean=%+.4f  (must be near 0)", sh_mean)
    log.info("VERDICT: %s (threshold: |IC| < 0.01 means signal isn't from noise)",
             "✓ PASS" if abs(sh_mean) < 0.01 else "✗ FAIL — even shuffled labels yield IC")

    # ── 3. Time-shift placebo ──────────────────────────────────────────────
    # Use fwd_60d shifted by additional 60 days (i.e., predict t+120 returns)
    # If this gives IC, it's regime persistence not real causal signal
    log.info("\n══ TEST 3: Time-shift placebo (shift labels +60d) ══")
    ts_ics = run_wf(panel, feat_cols, shift_days=60, seed=42)
    ts_mean = np.mean([x for x in ts_ics if not np.isnan(x)])
    log.info("  Time-shifted IC=%+.4f  per-cut=[%s]",
             ts_mean, ", ".join(f"{x:+.3f}" if not np.isnan(x) else "NA" for x in ts_ics))
    log.info("VERDICT: %s (threshold: |IC| < 0.015 means regime persistence is small)",
             "✓ PASS" if abs(ts_mean) < 0.015 else "✗ FAIL — high IC on shifted labels suggests regime persistence")

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n══ SANITY SUITE SUMMARY ══")
    log.info("Real baseline:   IC=+0.0660  (from Wave 1 result)")
    log.info("A/A reproduced:  IC=%+.4f ± %.4f", np.mean(aa_ics), aa_std)
    log.info("Shuffled label:  IC=%+.4f", sh_mean)
    log.info("Shifted +60d:    IC=%+.4f", ts_mean)
    pass_aa = aa_std < 0.005
    pass_sh = abs(sh_mean) < 0.01
    pass_ts = abs(ts_mean) < 0.015
    if pass_aa and pass_sh and pass_ts:
        log.info("✓ ALL 3 TESTS PASSED — baseline is real signal, not artifact")
    else:
        log.info("✗ TESTS FAILED: A/A=%s shuffle=%s shift=%s",
                 pass_aa, pass_sh, pass_ts)


if __name__ == "__main__":
    main()
