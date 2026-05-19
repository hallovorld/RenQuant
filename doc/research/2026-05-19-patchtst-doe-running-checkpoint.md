# HF PatchTST DOE — Running Checkpoint (Crash-Safe Verdict)

**Updated**: cron every 30min during DOE run
**Source raw data**: `artifacts/patchtst_doe_hf/runs_partial.csv`
**Source summary**: `artifacts/patchtst_doe_hf/summary_full.md`

This doc preserves the "best so far" verdict across crashes. If laptop
dies, this is the recovery breadcrumb. The script that maintains it:

  ```bash
  .venv/bin/python scripts/postprocess_doe_hf.py \
      --doe-dir artifacts/patchtst_doe_hf --partial
  ```

reads completed `*val_preds.parquet` files, ensembles per (point, cut),
computes bull_regime_IC + DSR.

---

## Update 2026-05-19 04:27 PT (21/81): pt_02 takes lead (still fails XGB)

| Point | lr | wd | seq | bull_ic_mean | DSR | n_cuts | verdict |
|---|---|---|---|---|---|---|---|
| **2** | 1.0e-05 | 3.0e-01 | 24 | **+0.085** | −0.21 | 2 | NEW leader, still fails |
| 1 | 1.0e-04 | 1.0e-02 | 24 | +0.058 | −0.78 | 3 | failed cut5 |
| 0 | 1.0e-05 | 1.0e-02 | 8 | +0.014 | −0.98 | 3 | underfit |

XGB baseline pool_ic = +0.094. NONE of pt_0/1/2 beats it yet.

**Status (post 21/81)**: 6/9 points still pending. pt_02 leadership
hint suggests **high wd + lower lr** direction. pt_06 (wd=3e-1, seq=8)
+ pt_07 (high wd + long seq) yet to come might show similar pattern.

**Shadow training (PID 25572) KILLED at 04:27** — 57 min/epoch was too
slow, pt_01 already failed gates so artifact unneeded. Freed 5GB RAM
+ CPU to speed up DOE.

## 🚨 Earlier REVERSAL: Point 1 NO LONGER passes (16/81 trials)

cut5_unwind data flipped pt_01 verdict. Earlier 2-cut data was misleading.

### Point 1 — current 3-cut verdict (still leading but now FAILS gates)

| | |
|---|---|
| lr | 1.0e-04 |
| weight_decay | 1.0e-02 |
| warmup_epochs | 4 (NOTE: knob is decorative, not wired in trainer) |
| seq_len | 24 |
| **bull_regime_IC mean (3 cuts)** | **+0.058** ← down from +0.103 |
| **DSR (Bailey-LdP 2014)** | **−0.78** ❌ (was +21.9 with 2 cuts) |
| n_cuts | 3 (cut1, cut3, cut5) |

**Per-cut breakdown reveals regime fragility**:
- cut1_covid (2020 Q1 COVID): BULL_VOLATILE +0.107 ✅
- cut3_inflpk (2022 Q4 inflation): BULL_VOLATILE +0.100 ✅
- **cut5_unwind (2024 Q3 unwind): BULL_VOLATILE −0.033 ❌**

**Implication**: pt_01 fails in 2024 carry-trade unwind regime. NOT
ready for shadow primary candidate. Continue DOE to find pt_X that
holds up across ALL 3 cuts.

This is a textbook PRIME DIRECTIVE finding — partial data (2 cuts) gave
misleading +0.103 verdict; full data (3 cuts) reveals regime-conditional
failure.

Pending: pt_2-pt_8 cut5 data + full main effects fit

### ❌ Point 0 — confirmed underfit

| | |
|---|---|
| lr | 1.0e-05 (too small) |
| bull_regime_IC | +0.014 ± 0.025 (essentially 0) |
| DSR | −0.98 (fails significance) |

Underfit confirmed via narrow negative prediction range `[-0.2, -0.0]`
across all 9 seeds — model hasn't learned ranking signal with this lr.

---

## XGB baseline (current prod)

`backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json::metadata.pool_ic = +0.094`

Note: XGB pool_ic is pooled across all regimes, not stratified.

---

## Pass-gate status (CLAUDE.md §5.14.4 + §5.13.4a)

| Gate | Threshold | Best so far (3-cut) | Pass? |
|---|---|---|---|
| Best DSR > 0 | required | −0.78 | ❌ |
| Best bull_ic > 0 | required | +0.058 | ✅ marginal |
| Best bull_ic ≥ XGB pool_ic | criterion for swap | +0.058 vs +0.094 (−38%) | ❌ |
| PBO < 0.5 | required | nan (only 2 points done) | 🟡 wait |
| Main effects regression | ≥5 points | only 2 points | 🟡 wait |

**Updated verdict (3-cut data)**: pt_01 **FAILS** DSR + XGB-comparison
gates. Earlier 2-cut "+9.6% over XGB" was a partial-data illusion
masking cut5 regime failure.

**Shadow promote DECISION (per user 2026-05-19 03:27 update)**:
- Shadow training artifact (pt_01 cut5 train) will COMPLETE for data
  collection, but golden.shadow_models entry NOT committed yet.
- Wait for pt_02-pt_08 data to find a config that holds across cuts.
- pt_02-pt_08 first cut1+cut3+cut5 results will determine real winner.

**Crash-resume invariant**: if killed now, best 3-cut config is pt_01
(+0.058) but FAILS XGB comparison → keep XGB primary, no shadow until
better DOE point emerges.

---

## Crash-recovery instructions

If laptop reboots or DOE killed:

1. Read this doc + `artifacts/patchtst_doe_hf/summary_full.md`
2. If best DSR > 0 and best bull_ic > XGB pool_ic: proceed with Step 4
   (SWA confirmatory) and Step 5 (promote) at the **best config from
   THIS doc** — no need to redo DOE.
3. If best DSR < 0 or bull_ic < XGB: keep XGB primary, register HF as
   shadow per Step 5 ELSE branch.

The val_preds parquet files in `artifacts/patchtst_doe_hf/pt_*/` are
GITIGNORED (large). Only aggregate CSVs + this doc are versioned.
To re-derive from val_preds after crash, re-run the post-hoc with
--partial flag.
