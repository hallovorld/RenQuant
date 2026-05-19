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

## Current best (as of 2026-05-19 02:27 PT, 14/81 trials done)

### 👑 Point 1 — leading candidate

| | |
|---|---|
| **lr** | 1.0e-04 |
| **weight_decay** | 1.0e-02 |
| **warmup_epochs** | 4 |
| **seq_len** | 24 |
| **bull_regime_IC mean** | **+0.103** |
| **bull_regime_IC std** | 0.004 |
| **DSR (Bailey-LdP 2014)** | +21.9 (very significant) |
| **n_cuts** | 2 (cut1_covid + cut3_inflpk) |

**Per-cut, per-HMM-regime breakdown**:
- cut1_covid (2020 Q1 COVID):
  - BULL_VOLATILE: **+0.107** ← bull contribution
  - BEAR: **+0.179** (model strong in bear too)
  - CHOPPY: +0.066
- cut3_inflpk (2022 Q4 inflation peak):
  - BULL_VOLATILE: **+0.100** ← bull contribution
  - BEAR: -0.025

Pending: cut5_unwind seeds + pt_2 through pt_7 + pt_8 (center)

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

| Gate | Threshold | Best so far | Pass? |
|---|---|---|---|
| Best DSR > 0 | required | +21.9 | ✅ |
| Best bull_ic > 0 | required | +0.103 | ✅ |
| Best bull_ic ≥ XGB pool_ic | criterion for swap | +0.103 vs +0.094 (+9.6%) | ✅ |
| PBO < 0.5 | required | nan (only 2 cuts) | 🟡 wait for cut5 |
| Main effects regression | ≥5 points | only 2 points | 🟡 wait |

**Tentative verdict**: pt_01 config wins on partial data. **If DOE
killed now, this becomes the verdict** — HF PatchTST primary swap
approved with lr=1e-4, wd=1e-2, warmup=4, seq=24.

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
