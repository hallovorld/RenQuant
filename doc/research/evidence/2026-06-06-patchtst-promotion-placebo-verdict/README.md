# 2026-06-06 — PatchTST primary promotion: placebo/triad verdict

**Verdict (operator decision 2026-06-06): KEEP PatchTST primary. Do NOT roll
back to XGB.** PatchTST is leakage-clean (time-shift placebo ≈0 at the true
horizon) and strong in BEAR. Its BULL_CALM IC is ~zero — but so is XGB's, so a
rollback fixes nothing and discards a placebo-cleaner, BEAR-stronger model. The
real blocker is that **no scorer — global or specialist — has clean BULL_CALM
alpha** (§3/§4); that, not the global-scorer choice, is where the work goes next.

> **Retraction:** an earlier draft of this doc recommended rolling back to XGB.
> Withdrawn — the XGB "advantage" was a confounded mixed-window artifact (its
> BEAR number came from a longer window including the 2024 bear), and XGB is no
> better in the dominant BULL_CALM regime.

This is **evidence only**; it does not by itself flip the live scorer.

## What was run

The promoted checkpoint scored through the same point-in-time WF manifest
contract as `run_wf_gate`, then IC decomposed into real / time-shift-placebo /
label-autocorrelation buckets and stratified by regime.

- Artifact: `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`
- Manifest (single-cutoff, built for this run): `artifacts/walkforward_manifest_patchtst_seed44_pt07.json`
  (cutoff/effective_train_cutoff = 2024-11-13, lookahead 60, paired with the
  production PatchTST calibration)
- Label: `fwd_60d_excess`; OOS validation window **2025-02-06 → 2026-02-10**
  (254 dates, 73,081 rows) — genuinely out-of-sample (train ended 2024-11-13,
  embargo to 2025-02-05).
- Raw output: `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_patchtst_seed44_20260606/`
- Reproduce:
  ```bash
  PYTHONPATH=$(for d in .subrepo_runtime/repos/*/src; do printf "%s:" "$PWD/$d"; done) \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  RENQUANT_SUBREPO_ROOT="$PWD/.subrepo_runtime/repos" \
  .venv/bin/python scripts/analyze_manifest_sanity_placebo.py \
    --artifact artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt \
    --manifest artifacts/walkforward_manifest_patchtst_seed44_pt07.json \
    --output-dir backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_patchtst_seed44_20260606
  ```

## 1 · Per-regime IC FIRST (§1 PRIME DIRECTIVE)

| Regime | IC | Hit rate | Dates | Share of window | 60d model-placebo IC |
|---|---:|---:|---:|---:|---:|
| BEAR | **+0.1215** | 0.95 | 43 | 17% | +0.0616 |
| **BULL_CALM** | **+0.0064** | 0.59 | 187 | **74%** | −0.0152 |
| BULL_VOLATILE | −0.0863 | 0.11 | 9 | 4% | −0.0515 |
| CHOPPY | +0.0408 | 0.60 | 15 | 6% | −0.0763 |

**Read:** real, strong alpha in BEAR; essentially **zero in BULL_CALM, the
regime the system occupies 74% of the time**; negative (tiny sample) in
BULL_VOLATILE. The dominant-regime cell is the operative one for day-to-day
live trading, and it carries no signal.

## 2 · Time-shift placebo — CLEAN (the one unambiguous positive)

| Shift (days) | Aligned real IC | Model-placebo IC | Label-autocorr IC |
|---:|---:|---:|---:|
| 5  | +0.0208 | +0.0277 | +0.8843 |
| 20 | +0.0265 | +0.0333 | +0.6396 |
| **60 (true horizon)** | **+0.0566** | **−0.0012** | +0.0869 |
| 80 | +0.0701 | −0.0014 | +0.0853 |
| 120 | +0.0781 | −0.0005 | +0.0536 |
| 180 | +0.0867 | −0.0338 | −0.0046 |

At the true 60d horizon real IC is +0.057 while the time-shifted placebo is
≈0, and the placebo stays ≈0 at 60/80/120 as real IC rises — the signature of
**genuine alpha, not overlapping-horizon leakage**. The promotion is **not
built on a leak**. (Short-shift placebo ~0.03 with label-autocorr ~0.88 is
ordinary 60d-label autocorrelation, not model leakage.)

## 3 · Head-to-head vs the demoted XGB primary

Baseline `panel-ltr.alpha158_fund.json` (prior runs,
`sanity_placebo_baseline_20260605/`). **Caveat: windows differ** — XGB
2024-02→2026-02 (508 dates) vs PatchTST 2025-02→2026-02 (254 dates).

| Regime | XGB IC | PatchTST IC |
|---|---:|---:|
| BEAR | +0.3072 | +0.1215 |
| BULL_CALM | +0.0106 | +0.0064 |
| BULL_VOLATILE | −0.0241 | −0.0863 |
| CHOPPY | +0.0172 | +0.0408 |
| pooled aligned 60d | +0.0655 | +0.0566 |

The pooled IC gap (+0.0655 vs +0.0566) is **not a clean XGB win**: XGB's number
comes from a longer window whose extra 2024 bear inflates its BEAR cell, and the
two **tie at ≈0 in the dominant BULL_CALM regime**. XGB is no better where it
matters, and PatchTST is the placebo-cleaner of the two (XGB's window flags
`promotion_evidence=false`). → no basis for rollback.

## 4 · The actual blocker — BULL_CALM, and the specialist lead (#233)

The same-window (2024-02→2026-02, apples-to-apples) Track-C **BULL_CALM
specialist** vs XGB:

| Regime | XGB | BULL_CALM specialist |
|---|---:|---:|
| BEAR | +0.3072 | +0.2950 |
| **BULL_CALM** | +0.0106 | **+0.0241** |
| CHOPPY | +0.0172 | +0.0247 |
| pooled aligned 60d | +0.0655 | +0.0636 |

The specialist **doubles XGB's raw BULL_CALM IC** — the right direction. **But
it does not survive the placebo:** the specialist's BULL_CALM 60d model-placebo
IC is **+0.0178**, so net-of-placebo clean alpha ≈ **+0.006** — back in the same
~zero band as every other scorer. The "FIRST POSITIVE" is real as a raw number
but ~¾ of it is persistence/overlapping-horizon structure, not clean alpha.

**This is the real finding:** no scorer — PatchTST, XGB, or the BULL_CALM
specialist — has *placebo-robust* BULL_CALM alpha. That is the open problem.

## Verdict

- **Leakage:** none — PatchTST is placebo-clean. ✓
- **Keep PatchTST primary.** XGB is not better (no clean pooled edge; ties at ≈0
  in BULL_CALM) and is placebo-dirtier. Rollback rec **withdrawn**.
- **Open problem (next work):** placebo-robust BULL_CALM alpha. The Track-C
  specialist (#233) is the best lead but its raw +0.024 collapses to ~+0.006
  under time-shift — closing that gap is the objective.
- Re-enabling runtime regime admission still requires backfilling WF
  regime-admission metadata regardless.

Agent-Origin: Claude
