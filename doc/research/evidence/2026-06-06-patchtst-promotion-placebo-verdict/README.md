# 2026-06-06 — PatchTST primary promotion: placebo/triad verdict

**Verdict:** `promotion_evidence` (pooled) = **true**, but per §1 PRIME DIRECTIVE
the per-regime read is **NOT promote-grade** — the pooled positive is a
regime-mix artifact carried by BEAR, and PatchTST shows **no edge over the
demoted XGB**. Recommendation: treat the live PatchTST promotion (#222) as
**unsupported by evidence**; either roll back to XGB primary or keep PatchTST
only after the matched-window tiebreaker below confirms a real BULL_CALM edge
(it does not, on current evidence).

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

PatchTST is **not an improvement**: XGB has higher pooled aligned IC and far
stronger BEAR; the two tie at ≈0 in BULL_CALM. PatchTST's only wins are CHOPPY
(small sample) and being placebo-cleaner pooled (XGB's longer window flags
`promotion_evidence=false`). XGB's BEAR edge may be partly the 2024 bear that
PatchTST's window excludes — hence the tiebreaker.

## 4 · Open tiebreaker (to make this airtight)

Score XGB on the **identical** 2025-02→2026-02 window for a clean per-regime
comparison. On all current evidence PatchTST shows no BULL_CALM edge, so the
matched-window run is expected to confirm the rollback case, not overturn it.

## Verdict

- **Leakage:** none — PatchTST is placebo-clean. ✓
- **Promotion (§7.4 / §1):** NOT supported as a global primary. Real only in
  BEAR; ~zero in the 74%-of-time BULL_CALM regime; no edge over XGB.
- **Action:** the operator-directed promotion (#222) lacks supporting evidence.
  Recommend rolling back to XGB primary (revert #222 + renquant-strategy-104#8),
  or repositioning PatchTST as a BEAR-regime specialist rather than the global
  primary. Re-enabling runtime regime admission still requires backfilling WF
  regime-admission metadata regardless.

Agent-Origin: Claude
