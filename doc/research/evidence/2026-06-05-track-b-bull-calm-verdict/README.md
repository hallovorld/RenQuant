# 2026-06-05 — Track B (momentum features) verdict: REJECT in BULL_CALM

**Status**: do not promote. The Track-B momentum/low-beta features make the
dominant regime (BULL_CALM) **worse**. The available shift-60 placebo is a
strong negative diagnostic, but it is **not** the full §7.2.1 R2 placebo
contract because the Track-B fire instructions require a 120-day time shift
(2x the fwd_60d horizon). This is the 4th consecutive NEGATIVE on the
"weak BULL_CALM signal" problem, but the evidence is recorded as a
directional reject, not R2-compliant promotion evidence.
**Owner**: Claude. **Reviewer requested**: codex (operator suspects
methodology issues — caveats called out explicitly in §5, please attack them).

---

## 1 · What was tested

**Hypothesis** (from [`2026-06-05-bull-calm-signal-is-catchable-momentum-diagnostic.md`](../../2026-06-05-bull-calm-signal-is-catchable-momentum-diagnostic.md)):
a naive momentum factor lands IC +0.039 in BULL_CALM (3.5× the model), so
adding momentum / low-beta features should lift the panel-LTR model's
BULL_CALM IC from +0.011 toward the Tier-1 bar +0.030.

**Track B features** (4, Kelly-Gu-Xiu / Frazzini-Pedersen): `mom_carry_12_1`
(12-1 momentum), `beta_dm` (BAB), `rvar_total`, `idio_vol_market`.

**Procedure**:
1. Rebuilt the panel with the 4 Track-B columns (`--include-track-b`).
2. Walk-forward retrain, 34 cuts, 2024-01-01 → 2025-11-30, 21-day cadence,
   `--include-features mom_carry_12_1,beta_dm,rvar_total,idio_vol_market`.
   Result: 176-feature models (172 baseline + 4). Manifest
   `walkforward_manifest_track_b.json`.
3. Per-regime IC + shift-60 placebo via `analyze_manifest_sanity_placebo.py`,
   leak-free WF scoring (each OOS date scored by its point-in-time cut model).
   Caveat: shift-60 is 1x the forward-label horizon; the pre-registered R2
   contract for Track B requires shift-120 before any promotion-grade IC claim.
4. **Identical script on the 172-feature baseline** (`walkforward_manifest_v2_20260602`,
   39 cuts) — same panel, same last-20% val window, same placebo battery — so
   the comparison is apples-to-apples, not against the doc-quoted +0.011.

Validation window (both): ~2024-02-02 → 2026-02-11. The baseline run
**reproduced the recovery-plan anchor** (BULL_CALM +0.011), confirming the
method before the Track-B comparison.

## 2 · The data — per-regime IC (baseline 172-feat → Track B 176-feat)

| Regime | n_dates | baseline mean_ic | Track B mean_ic | Δ | notes |
|---|--:|--:|--:|--:|---|
| **BULL_CALM** | ~400 | **+0.0106** | **−0.0049** | **−0.0155** | target was ≥ +0.030; went the WRONG way |
| BEAR | 50 | +0.307 | +0.315 | +0.008 | already strong; negligible move |
| CHOPPY | 39/40 | +0.017 | +0.035 | +0.018 | crosses +0.030 but n=40, not the bottleneck |
| BULL_VOLATILE | 19 | −0.024 | −0.037 | −0.013 | worse (momentum reverses here) |
| **pooled** | — | **+0.039** | **+0.029** | **−0.010** | Track B is worse pooled too |

BULL_CALM is ~78% of trading days. Track B made it **negative**.

## 3 · The shift-60 placebo — strong negative, not full R2

Shift-60 placebo (label shifted by the full forward-label horizon; if the
"signal" survives, it's not signal). BULL_CALM cell:

| | aligned_real_ic | model_placebo_ic | placebo/real ratio | label_autocorr |
|---|--:|--:|--:|--:|
| baseline | +0.0361 | +0.0222 | 0.62 | −0.04 |
| **Track B** | **+0.0050** | **+0.0323** | **6.52** | −0.05 |

The baseline already had a weak BULL_CALM signal (placebo keeps 62%). For
**Track B the real signal at the placebo alignment is ~0 (+0.005) while the
placebo IC is +0.032 — the fake signal is 6.5x the real one.** Label
autocorrelation is ~0, so this is a strong negative diagnostic. However,
because it uses shift-60 rather than the pre-registered shift-120, this block
does not satisfy the Track-B R2 compliance proof. Treat it as
`promotion_evidence = False`; do not treat it as a complete R2 verdict.

## 4 · Verdict — do not promote; 120d placebo still pending

The mean IC delta is negative in the target regime and the available shift-60
diagnostic is worse than the real signal, so there is no basis to promote or
continue with a prod/shadow switch. No config change. Track B features are not
added to production.

This is intentionally weaker than a full R2 statement: before quoting these IC
numbers as promotion-grade evidence, rerun the time-shift placebo at 120 days
per [`2026-06-03-track-b-fire-instructions.md`](../../2026-06-03-track-b-fire-instructions.md).

**Why the diagnostic was wrong** (hypothesis, not proven): the +0.039 naive
momentum IC was computed on the `sim_runs.db` ticker subset, not the full
panel (the diagnostic's own §6 caveat). More fundamentally, **handing 4
momentum features to a single pooled `rank:pairwise` model ≠ the model using
them like the naive factor**. Momentum works in BULL_CALM and reverses in
BULL_VOLATILE (−0.024 → −0.037 confirms the reversal got worse); a pooled
model learns ONE global momentum loading that can't serve both regimes, so the
features dilute into noise rather than concentrate signal where they help.

**The standing conclusion across 4 negatives** (Kelly σ-horizon, cash overlay,
QP allocator, Track B): at IC ≈ 0.01–0.04, neither downstream sizing nor
feature-stuffing the pooled model lifts BULL_CALM. The only structurally
different lever left is a **per-regime specialist model** (a BULL_CALM-only
model, §1.2/§1.5), not more features on the shared model.

## 5 · Caveats — where this could be wrong (codex: please attack)

1. **Model-freshness asymmetry in the val tail.** The baseline manifest has
   cuts through 2026-03-09; Track B only through 2025-11-30 (34 vs 39 cuts).
   For val dates after 2025-11, baseline scores with fresher point-in-time
   models while Track B reuses its last (2025-11-24) model. This mildly
   disadvantages Track B on the tail. n_dates is near-identical (399 vs 400)
   so coverage matches, and the BULL_CALM gap (−0.0155) + shift-60 placebo
   (ratio 6.5) are large enough to block promotion now, but it is a real
   asymmetry. A like-for-like fix is to truncate the baseline manifest to
   2025-11-30 and re-run.
2. **Feature source for the 4 added columns.** Baseline reads all 172 features
   from the rawlabel sanity panel; Track B reads the same 172 from rawlabel
   plus the 4 Track-B columns supplemented from the production training panel
   (the rawlabel panel lacks them — see the `_load_sanity_panel` fix in this
   PR). The 172 base features are byte-identical between the two runs; only the
   4 add-ons come from a different file. If those 4 columns are normalized
   differently in the training panel vs how the WF models were trained, the
   Track-B IC could be understated. (The WF models WERE trained on the training
   panel, so this should be consistent — but worth a second look.)
3. **CHOPPY crossed +0.030 (n=40).** Per §1.5 this could in principle be a
   CHOPPY-conditional win, but n=40 is too small and CHOPPY is not the
   bottleneck regime. Not promoted; flagged for completeness.
4. **No R2-compliant 120d placebo yet.** The current negative uses shift-60,
   which is useful but below the Track-B pre-fire contract. A 120d rerun is
   required before any final R2-compliant experiment record.
5. **No multi-seed** (§7.3). Single WF retrain per arm. A 5-seed repeat would
   tighten the negative.

## 6 · Artifacts

- Track B per-regime + placebo: `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_track_b_20260605/panel-ltr.json`
- Baseline: `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_baseline_20260605/panel-ltr.json`
- Track B WF manifest: `artifacts/walkforward_manifest_track_b.json` (34 cuts)
- Eval logs: `logs/track_b_wf/trackb_eval_*.log`, `logs/track_b_wf/baseline_eval2_*.log`

(JSON artifacts live under gitignored `artifacts/`; the numbers above are
transcribed from them. Reproduce with the two `analyze_manifest_sanity_placebo.py`
invocations in the eval logs.)
