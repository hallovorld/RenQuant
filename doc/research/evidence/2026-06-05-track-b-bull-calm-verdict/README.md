# 2026-06-05 — Track B (momentum features) verdict: REJECT in BULL_CALM

**Status**: do not promote. The Track-B momentum/low-beta features make the
dominant regime (BULL_CALM) **worse**. The 120-day time-shift placebo
(2x the fwd_60d horizon) now satisfies the §7.2.1 R2 pre-fire check and
also fails: Track B's BULL_CALM placebo IC is 16.39x the aligned real IC.
This is the 4th consecutive NEGATIVE on the "weak BULL_CALM signal"
problem and is now recorded as an R2-compliant do-not-promote verdict.
**Owner**: Claude. **Reviewer requested**: codex (operator suspects
methodology issues — caveats called out explicitly in §6, please attack them).

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
5. 2026-06-06 closeout: read the shift-120 diagnostics already emitted by
   `analyze_manifest_sanity_placebo.py` and apply the Track-B §7.2.1 R2
   placebo gate.

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

### 2.1 · CORRECTION (2026-06-06) — the −0.0155 was partly a model-freshness artifact

The baseline above (39 cuts, through 2026-03-09) had **fresher point-in-time
models on the val tail** than Track B (34 cuts, through 2025-11-24) — exactly
caveat §5.1. Re-running the baseline on a manifest **truncated to Track B's
34-cut range** (2024-01-01 → 2025-11-24) gives the promotion decision a matched
baseline/Track-B comparison:

| BULL_CALM | mean_ic | n_dates | vs Track B |
|---|--:|--:|--:|
| full baseline (39 cuts, fresher tail) | +0.0106 | 400 | — |
| **truncated baseline (34 cuts, MATCHED)** | **+0.0039** | 399 | — |
| Track B (34 cuts) | −0.0049 | 399 | **like-for-like Δ = −0.0088** |

**The honest matched-run effect of Track B is −0.0088, not −0.0155**. The old
headline mixed the full-baseline diagnostic with a shorter Track-B run and
overstated the damage by ~1.8× relative to the matched comparison.

Do **not** treat the full-baseline → truncated-baseline drop as a pure
model-recency estimate: the archived full baseline diagnostic used a narrower
142-ticker transformer-panel merge, while the corrected truncated baseline and
Track-B diagnostics use the current 292-ticker rawlabel/training-panel supplement
path. Pooled/regime moves on truncation are context, not a standalone model
recency finding. Compact evidence is archived in `matched_freshness_summary.json`.

**The do-not-promote verdict is unchanged**: even matched-freshness, Track B
(−0.0049) is still negative and still below the matched baseline (+0.0039);
neither is near the +0.030 target; and the §3 shift-60 diagnostic (computed on
Track B alone, unaffected by this) remains worse than the real aligned signal.
Only the *magnitude* claim changes. Artifacts:
`sanity_placebo_baseline_trunc_20260605/panel-ltr.json`,
`walkforward_manifest_v2_20260602.trunc_to_trackb.abspath.json`.

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

## 4 · The shift-120 placebo — R2 failed

The §7.2.1 R2 contract for Track B requires a 120-day time shift (2x the
`fwd_60d` horizon). The emitted diagnostics include that shift. BULL_CALM:

| | aligned_real_ic | model_placebo_ic | placebo/real ratio | label_autocorr | n_dates |
|---|--:|--:|--:|--:|--:|
| matched baseline | +0.0313 | +0.0421 | 1.35 | +0.0432 | 302 |
| **Track B** | **+0.0044** | **+0.0726** | **16.39** | +0.0432 | 302 |

Pooled Track B also fails the placebo sanity check: aligned_real_ic +0.0451,
model_placebo_ic +0.0667, ratio 1.48x over 388 dates. In the target
BULL_CALM regime, the fake 120d-shift signal dwarfs the aligned real signal.
This closes the earlier 120d gap and makes the reject R2-compliant. Compact
machine-readable evidence is archived in `matched_freshness_summary.json`.

## 5 · Verdict — do not promote

The mean IC delta is negative in the target regime and the available shift-60
diagnostic is worse than the real signal. The shift-120 R2 placebo is worse
again, so there is no basis to promote or continue with a prod/shadow switch.
No config change. Track B features are not added to production.

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

## 6 · Caveats — where this could be wrong (codex: please attack)

1. **Model-freshness asymmetry in the val tail.** ✅ **RESOLVED (2026-06-06)** —
   re-ran the baseline on a manifest truncated to Track B's 34-cut range; see
   §2.1. The matched comparison corrects the magnitude to Δ −0.0088, not
   −0.0155. The do-not-promote verdict stands (Track B still negative, still
   below the matched baseline, and the shift-60 diagnostic remains negative).
   Remaining attribution caveat: the old full-baseline diagnostic and corrected
   matched diagnostics also differ in panel/universe coverage, so the
   full→truncated drop is not a clean estimate of model recency alone.
   Original note: baseline had cuts through 2026-03-09, Track B through
   2025-11-24, so baseline scored the val tail with fresher point-in-time
   models.
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
4. **R2-compliant 120d placebo.** ✅ **RESOLVED (2026-06-06)** — the Track-B
   diagnostics include shift-120. BULL_CALM aligned_real_ic is +0.0044 while
   model_placebo_ic is +0.0726 (16.39x). Pooled Track B is also placebo-heavy
   (+0.0451 real vs +0.0667 placebo, 1.48x). This confirms the do-not-promote
   verdict under the Track-B pre-fire contract.
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
