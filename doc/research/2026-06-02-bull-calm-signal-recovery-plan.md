# Research plan — BULL_CALM signal recovery

**Date**: 2026-06-02
**Status**: Plan (codex review pending)
**Parent diagnostic**: [`2026-06-02-bull-calm-no-signal-diagnostic.md`](./2026-06-02-bull-calm-no-signal-diagnostic.md)
**Owner**: Claude (mainline) + Codex (PR review)
**Strategy**: renquant_104

## Problem statement

Tonight's clean WF gate verdict for the 172-feature alpha158+fund GBDT
candidate localized the production silence to **BULL_CALM regime
no-signal**:

| Regime | n_dates | mean_ic | hit_rate | passed |
|---|---:|---:|---:|---|
| BEAR | 50 | +0.307 | 96.0% | FAIL (placebo edge 52%) |
| **BULL_CALM** | **400** | **+0.011** | **48.3%** | **FAIL (under min_ic 0.02)** |
| BULL_VOLATILE | 19 | −0.024 | 42.1% | (n<30 ineligible) |
| CHOPPY | 39 | +0.017 | 58.9% | FAIL (under min_ic) |

The live `regime_admission` gate correctly blocks BULL_CALM buys (379
blocks observed today). Production isn't broken — the model has no
ranking signal in the regime that dominates 2024-2025 markets (~78% of
trading days), so the gate keeps the strategy mostly silent. The 5/18
prod artifact has the same recipe and the same BULL_CALM weakness.

This plan lays out three research tracks to recover signal in BULL_CALM,
ordered by cost. **Goal**: at least one of the three tracks lifts
`sanity_regime_ic.regimes.BULL_CALM.mean_ic` from +0.011 to **≥ +0.030**
(50% above the gate's 0.02 floor) WITHOUT regressing any other regime's
mean_ic by > 0.02.

## Why this is regime-conditional research, not a model bug

Per CLAUDE.md §1 PRIME DIRECTIVE: "Every feature, knob, and experiment
is designed and evaluated through a regime-conditional lens. Pooled-mean
metrics across regimes are MISLEADING and produce false NEITHER
verdicts." The candidate's pooled mean_ic (+0.039 across all regimes)
hides the regime split. Pooled-mean training is exactly how the model
ended up with this profile — BEAR rows dominate gradient because they
have high label dispersion, while BULL_CALM gets averaged into noise.

All three tracks below take BULL_CALM as a first-class object and treat
the existing model as the BEAR/CHOPPY incumbent.

---

## Track A — Per-regime calibrator on BULL_CALM

**Hypothesis**: If we fit a BULL_CALM-only calibrator on the existing
GBDT scores, the calibrator's output (probability + expected return) in
BULL_CALM will compress to a near-flat band → QP's `delta_below_min_dw`
will shrink BULL_CALM positions toward zero size, AND the calibrator's
own flat-region acceptance gate will surface this as a known phenomenon
rather than a runtime mystery.

**This is defensive, not offensive**: Track A doesn't generate new
signal in BULL_CALM. It makes the model's BULL_CALM "I don't know"
state legible to downstream. If `regime_admission` is later relaxed,
the per-regime calibrator becomes the second line of defense.

**Scope (≤ 1 day)**:

1. Add `calibrator_per_regime: dict[regime → calibrator_path]` to the
   panel-scoring config schema; consumer-side fallback to global when
   per-regime missing (no breakage on existing artifacts).
2. Fit BULL_CALM-only calibrator via `scripts/fit_calibrator_alpha158_fund.py
   --regime-filter BULL_CALM`. Uses Platt method (the global calibrator
   converged on Platt tonight).
3. Wire the regime detector's `final_regime` field into the calibrator
   lookup at scoring time. Falls back to global if regime is missing.
4. Run §7.2 sanity triad on the per-regime calibrator: a/a, shuffled,
   placebo at 2× label_horizon (PR #31's metric).
5. Add tests: per-regime artifact loading + fallback semantics + scoring
   smoke on BULL_CALM rows.

**Files**: `kernel/panel_pipeline/panel_scorer.py`,
`scripts/fit_calibrator_alpha158_fund.py`, `strategy_config.json`,
`tests/test_panel_scorer_per_regime_calibrator.py` (new).

**Success criteria**:
- BULL_CALM calibrated `expected_return` standard deviation < 0.005
  (compressed; signals "I don't know").
- Sanity triad clean on the per-regime calibrator artifact.
- WF gate rerun shows BULL_CALM buys drop from 85 to < 10 across 3 cuts
  AND Sharpe doesn't degrade (since those 85 buys were losing anyway).

**Failure mode → escalate to Track B**.

**Compute**: ~2 min per calibrator fit; ~15 min WF gate rerun for
validation. Total: < 1 hour wall.

**Rollback**: revert the per-regime config section. Calibrator artifact
files are added, not edits; safe to leave on disk.

---

## Track B — BULL_CALM-specific features (Kelly-Gu-Xiu Table 9)

**Hypothesis**: The alpha158+fund recipe is dominated by momentum
features that work in dispersion-heavy regimes (BEAR, BULL_VOLATILE).
In BULL_CALM, the canonical persistence factors are **low-volatility
premium, momentum carry, and earnings-quality** — features that capture
the calm-regime persistence Kelly-Gu-Xiu 2020 RFS Table 9 documents as
strongest in low-volatility periods (their `MOM12_m` + `BETA_DM` + `RVAR`
factors carry IC of +0.05 to +0.08 in their lowest-vol bucket vs
+0.02 to +0.04 in the highest-vol bucket).

**Scope (~2-3 days)**:

1. Implement 4 new features against the existing alpha158+fund panel:
   - `mom_carry_12_1`: 12-month-minus-1-month returns (skipping the
     short-term reversal window). Kelly-Gu-Xiu eq. (4) "MOM12_m".
   - `beta_dm`: Frazzini-Pedersen "betting-against-beta" beta vs SPY,
     daily rolling 252d.
   - `rvar_total`: total realized variance over 60d. The low-vol
     anomaly's standard proxy.
   - `idio_vol_3f`: idiosyncratic vol after regressing daily returns on
     3-factor (SPY, sector ETF, size proxy). Per Ang-Hodrick-Xing-Zhang
     2006 J. Finance — strongest in calm periods.

2. Recipe stamping: add `feature_addendum_v1` field to artifacts so we
   can compare 172-feature baseline vs 176-feature variant cleanly.

3. Walk-forward retrain over 39 cuts with the new feature set
   (`train_walkforward_panel.py --include-features feat_v2_carrysv`).

4. **CRITICAL** §7.2 sanity triad BEFORE evaluating IC numbers:
   shuffled-label, time-shift placebo at shift=2×label_horizon (per
   PR #31), and ≥ 1 leakage audit memo (per §7.2.1 R4 mandate) for each
   new feature.

5. Compare per-regime IC against baseline. **Promotion criterion**:
   BULL_CALM mean_ic improves by ≥ +0.02 (absolute) without any other
   regime dropping more than 0.02.

**Files**: `scripts/build_alpha158_fund_panel.py` (feature definitions),
`renquant-base-data/src/renquant_base_data/...` (paired feature lift),
`scripts/train_walkforward_panel.py --include-features` (new flag),
`memory/feedback_research_pipeline_must_gate_with_sanity_triad.md`
references for the validation contract.

**Success criteria** (per §7.4 promotion tiers):
- Tier 1 (research-only): BULL_CALM mean_ic ≥ +0.030, sanity triad clean.
- Tier 2 (screen): + above + no regime regression > 0.02 + ΔSPY-α ≥ 0
  in BULL_CALM-dominant cuts.
- Tier 3 (live-promotable): + above + DSR > 0.5 OR PBO < 0.5 OR n ≥ 30
  with t > 3 on the per-regime IC.

**Compute estimate**:
- Feature build on 716k-row panel: ~5 min.
- WF retrain × 39 cuts × Platt calibrators: ~10 min (matches tonight's
  cadence on M4 Pro 14c).
- WF gate evaluation: ~15 min (3 cuts × 12 min, parallelized).
- Total per iteration: ~30-40 min wall. Budget for 3-5 iterations as
  feature definitions get refined: ~3 hours.

**Failure mode → escalate to Track C**.

**Rollback**: revert the panel-builder commit; the v1 (172-feat) panel
file stays untouched on disk.

---

## Track C — Specialist regime models

**Hypothesis**: A single GBDT pooled across regimes is structurally
unable to express the BULL_CALM-specific signal because the gradient
training averages BEAR-dispersion gradients with BULL_CALM-calm
gradients. Training 4 specialist GBDTs (one per detector regime) and
ensembling at inference via `regime_detector.confidence × specialist_score`
addresses the root cause — each specialist optimizes for its regime's
return distribution.

**Scope (~1 week)**:

1. New training pipeline `scripts/train_per_regime_panel.py` — wraps
   `train_production_model.py` with a `--regime-filter` flag that
   restricts training rows to the regime BEFORE the
   `rank:pairwise` group construction (groups are intra-regime).
2. 4 artifacts: `panel-ltr.alpha158_fund.bear.json`,
   `panel-ltr.alpha158_fund.bull_calm.json`,
   `panel-ltr.alpha158_fund.bull_volatile.json`,
   `panel-ltr.alpha158_fund.choppy.json`. Recipe fingerprinted with
   regime suffix.
3. New scorer `RegimeEnsemblePanelScorer` that:
   - At scoring time, reads `today_regime` from context.
   - Computes specialist score with `confidence`-weighted average if
     transition; otherwise hard-uses the dominant specialist.
   - Backwards-compatible artifact format (consumers don't need
     specialist artifacts — fall back to global panel scorer if any
     specialist missing).
4. WF training pipeline `train_per_regime_walkforward.py` that produces
   39 cuts × 4 specialists per recipe.
5. **CRITICAL** §7.2 sanity triad PER SPECIALIST per cut. This is a
   ~156-element evaluation matrix — must be automated as a Job/Task
   chain per CLAUDE.md §5.1.

**Files**: `kernel/panel_pipeline/regime_ensemble_scorer.py` (new),
`scripts/train_per_regime_panel.py` (new), 4 new artifact paths,
`tests/test_regime_ensemble_scorer.py` (new — verify fallback semantics
when one specialist is missing or stale-fingerprint).

**Success criteria** (Tier 3 promotion):
- Per-regime mean_ic ≥ baseline + 0.02 for EACH of BEAR / BULL_CALM /
  CHOPPY individually.
- BULL_CALM specialist's mean_ic ≥ +0.030 AND placebo at 2× horizon < 50%.
- WF 3-cut Sharpe beats SPY in at least 1/3 cuts (current candidate
  beats 0/3). This is the live-promotability threshold.

**Compute estimate**:
- Training: 4 specialists × 39 cuts each = 156 models × ~25s = ~65 min
  serial; with jobs=8 → ~10 min wall.
- Calibrator fits: 156 calibrators × ~5s = ~13 min serial; ~2 min jobs=8.
- WF gate evaluation: ~15 min (same 3-cut sim, but 4× the model loads).
- Total per iteration: ~30 min wall.
- Budget: ~6 hours total including iteration on specialist
  hyperparameters (depth, eta) since each regime may want different params.

**Failure mode**: if BULL_CALM specialist's mean_ic doesn't lift above
+0.020, the alpha158+fund feature set itself is the constraint — go
back to Track B's feature engineering or accept that the strategy will
only fire in BEAR/CHOPPY.

**Rollback**: revert the new scorer config; existing global panel
scorer continues to load. Specialist artifacts are additive.

---

## Sequencing decision

Run **Track A first**. Cheapest, defensive, and quantifies the BULL_CALM
problem in production-visible artifacts. Track A doesn't generate signal,
but it makes the no-signal state explicit (rather than implicit via
`regime_admission` blocking).

If Track A confirms the BULL_CALM no-signal hypothesis (it will — we
already have the diagnostic), proceed to **Track B** for genuine new
signal. Track B's 4 features have published canonical references and
are cheap to compute.

If Track B's per-regime IC lift is below +0.020, escalate to **Track C**.
Track C is structurally correct (regime-conditional gradient training)
but expensive — only pay that cost if Track B's feature engineering
proves insufficient.

```
[Track A: per-regime calibrator]     [<1 hr] → defensive baseline
            │
            ↓ confirm BULL_CALM no-signal stamped on artifact
            │
[Track B: BULL_CALM-specific features] [~3 hr] → primary attempt
            │
            ├─ BULL_CALM mean_ic ≥ +0.030 → STOP; promote via Tier 3
            │
            └─ < +0.030 → escalate
                            ↓
[Track C: 4 regime specialists]        [~6 hr] → root-cause fix
            │
            ├─ each regime mean_ic ≥ baseline + 0.02 → promote via Tier 3
            │
            └─ BULL_CALM specialist still < +0.020 → recipe limit;
                escalate to alternate data source (intraday, sentiment, etc.)
```

## What this plan does NOT do

- ❌ Does NOT loosen `regime_admission` to force trades in BULL_CALM —
  that's symptomatic, not causal.
- ❌ Does NOT touch the gate thresholds (placebo, monotonicity) — they
  are correct as-is per PR #31's calibration.
- ❌ Does NOT propose new feature families beyond Kelly-Gu-Xiu's
  documented low-vol set. If that family fails, the answer is "this
  data source can't predict BULL_CALM" → switch data (PatchTST on
  intraday, or sentiment as separate feature stream — separate plan).
- ❌ Does NOT promise live trading recovery. The model may legitimately
  have no signal in BULL_CALM markets; in that case the strategy will
  fire ~22% of trading days (BEAR + CHOPPY + BULL_VOLATILE) and be
  silent the rest of the time. Track C's failure mode formalizes that
  acceptance.

## §7.1 + §7.2 + §7.13 compliance summary

| CLAUDE.md rule | Where addressed in this plan |
|---|---|
| §1 PRIME DIRECTIVE (regime-conditional) | All 3 tracks treat BULL_CALM as a first-class object; per-regime mean_ic is the success metric |
| §6.3 No-run path first | Track A is the no-new-experiments path — uses existing GBDT scores + new calibrator only |
| §7.2 Sanity triad | Each track lists sanity battery as a mandatory pre-IC step |
| §7.2.1 R1-R5 | R2 (placebo verdict block) is required for every IC number quoted; R4 (audit memo per hypothesis) is required at Track B/C boundary |
| §7.3 Multi-measurement | Track B/C promotion requires ≥ 5 runs (different seeds) per CLAUDE.md §7.3 |
| §7.4 Promotion gating | Tier 1/2/3 explicitly cited in each track's success criteria |
| §7.10 Canonical references | Kelly-Gu-Xiu 2020 RFS Table 9 + Frazzini-Pedersen + Ang-Hodrick-Xing-Zhang cited for Track B features |
| §7.11 Experiment design | Range-finding first (Track A's defensive check), then optimization (Track B feature variants), then root-cause (Track C ensemble) |
| §9 DOE methodology | Track B/C feature/hyperparam sweeps use Plackett-Burman screening if > 2 knobs vary |

## References

- Kelly, B. T., Gu, S., & Xiu, D. (2020). "Empirical Asset Pricing via
  Machine Learning." *Review of Financial Studies* 33(5), 2223–2273.
  Tables 7 (alpha decay) + 9 (factor IC by volatility regime).
- Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta."
  *Journal of Financial Economics* 111(1), 1–25.
- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The
  Cross-Section of Volatility and Expected Returns." *Journal of
  Finance* 61(1), 259–299.
- CLAUDE.md §1 (PRIME DIRECTIVE), §7.2-§7.4, §7.10, §7.11, §9 — internal
  operating model.

## Cross-refs

- Parent diagnostic: [`2026-06-02-bull-calm-no-signal-diagnostic.md`](./2026-06-02-bull-calm-no-signal-diagnostic.md)
- Memory: [`project_bull_calm_no_signal_2026-06-02`](../../memory/project_bull_calm_no_signal_2026-06-02.md)
- Memory: [`feedback_regime_conditional_strategy`](../../memory/feedback_regime_conditional_strategy.md) — PRIME DIRECTIVE
- Memory: [`project_perf_wall_realized_ic_2026-05-27`](../../memory/project_perf_wall_realized_ic_2026-05-27.md) — 5/27 wall (this plan's predecessor diagnostic)
- Memory: [`feedback_research_pipeline_must_gate_with_sanity_triad`](../../memory/feedback_research_pipeline_must_gate_with_sanity_triad.md) — sanity contract
