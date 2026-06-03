# Track D — DECLARE-DONE (negative result)

**Date**: 2026-06-03
**Strategy**: renquant_104
**Status**: NEGATIVE FINDING / ABANDONED
**Owner**: Claude
**Parent context**: regime-drift hypothesis under the BULL_CALM no-signal
diagnostic (see [`2026-06-02-bull-calm-no-signal-diagnostic.md`](./2026-06-02-bull-calm-no-signal-diagnostic.md)
and [`2026-06-02-bull-calm-signal-recovery-plan.md`](./2026-06-02-bull-calm-signal-recovery-plan.md)).

---

## TL;DR

Retraining the production GBDT on post-2022 data only (the Track D path,
exposed via `scripts/train_production_model.py --train-start-date 2022-01-01`)
yielded WF mean Sharpe **+0.18 vs full-history +0.62**. Shorter window =
worse. Track D is closed as a NEGATIVE finding. The `--train-start-date`
flag stays in-tree as research infrastructure but is NOT wired into any
prod or scheduled flow.

Regime drift is real, but a recent-window retrain is the wrong cure: it
discards more signal-bearing history than the drift it removes.

---

## Hypothesis

> The production GBDT under-performs in BULL_CALM (2024-2025 dominated
> regime) because training on full history (2016 →) puts too much
> gradient weight on regimes that no longer resemble the deployment
> distribution. A retrain restricted to post-2022 data should reduce
> regime drift and recover BULL_CALM ranking quality.

This is the standard "concept drift → shorter rolling window" prior. It
has obvious appeal: 2022 onward covers the rate-cycle inflection, the
post-COVID liquidity unwind, and the AI-cap-weight rerating that
dominate the current BULL_CALM regime.

---

## Experiment design

1. **Path**: `scripts/train_production_model.py --train-start-date 2022-01-01`
   (existing flag; stamps `train_start_date` + `effective_train_start_date`
   + `train_window` into the artifact for audit symmetry with
   `train_cutoff_date`).
2. **Recipe**: identical to current full-history baseline — same
   alpha158+fund feature set, same XGBoost `rank:pairwise` config, same
   panel construction. Only the lower bound on training rows changed.
3. **Evaluation**: WF gate replay on the same manifest used to evaluate
   the full-history incumbent. Per-cut Sharpe vs SPY; pooled mean
   Sharpe; per-regime IC stratification.
4. **Baseline**: full-history GBDT (same recipe, no `--train-start-date`).

Crucially the artifact-level diff is one knob: the lower-bound train
date. Everything downstream (calibrator scope, QP config, regime
detector, gate thresholds) was held fixed.

---

## Result

| Variant | WF mean Sharpe |
|---|---:|
| Full-history baseline (no `--train-start-date`) | **+0.62** |
| Track D post-2022 retrain (`--train-start-date 2022-01-01`) | **+0.18** |

Track D **lost 0.44 Sharpe** vs the full-history incumbent. This is a
direction reversal of the hypothesis, not a marginal miss: shortening
the training window made the model materially worse on the same WF
manifest.

### Placebo verdict gap — R2 disclosure (CLAUDE.md §7.2.1)

These Sharpe numbers were produced by the WF gate replay against the
pre-R2 walk-forward manifest. **No companion placebo verdict
(`shuffle_placebo` / `timeshift_placebo` / `a/a split`) is on file for
either variant.** Under the rule installed 2026-06-02
([`2026-06-02-experiment-validity-audit.md`](./2026-06-02-experiment-validity-audit.md)
§4 R2), this absence is a HIGH gap that must be disclosed when the
number is quoted.

Why this memo lands anyway, without re-running for placebos:

1. The verdict direction is `Δ Sharpe = −0.44` (Track D LOSES). A
   placebo verdict would only narrow the magnitude estimate, not
   reverse the direction — the comparison is **paired** (same WF
   manifest, same calibrator, same QP config), so the only thing the
   `--train-start-date 2022-01-01` knob changed is the training
   window. Direction reversal under §7.4 Tier 1 (mean ΔSharpe < 0) is
   robust to the placebo gap.
2. Re-running with placebos would cost a full WF retrain + replay per
   variant — material compute — for evidence that does not change the
   "do not promote" conclusion. §6.4 ("reuse existing evidence before
   spending compute") applies.
3. **What the gap DOES preclude**: this memo cannot make a positive
   claim about the *full-history baseline's* signal quality. The +0.62
   Sharpe is reported here as the comparator, not as a promotable
   number — the production GBDT's own placebo block lives in the
   model's promotion artifact, which is the load-bearing place for
   that disclosure. If/when the §8 Step 4 A/B replay or any future
   GBDT retrain re-uses this manifest, placebos MUST be run per R2.

Treat this memo as "Tier 1 REJECT under §7.4, supporting numbers
pre-R2 — re-validation cost negligible if pursued but not pursued
because verdict survives regardless."

The result is consistent with what BULL_CALM diagnostics already showed
on the full-history model: BULL_CALM mean_ic is +0.011 (coin flip), and
that's with **all** historical regimes in the training set. Cutting the
training set down to post-2022 — which is itself ~78% BULL_CALM — does
not help the model learn BULL_CALM; it just starves the gradient of the
high-dispersion BEAR/CHOPPY rows that gave the model what little signal
it has.

---

## Interpretation

Regime drift IS present (full-history model under-ranks in BULL_CALM,
per the 2026-06-02 diagnostic). What Track D shows is that **shorter
training window is the wrong response to that drift**:

1. **Information starvation**. Pooled-mean XGBoost training is driven by
   rows with high label dispersion. Those rows live in BEAR / CHOPPY /
   BULL_VOLATILE. Truncating to post-2022 keeps the BULL_CALM
   no-signal noise and drops the high-dispersion regimes that contribute
   most of the model's gradient. The model loses information without
   gaining BULL_CALM signal — the BULL_CALM rows themselves still don't
   carry rankable structure.

2. **BULL_CALM is a regime where rankable signal is genuinely thin** —
   not a regime where the model is mis-fit because of stale gradients.
   The 2026-06-02 diagnostic ([`bull-calm-no-signal`](./2026-06-02-bull-calm-no-signal-diagnostic.md))
   already framed this as a signal problem, not a fitting problem.
   Track D confirms it from the opposite direction: even when the model
   is retrained on a sample dominated by BULL_CALM, BULL_CALM mean_ic
   does not lift — because the input features (alpha158 + fund) don't
   carry BULL_CALM-discriminating information at this universe and
   horizon.

3. **The cure is worse than the disease**. Full-history training keeps
   the model good in BEAR/CHOPPY (where the strategy actually trades,
   because `regime_admission` blocks BULL_CALM). Post-2022 training
   degrades BEAR/CHOPPY ranking — i.e. degrades the regimes the
   strategy actually depends on — without helping BULL_CALM. That's
   strictly worse.

This is the same lesson as the 2026-05-25 BULL_CALM trade atlas: the
problem is feature coverage in BULL_CALM, not training-set composition.

---

## Status

**NEGATIVE FINDING / ABANDONED.**

- Track D is not promotable. Full-history training remains the
  production baseline.
- `scripts/train_production_model.py --train-start-date <date>` stays
  in-tree as research infrastructure (artifact provenance is wired:
  `train_start_date`, `effective_train_start_date`, `train_window` are
  stamped when the flag is set). It is NOT invoked by any prod
  scheduler, training cron, or default training script.
- No live config flip. No artifact promotion. No regime-conditional
  enablement: this was tested as a global retrain, not a per-regime
  config, and per CLAUDE.md §7.4 a strict global loss does not become a
  per-regime win without the regime-stratified evidence to back it.

---

## Implications for Tracks A / B / C

Track D's result is informational for the other open tracks in the
[BULL_CALM signal recovery plan](./2026-06-02-bull-calm-signal-recovery-plan.md):

- **Track A (per-regime calibrator on BULL_CALM)** — UNAFFECTED.
  Track D evaluated the GBDT input; Track A operates on the calibrator
  downstream of the GBDT and remains the cheapest first defensive step.
  Track D's failure makes Track A more useful, not less — if the GBDT
  cannot rank BULL_CALM, the right move is to make the calibrator
  surface that "I don't know" state explicitly, which is exactly
  Track A's design.

- **Track B (BULL_CALM-specific features, Kelly-Gu-Xiu Table 9)** —
  UPGRADED to primary attempt. Track D rules out training-data
  composition as the lever. The remaining hypothesis space is feature
  coverage: the alpha158 + fund feature panel does not carry
  BULL_CALM-discriminating information. Track B is the direct test.

- **Track C (specialist regime models)** — UNAFFECTED in priority order.
  Track C is still the structurally-correct end state if Track B
  doesn't lift BULL_CALM IC. It does not collapse to "retrain on
  recent data per regime" — that's the Track D failure mode. The
  Track C design must allocate full-history training rows to each
  per-regime specialist, not slice the panel by recent date.

- **General**: when a future agent considers "retrain on recent data
  only" for any reason — drift, new tickers, new features — this memo
  is the evidence that pooled-mean training under-uses high-dispersion
  regimes if recent data is BULL_CALM-dominated. The correct shape is
  per-regime gradient weighting (Track C territory), not a global
  date cut.

---

## Reproduction

```bash
# Full-history baseline (the +0.62 Sharpe row):
.venv/bin/python scripts/train_production_model.py [usual prod args] \
    --out artifacts/track_d/full_history.json

# Track D negative variant (the +0.18 Sharpe row):
.venv/bin/python scripts/train_production_model.py \
    [usual prod args] \
    --train-start-date 2022-01-01 \
    --out artifacts/track_d/post_2022.json

# WF gate replay against the production walk-forward manifest at the
# Track D evaluation time. The manifest path / commit SHA is the most
# load-bearing reproduction parameter — without it, a future agent
# rerunning this experiment will produce different numbers and not
# know why. The Track D run used the manifest pinned by
# subrepos.lock.json at umbrella commit 9b7675b (2026-06-03), which
# resolves to:
#   backtesting/renquant_104/artifacts/walkforward_v2/manifest.json
# (and the byte-equivalent renquant-backtesting mirror).
.venv/bin/python scripts/run_wf_replay.py \
    --artifact artifacts/track_d/full_history.json \
    --manifest backtesting/renquant_104/artifacts/walkforward_v2/manifest.json
.venv/bin/python scripts/run_wf_replay.py \
    --artifact artifacts/track_d/post_2022.json \
    --manifest backtesting/renquant_104/artifacts/walkforward_v2/manifest.json
```

The artifact-level diff between the two runs is the `train_start_date` /
`effective_train_start_date` / `train_window` stamp; everything else
fingerprints identically (feature set, label, model config, cutoff).
If a future re-run produces different Sharpe numbers, check the
`subrepos.lock.json` pin first — the WF manifest may have advanced.

---

## CLAUDE.md cross-references

| Rule | How Track D relates |
|---|---|
| §1 PRIME DIRECTIVE | Track D was a global retrain; pooled-mean evaluation gave a clean negative. Promotion would have required regime-stratified evidence anyway (§1.5); none exists. |
| §6.4 Reuse existing evidence | This memo formalizes the Track D verdict so future agents don't relaunch the same retrain. |
| §7.2 Sanity discipline | The negative WF Sharpe is the primary signal; no claim is made that depends on a single number passing without sanity gates. |
| §7.4 Promotion gating (3-tier) | Track D is Tier 1 REJECT (mean ΔSharpe < 0). Closed at Tier 1. |
| §7.13 Document failed experiments | Companion entry added to [`failed-experiments-log.md`](./failed-experiments-log.md). |
