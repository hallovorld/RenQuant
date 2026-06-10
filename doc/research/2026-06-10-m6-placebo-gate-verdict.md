# M6 verdict — the weekly_wf_promote time-shift placebo FAILURE is consistent with an overlapping-label diagnostic confound

**Date:** 2026-06-10 · **Author:** Claude (Fable 5) · **Status:** diagnosis + fix-path, for review
**Decision owner:** operator (gate-architecture call) · **Scope:** sanity battery / WF-gate / panel-builder only

---

## 0 · Verdict (one line)

The §5.2 time-shift placebo FAIL on the GBDT `alpha158_fund` panel
(`placebo_ic=+0.0359` vs `threshold=+0.0295`) is **consistent with a
placebo-threshold / overlapping-label confound**, and no direct leakage path has
been identified in the reviewed evidence. That is weaker than proving all
label/feature leakage is absent. The gate's `placebo < 0.5×aligned_real_ic` rule
is mis-specified for a daily-sampled 60-day-horizon label. **Do NOT relax the
threshold — replace it** with the persistence-baselined Layer-1b gate (the data
to calibrate it is already stamped). Risk of the fix: the GBDT only *barely*
clears a correct gate and is **momentum/persistence-tilted with near-zero genuine
alpha in the production-dominant BULL_CALM regime** — that residual is an
operator trading-risk call, kept separate from the leakage decision.

---

## 1 · The failure (reproduced)

From `RenQuant/logs/weekly_wf_promote/2026-06-09.log` (and `2026-06-10.log`),
three identical runs:

```
§5.2 sanity battery (shuffled-label + time-shift placebo)...
  shuffled_ic = -0.0005                                    → PASS (signal is REAL)
  placebo_ic  = +0.0359 at gate_shift=120d (2×horizon=60d) → FAIL
  threshold   = +0.0295 (= 0.5×|aligned_real_ic|, aligned_real_ic=+0.0590)
Sanity result: FAIL: placebo_ic=+0.0359 (must be < +0.0295)
```

Gate code path: `renquant-backtesting`
`src/renquant_backtesting/wf_gate/runner.py` — `_placebo_ic_threshold()`
(L154-156: `max(0.005, 0.5*|aligned_real_ic|)`) and the time-shift loop (L2281-2361,
gate shift = `2 × label_horizon`).

**Only the time-shift placebo fails; the shuffle placebo PASSES.** That asymmetry
is the first tell: a shuffled label kills the target structure under this
diagnostic, so a passing shuffle is evidence against fitting a fully shuffled
target. It does not by itself prove production-useful alpha or rule out every
leakage/selection path. A failing time-shift placebo *alone* is consistent with a
**persistent target**, not necessarily a leaky model.

## 2 · Evidence — supports an overlapping-label autocorrelation confound

### 2.1 · The label is autocorrelated exactly at the gate's shift point (reproduced)

Reproduced the §2 measurement from the 2026-06-08 overlapping-label RFC on
`data/alpha158_291_fundamental_dataset_rawlabel.parquet`
(cross-sectional per-date `corr(label_t, label_{t-lag})`, n≈2,400+ dates):

| label | horizon | AC@1×h | **AC@2×h (gate shift)** | AC@3×h |
|---|--:|--:|--:|--:|
| fwd_5d_excess | 5 | −0.0113 | **−0.0009** | −0.0019 |
| fwd_20d_excess | 20 | −0.0137 | **+0.0093** | +0.0083 |
| **fwd_60d_excess** (prod) | 60 | +0.0362 | **+0.0489** | +0.0356 |

**Only the production label is autocorrelated at the gate's 120d shift point**
(+0.0489). `fwd_60d_excess` is a 60-trading-day forward return sampled daily →
60-day **overlapping** windows → adjacent observations of the same name share up
to 59/60 of their realization path → strong serial correlation. The gate shifts
labels 120d forward and asserts the model's IC collapses; for fwd_5d/fwd_20d the
shifted label is decorrelated (≈0) so the placebo cleanly isolates leakage, but
for **fwd_60d the shifted label still correlates at +0.0489**, so a model that
loads on persistent forward-return structure can score on the shifted labels too.
This is the textbook overlapping/concurrent-outcomes pathology
(López de Prado, *Advances in Financial Machine Learning*, 2018, Ch. 4).

### 2.2 · Supporting discriminator: placebo IC tracks label autocorr, r=+0.993

The model's Layer-1a `model_placebo_profile` is **already stamped** on the failing
artifact
(`backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260610T201007Z.staging.json`,
`metadata.wf_gate_metadata.model_placebo_profile`). Decomposing the placebo IC by
regime at the gate shift (2×h = 120d), for regimes with enough OOS dates:

| regime | placebo_ic | label_autocorr_ic | aligned_real_ic | genuine_ic | n_dates |
|---|--:|--:|--:|--:|--:|
| BEAR | −0.0206 | +0.0223 | +0.2719 | +0.2925 | 49 |
| BULL_CALM | +0.0413 | +0.0422 | +0.0302 | **−0.0112** | 302 |
| CHOPPY | +0.0433 | +0.0403 | −0.0097 | **−0.0530** | 26 |

**`corr(placebo_ic, label_autocorr_ic)` across regimes = +0.993.** Across the
three eligible regimes, the model's placebo IC moves with the *target's own*
autocorrelation. This is strong supporting evidence for a persistence-confounded
time-shift diagnostic. It is not a standalone leak/no-leak classifier: a leaking
feature path could still be regime-correlated with target persistence, and this
sample has only three eligible regime points.

### 2.3 · Contrast with the confirmed PatchTST leak (different class)

The 2026-06-02 experiment-validity audit found PatchTST B_tuned leak-contaminated
with `timeshift_placebo +0.067 > real_ic +0.044` — placebo **EXCEEDS** real (ratio
>1, a sequence-boundary leak crossing the seq_len=24 window across the train/val
cut). The GBDT here has a different signature: `placebo +0.0359 < aligned_real
+0.0590` (ratio 0.63), placebo tracks label persistence across the eligible
regimes, and the per-fold OOS IC is positive in 2/3 folds. **The two failures
should not be treated as the same class without more evidence.**

### 2.4 · The CV setup does not show an obvious leakage path

From the failing artifact: `cv_method=purged_walk_forward`, `cv_embargo_days=60`,
`lookahead_days=60`. **Embargo = label horizon = 60d**, which is the
AFML-Ch.7-correct minimum to prevent the common train/test overlap contamination
for a 60d-horizon label (embargo must be ≥ horizon). `oos_per_fold_ic =
[+0.087, −0.011, +0.056]` (positive in 2/3 folds), `training_train_ic=0.124` vs
`oos=0.044` (2.8× gap). This reduces concern about the known CV-overlap failure
mode, but it does not prove every panel-builder or feature-path leakage mode is
absent.

**Conclusion: the evidence supports treating this as a persistence-confounded
placebo diagnostic and not as proof of GBDT panel leakage. The evidence does not
fully exonerate every possible leakage path.**

## 3 · Why the threshold is wrong (not "just relax it")

`placebo < 0.5 × aligned_real_ic` implicitly assumes the time-shifted label is
**decorrelated** so any placebo IC is leakage. That assumption holds for
fwd_5d/fwd_20d but is **false for the daily-sampled fwd_60d** label (§2.1). The
rule therefore charges the model for the *label's* persistence floor (+0.049
autocorr → +0.036 placebo) as if it were the *model's* leakage. The fix is not to
loosen the multiplier (a relaxation that would also pass a genuinely-leaky model)
— it is to **subtract the measured persistence baseline** and gate on what's left.

**Literature backing (do NOT derive a closed form):** a naive closed form
`placebo < label_autocorr_2h × |real_ic| + tol` does **not** hold —
`0.049 × 0.059 = 0.0029 ≪ 0.0359` (the RFC already showed this via codex review).
A model's placebo IC depends on its own ranking persistence and its loading on
persistent features, not just `real_ic × autocorr`. So the baseline must be
**estimated empirically** from the same panel the gate scores, per
Bailey & López de Prado (2014, *DSR/PBO*) — control for the data-generating
process's own structure rather than a hand-picked multiplier, and require the
*lower confidence bound* of the de-confounded statistic to clear a floor.

## 4 · Recommended fix path — Layer 1b distribution-calibrated gate

This is RFC #259 Layer 1b, made concrete. The Layer-1a diagnostics it needs are
**already computed and stamped** (`model_placebo_profile` with per-date n_dates;
`summarize_ic` already returns per-date IC series + std), so no new heavy compute.

**Replace** `_sanity_placebo_passed`'s single `placebo < 0.5×aligned_real` rule
with a genuine-IC lower-bound gate at the 2×-horizon shift:

1. Form per-date `genuine_ic_d = aligned_real_ic_d − placebo_ic_d` over the
   aligned OOS dates (the placebo's own per-date series; both already produced by
   `_cs_ic_series`).
2. Bootstrap (block bootstrap over dates, ~2000 resamples, seed-pinned) the mean
   of `genuine_ic_d`; take the **5% lower confidence bound (LCB)**.
3. **PASS iff** `LCB(genuine_ic) > genuine_ic_floor`, where `genuine_ic_floor` is a
   fixed minimum set from the label's empirical persistence distribution **and**
   trading economics (NOT reverse-engineered to pass today's model). A defensible
   starting floor is **+0.01** (≈ half the current pooled genuine_ic of +0.021,
   above zero with margin), to be finalized from the per-regime distribution.
4. Keep `shuf_ic` (|·|<0.005) as an **independent** hard gate — it stays; it is
   the less persistence-confounded placebo and remains a separate sanity check.
5. **Per-regime:** require the LCB-positive condition in the production-dominant
   regimes the candidate will actually trade in; do not pool-average away a
   regime where genuine_ic is negative (see §5).

This keeps the gate fail-closed (a genuinely-leaky model with placebo > real has
`genuine_ic ≤ 0` → LCB ≤ 0 → FAIL), removes the persistence penalty (genuine_ic
nets out the label's own autocorr by construction), and is calibrated to the
empirical baseline rather than a multiplier.

**Complementary architectural follow-ups (separately validated, not blockers):**
- **Layer 2a — add fwd_20d as a *secondary* sanity acceptance target.** Its 2×h
  autocorr is +0.009 (≈ decorrelated), so its time-shift placebo is *valid by
  construction*; it is a clean second trust boundary the candidate must also
  clear. Cheapest correct hardening; the fwd_60d *training* label stays.
- **Layer 2b — sample-uniqueness weighting (AFML Ch.4)** on the fwd_60d training
  set, to stop the 60× overlap inflating apparent fit.

## 5 · The residual trading-risk flag (operator decision, NOT a gate question)

Even with a correct Layer-1b gate, the per-regime decomposition (§2.2) shows the
GBDT's genuine alpha is **concentrated in BEAR** (+0.29 genuine_ic, 49 dates) and
is **negative in BULL_CALM (−0.011, 302 dates) and CHOPPY (−0.053, 26 dates)** —
i.e. in the regime where the strategy places most of its buys, this model carries
**no genuine forward alpha beyond momentum persistence**, and the live regime-IC
gate (`P-REGIME-IC`) already fails BULL_CALM/CHOPPY for exactly this reason
(`sanity_regime_ic.reason = "regime sanity IC failed: BULL_CALM,CHOPPY"`).

So a corrected placebo gate by itself does **not** auto-resume buys: the honest
near-term picture is that this GBDT is a BEAR-regime specialist, and being
in-market with a momentum-tilted model in BULL_CALM is a **capital-allocation
choice** that must not be smuggled in by tuning the leakage gate. The lever to
earn down that tilt is regime-specialist signal research (Track C), not gate
relaxation. This keeps the two decisions cleanly separated (RFC §7).

## 6 · What would falsify this verdict

- If, after Layer 1b, the GBDT's pooled `LCB(genuine_ic)` is **not** > 0 (or below
  the economics floor), the honest conclusion is **no tradeable alpha beyond
  momentum** → signal research, not gate-tuning.
- If a future panel build shows `embargo < horizon` or a feature computed with a
  forward window, that *would* be genuine leakage — re-open. Current reviewed
  evidence has `embargo = horizon = 60`; do not treat that as a blanket clean
  stamp for every feature path.
- The +0.0489 autocorr is the *target's* property; the +0.0359 placebo is the
  *model's* measured persistence-contaminated IC inheriting it. They are related
  but distinct — there is **no closed-form floor** (codex-confirmed). Layer 1b must
  estimate the empirical baseline from the same panel, not a derived number.

## 7 · Reproduction

```bash
PY=/Users/renhao/git/github/RenQuant/.venv/bin/python

# (a) label-autocorr decay — the root data (§2.1)
$PY renquant-backtesting/.../analysis/repro_m6_placebo_confound.py --mode autocorr \
    --rawlabel data/alpha158_291_fundamental_dataset_rawlabel.parquet \
    --out backtesting/renquant_104/artifacts/diagnostics/m6_placebo/autocorr.json

# (b) regime placebo↔autocorr discriminator from the STAMPED artifact (§2.2)
$PY renquant-backtesting/.../analysis/repro_m6_placebo_confound.py --mode regime \
    --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260610T201007Z.staging.json \
    --out backtesting/renquant_104/artifacts/diagnostics/m6_placebo/regime.json

# gate numbers: logs/weekly_wf_promote/2026-06-09.log (Sanity result line)
```

(Script: `renquant-backtesting/src/renquant_backtesting/analysis/repro_m6_placebo_confound.py`;
`--out` JSON support is added by `renquant-backtesting#53`.)

**Related:** RFC `doc/research/2026-06-08-overlapping-label-and-gate-architecture/`
(architecture-level answer; this doc is the M6 verdict applying it to the GBDT);
`doc/research/2026-06-02-experiment-validity-audit.md` (the *different-class*
PatchTST leak); `doc/research/2026-05-24-placebo-ic-debug.md` (the aligned-sample
fix this builds on).
