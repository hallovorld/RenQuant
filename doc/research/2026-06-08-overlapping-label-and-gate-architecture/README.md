# The overlapping fwd_60d label is the root pathology — gate + label architecture proposal

**Date:** 2026-06-08 · **Author:** Claude · **Status:** architecture RFC, for review
**Decision owner:** operator (trading-risk + label-architecture call)

---

## 0 · Executive summary

The live daily trader has placed **no buys for ~2 weeks** because no production
model carries `wf_gate_metadata{passed:true}`, which the hard preflight gates
(`P-WF-GATE`, `P-REGIME-IC`) require. The proximate cause is that the weekly
promote pipeline was broken (now fixed, PRs #247/#249/#43/#253/#254); the
**deeper** cause is that **no model can pass the §7.2 sanity gate** — and this
doc shows that is **not primarily a model-quality problem but a label-and-gate
architecture problem**.

The production label `fwd_60d_excess` is a 60-trading-day forward return computed
daily → **overlapping windows → cross-sectional autocorrelation of +0.049 at
lag-120** (the gate's time-shift point). The §7.2 time-shift placebo therefore
measures the *label's persistence*, not the *model's leakage* — so it rejects
every model whose IC contains the unavoidable persistence floor. Shorter-horizon
labels do **not** have this problem (fwd_20d: +0.009, fwd_5d: −0.001 at
2×-horizon). This is a textbook overlapping-outcomes pathology
(López de Prado, *Advances in Financial Machine Learning*, 2018, Ch. 4).

**Recommendation:** a three-layer fix — (1) make the placebo gate
label-autocorrelation-aware (immediate, bounded, unblocks the daily); (2)
migrate the sanity/training label off the overlapping fwd_60d horizon
(architectural root fix); (3) unify the promote pipeline so both model families
(GBDT, PatchTST) earn metadata the same way (no more manual promotion).

## 1 · The failure chain (evidence)

```
daily full → live.runner → preflight
  ✗ P-WF-GATE   [HARD]  "WF gate metadata absent" (prod model has none)
  ✗ P-REGIME-IC [HARD]  "regime-layered IC evidence absent"
  → aborts all buys → sell-only → account in cash
```

Both production artifacts (PatchTST primary, GBDT shadow) have
`wf_gate_metadata: False`. Metadata is stamped only on a *gate pass*. The GBDT
retrain (now runs end-to-end) **fails the gate**:

```
WF config parity: PASS
shuf_ic    = -0.0005   (cross-sectional shuffle)        → PASS
placebo_ic = +0.0359   at gate_shift=120d (2×horizon)   → FAIL
threshold  = +0.0295   (= 0.5 × aligned_real_ic 0.0590)
```

The shuffle placebo passes (signal is **real**); only the **time-shift** placebo
fails. That asymmetry is the tell.

## 2 · Root cause — the label is autocorrelated at the gate's shift point

`fwd_60d_excess` = 60-trading-day forward excess return, sampled **daily**. Daily
sampling of a 60-day-horizon outcome produces 60-day **overlapping** windows, so
consecutive observations of the same name share up to 59/60 of their realization
path → strong serial correlation. Measured directly (cross-sectional, per-date
corr of label_t vs label_{t−lag}, n≈2,400 dates):

| label | horizon | autocorr @ 1×-horizon | autocorr @ **2×-horizon** (gate shift) |
|---|--:|--:|--:|
| fwd_5d_excess | 5 | −0.011 | **−0.001** |
| fwd_20d_excess | 20 | −0.014 | **+0.009** |
| **fwd_60d_excess** (prod) | 60 | +0.036 | **+0.049** |

**Only the production label is autocorrelated at the gate's shift point.** The
§7.2 time-shift placebo shifts labels by 2×horizon (=120d) and asserts the
model's IC on the shifted labels collapses. For fwd_5d/fwd_20d the shifted label
is decorrelated (≈0), so the placebo cleanly isolates leakage. For **fwd_60d the
shifted label still correlates at +0.049**, so a model that genuinely predicts
forward returns *necessarily* scores on the shifted labels too — the placebo
measures the **target**, not the **model**.

### 2.1 · IC decomposition (what the model actually has)

| component | IC |
|---|--:|
| real (aligned) | +0.059 |
| time-shift placebo = **persistence floor** | +0.036 |
| **genuine forward alpha** (real − placebo) | **+0.023** |
| shuffle placebo (noise) | −0.0005 |

The model has **+0.023 of genuine, non-persistence forward alpha** — real (the
shuffle confirms it) but buried under a +0.036 persistence floor that is a
property of the *label*. The gate's `placebo < 0.5×real` rule fails it because
0.036/0.059 = 61% > 50% — i.e. it fails on the label's autocorrelation, not on
model leakage.

## 3 · Why this matters (and why it's not "just relax the gate")

The gate is a live-money trust boundary; loosening it naively would let a leaky
model through. The point is **specificity**: the current test cannot distinguish
"model leaks / is pure momentum" from "the target is persistent." We need a gate
that controls for the **label's own autocorrelation floor** — which is
measurable and stable (§2). This is the difference between a *valid* placebo
(decorrelated target) and a *confounded* one (autocorrelated target).

## 4 · Literature grounding (§7.10)

- **López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 4
  "Sample Weights":** overlapping outcomes break the IID assumption;
  observations must be weighted by **average uniqueness** (4.3) and resampled via
  **sequential bootstrap** (4.5), or sampled non-overlapping. Our daily-sampled
  fwd_60d is exactly the "concurrent labels" case Ch. 4 warns about.
- **Ch. 7 "Cross-Validation in Finance":** purging + embargo (we have embargo;
  the splitter invariant is pinned). Purging assumes the *label horizon*; a
  60-day horizon forces large embargoes and few independent folds.
- **Ch. 3 "Labeling":** fixed-horizon labels are inferior to **triple-barrier /
  event-based** labels that produce more independent, less-autocorrelated
  outcomes.
- **Qlib (microsoft/qlib) Alpha158:** the canonical reference pairs Alpha158
  features with a **short-horizon** label (e.g. `Ref($close,-2)/Ref($close,-1)-1`,
  next-day) precisely to keep outcomes near-independent. Our 60-day horizon is a
  large divergence from the canonical recipe (cf. §7.10 — divergences must be
  justified; this one is not).
- **Bailey & López de Prado (2014), DSR/PBO:** already in §9; complementary —
  controls multiple-testing, not label autocorrelation.

## 5 · Proposal — three layers

### Layer 1 — Label-autocorrelation-aware placebo gate (IMMEDIATE, bounded)
Replace the absolute `placebo_ic < 0.5 × real_ic` with a test against the
**label's measured 2×-horizon autocorrelation floor**:

```
floor   = label_autocorr_2h × |aligned_real_ic|     # persistence baseline
genuine = aligned_real_ic − placebo_ic              # alpha beyond persistence
PASS if  genuine ≥ margin  AND  placebo_ic ≤ floor + tol
```

This passes a model whose alpha *exceeds the label's persistence floor* and fails
one that doesn't beat persistence — the correct specificity. For the current
GBDT: genuine = +0.023 vs a persistence floor set by the +0.049 label autocorr →
the gate becomes a real test of "beats momentum," not "has zero persistence."
Surfaces `placebo / label_autocorr` in the metadata so a reject reads as
"weak model" vs "autocorrelated target." (renquant-backtesting `wf_gate`; ~1 file
+ tests; companion to RFC #257.)

### Layer 2 — Migrate off the overlapping 60-day label (ARCHITECTURAL root fix)
Options, in increasing order of change:
- **2a. Shorter sanity/eval label** — run the §7.2 sanity battery on `fwd_20d`
  (autocorr +0.009) instead of fwd_60d, so the placebo is *valid* by
  construction. Cheapest correct fix; the training label can stay fwd_60d.
- **2b. Sample-uniqueness weighting** (AFML Ch. 4) on the fwd_60d training set —
  weight each (ticker, date) by average uniqueness so the 60× overlap stops
  inflating apparent fit. Keeps the horizon, fixes the IID violation.
- **2c. Shorten the production horizon** to fwd_20d (or triple-barrier) — the
  fullest fix; aligns with canonical Qlib practice and removes the persistence
  floor at the source. Largest change (re-train, re-tune, re-validate both model
  families), but it is the architecturally correct target and the operator has
  signalled openness to architecture change.

### Layer 3 — Unified gate→stamp→promote for BOTH model families (SYSTEMIC)
The PatchTST was promoted manually on 2026-06-05 with **no** manifest, gate, or
metadata — which is *why* the daily can't validate it. Generalize
`weekly_wf_promote` (today GBDT-only) into a model-family-parameterized
gate→stamp→promote pipeline (renquant-orchestrator owns the workflow;
renquant-backtesting owns the gate) so PatchTST and GBDT both earn
`wf_gate_metadata` the same way. No artifact reaches `panel_scoring.artifact_path`
without passing the gate. (Closes the manual-promotion hole that started this.)

## 6 · Recommended path + sequencing

```
[now]                    [Layer 1 PR]            [Layer 2a]           [Layer 3]
gate confounded ───────► autocorr-aware ───────► fwd_20d sanity ────► unified promote
by label persistence     placebo gate            (valid placebo)      pipeline
                          → GBDT can pass         → root fix           → no manual promote
                          → daily buys resume      (training stays)     (both families)
```

1. **Layer 1** (this RFC + PR) — unblocks the daily *now* with a *correct*
   (not loosened) gate. Lowest risk, bounded.
2. **Layer 2a** — switch the sanity-battery label to fwd_20d; the placebo
   becomes valid by construction (no autocorrelation confound).
3. **Layer 3** — unify promotion so this class of "unvalidated model in prod"
   cannot recur.
4. **Layer 2c** (optional, larger) — evaluate shortening the production horizon
   to fwd_20d/triple-barrier as the canonical end-state.

## 7 · Trade-off the operator must weigh

Even a *correct* Layer-1 gate will pass a model whose edge is **+0.023 genuine
alpha on top of momentum persistence**. That is a real, positive, shuffle-clean
signal — but it is momentum-tilted. Promoting it means **being in-market with a
momentum-heavy model**, which is a deliberate trading-risk choice, not a pure
engineering one. Layers 2–3 reduce reliance on that tilt over time
(per-regime specialists, Track C, +0.0241 placebo-clean BULL_CALM IC, are the
genuine-alpha lever).

## 8 · Caveats / what would falsify this

- The +0.049 autocorr is measured on the rawlabel sanity panel; the +0.036
  placebo is the model inheriting it. They are consistent (placebo ≈
  real × persistence-share) but distinct quantities; Layer 1 should compute the
  floor from the *same* panel the gate scores.
- Layer 2c (shorten horizon) changes the strategy's holding period and tax/turn
  profile — not just a label swap; it needs a full WF + cost re-validation.
- If, after Layer 1, the GBDT's `genuine` (+0.023) still fails a sensible
  `margin`, then the honest conclusion is **the model has no tradeable alpha
  beyond momentum** and the right action is signal research (Track C), not
  gate-tuning. Layer 1 makes that determination *correctly*; today's gate cannot.

## 9 · Reproduction

```bash
# label autocorr decay (the root data, §2)
python - <<'PY'
import pandas as pd, numpy as np
df=pd.read_parquet('data/alpha158_291_fundamental_dataset_rawlabel.parquet',
    columns=['ticker','date','fwd_5d_excess','fwd_20d_excess','fwd_60d_excess'])
df['date']=pd.to_datetime(df['date'])
def ac(lab,lag):
    p=df.dropna(subset=[lab]).pivot_table(index='date',columns='ticker',values=lab).sort_index()
    return float(np.mean([np.corrcoef(p.iloc[i][m],p.iloc[i-lag][m])[0,1]
        for i in range(lag,len(p)) for m in [p.iloc[i].notna()&p.iloc[i-lag].notna()] if m.sum()>=20]))
for lab,h in [('fwd_5d_excess',5),('fwd_20d_excess',20),('fwd_60d_excess',60)]:
    print(lab, round(ac(lab,2*h),3))
PY
# gate numbers: logs/weekly_wf_promote/2026-06-08.log (Sanity result line)
```

**Related:** evidence #256 (persistence decomposition), RFC issue #257 (threshold
question — this doc is the architecture-level answer), Track C specialist work.
