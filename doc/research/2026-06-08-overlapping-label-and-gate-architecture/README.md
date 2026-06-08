# The overlapping fwd_60d label is the root pathology — gate + label architecture proposal

**Date:** 2026-06-08 · **Author:** Claude · **Status:** architecture RFC, for review
**Decision owner:** operator (trading-risk + label-architecture call)

---

## 0 · Executive summary

The live daily trader has placed **no buys for ~2 weeks** because no production
model carries `wf_gate_metadata{passed:true}`, which the hard preflight gates
(`P-WF-GATE`, `P-REGIME-IC`) require. The proximate cause is that the weekly
promote pipeline was broken (now fixed, PRs #247/#249/#43/#253/#254); the
**deeper** issue is that the §7.2 sanity gate's time-shift placebo is
**confounded** for the production label — it cannot distinguish a leaky/overfit
model from a model picking up the label's own persistence.

The production label `fwd_60d_excess` is a 60-trading-day forward return computed
daily → **overlapping windows → cross-sectional autocorrelation of +0.049 at
lag-120** (the gate's time-shift point). The §7.2 time-shift placebo therefore
partly measures the *label's persistence*, not purely the *model's leakage*.
Shorter-horizon labels do **not** have this confound (fwd_20d: +0.009, fwd_5d:
−0.001 at 2×-horizon). This is a textbook overlapping-outcomes pathology
(López de Prado, *Advances in Financial Machine Learning*, 2018, Ch. 4). It does
**not** prove every useful model must show `placebo_ic ≈ 0.036` — it proves the
*diagnostic is confounded* and needs an empirically-calibrated baseline.

**Recommendation (revised after codex review):** a fail-closed, diagnostic-first
sequence — (P0) **unify the promote pipeline** so no primary scorer reaches
production unvalidated (the systemic fix, independent of the label debate);
(P1) **stamp gate diagnostics with no threshold change**; (P2) **calibrate** a
distribution-based gate from that data; (P3) add a **fwd_20d secondary** sanity
target. This is leakage-control done correctly — explicitly separated from the
operator's trading-risk decision (§7).

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
| time-shift placebo (persistence-contaminated component) | +0.036 |
| **genuine forward alpha** (real − placebo) | **+0.023** |
| shuffle placebo (noise) | −0.0005 |

This model carries **+0.023 of genuine, non-persistence forward alpha** — real
(the shuffle confirms it) — alongside a +0.036 placebo component that the label's
autocorrelation makes hard to attribute cleanly to the model. The gate's
`placebo < 0.5×real` rule fails it (0.036/0.059 = 61% > 50%), but on a metric the
label's persistence confounds. **This is evidence of a confounded diagnostic and
a persistence-heavy model — not proof that no useful model could ever clear a
properly-calibrated gate.** That is what Layer 1a/1b determine empirically.

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

## 5 · Proposal — layered, diagnostic-first

> **Revised after codex review (2026-06-08).** The first draft proposed a
> closed-form gate `placebo < label_autocorr_2h × |real_ic| + tol`. Codex
> correctly showed it does **not** hold: `0.049 × 0.059 = 0.0029`, far below the
> observed `placebo_ic = 0.0359`, so the model only "passes" with a huge `tol` —
> a relaxation, not a calibrated correction. A model's placebo IC is **not**
> `real_ic × label_autocorr` (it also depends on the model's own ranking
> persistence and its loading on persistent features), so the baseline must be
> **estimated empirically**, not derived in closed form. The corrected,
> fail-closed sequence below replaces it.

### Layer 1a — Diagnostic-only, fail-closed (FIRST; zero behavior change)
Stamp, on the **exact sanity panel the gate already scores**, with **no** change
to any pass/fail:
- `label_autocorr_profile` — cross-sectional label autocorr at {1×, 2×, 3×}
  horizon, pooled **and** per-regime.
- `model_placebo_profile` — `aligned_real_ic`, `placebo_ic`, `shuf_ic`, and
  `genuine_ic = aligned_real_ic − placebo_ic` at the same shifts, pooled +
  per-regime.
This produces the data to **calibrate** a real gate and lets every reject be read
as "weak model" vs "confounded by an autocorrelated target." No threshold moves.
(renquant-backtesting `wf_gate`; additive metadata + tests.)

### Layer 1b — Distribution-calibrated gate (ONLY after 1a data exists)
Replace the single-mean binary rule with a **bootstrapped confidence bound** over
per-date IC (not a global multiplier):
- require the **lower confidence bound** of `genuine_ic = real_ic − placebo_ic`
  to be **> 0** (alpha beyond the *measured* persistence baseline is positive
  with CI), **and**
- an explicit **minimum genuine-IC** floor set from the baseline distribution and
  trading economics — *not* reverse-engineered to pass today's model.
This is leakage control calibrated to the label's empirical persistence, not a
relaxation. **Whether the current GBDT clears it is an open empirical question
the 1a diagnostics must answer — this doc does not claim it will.**

### Layer 2 — Reduce reliance on the overlapping 60-day label (ARCHITECTURAL)
A 60-day model can **validly optimize 60-day returns** — the issue is the
*sanity/leakage gate*, not the training objective. So treat the short-horizon
label as an **additional acceptance target**, not a forced training swap:
- **2a. fwd_20d as a secondary sanity acceptance target** — add a `fwd_20d`
  (autocorr +0.009) placebo as a *second* trust boundary the candidate must also
  clear. Since live rebalancing is daily, near-term sanity is genuinely
  informative, and its placebo is valid by construction. The fwd_60d training
  label stays. Cheapest correct addition.
- **2b. Sample-uniqueness weighting** (AFML Ch. 4) on the fwd_60d training set —
  weight each (ticker, date) by average uniqueness so the 60× overlap stops
  inflating apparent fit. Keeps the horizon, fixes the IID violation in training.
- **2c. (Larger, evaluate later) shorten the production horizon** to fwd_20d /
  triple-barrier — aligns with canonical Qlib practice but changes holding
  period, turnover, and tax profile; needs a full WF + cost re-validation of
  both model families. Not a label swap — a strategy change.

### Layer 3 — Unified gate→stamp→promote for BOTH model families (SYSTEMIC)
The PatchTST was promoted manually on 2026-06-05 with **no** manifest, gate, or
metadata — which is *why* the daily can't validate it. Generalize
`weekly_wf_promote` (today GBDT-only) into a model-family-parameterized
gate→stamp→promote pipeline (renquant-orchestrator owns the workflow;
renquant-backtesting owns the gate) so PatchTST and GBDT both earn
`wf_gate_metadata` the same way. No artifact reaches `panel_scoring.artifact_path`
without passing the gate. (Closes the manual-promotion hole that started this.)

## 6 · Recommended sequence (revised per codex — diagnostic-first, fail-closed)

```
[P0: systemic]        [P1: diagnostic]      [P2: calibrate]       [P3: 2nd target]
Layer 3               Layer 1a              Layer 1b              Layer 2a
unified promote ────► gate diagnostics ───► distribution-        ► fwd_20d
(both families)       (no behavior change)   calibrated gate        secondary sanity
no manual promote     stamp profiles         (CI lower-bound       (valid placebo)
                                              genuine_ic > 0)
```

1. **Layer 3 — unified gate→stamp→promote (highest priority).** Independent of
   the label debate, no primary scorer should ever reach production without
   manifest-based sanity + regime-IC + `wf_gate_metadata`. This closes the
   manual-promotion hole that started the incident and is the right fix
   regardless of the threshold question. (codex strongly endorses.)
2. **Layer 1a — diagnostic-only metadata (fail-closed).** Ship the
   autocorr/placebo/genuine-IC profiles with **no threshold change**, so the
   calibration in 1b is data-driven, not hand-waved.
3. **Layer 1b — distribution-calibrated gate**, only after 1a data exists.
4. **Layer 2a — fwd_20d secondary sanity target.** Then 2b/2c as larger,
   separately-validated changes.

**The daily-buy unblock is NOT a single-gate fix.** Today's daily also fails
`P-REGIME-IC` (BEAR/BULL_CALM) and the PatchTST primary carries no metadata at
all — so even a perfect placebo gate does not, by itself, resume buys. The
honest near-term unblock is **Layer 3** (get a *validated* model — whichever
family — properly stamped into production); Layers 1–2 make the gate that
validates it *correct*.

## 7 · Two separate decisions — do not conflate them

**(a) Leakage control (engineering).** "Is the model's IC an artifact of leakage
or label autocorrelation?" — answered by the diagnostic-calibrated gate (Layers
1a/1b). This is mechanical and belongs to the gate.

**(b) Trading risk (operator).** *Given* a model that is leakage-clean but whose
residual edge is **+0.023 genuine IC on top of momentum persistence** — do we
want to be **in-market with a momentum-tilted model**? That is a deliberate
capital-allocation choice, not an engineering one, and must not be smuggled in by
tuning a leakage gate. Per-regime specialists (Track C, +0.0241 placebo-clean
BULL_CALM IC) are the lever to earn down that tilt over time.

Conflating (a) and (b) is exactly the trap the first draft fell into (using a
leakage threshold to make a trading-risk call). Keep them separate.

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
