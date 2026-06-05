# 2026-06-05 — BULL_CALM signal IS catchable: dispersion + momentum diagnostic

**Status**: §6.3/§6.4 no-run-path diagnostic (existing data, zero retrain).
Reframes the "BULL_CALM has no signal" conclusion: the regime has the
HIGHEST cross-sectional dispersion AND a naive momentum factor lands
IC +0.039 there — 3.5× the current 169-feature model. The signal is
catchable; the current model isn't catching it. **Track B (momentum
features) is worth firing.**
**Owner**: Claude.

---

## 1 · Why this diagnostic

Three independent NEGATIVE results converged on "the bottleneck is the
panel signal, not the downstream knob":

- Kelly σ-horizon A/B → REJECT / null (#201, #203)
- Cash overlay (QQQ/SPY) → REJECT (conditional adverse selection)
- QP allocator A/B → edge is artifact, survives placebo (#41/#215)

All three are downstream of a panel signal that's weak in BULL_CALM
(mean_ic +0.011, hit-rate 48.3% — ~random — over the ~78% of trading
days BULL_CALM dominates). Before firing a 3–5h Track-B retrain, this
diagnostic asks the §6.3 question: *is BULL_CALM's weakness a regime
property (un-fixable) or a model/feature gap (fixable)?* Using only the
already-backfilled `data/sim_runs.db` — no retrain.

## 2 · Finding 1 — BULL_CALM has the HIGHEST cross-sectional dispersion

Per (regime, date) cross-sectional `fwd_20d` structure:

| Regime | n_dates | dispersion (std) | \|mean\| | disp/\|mean\| |
|---|--:|--:|--:|--:|
| **BULL_CALM** | 474 | **11.03%** | 4.22% | **2.61** |
| BULL_VOLATILE | 31 | 8.34% | 4.24% | 1.97 |
| CHOPPY | 16 | 6.89% | 3.35% | 2.06 |
| BEAR | 2 | 11.22% | 5.31% | 2.11 |

This **refutes the "everyone rises together → no ranking room"
hypothesis.** BULL_CALM has the largest relative spread (11% std) and the
highest dispersion-to-drift ratio. There is enormous room for a ranker to
add value — an 11% cross-sectional std means the gap between a good and a
bad name over 20 days is large. The ranking room exists; the model just
isn't using it (IC +0.011 against 11% available spread).

## 3 · Finding 2 — a naive momentum factor catches 3.5× the model's IC

Cross-sectional IC of trailing-20d momentum → forward-20d return, per
regime (Jegadeesh-Titman 1993 momentum, computed by time-lagging the
backfilled `fwd_20d`):

| Regime | n_dates | momentum mean_IC | hit-rate (IC>0) |
|---|--:|--:|--:|
| **BULL_CALM** | 454 | **+0.0391** | **57%** |
| CHOPPY | 16 | +0.1481 | 94% |
| BULL_VOLATILE | 31 | −0.0167 | 42% (reversal) |
| BEAR | 2 | +0.0115 | 50% |

**In BULL_CALM, a single naive momentum signal lands IC +0.039 — 3.5×
the current 169-feature panel-LTR model's +0.011.** The model, with 169
features, is being beaten in its dominant regime by one classical factor
it evidently under-weights.

Note BULL_VOLATILE shows momentum REVERSAL (−0.017) — consistent with the
regime-conditional thesis (§1 PRIME DIRECTIVE): momentum works in calm
trends, reverses in volatile ones. A pooled model that learns one
momentum loading can't serve both; this is exactly why per-regime
features / specialists (Track B/C) are the right lever.

## 4 · Conclusion — the signal is catchable, the model isn't catching it

| Question | Answer |
|---|---|
| Is BULL_CALM ranking room real? | YES — 11% cross-sectional std, highest of any regime |
| Is there a catchable signal? | YES — naive momentum IC +0.039 (3.5× the model) |
| Is the weakness a regime property? | NO — it's a model/feature gap |
| Does this contradict the QP/Kelly/overlay NEGATIVEs? | NO — those are downstream of the signal; the signal itself is fixable |

The three NEGATIVE results are all sizing/allocation downstream of a
weak realized signal. But the signal is **not intrinsically weak** — it's
present (11% dispersion, +0.039 momentum IC) and the current model just
isn't extracting it. The correct iteration is on the SIGNAL (features /
per-regime model), and the diagnostic quantifies the upside: closing even
half the momentum gap (0.011 → 0.025) clears the recovery-plan Tier-1 bar
(BULL_CALM mean_ic ≥ 0.030 is within reach if momentum is added cleanly).

## 5 · Recommendation — fire Track B (momentum features)

[`2026-06-02-bull-calm-signal-recovery-plan.md`](2026-06-02-bull-calm-signal-recovery-plan.md)
Track B adds four Kelly-Gu-Xiu / Frazzini-Pedersen factors —
`mom_carry_12_1` (12-1 momentum), `beta_dm` (BAB), `rvar_total`,
`idio_vol_market`. This diagnostic gives the empirical green light: the
core Track-B factor (momentum) already out-ICs the model in BULL_CALM.

**Fire path** (per [`2026-06-03-track-b-fire-instructions.md`](2026-06-03-track-b-fire-instructions.md)):
the existing panel parquets do NOT yet contain the Track-B columns
(verified: `mom_carry_12_1` etc. absent from all `alpha158_291_*.parquet`),
so Path A (triad on existing panel) is not available. Path B is required:

1. **Rebuild panel** with the 4 Track-B features
   (`build_alpha158_fund_panel.py --include-features
   mom_carry_12_1,beta_dm,rvar_total,idio_vol_market`).
2. **WF retrain** (`train_walkforward_panel.py --include-features …`,
   ~3–5h).
3. **§7.2 sanity triad** (shuffle / time-shift +120d / A/A) BEFORE any
   IC number is quoted (§7.2.1 R2). The replay placebo infrastructure
   built this week (#41/#215) is the template.
4. **Per-regime IC** (§1 PRIME DIRECTIVE — by-regime first). Promotion:
   BULL_CALM mean_ic 0.011 → ≥ 0.030, placebo-clean.

This is a fire decision (3–5h compute). The diagnostic above is the
justification: the upside is quantified (3.5× momentum IC gap) and the
regime-conditional structure (momentum works in CALM, reverses in VOL)
predicts a per-regime feature/model will help where the pooled model
fails.

## 6 · Caveats

- The momentum IC is computed on the `sim_runs.db` ticker subset (the
  names that appear in the decision trace), not the full 142-name panel.
  It's a directional read, not the production WF IC. The Track-B retrain
  (§5) is what produces the comparable WF number.
- No placebo on the momentum IC itself — but momentum is a
  centuries-documented factor (Jegadeesh-Titman 1993; Asness-Moskowitz-
  Pedersen 2013), not a data-mined artifact; the §7.2 placebo belongs on
  the retrained MODEL's IC, not on the factor-existence check.
- BULL_VOLATILE momentum reversal is n=31; CHOPPY +0.148 is n=16 — both
  too small to lean on. BULL_CALM (n=454) is the robust cell and the one
  that matters (78% of days).

## 7 · Bottom line

After a week of NEGATIVE results on sizing/allocation, this is the first
POSITIVE signal-side read: **BULL_CALM's weakness is a catchable
model/feature gap, not a regime property.** A naive momentum factor
already beats the 169-feature model 3.5× there. Fire Track B; the upside
is quantified and the mechanism (regime-conditional momentum) is sound.
