# G1 v6 — BCa bound empirical calibration validation

Date: 2026-07-19
PR: this one (author hallovorld; reviewer haorensjtu-dev)
Amendment under test: `doc/experiments/2026-07-19-equal-weight-deployment-prereg-v6.md`
(PR #512). This PR is the §E-item-2 acceptance evidence for that amendment.
Extends the merged v5 §4.6 sim (RenQuant#511).

## Bottom line

**BCa does NOT control the boundary type-I.** Under the frozen v6 method the deployment
rule's boundary type-I at μ = MEE = 3 is essentially UNCHANGED from the anti-conservative
v5 percentile bound (~0.14 at T=240, ~0.12 at T=480), under BOTH the iid control and the
provisional AR(1) null, at BOTH horizons. Every cell's 95% CI lower bound sits ABOVE the
nominal α = 0.10; every CI upper exceeds the 0.12 margin. The v6 §4.6-step-2 gate therefore
returns **`INFEASIBLE — INFERENCE METHOD UNCALIBRATED`**.

Decision needed: v6's own fallback fires — escalate to a **v7 studentized-t block
bootstrap**, NOT an α / MEE / block-length relaxation. (Everything else in v5/v6 —
MEE=3/PE=6, the rule form, the pilot contract, the two-stage start — stays.)

This is the FOURTH pre-activation catch in the v3→v7 lineage; prereg discipline again
blocked activation of a mis-calibrated rule before any capital, at zero cost. `[VERIFIED]`
by 8000-trial / 10000-bootstrap Monte-Carlo, fresh per-cell seeds (seed base 20260719).

## The decisive numbers — percentile vs BCa boundary type-I (μ = MEE)

8000 trials, B = 10000, MBB block b = 35, σ_d(marginal) = 25 bps. Rule: LB90 > MEE.

| DGP | T | percentile rate (95% CI) | BCa rate (95% CI) | BCa calibrated? |
|-----|-----|--------------------------|-------------------|-----------------|
| iid (φ=0)      | 240 | 0.1445 [0.1368, 0.1522] | 0.1459 [0.1381, 0.1536] | NO |
| iid (φ=0)      | 480 | 0.1177 [0.1107, 0.1248] | 0.1187 [0.1117, 0.1258] | NO |
| AR(1) φ=0.35   | 240 | 0.1487 [0.1410, 0.1565] | 0.1514 [0.1435, 0.1592] | NO |
| AR(1) φ=0.35   | 480 | 0.1195 [0.1124, 0.1266] | 0.1209 [0.1137, 0.1280] | NO |

Gate = point estimate ≤ 0.10 AND 95% CI upper ≤ 0.12, required in EVERY cell. All four
cells FAIL both bars. BCa moves the rate by −0.0014 to +0.0027 — i.e. it does not move it
at all, and at T=240 it is marginally WORSE.

Reproduced the merged v5 percentile number exactly (paired arm, seed 23, 1500/1500 →
0.1393; the v5 test's frozen 0.1393), confirming the comparison shares the same resamples.
Deep-null μ=0 rejection = 0.0120 (< 0.05) → the sim is sound; the elevated boundary rate is
genuine, not MC noise.

## Why BCa cannot fix it (the mechanism)

The percentile MBB bound's anti-conservatism is a **variance-scale** defect: with a large
block (b=35) over a short series (T=240 ⇒ only ~7 blocks), the block bootstrap
UNDER-estimates the sampling variance of the sample mean, so the one-sided lower bound sits
too high and over-rejects. BCa corrects **median-bias** (z0, from the fraction of bootstrap
means below the observed mean) and **skewness** (a, from the delete-one-block jackknife) —
for the sample-mean statistic BOTH are ≈ 0 on average, so BCa ≍ percentile. It corrects the
wrong thing. The remedy is a **pivotal / studentized** statistic (bootstrap the
studentized-t of the mean with a block-robust SE), which is what re-scales the interval to
its true width. → v7.

## Feasibility is moot but also fails (recorded for completeness)

Because the inference is uncalibrated, the §4.6-step-3 power check is not reached. For the
record, at the same σ_d = 25 the BCa power at μ = PE = 6 tops out at 0.732 at T = 480
(0.546 @ 240, 0.649 @ 360), never reaching 0.80 within T ≤ 480. So even a hypothetically
calibrated v7 bound would then return `INFEASIBLE AT CONSERVATIVE PILOT NOISE` at this
book's provisional noise. G1 is doubly blocked at σ_d = 25.

## Resolved G1 direction

**NEEDS-V7 (immediate), trending INFEASIBLE-AT-NOISE.** The binding gate today is inference
calibration: BCa fails, so v6 cannot activate and the honest next step is a v7 studentized-t
amendment (design PR, drafted personally per design-review policy). The provisional power
ceiling (~0.73 < 0.80) further implies that even a calibrated bound is unlikely to make the
equal-weight experiment testable at σ_d = 25 without variance reduction or a longer horizon
— which are each a NEW amendment with a NEW pilot, never a parameter walk. G1 is NOT
feasible-pending-pilot at the measured/provisional noise.

## What this PR adds

- `scripts/prereg_ew_power_simulation_v6.py` — the v6 BCa block-bootstrap lower bound
  (`mbb_bca_lower_bound`) alongside the retained v5 percentile bound; `simulate_ar1` and the
  MBB resampling are REUSED from the merged `prereg_ew_power_simulation` module (a pinned
  test proves the resample means are byte-identical). Adds the paired validation
  (`validate_boundary_type_i`), the v6 §4.6 calibration gate (`type_i_calibrated`,
  `v6_outcome`), and the full `run_protocol_v6`.
- `tests/test_prereg_ew_power_simulation_v6.py` — 31 tests: resampling-reuse equivalence,
  BCa reduces to the percentile bound under symmetry+zero-acceleration, the two-bar
  calibration gate (incl. the CI-upper margin), the v6 outcome branch map, determinism,
  deep-null sanity, and the pinned finding (BCa does not control; ~0.14).
- `doc/experiments/2026-07-19-ew-prereg-v6-bca-validation.json` — the frozen 8000/10000
  validation evidence (code_commit-pinned).

Full v5 + v6 prereg suites green (58 passed) under the reduced-count fixed seeds. (The local
Python 3.9 env cannot collect ~98 unrelated repo modules that use runtime `X | Y` unions;
that is pre-existing and independent of these two added files, which collect cleanly.)

## Nothing walked

MEE=3, PE=6, block-length rule, α=0.10, the pilot contract, the two-stage start — all
unchanged. No production paths read or written. Research-only, run in a scratch clone.
