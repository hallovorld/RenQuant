# Progress: G1 v7 studentized-t bound — pre-activation validation

Date: 2026-07-19

## What

`scripts/prereg_ew_power_simulation_v7.py` + `tests/` + frozen evidence
`doc/experiments/2026-07-19-ew-prereg-v7-validation.json`. Acceptance evidence
for the v7 amendment (PR #512, §G item 2): extends the merged v5 §4.6 sim with
the studentized-t (bootstrap-t) block-bootstrap lower bound and the strengthened
§4.6 calibration gate, then EMPIRICALLY tests whether studentizing fixes the
variance-scale anti-conservatism the percentile (#511) and BCa (#513) bounds had.

## Decisive result — studentized-t does NOT control the boundary type-I

Frozen config: B=10,000, N=5,000 trials, fresh seeds. Rule LB90 > MEE, boundary
null μ = MEE = 3, σ_d = 25 bps, b = 35.

| DGP | T | percentile (control) | studentized-t (95% CI) | CP-95% upper | accepted? |
|-----|-----|----------------------|------------------------|--------------|-----------|
| iid (φ=0)    | 240 | 0.1420 | **0.1168** [0.1079, 0.1257] | 0.1245 | NO |
| iid (φ=0)    | 480 | 0.1238 | **0.1050** [0.0965, 0.1135] | 0.1124 | NO |
| AR(1) φ=0.35 | 240 | 0.1388 | **0.1158** [0.1069, 0.1247] | 0.1235 | NO |
| AR(1) φ=0.35 | 480 | 0.1216 | **0.1116** [0.1029, 0.1203] | 0.1192 | NO |

Studentized-t roughly HALVES the excess (percentile ~0.14 → ~0.11) but every
core cell fails the gate (point > 0.10, and the one-sided Clopper-Pearson 95%
upper bound > α). The broadened suite (persistence φ=0.60, GARCH vol-clustering,
Student-t(5) heavy tails, skew, missing sessions) is no better — ALL 14 cells
fail, worst-cell studentized type-I 0.1180 (GARCH, T=240); see the frozen
evidence JSON for the full grid. Positive control: the percentile arm reproduces
the merged ~0.14 number. Deep-null (μ=0) rejection ~0.008 (sanity test) → the
sim is sound, so the residual is genuine, not MC noise.

**Mechanism:** the residual is a FEW-BLOCKS finite-sample floor — b=35 over
T=240 leaves only ~7 blocks and the SE uses L=6 batches, so even a scale-free
pivot's bootstrap distribution under-estimates the true upper tail. It is
INVARIANT to the block-based SE estimator (batch-means == delete-one-block
jackknife; the overlapping-block LRV variant is worse), so it is not an
SE-estimator artifact. → `INFEASIBLE — INFERENCE METHOD UNCALIBRATED`, escalate
to a **v8** (a different pivot or a calibrated / iterated double bootstrap),
NEVER an α / MEE / block relaxation.

For the record, even a perfectly calibrated bound tops out at power ~0.73 < 0.80
at PE=6 within T ≤ 480 at σ_d = 25 (established in #513), so G1 is ALSO
`INFEASIBLE AT CONSERVATIVE PILOT NOISE`. G1 is doubly blocked at the provisional
noise; a clean feasibility answer awaits the REAL σ_d from a ≥40-session blinded
pilot (calendar, not today).

## Contents

- `scripts/prereg_ew_power_simulation_v7.py` — `studentized_block_lower_bound`
  (frozen batch-means / delete-block SE; numpy-linear pivot quantile; SE floor;
  L≥2 guard) ALONGSIDE the retained percentile control; the broadened DGP suite
  (`simulate_dgp`); the strengthened acceptance gate (`type_i_accepted` with the
  Clopper-Pearson upper rule); `validate_boundary_type_i`, `feasibility_preview`,
  and `run_protocol_v7`. The MBB resample draw structure is reused byte-identical.
- `tests/test_prereg_ew_power_simulation_v7.py` — 42 tests (draw-structure reuse;
  batch-means == delete-block equivalence; CP gate incl. the multiplicity/margin
  logic; DGP-suite properties; determinism; deep-null; the pinned finding).
- `doc/experiments/2026-07-19-ew-prereg-v7-validation.json` — frozen evidence.

## Reproducibility

The evidence is deterministic under the fixed seed — reproduce byte-identically with
`python scripts/prereg_ew_power_simulation_v7.py --validate --seed 20260719 --trials 5000 --boots 10000 --out <path>`
(the JSON's `code_commit` records the scratch-clone HEAD at generation time; the
SEED, not the commit hash, is the reproducibility guarantee).

## Status

Validation only. NOT merged — per the v7 §C discipline a method that fails its
own calibration gate is not ratified. Fifth pre-activation catch in the v3→v7
lineage; prereg discipline blocked a mis-calibrated rule before any capital, at
zero cost. Nothing walked; no production paths touched; run in a scratch clone.
Open for the operator's v8 decision. No activation, no pilot, no capital.
