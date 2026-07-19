# Amendment v7: studentized-t lower bound + strengthened calibration gate

Date: 2026-07-19
Status: RFC amendment to the merged v5 prereg
(`2026-07-17-equal-weight-deployment-prereg-v5.md`). v5 remains the base;
this amendment REPLACES only the §4.1 inference method (the MBB lower-bound
computation) and STRENGTHENS the §4.6 activation-blocking type-I calibration
check. Supersedes the v6 (BCa) amendment: the v6 BCa bound was empirically
disproven (RenQuant#513) — it does not correct the variance-scale flaw.
No activation, no capital, no observation start is authorized. This is a
pre-activation method revision, not a mid-flight rule change (no observation
has begun under any version).

## A. Why v7 exists — the flaw is VARIANCE-SCALE, and it survives BCa

Two merged pre-activation simulations proved the earlier bounds are
anti-conservative at the deployment boundary (rule `LB90 > MEE`, boundary
null μ = MEE):

- **v5 percentile MBB** (10th pct of the bootstrap means): boundary type-I
  ~0.13–0.15 (T=240), ~0.12–0.13 (T=480) — RenQuant#511 (merged,
  independently reproduced).
- **v6 BCa** (bias-corrected + accelerated block bootstrap): 0.119–0.151,
  essentially UNCHANGED from the percentile bound and marginally worse at
  T=240 — RenQuant#513 (merged). BCa moved the rate by only −0.001…+0.003.

Mechanism (now established by two sims): the anti-conservatism is a
**variance-scale** error. A large block (`b = 35`) over a short series
(T = 240) leaves only ~⌈T/b⌉ = 7 blocks, so the bootstrap distribution of
the sample MEAN under-estimates that mean's true sampling variance; the
lower bound is therefore too tight and rejects the boundary null too often.
BCa corrects median-bias (z0) and skewness/acceleration (a) — NOT the
variance scale — so it corrects the wrong thing. The v6 §B fallback (named
in advance) is exactly this amendment.

The prereg discipline worked as intended: the flaw was caught by simulation
BEFORE activation, before any capital, at zero cost. This is the fourth
pre-activation catch in the v3→v7 lineage (v3 unsatisfiable gate; v4
positivity≠materiality; v5 percentile under-controls; v6 BCa doesn't fix it).

## B. The repaired inference (v7) — a studentized-t block bootstrap

**§4.1 inference (replaced):** the one-sided 90% lower confidence bound for
mean(d_t) is computed by a **studentized-t (bootstrap-t) block bootstrap** —
the textbook fix for a variance-scale coverage error, because it makes the
statistic approximately scale-free by rescaling each resample by its OWN
block-based standard error. The construction is FROZEN and fully executable
(responding to review point 2 below):

1. **Resampling (unchanged):** the moving-block bootstrap with the v5 block
   length `b = min(⌈1.75 × max_holding_days⌉, 40)` sessions and B = 10,000
   resamples. The block-start draw structure is byte-identical to the merged
   v5/v6 primitive (`rng.integers(0, T − b + 1, size=(B, ⌈T/b⌉))`); a pinned
   test asserts it.
2. **Per-resample block-based standard error `se*` (FROZEN):** the
   **non-overlapping batch-means** estimator. Partition the series into
   `L = ⌊T/b⌋` consecutive length-`b` batches (the final `T mod b` sessions
   are dropped from the SE computation ONLY; they still enter the point
   estimate). With batch means Y_1..Y_L, `se = sd(Y; ddof=1)/√L`. For the
   MEAN this is algebraically identical to the delete-one-block jackknife SE
   over the same blocks (a pinned test asserts the equality) — so "block-of-
   blocks / delete-block variance" and "batch-means" name the same frozen
   object. Batch length = the MBB block length `b` (the explicit relation
   review point 2 asks for). Requires `L ≥ 2`; a degenerate zero-dispersion
   resample floors `se` at `SE_FLOOR = 1e-9` bps so the pivot stays finite
   (never dropped — B is fixed).
3. **Pivot + bound:** for each resample r compute
   `t*_r = (mean*_r − mean) / se*_r`; take the one-sided lower bound
   `LB = mean − Q_{1−α}(t*) · se`, where `se` is the same batch-means SE on
   the ORIGINAL series and `Q_{1−α}` is the (1−α)-quantile of the pivot
   sample under numpy's default **linear interpolation** (frozen). The
   deployment rule `LB > MEE` is equivalent to `(mean − MEE)/se > Q_{1−α}(t*)`.

The percentile bound is retained in code as a positive control (it must
reproduce the merged v5 ~0.15 number on the same seeds), so the calibration
is COMPARED, never silently swapped.

## C. Strengthened HARD activation gate (responds to the v6 review)

The blocking design review of the v6 amendment (codex, PR #512) raised three
points; v7 adopts all three as frozen requirements. The gate no longer just
"reports a point estimate that could accept an over-α method."

**Review point 1 — the accept rule must actually control the target.**
§4.6 step 2 acceptance now requires, in EVERY frozen DGP cell:
- (retained bars, v6) MC point estimate ≤ 0.10 AND 95% MC-CI upper ≤ 0.12; AND
- (added) a **one-sided Clopper-Pearson 95% UPPER confidence bound** on the
  true boundary type-I ≤ α = 0.10.
The interval method (exact Clopper-Pearson), the replication count
(N = 5,000 trials, FROZEN), the seeds (fixed per cell, recorded), and the
**stopping prohibition** (no optional stopping — N is fixed before the run)
are all pre-specified. Multiplicity: the method is CALIBRATED only if the
conjunction holds across ALL cells (a single failing cell → uncalibrated).
N = 5,000 is sized so that a truly conservative method (true type-I ≤ ~0.083)
clears the CP-95% upper ≤ 0.10 bar, making the rule achievable rather than
vacuous.

**Review point 2 — an executable bound definition.** §B freezes the block
partition (non-overlapping `L = ⌊T/b⌋` batches of length `b`), the edge
handling (drop `T mod b` from the SE only; last MBB block truncated in the
point estimate), the quantile interpolation (numpy linear), the
degenerate/undefined behavior (`SE_FLOOR`, `L ≥ 2` guard), and the exact
relation between the MBB block length and the SE blocks (identical `b`).

**Review point 3 — a credible calibration domain.** The frozen DGP suite is
broadened from {iid, one AR(1)} to cover the intended dependence and
distributional risks:
- `iid` (φ=0) and `ar1` (φ=0.35) — the required core cells;
- `ar1_persist` (φ=0.60) — serial-dependence persistence;
- `garch` — GARCH(1,1) innovation variance (volatility clustering);
- `heavy_t5` — standardized Student-t(5) innovations (heavy tails);
- `skew` — standardized centred-exponential innovations (skew ≈ 2);
- `missing` — ~8% sessions missing at random (incomplete paired series).
All cells are matched to the SAME marginal std so they differ only in the
dependence/tail/skew structure. Each is evaluated at T = 240 and T = 480.
The **pilot-calibrated null** (§4.7) that supplies the activation-time
dependence/variance parameters must be derived from data INDEPENDENT of any
eventual test decision (blinded prospective pilot), so calibration is not a
fitted degree of freedom; the synthetic default is loudly NOT-FOR-ACTIVATION.

If the §4.6 re-run shows the method still does not control boundary type-I in
every cell → `INFEASIBLE — INFERENCE METHOD UNCALIBRATED`, escalate to the
next amendment; NEVER relax α / MEE / block length.

## D. Empirical result of the v7 method (RenQuant#514 sim — HONEST)

The v7 sim (`scripts/prereg_ew_power_simulation_v7.py`, PR #514) was built and
run at the frozen config (B = 10,000, N = 5,000 trials, fresh seeds). The
studentized-t bound materially improves on percentile/BCa but STILL does not
control the boundary type-I (core cells; full 14-cell grid in the evidence JSON):

| DGP | T | percentile (control) | studentized-t (95% CI) | CP-95% upper | accepted? |
|-----|-----|----------------------|------------------------|--------------|-----------|
| iid (φ=0)    | 240 | 0.1420 | **0.1168** [0.108, 0.126] | 0.1245 | NO |
| iid (φ=0)    | 480 | 0.1238 | **0.1050** [0.097, 0.114] | 0.1124 | NO |
| AR(1) φ=0.35 | 240 | 0.1388 | **0.1158** [0.107, 0.125] | 0.1235 | NO |
| AR(1) φ=0.35 | 480 | 0.1216 | **0.1116** [0.103, 0.120] | 0.1192 | NO |

Studentized-t roughly HALVES the excess (percentile ~0.14 → ~0.11) but every
core cell fails both the retained bars and the CP-95%-upper ≤ α rule; the
broader suite (persistence φ=0.60 / GARCH / t5 / skew / missing) is no better —
worst-cell studentized type-I 0.1180 (GARCH, T=240), and NO cell of the 14 is
accepted. The
residual is INVARIANT to the block-based SE estimator (batch-means ==
delete-block jackknife; the overlapping-block LRV variant is worse), so it is
a genuine **few-blocks finite-sample floor** (b=35 over T=240 ≈ 7 blocks; the
SE uses L=6 batches), not an SE-estimator artifact. The deep-null (μ=0)
rejection is negligible (~0.008; asserted in the sim's sanity test), and the
percentile control reproduces the merged ~0.14, so the sim is sound and the
residual is genuine, not MC noise.

**Verdict: `INFEASIBLE — INFERENCE METHOD UNCALIBRATED`.** v7 is NOT ratified.
The gate (as strengthened in §C) correctly refuses it. This amendment stays
OPEN pending a v8 decision.

## E. v8 direction (for the operator's decision — NOT executed here)

The residual is a few-blocks property, so a further SE choice will not fix it;
v8 needs a DIFFERENT pivot or a small-sample calibration layer:
1. a **calibrated / iterated (double) block bootstrap** that recalibrates the
   nominal level so realized one-sided coverage hits 90% (the standard fix for
   residual bootstrap-t under-coverage; heavier compute);
2. a subsampling bound, or a symmetric-t / small-sample-df reference;
3. accepting that at b=35 with T ≤ 480 (≈ 7–13 blocks) NO bound may achieve
   exact one-sided 90% coverage — i.e. G1 may be untestable at this horizon
   under the block rule, independent of pivot.
Any of these is a NEW amendment with the SAME §C gate; none walks α/MEE/block.

Separately and already established (RenQuant#513): even a perfectly calibrated
bound tops out at power ~0.73 < 0.80 at PE = 6 within T ≤ 480 at the
provisional σ_d = 25 bps → G1 is **also** `INFEASIBLE AT CONSERVATIVE PILOT
NOISE`. So G1 is doubly blocked at the provisional noise; a clean feasibility
answer awaits the REAL σ_d from a ≥40-session blinded pilot (calendar, not
today). This amendment does not activate anything — no capital, no pilot start.

## F. What v7 does NOT change

MEE = 3 / PE = 6 and their economic rationale (#500); the deployment rule
`LB90 > MEE` (decides) vs efficacy `LB90 > 0` (reported); the two-stage
pilot-registration/activation start; §4.7 blinded pilot + data hygiene
(pre-freeze burn, single-epoch pooling); the arms, g_t self-financing, and
frozen schemas. Only the bound COMPUTATION and the method-calibration gate
change.

## G. Acceptance for this amendment

1. Codex adversarial review of this document (re-requested; supersedes the v6
   review).
2. The §4.6 sim EXTENDED with the studentized-t bound (RenQuant#514): DONE —
   it empirically establishes the method does NOT clear the strengthened gate
   (§D). Per the §C discipline the honest outcome is `INFEASIBLE — INFERENCE
   METHOD UNCALIBRATED`, so neither this amendment nor the v7 sim is merged as
   the ratified method; both stay open for the v8 decision.
3. No activation, no pilot, no capital under any version.
