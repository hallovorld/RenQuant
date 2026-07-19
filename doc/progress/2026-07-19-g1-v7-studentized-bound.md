# Progress: G1 v7 studentized-t bound amendment (supersedes v6 BCa)

Date: 2026-07-19

## What

`doc/experiments/2026-07-19-equal-weight-deployment-prereg-v7.md` — amendment
replacing v5's §4.1 percentile MBB lower bound (and superseding the v6 BCa
amendment, disproven by #513) with a **studentized-t (bootstrap-t) block
bootstrap** lower bound, and STRENGTHENING the §4.6 activation gate per the
codex review of v6.

This revises PR #512 from BCa → studentized-t. The v6 BCa content is dropped.

## Why

Two merged sims (#511 percentile, #513 BCa) established the boundary
anti-conservatism is a **variance-scale** error (a large block b=35 over a
short T=240 ≈ 7 blocks under-estimates the sample mean's sampling variance),
NOT the median-bias/skewness BCa corrects. The textbook fix is to studentize:
rescale each resample by its own block-based SE so the pivot is scale-free.

## How it answers the v6 blocking review (all three points)

1. **Accept rule that controls the target** — §4.6 step 2 now also requires a
   one-sided Clopper-Pearson 95% upper bound ≤ α = 0.10 in every cell, at a
   FROZEN N=5000 with no optional stopping, conjunction across cells
   (multiplicity). The v6 point≤0.10 + CI-upper≤0.12 bars are retained on top.
2. **Executable bound** — §B freezes the block partition (non-overlapping
   L=⌊T/b⌋ batches of length b), edge handling (drop T mod b from the SE
   only), quantile interpolation (numpy linear), degenerate behavior
   (SE_FLOOR, L≥2), and the batch-length = MBB block-length relation. The
   batch-means SE == the delete-one-block jackknife SE for the mean (asserted).
3. **Credible calibration domain** — the frozen DGP suite is broadened to
   {iid, AR(1) φ=0.35, AR(1) φ=0.60, GARCH vol-clustering, t5 heavy tails,
   skew, missing sessions}, all matched to the same marginal std; the
   pilot-calibrated null must be blinded/decision-independent.

## Result (HONEST — the gate bites)

The v7 sim (#514) run at the frozen config shows studentized-t roughly HALVES
the excess (percentile ~0.15 → studentized ~0.11) but STILL does not clear
≤ 0.10 at the binding few-blocks regime: boundary type-I 0.1156 / 0.1064 /
0.1128 / 0.1050 (iid-240 / iid-480 / ar1-240 / ar1-480), every cell rejected
by the strengthened gate. The residual is invariant to the block-based SE
estimator, so it is a genuine few-blocks finite-sample floor, not an artifact.

Verdict: `INFEASIBLE — INFERENCE METHOD UNCALIBRATED` → v8 needs a different
pivot or a calibration/double-bootstrap layer (NOT an α/MEE/block relaxation).

## Status

RFC amendment only. NOT merged — per the §C discipline a method that fails its
own calibration gate is not ratified. Stays open for the operator's v8
decision. No activation, no pilot, no capital. Drafted per explicit operator
instruction to revise #512 to studentized-t.
