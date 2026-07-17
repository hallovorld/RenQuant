# Amendment v4: repair of the decision-rule structure (equal-weight deployment prereg)

Date: 2026-07-17
Status: RFC amendment to `2026-07-13-equal-weight-deployment-prereg.md` (v3).
v3 remains the base document; this amendment REPLACES §4.3-G1, §4.6 step 3,
and adds §4.7. Drafted personally per design-review policy.
Prereg discipline: v3 was never activated; no observation has begun; this
amendment is therefore a pre-activation design revision, not a mid-flight
rule change.

## A. Why v4 exists (the §4.6 finding)

The v3 pre-activation simulation (RenQuant#485, merged 2026-07-16) exposed
a STRUCTURAL contradiction, not a sample-size problem:

- v3 §4.3-G1 required the one-sided 90% MBB lower bound to EXCEED the
  MDE (3 bps/session);
- v3 §4.6 evaluated power at true effect = MDE.

For any calibrated one-sided 90% bound, P(LB > θ_true) ≈ α ≈ 0.10 by the
definition of coverage — measured power was flat at 0.125–0.147 across
T = 240…480 and can never approach 0.80. The gate as written is
unsatisfiable at its own design point. §4.6 did its job: the flaw was
caught before activation.

## B. The repaired decision rule (Neyman–Pearson form)

**G1 (replaced):** the experiment GOES iff the one-sided 90% MBB lower
bound for `mean(d_t)` exceeds **0** (test of H0: μ ≤ 0 at α = 0.10).

**MDE's role (clarified):** 3 bps/session is NOT the pass line; it is the
alternative at which the design must demonstrate power ≥ 0.80 in the
pre-activation simulation. Economic materiality is reported separately:
alongside the binary verdict, the memo reports the point estimate and the
full lower-bound curve so a "significant but < MDE" outcome is visibly
weak, and §4.3-G6 attribution still applies.

This is the standard testing structure v3 conflated: size at the null,
power at the alternative.

## C. Feasibility is an empirical question — the σ_d dependency (§4.7 NEW)

The #485 simulation shows LB>0 power depends decisively on the paired
daily noise σ_d, which v3 parameterized PROVISIONALLY (AR(1),
σ = 25 bps/session) from ~4 real paired sessions:

- σ_d = 25 bps → T ≈ 649 sessions for power 0.80 (exceeds the v3 480 cap)
- σ_d = 15 bps → T ≈ 234 sessions (feasible inside the cap)

Therefore v4 makes null calibration a HARD activation prerequisite:

1. **Accumulate ≥ 40 real paired sessions** from the running two-arm
   harness (no activation before then; the harness itself is
   observation-free plumbing, not the experiment).
2. **Re-fit the null** (σ_d, autocorrelation → MBB block length b) from
   those sessions; freeze the fitted parameters + their sample manifest in
   the activation commit.
3. **Re-run §4.6** with the fitted null. Activation requires type-I ≤ 0.10
   AND power ≥ 0.80 at MDE within T ≤ 480. If the fitted σ_d makes that
   infeasible, the experiment is NOT activated and the recorded outcome is
   "infeasible at measured noise — redesign required" (variance-reduction
   options: longer rebalance interval, paired-difference denoising, or a
   larger MDE — each a NEW amendment, never a silent parameter walk).

## D. Statistical discipline imported from the G2 review (methodology unification)

The 2026-07-17 codex review of the G2 gate memo (orch#532) set the house
standard; the same items bind here explicitly:

- **No post-selection inflation:** this prereg tests exactly ONE frozen
  spec (equal-weight vs conviction, one universe, one rebalance rule);
  any additional spec added later requires family-wise control declared
  BEFORE evaluation.
- **Inference model pinned:** daily paired differences d_t; MBB with
  b = ceil(1.75 × max_holding_days) unless the §4.7 fit indicates
  longer dependence, in which case b is set from the fitted
  autocorrelation length and frozen in the activation commit; report
  n valid sessions and per-session pairing integrity counts.
- **Executable timing convention:** d_t is computed from same-snapshot
  paired decisions (§3.2 g_t framework, unchanged from v3); no
  close-to-close double use is possible by construction, but the
  activation commit must state the exact snapshot timestamp convention.

## E. What v4 does NOT change

- The g_t self-financing exposure rule (§3.2), regime descriptive-only
  status (§4.2), frozen telemetry/verdict schemas (§6.5), and
  RFC-until-activation gating (§7) all stand as merged in v3.
- No capital, no arming, no observation start is authorized by this
  amendment. Activation remains operator-visible per §7.

## F. Acceptance for this amendment

1. Codex adversarial review of this document.
2. The #485 simulation code re-run demonstrating the REPAIRED rule's
   type-I ≤ 0.10 at the null (sanity: it is the same bound, so type-I is
   inherited) and the power curve vs T under both provisional σ values —
   updating the frozen results artifact with the v4 rule.
3. §4.7's session counter starts reporting in the two-arm harness
   telemetry (n_paired_sessions to date), so the ≥40 prerequisite is
   mechanically checkable at activation time.
