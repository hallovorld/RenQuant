# Progress: G1 v5 pre-activation power / type-I simulation (§4.6)

Date: 2026-07-18

## What

`scripts/prereg_ew_power_simulation_v5.py` (+ `tests/test_prereg_ew_power_simulation_v5.py`)
— the §4.6 pre-activation power/type-I simulation updated to the MERGED v5
decision rule (`doc/experiments/2026-07-17-equal-weight-deployment-prereg-v5.md`,
MEE=3/PE=6 rationale `doc/experiments/2026-07-18-g1-mee-pe-rationale.md`, #500).

Supersedes the #485 sim (`scripts/prereg_ew_power_simulation.py`), which tested
the v3 rule (LB > MDE, power sized AT the MDE ⇒ ≈ α by coverage). The sound
numeric primitives (AR(1) DGP, MBB lower bound) are REUSED from #485; only the
rule, the pilot input contract, the two-stage output emission, and the
infeasibility branch are new.

v5 operating characteristics under test:
- DEPLOYMENT rule (decides): one-sided 90% MBB lower bound of `mean(d_t)` > MEE (3 bps/session).
- TYPE-I at the LEAST-FAVOURABLE null μ = MEE (H0: μ ≤ MEE); must be ≤ 0.10, MC point estimate governs, 95% CI reported.
- POWER at the planning effect μ = PE = 6 bps/session (= 2·MEE), strictly ABOVE the threshold — so power is NOT coverage-capped (the v3 error), grows with T.
- Horizon search T ∈ 240..480 (step 20); MBB block length `b = ceil(1.75·max_holding_days)` capped at 40 SESSIONS (v5 caps `b`, not `max_holding_days` — differs from #485's cap for holdings > ~22 sessions).

## Two-stage awareness (§4.7)

This sim is the piece the ACTIVATION commit runs on the SEALED blinded pilot.
The pilot data does not exist yet, so the DGP dependence + the conservative
upper-90%-CL long-run variance are an EXPLICIT INPUT CONTRACT (`PilotNull`, from
`--pilot-fit` JSON) with a documented SYNTHETIC DEFAULT for testing/preview only
(loudly flagged NOT VALID FOR ACTIVATION). The AR(1) marginal variance is
back-solved so the DGP's long-run variance equals the pilot's conservative
bound. The sim emits the frozen §4.6-step-5 / §7-yaml outputs (type-I + CI,
power at PE + CI, final T, final n_blocks, `b`, pilot-fitted DGP params + pilot
manifest digest, sim code commit + seed) ready to paste into the activation
commit.

## Infeasibility (§4.6 step 4)

If power ≥ 0.80 at PE cannot be reached within T ≤ 480 under the conservative
pilot variance, the recorded outcome is `INFEASIBLE AT CONSERVATIVE PILOT
NOISE`. No MEE/PE/block parameter is walked to force feasibility.

## Finding (preview, provisional not-yet-pilot σ_d)

Run against the synthetic default = #485 provisional (marginal σ_d = 25 bps,
φ = 0.35, b = 35), a PREVIEW only (not the sealed pilot):

1. **Power infeasible at PE=6 under σ_d=25.** Power at PE rises 0.55 (T=240) →
   0.73 (T=480), never reaching 0.80 ⇒ `INFEASIBLE AT CONSERVATIVE PILOT NOISE`.
   Consistent with #485 (σ_d=25 pushes the required horizon beyond 480).
2. **The FROZEN §4.1 percentile MBB is mildly anti-conservative.** The
   deployment rule's boundary type-I (μ=MEE) runs ~0.13–0.15 (> the nominal
   0.10), worst with few blocks (b=35 ⇒ 7 blocks at T=240; ~0.145). The 95%
   simulation CI excludes 0.10 — a calibration fact, not MC noise. This is the
   SAME quantity #485 reported as "power-at-MDE ≈ α" for the v3 rule; v5's §4.6
   step 2 re-reads it as the deployment rule's type-I, and it exceeds α. The
   sim's gate correctly refuses activation (never silently passes). A real
   activation-ready GO therefore requires the sealed pilot's fitted null to
   clear BOTH bars, or a NEW amendment adopting a coverage-corrected bound
   (studentized-t / BCa) with a NEW pilot — not a silent method swap.

These are PREVIEW numbers on provisional, pre-pilot noise. The binding run is
the activation-time run on the sealed blinded pilot (§4.7).

## Tests

27 tests (this file), green under the repo xdist config. Cover: the v5 block
rule + its divergence from #485; the pilot input contract (long-run-variance
sizing, JSON round-trip, validation); power increases with T and effect size;
the infeasibility branch fires under high injected variance without walking
parameters; the v3-vs-v5 difference (v5 power at PE not coverage-capped);
determinism under fixed seed; the anti-conservative-type-I finding pinned; and
the full outcome-branch logic (GO_VALID / TYPE_I_EXCEEDED / INFEASIBLE) plus the
synthetic-never-activation-ready guard.

## Not done (calendar / data — by design)

This is the LAST CODE piece before G1 pilot registration. The pilot-registration
COMMIT and the ≥40-session blinded pilot are calendar/data, not code, and are
NOT part of this PR. The activation-valid sim run happens later, on the sealed
pilot's fitted null.
