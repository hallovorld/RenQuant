# G1: §4.6 pre-activation power simulation — FROZEN DESIGN FAILS ITS OWN GATE

STATUS: delivered
WHAT: Implements and runs the §4.6 pre-activation Monte Carlo required by
the equal-weight deployment prereg v3 (RenQuant#480, merged 2026-07-16).
`scripts/prereg_ew_power_simulation.py` faithfully implements the FROZEN
§4.1 inference (MBB, b = ceil(1.75·max_holding_days), 10,000 bootstrap
draws, one-sided 90% lower bound, GO iff lower bound > MDE = 3 bps/session)
and the §4.6 protocol (5,000-trial null for type-I, 5,000-trial power at
the MDE, 20-session horizon sweep 240→480, frozen JSON output artifact for
the activation commit). 13 component tests.

RESULT (full run, seed 20260716, results JSON committed alongside):
- Type-I at T=240: 0.0126 (95% CI [0.0095, 0.0157]; requirement
  <= 0.10) — passes trivially, because requiring the lower bound to clear
  3 bps under a zero-mean null is far stricter than nominal alpha.
- Power at MDE: 0.125–0.147, flat across T = 240…480 (0.1468 at T=240, 0.1246 at T=480) — nowhere near the
  required 0.80, and FLAT in T. **The §4.6 horizon adjustment cannot fix
  this.** Sensitivity grid (27 cells over sigma_d ∈ {10,25,40} bps,
  phi ∈ {0,0.35,0.6}, max_holding_days ∈ {10,20,40}): power range
  0.121–0.214 — the failure is structural, not a DGP artifact.

WHY THE DESIGN CANNOT PASS AS FROZEN: §4.6 step 3 evaluates power at a
true effect EQUAL to the MDE, while the §4.1 GO criterion requires the
one-sided 90% LOWER BOUND to EXCEED the MDE. For any calibrated 90%
lower bound, P(LB > true mean) ≈ 0.10 by the definition of coverage —
at ANY sample size. Power at the boundary is the miscoverage rate, not
a quantity that grows with T. The prereg conflates two conventions:
(a) GO = LB > 0 with the MDE embedded in the power analysis (standard),
vs (b) GO = LB > MDE (what §4.1 froze). Under (b), 80% power requires a
true effect ≈ MDE + 2.12·SE (≈ 7.9 bps/session at the provisional DGP,
T=240) — 2.6× the declared MDE.

CONSEQUENCE (per §4.6's own rule): "If this simulation fails, the
protocol's horizon or MDE is revised — the experiment does not start."
Activation is BLOCKED pending a design-level revision (separate design
PR; options quantified in the PR body: (i) change GO to LB > 0 —
feasibility then depends on the real sigma_d: T≈649 needed at
sigma=25 bps (exceeds the 480 cap), T≈234 at sigma=15 bps; (ii) keep
GO = LB > MDE and re-derive the power requirement at a detectable
effect; (iii) raise the MDE). No revision is made here — prereg
discipline: the raise happens BEFORE activation, which is exactly what
this gate is for.

DGP caveat: AR(1), sigma_d = 25 bps/session, phi = 0.35,
max_holding_days = 20 (b = 35) — PROVISIONAL PARAMETRIC, marked as such
in the output artifact. §4.6 asks for calibration from an A1-vs-B
replay; the two-arm harness (live 2026-07-11) has produced only ~4
paired sessions — far too few to fit an ACF — and no per-session paired
active-return series for these two arms exists in the artifact store.
The structural conclusion (power ≈ alpha at the MDE boundary) is
DGP-invariant (demonstrated by the 27-cell grid); the FEASIBILITY
numbers for option (i) are not, so a data-fitted DGP re-run is
mandatory input to the revision design PR.

EVIDENCE: `doc/experiments/2026-07-16-ew-prereg-power-simulation-results.json`
(frozen §4.6 output artifact: rates, CIs, per-T sweep, grid, DGP params,
seed, code commit); `pytest tests/test_prereg_ew_power_simulation.py`
→ 13 passed.
NEXT: design-level revision PR for the §4.1/§4.6 convention mismatch
(operator-visible decision); data-fitted DGP re-run once the two-arm
harness has accumulated enough paired sessions to estimate sigma_d/ACF.
