# 2026-08-04 — R4 gate provenance on the SERVING bundle producer (orch#564)

## The dark deployment, measured

AC6 R4 ("the run bundle answers what gate state was in force") landed in
`renquant_orchestrator/daily.py` — a bundle producer whose output surface
(`data/production_runs/`) stopped being exercised 2026-05-07 when the
daily-bridge default took the runner leg. Production persists its bundle via
umbrella `adapters/runner.py` → `kernel/artifact_contract.build_run_bundle` →
`pipeline_runs.run_bundle_json`. Measured today on the successful full run:
9,154 bytes, no `wf_gate_provenance` key. Function-level root cause is on
orch#564.

## Change

`build_run_bundle` gains a `wf_gate_provenance` block read from the resolved
`panel` artifact: same tri-state contract as the orchestrator module (present /
no_artifact_manifest / artifact_carries_no_gate_stamp, plus a
provenance_read_failed catch-all), same canonical-key-presence rule (an empty
canonical block never falls through to the legacy top-level copy — twin
registry R8), same never-raise recorder rule, field list mirrored in LOCKSTEP —
**plus** the two RFC#210 identity fields today made load-bearing:
`promotion_basis` (metadata) and `trained_date` (top-level). A run serving
under a freshness-fallback license can now answer "what license was in force"
from its own bundle.

## Verification

- `tests/test_run_bundle_wf_gate_provenance.py`: governance-served artifact →
  present block with license fields; no-artifact vs no-stamp are DISTINCT
  statuses; empty canonical block does not resurrect the legacy decoy;
  unreadable artifact never raises; `build_run_bundle` end-to-end carries the
  block. Plus the existing `tests/test_artifact_contract.py` — 21 passed.
- Acceptance after deploy: tomorrow's daily bundle carries the key; orch#564
  closes on that measurement (not before).

## Boundary note

`kernel.artifact_contract` exists ONLY in the umbrella (verified: no pipeline
copy, no live_bridge alias), so this is the correct home today. The A12 lift
(run-bundle writer → renquant-orchestrator) remains the durable destination;
this block is written to survive that lift verbatim.
