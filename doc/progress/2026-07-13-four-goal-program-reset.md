# Four-goal program reset

## Bottom line

**[VERIFIED]** The prior G1-G4 dashboard over-counted deliverables and did not
establish deployment safety or economic value. This PR adds the controlling
program design: G3 canonical-path evidence is gate zero; G1 separates
quantization from allocation; G2 is an offline falsification program; and G4
requires an immutable evidence graph plus an external confirmation window.

## Scope

- Documentation and cross-repository operating boundaries only.
- No scheduler, broker, model, strategy, data, or artifact behavior changes.
- No data/model/production files are created or modified.

## Verification

- Read against `doc/arch/subrepo-operating-model.md`: the owner matrix retains
  the model-factory, artifact, policy, decision, execution, and orchestration
  boundaries.
- Read against `doc/arch/goal-governance-process.md`: the reset replaces
  PR-count progress with named gates, immutable evidence, and stop conditions.
- Research rationale and primary references are recorded in
  `doc/arch/2026-07-13-four-goal-program-reset.md` section 8.

## Rollback

Revert this documentation-only PR. No runtime state is affected.
