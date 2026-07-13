# 2026-07-13 Relocate goal-governance and G2/G3 plan docs to umbrella

## Summary

Moved three documents from `renquant-orchestrator` to this repo's `doc/arch/`,
per Codex review on orchestrator PR #503/#507: they are cross-repo
operating-model material (apply to all goals/repos, not orchestrator-local),
and orchestrator was becoming a second source of truth for them.

- `doc/arch/goal-governance-process.md` (from orchestrator `doc/memory/goal-governance-process.md`)
- `doc/arch/2026-07-13-g2-phased-plan.md` (from orchestrator `doc/design/2026-07-13-g2-phased-plan.md`)
- `doc/arch/2026-07-13-g3-refactoring-plan.md` (from orchestrator `doc/design/2026-07-13-g3-refactoring-plan.md`)

Content is unchanged except: a provenance note on each file pointing back to
the orchestrator PR, and a status caveat on the G2 phased plan stating it
predates the SMA50 pivot (base-data#45) and that orchestrator#505's
data/strategy/pipeline ownership is still under active architectural review
(Codex CHANGES_REQUESTED as of 2026-07-13) -- the plan should not be read as
an approved, settled design.

The orchestrator-specific incident retrospective
(`doc/progress/2026-07-13-g2-premature-deployment-retro.md`) stays in
orchestrator, per Codex's review: it concerns that repo's own scheduler
deployment, not umbrella-level policy.

## Test plan

- [x] Docs-only change, no code touched
