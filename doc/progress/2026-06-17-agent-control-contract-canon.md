# Adopt the agent control contract into cross-repo canon (umbrella)

STATUS:    in-progress (awaiting Codex review; not self-merged)
WHAT:      Adds umbrella CLAUDE.md §3.0 referencing the agent control contract +
           externalised memory (defined in renquant-orchestrator `doc/AGENT-RETROSPECTIVE.md`
           + `doc/memory/`) so all 13 repos inherit it. Non-negotiables: bottom-line-first
           reports, evidence block before conclusions, never write production paths, per-PR
           progress doc, Codex review = the merge gate for agent PRs.
WHY/DIR:   The systemic agent failure (renquant-orchestrator retrospective) is repo-agnostic;
           the cross-repo canon (§3) is where it must be inherited. Aligns with §3.1 #5
           (agent auto-merge already requires an APPROVED review).
EVIDENCE:  n/a (docs/process; no model or data claim).
NEXT:      Codex review of this PR; if approved, all repos' agents inherit the contract.
