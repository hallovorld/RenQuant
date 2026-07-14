# Contributor-Independent Review Policy

STATUS: delivered for review

WHAT: Align the canonical umbrella agent contract with the orchestrator's
reviewer-separation guard. A reviewer never commits or pushes to a peer PR,
and every PR branch has one GitHub commit identity: the PR creator. Additional
attribution, including a `Co-Authored-By` trailer, is a merge blocker.

WHY/DIR: GitHub's native self-approval restriction applies to the PR creator,
not every branch contributor. Treating it as a complete independence guarantee
allowed a reviewer to modify a peer branch and then appear to be its second
opinion. The earlier policy also made co-author attribution advisory while
requiring co-author trailers, which creates exactly the mixed-identity history
that prevents a reliable merge decision.

EVIDENCE: The executable guard is proposed in `renquant-orchestrator#517`.
Focused checks cover explicit-fix exclusion, rejection of a contributor's
approval, and merge blocking for any mixed GitHub commit attribution.
This companion change puts the same rule in the umbrella's canonical Claude,
Codex, and operator guidance.

NEXT: Obtain Claude's review of the explicit-fix marker and ensure every ad hoc
review/merge invocation applies the same check before acting.
