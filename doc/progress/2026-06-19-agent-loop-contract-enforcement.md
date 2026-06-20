# agent loop contract enforcement

STATUS: delivered
WHAT: hardens the local `agent_pr_loop` so each unattended cycle bootstraps local SHORT memory, instructs agents to read the control contract before acting, enforces progress-doc / evidence / production-path review expectations in prompts, and runs strict post-merge audit every cycle.
WHY/DIR: aligns the umbrella loop with the merged agent-control contract so unattended review/fix/merge runs do not silently drift back to prompt-only behavior.
EVIDENCE: n/a
NEXT: keep the loop running under launchd and flip the GitHub repo rulesets to require the Codex actor where admin rights permit.
