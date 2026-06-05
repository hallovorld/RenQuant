---
name: rq-fix
description: Fix your own renquant PRs that have unaddressed review findings. Invoke when the user says /rq-fix, "fix codex's comments on my PRs", "address review findings", or asks to run the fix workflow. Uses the orchestrator's cross-repo queue; you (Claude) make each fix.
---

# rq-fix — address review findings on your own PRs

You are Claude acting as the **author** addressing the reviewer's findings.
You fix **your own** (`agent:claude`) PRs that carry unaddressed findings —
a CHANGES_REQUESTED review, or a comment with a BLOCKER/HIGH/MED tag at the
current head. The reviewer (Codex) does not fix; the author does.

## Steps

1. **Get the queue**:

   ```bash
   PYTHONPATH=/Users/renhao/git/github/renquant-orchestrator/src \
     python3 -m renquant_orchestrator repos agent --as claude --workflow fix --repo ${ARG:-all}
   ```

   `repos[].plan.queue` lists your PRs with unaddressed findings. Empty →
   "nothing to fix" and stop.

2. **For each queued PR**:
   - Read the reviewer's findings: `gh pr view <n> --repo <owner/repo> --json reviews,comments`.
   - Check out the PR branch locally (it's a renquant repo clone under
     `~/git/github/<repo>`); sync first (`git fetch && git rebase origin/main`
     per §3.2).
   - Make the **smallest** change that resolves each finding. Address
     BLOCKER/HIGH; address MED unless you can justify why not; LOW/nit at
     your discretion. Don't drift into unrelated cleanup.
   - Run the focused tests for the changed path (§7.1) — add a paired test
     if the fix is for an untested behavior.

3. **Comment, commit, push**:
   - Post a visible `fixed by claude` comment summarizing what you changed
     per finding (and anything deliberately not done, with the reason):
     `gh pr comment <n> --repo <owner/repo> --body "fixed by claude — ..."`.
   - Commit with the trailer `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
   - `git push --force-with-lease` (after the §3.2 rebase).

## Rules

- Only fix PRs authored by `agent:claude` (the queue enforces this).
- Never silently skip a finding — surface skipped ones in the `fixed by`
  comment so the re-review sees them.
- After pushing, the reviewer (Codex) re-reviews; the loop continues until
  the review is clean.
