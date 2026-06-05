---
name: rq-review
description: Review the OTHER agent's open PRs across renquant repos. Invoke when the user says /rq-review, "review codex's PRs", "do my reviews", or asks to run the review workflow. Uses the orchestrator's cross-repo queue; you (Claude) write each review.
---

# rq-review — review the peer agent's PRs

You are Claude acting as the **reviewer** in the orchestrator-driven
multi-agent PR model. You review **Codex's** open PRs (never your own —
GitHub blocks self-approve and the queue excludes self-authored).

## Steps

1. **Get the queue** from the control plane. Default to the whole manifest;
   if the user named a repo, pass `--repo <name>`:

   ```bash
   PYTHONPATH=/Users/renhao/git/github/renquant-orchestrator/src \
     python3 -m renquant_orchestrator repos agent --as claude --workflow review --repo ${ARG:-all}
   ```

   The JSON `repos[].plan.queue` lists each PR (number, head_ref, url) that
   needs your review. Empty queue → tell the user "nothing to review" and stop.

2. **For each queued PR**, in its repo:
   - Read the diff: `gh pr diff <n> --repo <owner/repo>`.
   - Read the PR body for intent.
   - Judge per CLAUDE.md §7 (correctness, data-flow safety, tests,
     leakage, operational fail-closed). Review the **diff**, not the codebase.

3. **Post ONE consolidated review** with your token, with severity tags
   (`BLOCKER` / `HIGH` / `MED` / `LOW`) and a clear verdict. The body MUST
   contain the visible attribution string **`reviewed by claude`**:

   ```bash
   gh pr review <n> --repo <owner/repo> <STATE> --body "reviewed by claude — ..."
   ```

   Choose STATE by highest severity found:
   - `--request-changes` if any BLOCKER / HIGH / MED.
   - `--comment` if only LOW / nits.
   - `--approve` if zero findings (note: if the PR is under the same
     account you operate, GitHub blocks self-approve — then post
     `--comment` stating "VERDICT: APPROVE").

## Rules

- One review per PR, consolidated — not a thread of comments.
- Run the PR's focused tests when a claim hinges on them; cite the result.
- Never review a PR authored by `agent:claude` (the queue already excludes
  these; double-check the `author_agent` field).
- After finishing, report a one-line summary per PR (verdict + key finding).
