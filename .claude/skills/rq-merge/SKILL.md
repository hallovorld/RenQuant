---
name: rq-merge
description: Merge your own approved + green renquant PRs. Invoke when the user says /rq-merge, "merge my approved PRs", or asks to run the merge workflow. The orchestrator does the deterministic merge (no judgment needed); you just surface the result.
---

# rq-merge — merge your own approved + green PRs

The orchestrator merges deterministically — APPROVED-at-head + all checks
green + no stop label (`agent:manual-hold` / `agent:cost-cap` /
`agent:rebase-conflict`). It posts a visible `merged by claude` audit
comment **before** each `gh pr merge`, and fails closed if that comment
can't be written. You do not make judgment calls here; you run it and
report.

## Steps

1. **Single repo (default-safe):**

   ```bash
   PYTHONPATH=/Users/renhao/git/github/renquant-orchestrator/src \
     python3 -m renquant_orchestrator repos agent --as claude --workflow merge \
       --repo ${ARG:-hallovorld/RenQuant} --execute
   ```

2. **Cross-repo sweep (explicit blast-radius opt-in):** a `--repo all`
   merge that executes REFUSES unless you pass `--allow-all` and a bounded
   `--max-merges`. Only do this when the user explicitly asks for an
   all-repo merge sweep:

   ```bash
   PYTHONPATH=/Users/renhao/git/github/renquant-orchestrator/src \
     python3 -m renquant_orchestrator repos agent --as claude --workflow merge \
       --repo all --execute --allow-all --max-merges 5
   ```

3. **Dry run first** (recommended): omit `--execute` to see the merge queue
   (what WOULD merge) before actually merging.

## Notes

- `--allow-no-checks` opts a repo with no PR-level CI into mergeability
  (default fails closed). Use only when the user confirms a repo legitimately
  has no checks.
- Report per-PR: merged? + the audit-comment status. If a PR was approved
  but not merged, say why (failing/pending check, stop label, or no checks).
- This never merges a PR you authored-and-self-approved — an APPROVED review
  is always a genuine peer review (GitHub blocks self-approve).
