# Codex Agent Automation Design

Date: 2026-05-31

Status: design for review. This document specifies the RenQuant-wide mechanism
for Codex-owned pull requests, automatic review of non-Codex pull requests, and
automatic response to reviews on Codex-owned pull requests.

## Goals

The automation has three hard goals:

1. Any branch or pull request created by Codex must identify itself as Codex.
2. Any open pull request not created by Codex should receive a Codex review.
3. Any trusted review on a Codex-owned pull request should trigger a bounded
   Codex fix pass against that same pull request branch.

The system must preserve the PR-based workflow in `CLAUDE.md` section 3.1. It
must never push directly to `main`, auto-merge, or run untrusted pull request
code with write-token secrets.

## Identity Contract

Labels are the source of truth because authorship is ambiguous: a human may
open a Codex-assisted PR, and automation may push through a bot token.

Codex-created branches use:

```text
codex/<task-slug>
```

Codex-created pull requests must include:

```text
labels:
  agent:codex
  agent:auto-fix

body footer:
  Agent-Origin: Codex
  Agent-Policy: auto-fix-on-review
```

Optional control labels:

```text
agent:manual-hold    # stop all Codex automation on this PR
agent:needs-review   # force a Codex review even if normally skipped
agent:reviewed       # Codex review already posted for the current head
agent:fix-attempt-1  # incremented by auto-fix workflow
agent:fix-attempt-2
agent:fix-attempt-3
```

## Components

Add these files to each active RenQuant repo, or centralize them as reusable
workflows once the first repo proves the design.

```text
.github/workflows/codex-auto-review.yml
.github/workflows/codex-auto-fix.yml
scripts/codex_pr_create.sh
```

The local `AGENTS.md` remains the repo-specific instruction surface: tests,
layout, forbidden imports, and rollback notes. Cross-repo behavior stays here in
`RenQuant/doc/arch/` to avoid duplicated stale policy.

## Pull Request Creation

`scripts/codex_pr_create.sh` is the only supported helper for Codex-created
pull requests. It should:

1. `git fetch origin`.
2. Create or reuse `codex/<task-slug>` from latest `origin/main`.
3. Commit with a normal project commit message.
4. Push the branch.
5. Create the PR with `agent:codex` and `agent:auto-fix` labels.
6. Add the `Agent-Origin` and `Agent-Policy` footer.

Template:

```bash
gh pr create \
  --base main \
  --head "$BRANCH" \
  --label agent:codex \
  --label agent:auto-fix \
  --title "$TITLE" \
  --body-file "$BODY_FILE"
```

## Auto Review Workflow

Trigger:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
```

Decision logic:

```text
skip if draft
skip if label agent:manual-hold exists
skip if label agent:codex exists
review if label agent:needs-review exists
review otherwise, because the PR is non-Codex
```

The review job checks out the pull request merge ref in read-only mode, runs the
repo's configured inspection/test commands where safe, and posts one review
comment summarizing findings. The review must use normal code-review severity:
blockers and high-risk issues first, then lower-risk notes. It must not rewrite
the branch.

After posting a review for a given head SHA, the workflow records that state by
comment marker or label metadata. A later `synchronize` event should re-review
only when the head SHA changes.

## Auto Fix Workflow

Triggers:

```yaml
on:
  pull_request_review:
    types: [submitted]
  pull_request_review_comment:
    types: [created]
  issue_comment:
    types: [created]
```

Decision logic:

```text
skip if PR lacks agent:codex
skip if PR lacks agent:auto-fix
skip if label agent:manual-hold exists
skip if actor is not trusted
skip if comment/review came from Codex automation itself
run if review state is CHANGES_REQUESTED
run if top-level PR comment starts with "@codex fix"
run if review comment starts with "@codex fix"
stop after 3 attempts
```

Trusted actors are repository owners, members, and collaborators with write
access. For public forks, never expose write-token secrets to untrusted code.

The fix job should:

1. Fetch PR metadata.
2. Check out the PR head branch only after trust checks pass.
3. Collect review body plus inline review comments.
4. Run Codex with a narrow prompt: address only the review feedback.
5. Run the repo's required tests from `AGENTS.md`.
6. Commit and push to the same PR branch.
7. Post a summary comment with changes, tests, and unresolved items.

The job must not change unrelated files, alter production artifacts, or merge
the PR. Human approval remains required.

## State Machine

```text
non-Codex PR opened
  -> codex-auto-review posts review
  -> human author fixes manually

Codex PR opened
  -> skipped by auto-review
  -> human review requests changes
  -> codex-auto-fix pushes fix attempt 1
  -> CI/review repeats
  -> stop at approval, manual-hold, merge, close, or attempt 3
```

## Safety Rules

Use `pull_request` for read-only review jobs. Use `pull_request_target` only for
metadata operations such as labeling or commenting, and do not check out or run
the untrusted PR head in the workspace when secrets are available.

Use concurrency to prevent overlapping agents on the same PR:

```yaml
concurrency:
  group: codex-pr-${{ github.event.pull_request.number || github.event.issue.number }}
  cancel-in-progress: false
```

The automation must fail closed when it cannot determine ownership, trust, or
attempt count. Adding `agent:manual-hold` must immediately stop future automated
review and fix runs.

## Rollout Plan

1. Add labels to every active RenQuant repo.
2. Implement `scripts/codex_pr_create.sh` in `RenQuant`.
3. Pilot `codex-auto-review.yml` on one low-risk repo with comment-only output.
4. Add `codex-auto-fix.yml` for Codex-labeled PRs only.
5. Promote workflows to reusable cross-repo workflows after two clean cycles.
6. Add CI checks that reject Codex PRs missing `Agent-Origin: Codex`.

## References

- GitHub Actions events: `pull_request`, `pull_request_review`,
  `pull_request_review_comment`, and `issue_comment`:
  https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- GitHub PR creation with CLI and API:
  https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
- OpenAI Codex overview:
  https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
- OpenAI Codex CLI overview:
  https://help.openai.com/en/articles/11096431-openai-codex-cli-getting-started
