# RenQuant umbrella · reusable workflows

Templates referenced by every renquant repo for agent automation. Design
canon: [`doc/ops/agent-automation.md`](../../doc/ops/agent-automation.md).

## Templates here

| File | What |
|---|---|
| `_agent-attribution-check-template.yml` | G1 (§4.1 Layer C) — verifies the canonical `agent:*` label, `Agent-Origin` body footer, and per-commit `Co-Authored-By` trailer match. Bash + `gh` + `git` only — no LLM. |
| `_agent-review-template.yml` | G2 (§4.2) — auto-reviews non-self PRs using the agent indicated by `inputs.agent` (claude or codex). |
| `_agent-fix-template.yml` | G3 (§4.3) — auto-fixes the agent's own PRs when the reviewer requests changes or invokes `@<agent> fix`. Per-executor attempt counter via labels. |

## How each subrepo wires these in

Each renquant repo adds ~25-line wrappers in its own `.github/workflows/`
that invoke the templates via `uses:`:

```yaml
# In every repo: .github/workflows/agent-review.yml
name: agent-review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
    paths-ignore:    # mandatory cost gates per §3.1
      - 'doc/**'
      - '**/*.md'
    # NB: do NOT add '.github/**' to paths-ignore — that's the
    # security control plane (workflows, CODEOWNERS).
concurrency:
  group: agent-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true
jobs:
  claude-review:
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: claude, model: claude-sonnet-4-6 }
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
  codex-review:
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: codex, model: gpt-5-codex }
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
```

```yaml
# .github/workflows/agent-autofix.yml
name: agent-autofix
on:
  pull_request_review:
    types: [submitted]
  issue_comment:
    types: [created]
jobs:
  claude-fix:
    uses: hallovorld/RenQuant/.github/workflows/_agent-fix-template.yml@main
    with: { agent: claude, model: claude-opus-4-7 }
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
      git_push_token: ${{ secrets.AGENT_GIT_PUSH_TOKEN }}
  codex-fix:
    uses: hallovorld/RenQuant/.github/workflows/_agent-fix-template.yml@main
    with: { agent: codex, model: gpt-5-codex }
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
      git_push_token: ${{ secrets.AGENT_GIT_PUSH_TOKEN }}
```

```yaml
# .github/workflows/agent-attribution-check.yml
name: agent-attribution-check
on:
  pull_request:
    types: [opened, edited, synchronize, labeled, unlabeled]
jobs:
  check:
    uses: hallovorld/RenQuant/.github/workflows/_agent-attribution-check-template.yml@main
```

## Repo secrets each subrepo needs

| Secret | Used by | Source |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude review + fix | console.anthropic.com → API Keys |
| `OPENAI_API_KEY` | Codex review + fix | platform.openai.com → API keys |
| `AGENT_GIT_PUSH_TOKEN` | Fix template (push commits) | PAT or GitHub App token with `contents:write` + `pull-requests:write` scoped to this repo |

Set via repo Settings → Secrets and variables → Actions, OR as an
org-wide secret accessible to all renquant repos.

## Per-repo customization point

`.github/agent-review-prompt.md` in each subrepo. The umbrella ships a
[default](../agent-review-prompt.md) that uses `CLAUDE.md §7` as the
review canon. Subrepos can override with repo-specific framing
(e.g. backtesting-specific data-flow gotchas, model-specific
calibrator invariants) — the templates read whichever file is at
`.github/agent-review-prompt.md` (or pass a different path via the
`prompt_path` input).

## Open verification needed before pilot ships

The G2 and G3 templates invoke `anthropics/claude-code-action@v1` and
`openai/codex-action@v1` with assumed input shapes. **Verify against
canonical action docs before merging into a repo that will actually
run them.** Specifically:

- Does `claude-code-action` accept `prompt` + `claude_args` + `--model`
  as documented? Or does it use `model` as a separate input?
- Does `codex-action` accept `prompt` (inline) or only `prompt_file`?
- Output names: `result` (Claude) vs `final-message` (Codex)?
- Does either action auto-post a PR comment when invoked under
  `pull-requests: write`, or does the wrapper need to handle that?

These are flagged inline as `TODO(P0 before pilot)` in the templates.
The G1 attribution-check template has no such uncertainty — it's pure
bash + git + gh.
