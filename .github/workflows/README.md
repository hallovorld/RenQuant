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
    paths-ignore:    # mandatory cost gate per §3.1
      - 'doc/**'
    # NB: do NOT add '.github/**' OR '**/*.md' — both are control
    # planes. `.github/**` houses workflows + CODEOWNERS;
    # `.github/agent-*.md` is the automation's own prompt template.
    # `doc/**` is project docs only and safe to skip for cost.
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
| `AGENT_GIT_PUSH_TOKEN` | Fix template (push commits) | Optional but recommended PAT or GitHub App token with `contents:write` + `pull-requests:write` scoped to this repo; if absent, same-repo fix pushes fall back to `github.token` |

Set via repo Settings → Secrets and variables → Actions, OR as an
org-wide secret accessible to all renquant repos. The reusable templates
intentionally declare these secrets as optional so repositories without
agent credentials skip agent jobs cleanly instead of failing workflow
startup before any job is created.

## Per-repo customization points

Each subrepo ships TWO prompt files for the umbrella templates to read:

| File | Used by | Default | Notes |
|---|---|---|---|
| `.github/agent-review-prompt.md` | G2 review template | umbrella's [agent-review-prompt.md](../agent-review-prompt.md) | sets review framing — CLAUDE.md §7 canon |
| `.github/agent-fix-prompt.md` | G3 fix template | umbrella's [agent-fix-prompt.md](../agent-fix-prompt.md) | sets fix-mode framing — minimal change, run tests, commit |

Subrepos can override either with repo-specific framing (backtesting
data-flow gotchas, model calibrator invariants, etc.) — the templates
read whichever file is at the documented path. Or pass a different
path via the `prompt_path` input.

## Action API contract (verified)

Verified 2026-06-01 against live `action.yml` for each action:

- **`anthropics/claude-code-action@v1`** uses underscored input names:
  - `prompt` (inline content — NO `prompt_file` variant)
  - `anthropic_api_key`
  - `claude_args` (CLI passthrough — includes `--model`, `--allowed-tools`, `--max-turns`)

- **`openai/codex-action@v1`** uses dashed input names:
  - `prompt` (inline) OR `prompt-file` (path) — either supported
  - `openai-api-key`
  - `model`
  - `codex-args`

Earlier drafts used the wrong convention on both (underscored
`openai_api_key` + `prompt_file` for both actions). GitHub Actions
silently ignores unknown inputs — would have run without the API
key or prompt and silently failed.

The G1 attribution-check template has NO LLM dispatch — pure bash +
git + gh.
