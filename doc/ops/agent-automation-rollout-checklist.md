# Agent automation · P1 fan-out checklist

Operator checklist for rolling [`doc/ops/agent-automation.md`](agent-automation.md) §5 P1 across the 11 public protected renquant repos (umbrella excluded — it already houses the templates).

## Pre-flight gates

Before running `scripts/fan_out_agent_automation.sh --execute`, ALL of these MUST be true. Skip none.

### Code gates

- [ ] RenQuant PR #19 (CLAUDE.md §3.7 pointer) merged on `main`
- [ ] RenQuant PR #20 (umbrella infra — reusable templates) merged on `main`
- [ ] The pilot wrapper PR in `renquant-model` (#20 there) merged on `main`
- [ ] The pilot has fired ON ≥ 1 real PR and completed without API-shape errors. Specifically the umbrella PR #20 `TODO(P0 before pilot)` items are resolved:
  - [ ] `anthropics/claude-code-action@v1` accepts `prompt` + `claude_args` + `--model` (or the templates are adjusted to the actual input shape)
  - [ ] `openai/codex-action@v1` accepts the configured input shape
  - [ ] Output names (`result` vs `final-message` etc.) match what the templates read

### Secrets gates (per target repo)

For each of the 8 P1 target repos, configure under repo Settings → Secrets and variables → Actions, OR (preferred) configure as an organization secret accessible to the renquant repo set:

- [ ] `ANTHROPIC_API_KEY` from console.anthropic.com
- [ ] `OPENAI_API_KEY` from platform.openai.com
- [ ] `AGENT_GIT_PUSH_TOKEN` — PAT or GitHub App token with `contents:write` + `pull-requests:write` scoped to that repo. Recommend a dedicated bot identity (e.g. `claude-code-bot` / `codex-bot`) rather than a personal PAT so commits + comments are attributable to the agent.

Verify per-repo with:

```bash
for r in renquant-artifacts renquant-backtesting renquant-base-data \
         renquant-common renquant-execution renquant-orchestrator \
         renquant-pipeline renquant-strategy-104; do
  echo "[$r]"
  gh secret list --repo "hallovorld/$r" | grep -E 'ANTHROPIC_API_KEY|OPENAI_API_KEY|AGENT_GIT_PUSH_TOKEN' || \
    echo "  ❌ missing secrets"
done
```

### Cost-gate spot-check

The wrappers each ship the mandatory cost gates from §6:
- `paths-ignore: ['doc/**']` (skip project docs only — NOT `.github/**`
  or `**/*.md`; those are control planes per the §3.1 update)
- `concurrency: cancel-in-progress: true` on the review workflow
- `if: github.event.pull_request.changed_files < 100` on each review job

Re-verify the wrappers in `renquant-model/.github/workflows/` still have all three before running the fan-out (a future change might drop one):

```bash
for f in agent-review.yml; do
  for gate in "paths-ignore:" "cancel-in-progress: true" "changed_files < 100"; do
    grep -q "$gate" "/Users/renhao/git/github/renquant-model/.github/workflows/$f" \
      || echo "::error::$f missing gate $gate"
  done
done
```

## Execute

```bash
cd /Users/renhao/git/github/RenQuant

# Dry-run first to see what would happen (no PRs, no clones):
./scripts/fan_out_agent_automation.sh

# Real run — clones each target into /tmp/fanout-<hash>/<repo>/
# (NEVER touches your working clones at ~/git/github/<repo>),
# opens 8 PRs:
./scripts/fan_out_agent_automation.sh --execute

# Or one at a time during verification:
./scripts/fan_out_agent_automation.sh --execute --only renquant-common

# Cleanup after: temp clones live at the workdir printed at the
# top of the run output. To remove:
rm -rf /tmp/fanout-agent-automation-XXXXX
```

**Non-destructive guarantee** (per PR #21 codex review): the script
clones each target repo into a fresh temp directory and operates
there. Your existing working clones at `~/git/github/<repo>` are
never touched. In-flight branches in those clones survive the
fan-out unchanged.

## Post-fan-out verification

For each opened PR:

- [ ] CI workflow runs (whatever existing CI the repo has). Agent workflows DON'T fire on the wrapper-adding PR itself (they fire on the NEXT PR after merge).
- [ ] Merge the wrapper PR.
- [ ] Open a small no-op test PR (e.g. a typo fix). Verify:
  - [ ] `agent-attribution-check` workflow fires (and skips, because the test PR has no `agent:*` label)
  - [ ] `agent-review` workflow fires AND posts a Claude review + Codex review (allowed to be empty / "looks good" — we're testing dispatch, not findings)
- [ ] If both reviews fire successfully, declare the repo migrated. If not, fix forward in the umbrella templates and re-run.

## Rollback

Per-repo rollback: close the PR opened by the fan-out script. Workflows haven't fired yet (they only fire post-merge).

Post-merge rollback: open a follow-up PR removing `.github/workflows/agent-*.yml` files. No cleanup needed elsewhere — the wrappers don't add state outside `.github/`.

## Cost guardrails post-fan-out

Set an org-level Anthropic API spend alert + OpenAI API spend alert at, say, $200/month (the doc's §6 envelope ceiling). If either crosses, kill the workflows in the offending repo:

```bash
# Disable the noisier workflow in a specific repo
gh workflow disable agent-review.yml --repo hallovorld/<repo>
```

Re-enable once the cost source is identified (usually a sweep of large docs-only PRs that bypassed `paths-ignore` somehow).
