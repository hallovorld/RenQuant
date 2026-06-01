#!/usr/bin/env bash
# P1 fan-out of doc/ops/agent-automation.md to the public protected
# renquant repos. Opens one PR per repo adding the three ~25-line
# wrapper workflows + the per-repo review prompt.
#
# DO NOT RUN until:
#   1. RenQuant PR #20 (umbrella infra) is merged on `main`
#      (the `uses:` references resolve against `hallovorld/RenQuant@main`)
#   2. The renquant-model pilot has demonstrated the LLM dispatch
#      works under real secrets (confirms the TODO(P0 before pilot)
#      assumptions in the templates)
#   3. Each target repo has been configured with the three secrets
#      (ANTHROPIC_API_KEY / OPENAI_API_KEY / AGENT_GIT_PUSH_TOKEN);
#      this script does NOT manage secrets (use `gh secret set` separately).
#
# Usage:
#
#     # Dry-run (default): prints what would be done without opening PRs
#     ./scripts/fan_out_agent_automation.sh
#
#     # Real run: actually opens the PRs
#     ./scripts/fan_out_agent_automation.sh --execute
#
#     # Run against a single repo (debugging):
#     ./scripts/fan_out_agent_automation.sh --execute --only renquant-common

set -euo pipefail

EXECUTE=0
ONLY_REPO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --only) ONLY_REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# 11 public protected per CLAUDE.md §3.1, minus renquant-model (pilot).
# renquant-model-gbdt / renquant-model-patchtst are P3-deprecated
# (merged into renquant-model) — verify via `gh repo view` before
# including in the fan-out.
REPOS=(
  renquant-artifacts
  renquant-backtesting
  renquant-base-data
  renquant-common
  renquant-execution
  renquant-orchestrator
  renquant-pipeline
  renquant-strategy-104
)

# Where the umbrella's reusable templates live — the wrappers reference these.
TEMPLATE_REF="hallovorld/RenQuant/.github/workflows"
TEMPLATE_BRANCH="main"

# Source of the wrapper YAML — copied verbatim from the pilot
# (`renquant-model/.github/workflows/agent-*.yml`).
PILOT_DIR="/Users/renhao/git/github/renquant-model/.github/workflows"
PILOT_PROMPT="/Users/renhao/git/github/renquant-model/.github/agent-review-prompt.md"

for f in agent-review.yml agent-autofix.yml agent-attribution-check.yml; do
  if [ ! -f "${PILOT_DIR}/${f}" ]; then
    echo "::error::pilot wrapper missing at ${PILOT_DIR}/${f}; cannot fan out" >&2
    exit 1
  fi
done

for repo in "${REPOS[@]}"; do
  if [ -n "$ONLY_REPO" ] && [ "$repo" != "$ONLY_REPO" ]; then
    continue
  fi

  REPO_PATH="/Users/renhao/git/github/$repo"
  if [ ! -d "${REPO_PATH}/.git" ]; then
    echo "::warning::$repo not cloned locally at ${REPO_PATH}; skipping" >&2
    continue
  fi

  BRANCH="feat/agent-automation-wire"
  echo ""
  echo "================================================"
  echo "  $repo"
  echo "================================================"

  if [ "$EXECUTE" -eq 0 ]; then
    echo "  [dry-run] would:"
    echo "    cd $REPO_PATH"
    echo "    git fetch origin -q && git checkout main && git reset --hard origin/main"
    echo "    git checkout -b $BRANCH"
    echo "    mkdir -p .github/workflows"
    echo "    cp ${PILOT_DIR}/{agent-review.yml,agent-autofix.yml,agent-attribution-check.yml} .github/workflows/"
    echo "    cp ${PILOT_PROMPT} .github/agent-review-prompt.md"
    echo "    # customize .github/agent-review-prompt.md for repo-specific gotchas"
    echo "    git add .github/"
    echo "    git commit -m 'feat(automation): wire agent automation (P1 fan-out)'"
    echo "    git push -u origin $BRANCH"
    echo "    gh pr create --base main --head $BRANCH ..."
    continue
  fi

  (
    cd "$REPO_PATH"
    git fetch origin -q
    git checkout main 2>&1 | tail -1
    git reset --hard origin/main 2>&1 | tail -1
    # Use -B so re-running on a fresh branch resets state cleanly.
    git checkout -B "$BRANCH" 2>&1 | tail -1
    mkdir -p .github/workflows
    cp "${PILOT_DIR}/agent-review.yml"             .github/workflows/agent-review.yml
    cp "${PILOT_DIR}/agent-autofix.yml"            .github/workflows/agent-autofix.yml
    cp "${PILOT_DIR}/agent-attribution-check.yml"  .github/workflows/agent-attribution-check.yml
    # NOTE: the prompt is intentionally a starting point — repo
    # maintainers should customize it for repo-specific review gotchas
    # before merging. The umbrella default works as a baseline.
    cp "$PILOT_PROMPT" .github/agent-review-prompt.md
    # Strip renquant-model-specific lines from the copied prompt;
    # leave the umbrella-default framing intact.
    sed -i.bak '/PR #17 review/d; /PR #9 placebo/d; /single-seed flakiness/d; /detector_version threading/d' .github/agent-review-prompt.md
    rm -f .github/agent-review-prompt.md.bak
    git add .github/
    git status --short
    git commit -m "feat(automation): wire agent automation (P1 fan-out)

Adds the three ~25-line wrapper workflows + repo-specific review
prompt. Wrappers invoke the umbrella's reusable templates from
hallovorld/RenQuant/.github/workflows/ — design canonicalized in
doc/ops/agent-automation.md.

Required secrets (set under repo Settings → Secrets and variables):
  ANTHROPIC_API_KEY       Claude review + fix
  OPENAI_API_KEY          Codex review + fix
  AGENT_GIT_PUSH_TOKEN    G3 fix workflow's commit/push

Until secrets are configured workflows fire but skip-or-error on
the LLM dispatch — safe, no production behavior depends on them yet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
    git push -u origin "$BRANCH"
    gh pr create --base main --head "$BRANCH" \
      --title "feat(automation): wire agent automation (P1 fan-out)" \
      --body "P1 fan-out of [\`doc/ops/agent-automation.md\`](https://github.com/hallovorld/RenQuant/blob/main/doc/ops/agent-automation.md). Wraps the umbrella's reusable templates per the pattern proven in [renquant-model #20](https://github.com/hallovorld/renquant-model/pull/20). See script: \`hallovorld/RenQuant/scripts/fan_out_agent_automation.sh\`.

**Required secrets** (manual, NOT in this PR):
- \`ANTHROPIC_API_KEY\`
- \`OPENAI_API_KEY\`
- \`AGENT_GIT_PUSH_TOKEN\`

Until configured, workflows fire but skip on LLM dispatch — safe.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  )
done

echo ""
if [ "$EXECUTE" -eq 0 ]; then
  echo "Dry run complete. Re-run with --execute to actually open the PRs."
else
  echo "Fan-out complete. Check \`gh pr list --search 'org:hallovorld is:open in:title agent-automation'\` for the opened PRs."
fi
