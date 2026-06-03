#!/usr/bin/env bash
# P1 fan-out of doc/ops/agent-automation.md to the public protected
# renquant repos. Opens one PR per repo adding the three ~25-line
# wrapper workflows + the generic umbrella review/fix prompts.
#
# DESIGN PRINCIPLES (per PR #21 codex review):
#
#   1. NON-DESTRUCTIVE — never `git reset --hard` operator's working
#      clones. Each target repo is cloned fresh into /tmp under a
#      dedicated workdir for the run.
#
#   2. PINNED INPUTS — wrapper YAML is embedded INLINE here (heredocs),
#      not copied from any sibling worktree. The script self-contains
#      its source-of-truth. Prompts are fetched from a pinned URL on
#      the umbrella main branch.
#
#   3. GENERIC PROMPTS — use the umbrella default agent-review-prompt.md
#      / agent-fix-prompt.md, not any repo-specific customization. Per-
#      repo customization is a follow-up PR each maintainer opens.
#
# DO NOT RUN until:
#   1. RenQuant umbrella infra PR is merged on `main` (reusable
#      templates exist at hallovorld/RenQuant/.github/workflows/_*@main)
#   2. The renquant-model pilot has demonstrated the LLM dispatch
#      works under real secrets
#   3. Each target repo has been configured with the three secrets
#      (use `gh secret set` separately — this script does NOT manage them)
#
# Usage:
#
#     # Dry-run (default): prints what would be done
#     ./scripts/fan_out_agent_automation.sh
#
#     # Real run: clones each target into /tmp, opens PR
#     ./scripts/fan_out_agent_automation.sh --execute
#
#     # Single repo (debugging):
#     ./scripts/fan_out_agent_automation.sh --execute --only renquant-common
#
#     # Custom workdir for temp clones (default: $(mktemp -d))
#     ./scripts/fan_out_agent_automation.sh --execute --workdir /path/to/temp

set -euo pipefail

EXECUTE=0
ONLY_REPO=""
WORKDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --only) ONLY_REPO="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Target repos — 11 public protected per CLAUDE.md §3.1 MINUS:
#   * renquant-model (pilot — separate PR)
#   * renquant-model-gbdt, renquant-model-patchtst (P3-merged into
#     renquant-model; not separately maintained)
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

# Set up temp workdir for the clones. NEVER touch the operator's
# working clones at ~/git/github/<repo>.
if [ -z "$WORKDIR" ]; then
  WORKDIR=$(mktemp -d -t fanout-agent-automation-XXXXXX)
fi
mkdir -p "$WORKDIR"
echo "Temp workdir: $WORKDIR"

# ── Inline wrapper YAML (embedded — no external file deps) ────────

readonly WRAPPER_REVIEW='name: agent-review
# Auto-review every non-self PR via Claude AND Codex.
# Wraps RenQuant umbrella'\''s reusable G2 template.

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
    # MANDATORY cost gate: skip project docs only.
    # NB: do NOT add .github/** or **/*.md — both are control planes.
    paths-ignore:
      - '\''doc/**'\''

concurrency:
  group: agent-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  claude-review:
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with:
      agent: claude
      model: claude-sonnet-4-6
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}

  codex-review:
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with:
      agent: codex
      model: gpt-5-codex
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
'

readonly WRAPPER_AUTOFIX='name: agent-autofix
# Auto-fix on reviewer findings. Wraps RenQuant umbrella'\''s G3 template.

on:
  pull_request_review:
    types: [submitted]
  issue_comment:
    types: [created]

concurrency:
  group: agent-autofix-${{ github.event.pull_request.number || github.event.issue.number }}
  cancel-in-progress: false

jobs:
  claude-fix:
    uses: hallovorld/RenQuant/.github/workflows/_agent-fix-template.yml@main
    with:
      agent: claude
      model: claude-opus-4-7
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
      git_push_token: ${{ secrets.AGENT_GIT_PUSH_TOKEN }}

  codex-fix:
    uses: hallovorld/RenQuant/.github/workflows/_agent-fix-template.yml@main
    with:
      agent: codex
      model: gpt-5-codex
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
      git_push_token: ${{ secrets.AGENT_GIT_PUSH_TOKEN }}
'

readonly WRAPPER_ATTRIBUTION='name: agent-attribution-check
# Server-side enforcement of agent-automation §4.1 Layer C.
# Pure bash + git + gh — no LLM dispatch.

on:
  pull_request:
    types: [opened, edited, synchronize, labeled, unlabeled]

jobs:
  check:
    uses: hallovorld/RenQuant/.github/workflows/_agent-attribution-check-template.yml@main
'

# v2 Phase A wrappers (per doc/ops/agent-automation-v2-design.md
# §3.1 + §3.2). Close the v1 gap where reviewer agents post COMMENTED
# with HIGH/MED findings (G3 gate misses) and where agent:fix:<name>
# executor-permission labels aren'\''t auto-added on PR open.

readonly WRAPPER_DEFAULT_LABELS='name: agent-default-labels
# Auto-adds agent:fix:<name> labels when authorship label is present.
# Wraps RenQuant umbrella'\''s reusable default-labels template.

on:
  pull_request:
    types: [opened, labeled, synchronize, reopened]

concurrency:
  group: agent-default-labels-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  ensure-fix-labels:
    uses: hallovorld/RenQuant/.github/workflows/_agent-default-labels-template.yml@main
'

readonly WRAPPER_REVIEW_CLASSIFY='name: agent-review-classify
# Bridges COMMENTED-with-HIGH/MED-findings reviews into the G3
# auto-fix loop. Wraps RenQuant umbrella'\''s reusable
# review-classify template.

on:
  pull_request_review:
    types: [submitted]

concurrency:
  group: agent-review-classify-${{ github.event.pull_request.number }}-${{ github.event.review.id }}
  cancel-in-progress: false

jobs:
  classify:
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-classify-template.yml@main
'

# Generic review + fix prompts fetched from the umbrella's canon
# copy at runtime so this script doesn't fork its own version of them.
# Per-repo customization is a follow-up PR each maintainer opens.
readonly UMBRELLA_REVIEW_PROMPT_URL='https://raw.githubusercontent.com/hallovorld/RenQuant/main/.github/agent-review-prompt.md'
readonly UMBRELLA_FIX_PROMPT_URL='https://raw.githubusercontent.com/hallovorld/RenQuant/main/.github/agent-fix-prompt.md'

# ── Per-repo execution ─────────────────────────────────────────────

for repo in "${REPOS[@]}"; do
  if [ -n "$ONLY_REPO" ] && [ "$repo" != "$ONLY_REPO" ]; then
    continue
  fi

  REPO_DIR="${WORKDIR}/${repo}"
  BRANCH="feat/agent-automation-wire"

  echo ""
  echo "================================================"
  echo "  $repo  ($([ $EXECUTE -eq 0 ] && echo "dry-run" || echo "execute"))"
  echo "================================================"

  if [ "$EXECUTE" -eq 0 ]; then
    cat <<EOF
  [dry-run] would:
    gh repo clone hallovorld/$repo $REPO_DIR
    cd $REPO_DIR
    git checkout -B $BRANCH origin/<default>
    mkdir -p .github/workflows
    write inline:  .github/workflows/agent-review.yml
                   .github/workflows/agent-autofix.yml
                   .github/workflows/agent-attribution-check.yml
                   .github/workflows/agent-default-labels.yml         (v2 Phase A)
                   .github/workflows/agent-review-classify.yml        (v2 Phase A)
    fetch umbrella default prompts → .github/agent-{review,fix}-prompt.md
    git add .github/ && git commit
    git push -u origin $BRANCH
    gh pr create --base <default> --head $BRANCH ...
EOF
    continue
  fi

  # CLONE INTO TEMP — never touch operator's working tree at
  # ~/git/github/<repo>. Past draft used git reset --hard there;
  # PR #21 codex review caught this as a HIGH destructiveness bug.
  if [ -d "$REPO_DIR" ]; then
    echo "::error::$REPO_DIR already exists; pick a fresh --workdir or delete it" >&2
    exit 1
  fi
  gh repo clone "hallovorld/$repo" "$REPO_DIR" -- --quiet

  (
    cd "$REPO_DIR"

    # Default branch detection — repos use either main or master.
    DEFAULT_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD | sed 's@^origin/@@')
    git checkout -B "$BRANCH" "origin/${DEFAULT_BRANCH}"

    mkdir -p .github/workflows
    printf '%s' "$WRAPPER_REVIEW"          > .github/workflows/agent-review.yml
    printf '%s' "$WRAPPER_AUTOFIX"         > .github/workflows/agent-autofix.yml
    printf '%s' "$WRAPPER_ATTRIBUTION"     > .github/workflows/agent-attribution-check.yml
    # v2 Phase A:
    printf '%s' "$WRAPPER_DEFAULT_LABELS"  > .github/workflows/agent-default-labels.yml
    printf '%s' "$WRAPPER_REVIEW_CLASSIFY" > .github/workflows/agent-review-classify.yml

    # Fetch umbrella default prompts — generic across all renquant
    # repos. Per-repo customization (backtesting data-flow gotchas,
    # model calibrator invariants, etc.) is a SEPARATE follow-up PR
    # each repo's maintainer opens.
    curl -fsSL "$UMBRELLA_REVIEW_PROMPT_URL" > .github/agent-review-prompt.md
    curl -fsSL "$UMBRELLA_FIX_PROMPT_URL"    > .github/agent-fix-prompt.md

    git add .github/
    git -c user.name='claude-code-bot' \
        -c user.email='noreply@anthropic.com' \
        commit -m "feat(automation): wire agent automation (P1 fan-out)

Adds the three ~25-line wrapper workflows + umbrella-default review
and fix prompts. Wrappers invoke the umbrella's reusable templates
from hallovorld/RenQuant/.github/workflows/ — design canonicalized
in doc/ops/agent-automation.md.

Required secrets (set under repo Settings → Secrets and variables):
  ANTHROPIC_API_KEY       Claude review + fix
  OPENAI_API_KEY          Codex review + fix
  AGENT_GIT_PUSH_TOKEN    G3 fix workflow's commit/push

Until secrets are configured workflows fire but skip-or-error on
the LLM dispatch — safe, no production behavior depends on them yet.

Per-repo prompt customization (replacing the umbrella defaults with
repo-specific framing — e.g. backtesting data-flow gotchas, model
calibrator invariants) is a follow-up PR each maintainer opens.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
    git push -u origin "$BRANCH"
    gh pr create --base "$DEFAULT_BRANCH" --head "$BRANCH" \
      --title "feat(automation): wire agent automation (P1 fan-out)" \
      --body "P1 fan-out of [\`doc/ops/agent-automation.md\`](https://github.com/hallovorld/RenQuant/blob/main/doc/ops/agent-automation.md). Wraps the umbrella's reusable templates per the pattern proven in [renquant-model #20](https://github.com/hallovorld/renquant-model/pull/20). Generated by \`hallovorld/RenQuant/scripts/fan_out_agent_automation.sh\`.

**Required secrets** (configure separately, not in this PR):
- \`ANTHROPIC_API_KEY\`
- \`OPENAI_API_KEY\`
- \`AGENT_GIT_PUSH_TOKEN\`

Per-repo prompt customization is a follow-up — this PR ships only the umbrella-default review and fix prompts.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  )
done

echo ""
if [ "$EXECUTE" -eq 0 ]; then
  echo "Dry run complete. Re-run with --execute to actually open PRs."
else
  echo "Fan-out complete. PRs opened under hallovorld/<repo>. Temp clones live at $WORKDIR."
  echo "To clean up: rm -rf $WORKDIR"
fi
