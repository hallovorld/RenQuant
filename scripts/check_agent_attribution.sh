#!/usr/bin/env bash
# Layer B of G1 attribution enforcement (doc/ops/agent-automation.md §4.1).
#
# Invoked from a per-agent CLI hook (Claude Code `PostToolUse` on `Bash`,
# Codex equivalent). Reads the bash command the agent just ran from the
# CLI's environment and warns/blocks on `gh pr create` or `git commit`
# without the canonical Claude/Codex attribution.
#
# Usage (called by the agent CLI, not directly):
#
#     scripts/check_agent_attribution.sh claude
#     scripts/check_agent_attribution.sh codex
#
# Exits:
#   0  — command was fine OR not one we gate on
#   2  — command was `gh pr create` / `git commit` without canonical
#        attribution; stderr explains how to fix (agent re-reads stderr
#        as feedback and self-corrects)
set -euo pipefail

AGENT="${1:-}"
if [ -z "$AGENT" ]; then
  echo "usage: $0 {claude|codex}" >&2
  exit 1
fi

# The CLI hook puts the executed bash command in $CLAUDE_TOOL_INPUT
# (Claude) or $CODEX_TOOL_INPUT (Codex equivalent — confirm name when
# wiring the Codex hook). Read whichever is set; bail quietly if neither.
CMD="${CLAUDE_TOOL_INPUT:-${CODEX_TOOL_INPUT:-}}"
if [ -z "$CMD" ]; then
  exit 0  # not invoked from a hook context we recognize
fi

# Per-agent expected strings (must match doc/ops/agent-automation.md §2.2 / §2.4).
# AGENT_TITLE is the capitalized form. NOT using ${AGENT^} — that's a Bash 4+
# feature, and macOS ships Bash 3.2 by default. This script runs as a Claude
# Code PostToolUse hook on operator macOS as well as Linux CI, so the
# implementation must be Bash 3.2-compatible.
case "$AGENT" in
  claude)
    AGENT_TITLE="Claude"
    EXPECTED_FOOTER="🤖 Generated with"
    EXPECTED_TRAILER="Co-Authored-By: Claude"
    EXPECTED_ORIGIN="Agent-Origin: Claude"
    ;;
  codex)
    AGENT_TITLE="Codex"
    EXPECTED_FOOTER="🤖 Generated with"
    EXPECTED_TRAILER="Co-Authored-By: Codex"
    EXPECTED_ORIGIN="Agent-Origin: Codex"
    ;;
  *)
    echo "unknown agent: $AGENT" >&2
    exit 1
    ;;
esac

# Check 1: `gh pr create` (or `gh pr edit --body ...`) — must include
# canonical body footer + Agent-Origin line.
if echo "$CMD" | grep -qE 'gh pr (create|edit)'; then
  # The body content might be on the same line (-b "..."), in a heredoc,
  # or in a --body-file. Match against the command string for the inline
  # cases; for --body-file we can't introspect without reading the file
  # (out of scope for a hook). Only check inline forms.
  if ! echo "$CMD" | grep -q "$EXPECTED_FOOTER"; then
    cat >&2 <<MSG
warning: \`gh pr create/edit\` invocation appears to lack the canonical
${AGENT} attribution footer (\`${EXPECTED_FOOTER} ...\`). Required by
RenQuant doc/ops/agent-automation.md §2.2. Add to the PR body before
opening, or the server-side attribution-check workflow will block merge.
MSG
    if ! echo "$CMD" | grep -q "$EXPECTED_ORIGIN"; then
      cat >&2 <<MSG
warning: also missing \`${EXPECTED_ORIGIN}\` machine-readable origin
line (§2.2). Body should end with both:

  ${EXPECTED_ORIGIN}
  Agent-Policy: auto-fix-on-review
  ${EXPECTED_FOOTER} [${AGENT_TITLE} Code](...)
MSG
    fi
    # Warning only — don't block, the agent should self-correct on
    # reading this stderr.
    exit 0
  fi
fi

# Check 2: `git commit` — must include `Co-Authored-By: <Agent> <email>`
# trailer. Same caveat about --file vs inline.
if echo "$CMD" | grep -qE 'git commit'; then
  # Skip amends (would re-trigger on the same commit unnecessarily).
  if echo "$CMD" | grep -q -- '--amend'; then
    exit 0
  fi
  if ! echo "$CMD" | grep -q "$EXPECTED_TRAILER"; then
    cat >&2 <<MSG
warning: \`git commit\` invocation appears to lack the canonical
${AGENT} \`${EXPECTED_TRAILER}\` trailer. Required by RenQuant
doc/ops/agent-automation.md §2.4. Add to the commit message:

  ${EXPECTED_TRAILER} <noreply@$(case "$AGENT" in claude) echo anthropic.com;; codex) echo openai.com;; esac)>

Without this, server-side attribution-check workflow blocks merge.
MSG
    exit 0  # warning only
  fi
fi

exit 0
