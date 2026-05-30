#!/usr/bin/env bash
# CLAUDE.md §3a — install pre-push hook into any local renquant repo to block
# direct push to main. Required on PRIVATE repos where GitHub free plan
# disallows server-side branch protection (renquant-model, renquant-state-backup).
#
# Usage:
#   bash scripts/install_pr_hook.sh /path/to/local/renquant-repo
#   # or for all known local renquant clones at once:
#   bash scripts/install_pr_hook.sh --all
#
# Idempotent: re-runs overwrite the hook with current version.

set -euo pipefail

HOOK_CONTENT='#!/usr/bin/env bash
# CLAUDE.md §3a — block direct push to main (CLAUDE.md §3a)
# Override via: PR_HOOK_BYPASS=1 git push origin main  (intentional emergency only)

while read local_ref local_sha remote_ref remote_sha; do
  if [[ "$remote_ref" == "refs/heads/main" ]]; then
    if [[ "${PR_HOOK_BYPASS:-0}" == "1" ]]; then
      echo "⚠️  PR_HOOK_BYPASS=1 — direct push to main allowed this once" >&2
      continue
    fi
    echo "❌ Direct push to main is BLOCKED (CLAUDE.md §3a)" >&2
    echo "   Use: git checkout -b feat/your-change && gh pr create --base main" >&2
    echo "   Emergency override: PR_HOOK_BYPASS=1 git push origin main" >&2
    exit 1
  fi
done
exit 0
'

install_hook() {
  local repo_dir="$1"
  if [[ ! -d "$repo_dir/.git" ]]; then
    echo "  skip $repo_dir (no .git)"
    return
  fi
  local hook="$repo_dir/.git/hooks/pre-push"
  printf '%s' "$HOOK_CONTENT" > "$hook"
  chmod +x "$hook"
  echo "  ✅ installed → $hook"
}

if [[ "${1:-}" == "--all" ]]; then
  echo "Installing pre-push hook into all local renquant clones under /Users/renhao/git/github/..."
  for d in /Users/renhao/git/github/RenQuant /Users/renhao/git/github/renquant-*; do
    install_hook "$d"
  done
else
  if [[ -z "${1:-}" ]]; then
    echo "Usage: $0 <repo-dir>  OR  $0 --all" >&2
    exit 2
  fi
  install_hook "$1"
fi
