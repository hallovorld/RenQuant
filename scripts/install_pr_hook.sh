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
# CLAUDE.md §3a — block direct push to main + scan pushed commits for leaked
# GitHub tokens (doc/ops/agent-token-storage.md).
# Override via: PR_HOOK_BYPASS=1 git push origin main  (intentional emergency only)

# Real-token shapes only: classic gh{p,o,u,s}_ + 36+ base62, or fine-grained
# github_pat_ + 50+. Bare prefix mentions in docs (e.g. the literal ghp underscore)
# do NOT match, so this SOP and the hook itself push cleanly.
SECRET_RE="(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})"
ZERO="0000000000000000000000000000000000000000"

while read local_ref local_sha remote_ref remote_sha; do
  if [[ "$remote_ref" == "refs/heads/main" ]]; then
    if [[ "${PR_HOOK_BYPASS:-0}" == "1" ]]; then
      echo "⚠️  PR_HOOK_BYPASS=1 — direct push to main allowed this once" >&2
    else
      echo "❌ Direct push to main is BLOCKED (CLAUDE.md §3a)" >&2
      echo "   Use: git checkout -b feat/your-change && gh pr create --base main" >&2
      echo "   Emergency override: PR_HOOK_BYPASS=1 git push origin main" >&2
      exit 1
    fi
  fi

  # Secret scan: diff of the commits actually being pushed.
  [[ "$local_sha" == "$ZERO" ]] && continue          # branch deletion
  if [[ "$remote_sha" == "$ZERO" ]]; then            # new branch: base on main
    base="$(git merge-base "$local_sha" refs/heads/main 2>/dev/null \
            || git hash-object -t tree /dev/null)"
    range="$base..$local_sha"
  else
    range="$remote_sha..$local_sha"
  fi
  if git diff "$range" 2>/dev/null | grep -nEq "$SECRET_RE"; then
    echo "❌ Push BLOCKED: a GitHub token pattern was found in the pushed diff." >&2
    echo "   Secrets must never be committed (doc/ops/agent-token-storage.md)." >&2
    echo "   Scrub it from history, then re-push. Override (NOT advised):" >&2
    echo "     PR_HOOK_BYPASS=1 git push ..." >&2
    [[ "${PR_HOOK_BYPASS:-0}" == "1" ]] || exit 1
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
