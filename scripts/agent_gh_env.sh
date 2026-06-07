#!/usr/bin/env bash
# Agent GitHub-token loader (SOP: doc/ops/agent-token-storage.md).
#
# Reads per-agent fine-grained PATs from the macOS login Keychain into the env
# vars the orchestrator + gh expect. The token is NEVER printed and NEVER
# written to disk — it lives only in this shell's environment.
#
# Usage — SOURCE this, do not execute:
#   source scripts/agent_gh_env.sh claude          # single agent: GH_TOKEN + RENQUANT_CLAUDE_GH_TOKEN
#   source scripts/agent_gh_env.sh codex
#   source scripts/agent_gh_env.sh --orchestrator  # both: RENQUANT_{CLAUDE,CODEX}_GH_TOKEN (no GH_TOKEN)
#
# Store the tokens first (interactive, hidden — run these YOURSELF in a terminal,
# never via an agent; the -w with no value prompts so the token stays out of
# argv / shell history / any transcript):
#   security add-generic-password -U -a "$USER" -s renquant-gh-claude -w
#   security add-generic-password -U -a "$USER" -s renquant-gh-codex  -w
#
# Rotation: revoke the PAT on GitHub, mint a new one, re-run the same
# add-generic-password -U command (it overwrites). No code/file changes.

_agt_die() { printf 'agent_gh_env: %s\n' "$1" >&2; return 1 2>/dev/null || exit 1; }

_agt_keychain() {  # $1=service -> token on stdout (empty if absent)
  security find-generic-password -s "$1" -w 2>/dev/null
}

_agt_missing_hint() {  # $1=service
  printf 'agent_gh_env: no Keychain token for %s. Add it (hidden prompt):\n  security add-generic-password -U -a "$USER" -s %s -w\n' "$1" "$1" >&2
}

case "${1:-}" in
  claude|codex)
    _agt_svc="renquant-gh-$1"
    _agt_tok="$(_agt_keychain "$_agt_svc")"
    if [ -z "$_agt_tok" ]; then _agt_missing_hint "$_agt_svc"; unset _agt_svc _agt_tok; _agt_die "missing token"; fi
    case "$1" in
      claude) export RENQUANT_CLAUDE_GH_TOKEN="$_agt_tok" ;;
      codex)  export RENQUANT_CODEX_GH_TOKEN="$_agt_tok" ;;
    esac
    export GH_TOKEN="$_agt_tok" GITHUB_TOKEN="$_agt_tok"
    unset _agt_tok _agt_svc
    printf 'agent_gh_env: loaded %s token (GH_TOKEN + RENQUANT_*_GH_TOKEN, hidden)\n' "$1" >&2
    ;;
  --orchestrator|all)
    _agt_c="$(_agt_keychain renquant-gh-claude)"
    _agt_x="$(_agt_keychain renquant-gh-codex)"
    _agt_n=0
    if [ -n "$_agt_c" ]; then export RENQUANT_CLAUDE_GH_TOKEN="$_agt_c"; _agt_n=$((_agt_n+1)); else _agt_missing_hint renquant-gh-claude; fi
    if [ -n "$_agt_x" ]; then export RENQUANT_CODEX_GH_TOKEN="$_agt_x";  _agt_n=$((_agt_n+1)); else _agt_missing_hint renquant-gh-codex; fi
    unset _agt_c _agt_x
    printf 'agent_gh_env: orchestrator mode — loaded %d/2 agent tokens (RENQUANT_*_GH_TOKEN, hidden)\n' "$_agt_n" >&2
    if [ "$_agt_n" -ne 2 ]; then unset _agt_n; _agt_die "not all agent tokens present"; fi
    unset _agt_n
    ;;
  *)
    printf 'usage: source scripts/agent_gh_env.sh <claude|codex|--orchestrator>\n' >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
