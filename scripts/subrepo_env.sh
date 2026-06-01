#!/usr/bin/env bash
# Shared shell helpers for resolving RenQuant subrepo source roots.
#
# Default root is the sibling GitHub directory. Production can override with
# RENQUANT_SUBREPO_ROOT or source .subrepo_assembly/current.env so scheduled
# wrappers import lock-pinned clones instead of developer worktrees.

renquant_load_subrepo_env() {
    local repo_dir="${1:?repo_dir required}"
    local env_path="${RENQUANT_SUBREPO_ENV:-$repo_dir/.subrepo_assembly/current.env}"
    if [ -f "$env_path" ]; then
        # shellcheck disable=SC1090
        source "$env_path"
    fi
}

renquant_subrepo_root() {
    local repo_dir="${1:?repo_dir required}"
    local default_root="${2:-}"
    if [ -z "$default_root" ]; then
        default_root="$(cd "$repo_dir/.." && pwd)"
    fi
    if [ -n "${RENQUANT_SUBREPO_ROOT:-}" ]; then
        printf '%s\n' "$RENQUANT_SUBREPO_ROOT"
    elif [ -n "${RENQUANT_ASSEMBLY_DIR:-}" ] && [ -d "$RENQUANT_ASSEMBLY_DIR/repos" ]; then
        printf '%s\n' "$RENQUANT_ASSEMBLY_DIR/repos"
    else
        printf '%s\n' "$default_root"
    fi
}

renquant_subrepo_src() {
    local root="${1:?root required}"
    local repo="${2:?repo required}"
    printf '%s/%s/src\n' "$root" "$repo"
}

renquant_subrepo_pythonpath() {
    local root="${1:?root required}"
    shift
    local out=""
    local repo
    for repo in "$@"; do
        if [ -n "$out" ]; then
            out="$out:"
        fi
        out="$out$(renquant_subrepo_src "$root" "$repo")"
    done
    printf '%s\n' "$out"
}
