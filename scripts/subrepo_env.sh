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
    local root
    local assembly_dir
    if [ -z "$default_root" ]; then
        default_root="$(cd "$repo_dir/.." && pwd)"
    fi
    if [ -n "${RENQUANT_SUBREPO_ROOT:-}" ]; then
        root="$RENQUANT_SUBREPO_ROOT"
        case "$root" in
            /*) ;;
            *) root="$repo_dir/$root" ;;
        esac
        printf '%s\n' "$root"
    elif [ -n "${RENQUANT_ASSEMBLY_DIR:-}" ]; then
        assembly_dir="$RENQUANT_ASSEMBLY_DIR"
        case "$assembly_dir" in
            /*) ;;
            *) assembly_dir="$repo_dir/$assembly_dir" ;;
        esac
        if [ -d "$assembly_dir/repos" ]; then
            printf '%s\n' "$assembly_dir/repos"
        else
            printf '%s\n' "$default_root"
        fi
    else
        printf '%s\n' "$default_root"
    fi
}

renquant_subrepo_src() {
    local root="${1:?root required}"
    local repo="${2:?repo required}"
    printf '%s/%s/src\n' "$root" "$repo"
}

renquant_strategy_config() {
    local root="${1:?root required}"
    local config_name="${2:-strategy_config.json}"
    local path="$root/renquant-strategy-104/configs/$config_name"
    if [ -f "$path" ]; then
        printf '%s\n' "$path"
        return 0
    fi
    return 1
}

renquant_strict_enabled() {
    local strict_env="${1:-}"
    local strict_value="0"
    if [ "${RENQUANT_OPS_FAIL_CLOSED:-0}" = "1" ]; then
        return 0
    fi
    if [ "${RENQUANT_STRICT_SUBREPO_PATHS:-0}" = "1" ]; then
        return 0
    fi
    if [ -n "$strict_env" ]; then
        strict_value="${!strict_env:-0}"
        if [ "$strict_value" = "1" ]; then
            return 0
        fi
    fi
    return 1
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
