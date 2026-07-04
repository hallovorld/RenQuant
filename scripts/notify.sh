#!/bin/sh
# scripts/notify.sh — canonical shell ntfy sender for the RenQuant fleet
# (compliance campaign B6, audit XC-4). Python twin: renquant_common.notify.
#
# Replaces the 8 hand-rolled `source .env` + curl blocks in the orchestrator
# ops wrappers (and is the target for future umbrella-script re-points).
# Sourceable from bash AND zsh wrappers running under `set -u`.
#
# Usage:
#   . "$RQ_ROOT/scripts/notify.sh"
#   rq_notify "Title" "body" [priority] [tags]
#
# Contract (must match renquant_common.notify):
#   * Topic resolution: $NTFY_TOPIC env > NTFY_TOPIC= line in $RQ_ROOT/.env
#     (default umbrella root when RQ_ROOT unset) > fleet default "renquant".
#     The .env is PARSED, not sourced — no caller-env pollution.
#   * RENQUANT_NO_NOTIFY truthy (1/true/yes/on, case-insensitive) suppresses
#     the send, ALWAYS.
#   * curl --max-time 5 (standardized timeout).
#   * Never fails the caller: always returns 0, all curl errors swallowed.

rq_notify() {
    _rqn_title="${1:-RenQuant}"
    _rqn_body="${2:-}"
    _rqn_priority="${3:-}"
    _rqn_tags="${4:-}"

    case "$(printf '%s' "${RENQUANT_NO_NOTIFY:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on)
            echo "[ntfy suppressed] ${_rqn_title}" >&2
            return 0
            ;;
    esac

    _rqn_topic="${NTFY_TOPIC:-}"
    if [ -z "${_rqn_topic}" ]; then
        _rqn_env_file="${RQ_ROOT:-/Users/renhao/git/github/RenQuant}/.env"
        if [ -f "${_rqn_env_file}" ]; then
            _rqn_topic="$(sed -n 's/^NTFY_TOPIC=//p' "${_rqn_env_file}" | head -1 | tr -d '"' | tr -d "'")"
        fi
    fi
    _rqn_topic="${_rqn_topic:-renquant}"

    if [ -n "${_rqn_priority}" ] && [ -n "${_rqn_tags}" ]; then
        curl -s --max-time 5 -H "Title: ${_rqn_title}" -H "Priority: ${_rqn_priority}" \
            -H "Tags: ${_rqn_tags}" -d "${_rqn_body}" \
            "https://ntfy.sh/${_rqn_topic}" >/dev/null 2>&1 || true
    elif [ -n "${_rqn_priority}" ]; then
        curl -s --max-time 5 -H "Title: ${_rqn_title}" -H "Priority: ${_rqn_priority}" \
            -d "${_rqn_body}" "https://ntfy.sh/${_rqn_topic}" >/dev/null 2>&1 || true
    elif [ -n "${_rqn_tags}" ]; then
        curl -s --max-time 5 -H "Title: ${_rqn_title}" -H "Tags: ${_rqn_tags}" \
            -d "${_rqn_body}" "https://ntfy.sh/${_rqn_topic}" >/dev/null 2>&1 || true
    else
        curl -s --max-time 5 -H "Title: ${_rqn_title}" \
            -d "${_rqn_body}" "https://ntfy.sh/${_rqn_topic}" >/dev/null 2>&1 || true
    fi
    return 0
}
