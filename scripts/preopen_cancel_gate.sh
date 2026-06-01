#!/bin/bash
# Pre-open cancel gate runner — invoked by launchd ~15 min before NYSE open.
# The original umbrella implementation remains in scripts/preopen_cancel_gate.py.
# Default path uses renquant-execution when available; set
# RQ_PREOPEN_GATE_RUNNER=umbrella for immediate rollback.

set -eu
cd /Users/renhao/git/github/RenQuant
mkdir -p logs/preopen_gate

notify_fallback_once() {
    local stamp="logs/preopen_gate/.subrepo_fallback_alert_stamp"
    local cooldown="${RQ_PREOPEN_GATE_FALLBACK_ALERT_COOLDOWN_SEC:-86400}"
    local now last age topic body

    now="$(date +%s)"
    last=0
    if [ -f "$stamp" ]; then
        last="$(cat "$stamp" 2>/dev/null || echo 0)"
    fi
    case "$last" in
        ''|*[!0-9]*) last=0 ;;
    esac
    age=$((now - last))
    if [ "$age" -lt "$cooldown" ]; then
        return 0
    fi

    printf '%s\n' "$now" > "$stamp" 2>/dev/null || true
    topic="${RENQUANT_NTFY_TOPIC:-${NTFY_TOPIC:-renquant}}"
    body="PREOPEN_GATE_FALLBACK: renquant_execution.preopen_cancel_gate unavailable on $(hostname); using umbrella scripts/preopen_cancel_gate.py. Set RQ_PREOPEN_GATE_STRICT=1 to fail closed."
    curl -sS -H "Title: RenQuant PREOPEN-GATE FALLBACK" \
        -H "Priority: low" \
        -d "$body" \
        "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
}

# shellcheck disable=SC1091
source .venv/bin/activate
set -a
# shellcheck disable=SC1091
source .env
set +a

RUNNER="${RQ_PREOPEN_GATE_RUNNER:-multirepo}"
if [ "$RUNNER" = "umbrella" ]; then
    exec python scripts/preopen_cancel_gate.py "$@"
fi

export RENQUANT_REPO_ROOT="$PWD"
GITHUB_DIR="$(cd "$PWD/.." && pwd)"
# shellcheck disable=SC1091
source "$PWD/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$PWD"
SUBREPO_ROOT="$(renquant_subrepo_root "$PWD" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
SUBREPO_SRC="$(renquant_subrepo_src "$SUBREPO_ROOT" renquant-execution)"
COMMON_SRC="$(renquant_subrepo_src "$SUBREPO_ROOT" renquant-common)"
export PYTHONPATH="$SUBREPO_SRC:$COMMON_SRC:${PYTHONPATH:-}"

if python - <<'PY' >/dev/null 2>&1
import renquant_execution.preopen_cancel_gate  # noqa: F401
PY
then
    python - <<'PY' >&2
import renquant_execution.preopen_cancel_gate as m
print(f"renquant_execution.preopen_cancel_gate={m.__file__}")
PY
    exec python -m renquant_execution.preopen_cancel_gate "$@"
fi

if [ "${RQ_PREOPEN_GATE_STRICT:-0}" = "1" ]; then
    echo "ERROR: renquant_execution.preopen_cancel_gate unavailable and RQ_PREOPEN_GATE_STRICT=1" >&2
    exit 1
fi

echo "WARN: renquant_execution.preopen_cancel_gate unavailable; falling back to umbrella script." >&2
notify_fallback_once
exec python scripts/preopen_cancel_gate.py "$@"
