#!/usr/bin/env bash
# Scheduled local agent PR loop.
# Fired by launchd every 5 minutes. Loads per-agent GitHub tokens from the
# Keychain, widens PATH so codex/claude CLIs are visible under launchd, then
# runs scripts/agent_pr_loop.py once. Fail-closed on missing tokens / CLIs.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$REPO_DIR/logs/agent_pr_loop"
LOCK_DIR="$LOG_DIR/.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

write_preflight_status() {
    local message="$1"
    local status_path="$LOG_DIR/status.json"
    python3 - "$status_path" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
message = sys.argv[2]
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps({
    "started_at": datetime.now(timezone.utc).isoformat(),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "ok": False,
    "error": message,
    "stage": "shell-preflight",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

keychain_token() {
    security find-generic-password -s "$1" -w 2>/dev/null || true
}

mkdir -p "$LOG_DIR"

# launchd PATH does not include the operator's Homebrew / node / local bins by
# default. Seed the common non-system roots explicitly, then layer user-local
# paths if present.
PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
for candidate in "$HOME/.local/bin" "$HOME/.nvm/versions/node"/*/bin; do
    if [ -d "$candidate" ]; then
        PATH="$candidate:$PATH"
    fi
done
export PATH

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" >"$LOCK_PID_FILE"
        return 0
    fi

    if [ -f "$LOCK_PID_FILE" ]; then
        local holder_pid
        holder_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
        if [ -n "$holder_pid" ] && kill -0 "$holder_pid" 2>/dev/null; then
            echo "agent_pr_loop: already running under pid $holder_pid; skipping overlapping launch"
            return 1
        fi
    fi

    rm -rf "$LOCK_DIR"
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" >"$LOCK_PID_FILE"
        echo "agent_pr_loop: cleared stale lock and resumed"
        return 0
    fi

    echo "agent_pr_loop: failed to acquire lock"
    return 1
}

release_lock() {
    rm -f "$LOCK_PID_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

if ! acquire_lock; then
    exit 0
fi
trap 'release_lock' EXIT

CLAUDE_TOKEN="$(keychain_token renquant-gh-claude)"
CODEX_TOKEN="$(keychain_token renquant-gh-codex)"
if [ -z "$CLAUDE_TOKEN" ] || [ -z "$CODEX_TOKEN" ]; then
    write_preflight_status "missing agent Keychain token(s): renquant-gh-claude and/or renquant-gh-codex"
    exit 44
fi
export RENQUANT_CLAUDE_GH_TOKEN="$CLAUDE_TOKEN"
export RENQUANT_CODEX_GH_TOKEN="$CODEX_TOKEN"
export RENQUANT_ORCHESTRATOR_ROOT="${RENQUANT_ORCHESTRATOR_ROOT:-$REPO_DIR/../renquant-orchestrator}"

exec "$PYTHON" "$REPO_DIR/scripts/agent_pr_loop.py"
