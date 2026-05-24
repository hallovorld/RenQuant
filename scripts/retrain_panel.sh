#!/usr/bin/env bash
# retrain_panel.sh — Compatibility wrapper for the old Sunday retrain agent.
#
# The active 104 promote trust boundary is now weekly_wf_promote.sh:
# it retrains alpha158+fund into staging, runs the strict WF/sanity gates,
# then swaps production only on pass. The old sunday_panel_sweep/train_104
# path uses the legacy 22-feature builder and is intentionally refused by
# train_104.py for the current 172-feature alpha158_fund production artifact.
#
# This wrapper remains so the existing launchd plist does not emit a stale
# ERROR every Sunday. If weekly_wf_promote already ran today, this is a no-op.
# If it did not run, delegate to weekly_wf_promote without adding a second
# wrapper ntfy; weekly_wf_promote owns the operator alert.
#
# Usage:
#   bash scripts/retrain_panel.sh
#   bash scripts/retrain_panel.sh --strategy renquant_104
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/retrain_panel"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

exec >> "$LOG" 2>&1
echo "=== retrain_panel started at $(date) ==="

# ── Lock file — prevent concurrent invocations ────────────────────────────────
LOCK_FILE="/tmp/renquant_104_retrain_panel.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    echo "Another retrain_panel run is active (PID=$EXISTING_PID) — skipping."
    exit 0
fi
trap "rm -f '$LOCK_FILE'" EXIT

cd "$REPO_DIR"

STRATEGY="renquant_104"
for arg in "$@"; do
    case "$arg" in
        --strategy)    shift; STRATEGY="$1"; shift ;;
        --strategy=*)  STRATEGY="${arg#--strategy=}"; shift ;;
    esac
done

WEEKLY_LOG="$REPO_DIR/logs/weekly_wf_promote/$DATE.log"
if [ -f "$WEEKLY_LOG" ]; then
    echo "weekly_wf_promote already ran today ($WEEKLY_LOG)."
    echo "retrain_panel104 is a compatibility no-op; no ntfy emitted."
    echo "=== retrain_panel finished as no-op at $(date) ==="
    exit 0
fi

echo "weekly_wf_promote has not run today; delegating to the strict trust boundary."
echo "No retrain_panel wrapper ntfy will be emitted; weekly_wf_promote owns alerts."
if bash scripts/weekly_wf_promote.sh; then
    echo "=== retrain_panel delegated weekly_wf_promote PASS at $(date) ==="
    exit 0
else
    echo "=== retrain_panel delegated weekly_wf_promote FAIL at $(date) ==="
    echo "Production preserved by weekly_wf_promote; check logs/weekly_wf_promote/$DATE.log."
    exit 1
fi
