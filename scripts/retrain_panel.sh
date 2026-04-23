#!/usr/bin/env bash
# retrain_panel.sh — Force a full renquant_104 retrain (tournament + panel-LTR
# + recalibrate), ignoring the training.cadence gate in strategy_config.json.
#
# Intended for explicit scheduling (e.g. a Sunday-only launchd agent) when
# you want to decouple "full retrain" from the daily trading script:
#   - daily_104.sh runs every trading day and is gated by training.cadence
#   - retrain_panel.sh always forces a retrain
#
# Usage:
#   bash scripts/retrain_panel.sh
#   bash scripts/retrain_panel.sh --strategy renquant_104
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
CONDA_PREFIX="/Users/renhao/miniconda3/envs/renquant"
PYTHON="$CONDA_PREFIX/bin/python"
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

echo "--- Forcing full retrain (strategy=$STRATEGY) ---"
if "$PYTHON" scripts/train_104.py --strategy "$STRATEGY" --force; then
    echo "=== retrain_panel finished at $(date) ==="
    notify "RenQuant 104 panel" "Sunday retrain done ($STRATEGY)"
else
    echo "=== retrain_panel FAILED at $(date) ==="
    notify "RenQuant 104 panel ERROR" "Forced retrain failed — check $LOG"
    exit 1
fi
