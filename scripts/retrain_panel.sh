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

echo "--- Sunday multi-backend sweep (strategy=$STRATEGY) ---"
echo "    backends: xgboost (production) → lightgbm → transformer"
echo "    expected wall time: ~75-90 min sequential"
if "$PYTHON" scripts/sunday_panel_sweep.py --strategy "$STRATEGY"; then
    echo "=== retrain_panel sweep finished at $(date) ==="
    REPORT_PATH=$(ls -t "$REPO_DIR/doc/panel_sunday_sweep_"*.md 2>/dev/null | head -1)
    if [ -n "$REPORT_PATH" ]; then
        BODY="Sunday sweep done — see $(basename "$REPORT_PATH")"
    else
        BODY="Sunday sweep done — XGBoost active"
    fi
    notify "RenQuant 104 panel" "$BODY"
else
    echo "=== retrain_panel sweep FAILED at $(date) ==="
    notify "RenQuant 104 panel ERROR" "Sunday sweep failed — check $LOG"
    exit 1
fi
