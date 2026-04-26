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

# Launch the sweep in the background so we can attach a resource monitor
# to its PID. monitor_training_resources.py samples cpu+rss every 5s
# (parent + recursive children) into a CSV next to this log. The
# plotter at the end turns it into a polished PNG.
"$PYTHON" scripts/sunday_panel_sweep.py --strategy "$STRATEGY" &
SWEEP_PID=$!
echo "sweep PID=$SWEEP_PID"

"$PYTHON" scripts/monitor_training_resources.py \
    --pid "$SWEEP_PID" --interval 5 \
    --out "logs/retrain_panel/$DATE.resources.csv" &
MON_PID=$!
echo "monitor PID=$MON_PID  → logs/retrain_panel/$DATE.resources.csv"

# Wait for the sweep, capture exit code, then kill the monitor (it'd
# otherwise spin until --max-duration; killing it cleanly here also
# flushes the CSV via the SIGTERM handler).
wait "$SWEEP_PID"
SWEEP_RC=$?
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

# Render the resource chart regardless of sweep success — partial data
# is still useful for diagnosing where a failure happened.
echo "--- rendering resource chart ---"
"$PYTHON" scripts/plot_training_resources.py --date "$DATE" || \
    echo "warn: plot_training_resources failed (non-fatal)"

if [ "$SWEEP_RC" -eq 0 ]; then
    echo "=== retrain_panel sweep finished at $(date) ==="
    REPORT_PATH=$(ls -t "$REPO_DIR/doc/panel_sunday_sweep_"*.md 2>/dev/null | head -1)
    PNG_PATH="$LOG_DIR/$DATE.resources.png"
    if [ -n "$REPORT_PATH" ]; then
        BODY="Sunday sweep done — see $(basename "$REPORT_PATH"); chart: $(basename "$PNG_PATH")"
    else
        BODY="Sunday sweep done — XGBoost active; chart: $(basename "$PNG_PATH")"
    fi
    notify "RenQuant 104 panel" "$BODY"
else
    echo "=== retrain_panel sweep FAILED at $(date) ==="
    notify "RenQuant 104 panel ERROR" "Sunday sweep failed — check $LOG"
    exit 1
fi
