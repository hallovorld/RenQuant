#!/usr/bin/env bash
# conditional_retrain_104.sh — if SPY or VIX shows a daily anomaly,
# trigger the weekly WF trust-boundary chain immediately. Otherwise exit
# quietly. The anomaly path must not call legacy train_104.py directly.
#
# Schedule via launchd at 13:10 PT (45 min ahead of daily_104.sh).
#
# Triggers:
#   * SPY |daily Δ| > 2%   → train_104.py --force --trigger=anomaly_spy_2pct
#   * VIX |daily Δ| > 5%   → train_104.py --force --trigger=anomaly_vix_5pct

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/conditional_retrain_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

exec >> "$LOG" 2>&1
echo "=== conditional_retrain_104 started at $(date) ==="

# NYSE calendar guard
if ! "$PYTHON" -c "
import sys, pandas_market_calendars as mcal
cal = mcal.get_calendar('NYSE')
sched = cal.schedule('$DATE', '$DATE')
sys.exit(0 if len(sched) > 0 else 1)
"; then
    echo "NYSE closed today — skipping"
    exit 0
fi

cd "$REPO_DIR"

# Detect triggers via check_retrain_triggers.py — exits 0 (no trigger)
# or 1 (trigger(s) fired). Captured stdout holds the trigger tag(s).
TRIGGER_OUT=$("$PYTHON" scripts/check_retrain_triggers.py 2>&1)
TRIGGER_RC=$?

echo "$TRIGGER_OUT"

if [ "$TRIGGER_RC" -eq 0 ]; then
    echo "No triggers fired — exiting"
    exit 0
fi

# Parse first trigger tag (the name printed to stdout by check_retrain_triggers)
# Prefer SPY if both fired.
TRIGGER=$(echo "$TRIGGER_OUT" | grep -E "^anomaly_" | head -1)
if [ -z "$TRIGGER" ]; then
    echo "Trigger detected but no tag parsed — defaulting to 'anomaly_unknown'"
    TRIGGER="anomaly_unknown"
fi

echo "=== Firing gated weekly promote chain: trigger=$TRIGGER ==="
notify "RenQuant 104 WF promote fired" "Trigger: $TRIGGER"

if RENQUANT_WEEKLY_TRIGGER="$TRIGGER" bash scripts/weekly_wf_promote.sh; then
    echo "=== Gated WF promote chain complete ($TRIGGER) at $(date) ==="
    notify "RenQuant 104 WF promote OK" "Anomaly-gated chain done: $TRIGGER"
else
    echo "=== Gated WF promote chain FAILED ($TRIGGER) at $(date) ==="
    notify "RenQuant 104 WF promote ERROR" "Anomaly-gated chain failed: $TRIGGER"
    exit 1
fi
