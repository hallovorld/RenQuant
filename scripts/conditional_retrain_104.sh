#!/usr/bin/env bash
# conditional_retrain_104.sh — if SPY or VIX shows a daily anomaly,
# trigger an immediate retrain of renquant_104 BEFORE the regular
# 13:55 PT daily_104.sh pass. Otherwise exit quietly.
#
# Schedule via launchd at 13:10 PT (45 min ahead of daily_104.sh).
#
# Triggers:
#   * SPY |daily Δ| > 2%   → train_104.py --force --trigger=anomaly_spy_2pct
#   * VIX |daily Δ| > 5%   → train_104.py --force --trigger=anomaly_vix_5pct

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"
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

echo "=== Firing retrain: trigger=$TRIGGER ==="
notify "RenQuant 104 retrain fired" "Trigger: $TRIGGER"

if "$PYTHON" scripts/train_104.py --strategy renquant_104 --force --trigger "$TRIGGER"; then
    echo "=== Retrain complete ($TRIGGER) at $(date) ==="
    notify "RenQuant 104 retrain OK" "Anomaly retrain done: $TRIGGER"
else
    echo "=== Retrain FAILED ($TRIGGER) at $(date) ==="
    notify "RenQuant 104 retrain ERROR" "Anomaly retrain failed: $TRIGGER"
    exit 1
fi
