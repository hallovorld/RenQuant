#!/usr/bin/env bash
# conditional_retrain_104.sh — if SPY or VIX shows a daily anomaly,
# trigger the weekly WF trust-boundary chain immediately. Otherwise exit
# quietly. The anomaly path must not call legacy train_104.py directly.
#
# Schedule via launchd at 13:10 PT (45 min ahead of daily_104.sh).
#
# Triggers:
#   * SPY daily move > 2% -> weekly_wf_promote.sh trigger=anomaly_spy_2pct
#   * VIX daily move > 5% -> weekly_wf_promote.sh trigger=anomaly_vix_5pct

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
GITHUB_DIR="$(cd "$REPO_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
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

run_trigger_check() {
    if [ "${RQ_CONDITIONAL_TRIGGER_RUNNER:-multirepo}" = "legacy" ]; then
        "$PYTHON" scripts/check_retrain_triggers.py
        return $?
    fi

    local orch_src
    orch_src="$(renquant_subrepo_src "$SUBREPO_ROOT" renquant-orchestrator)"
    if PYTHONPATH="$orch_src:${PYTHONPATH:-}" "$PYTHON" - <<'PY'
import renquant_orchestrator.anomaly_triggers  # noqa: F401
PY
    then
        PYTHONPATH="$orch_src:${PYTHONPATH:-}" "$PYTHON" -m renquant_orchestrator.anomaly_triggers
        return $?
    fi

    if [ "${RQ_CONDITIONAL_TRIGGER_STRICT:-0}" = "1" ]; then
        echo "ERROR: renquant_orchestrator.anomaly_triggers unavailable and RQ_CONDITIONAL_TRIGGER_STRICT=1"
        return 2
    fi

    echo "WARN: renquant_orchestrator.anomaly_triggers unavailable; falling back to umbrella trigger check."
    "$PYTHON" scripts/check_retrain_triggers.py
}

# Detect triggers via renquant-orchestrator by default. Exit code 0 means no
# trigger; 1 means trigger(s) fired. Captured stdout holds trigger tag(s).
TRIGGER_OUT=$(run_trigger_check 2>&1)
TRIGGER_RC=$?

echo "$TRIGGER_OUT"

if [ "$TRIGGER_RC" -eq 0 ]; then
    echo "No triggers fired — exiting"
    exit 0
fi
if [ "$TRIGGER_RC" -ne 1 ]; then
    echo "Trigger check FAILED with rc=$TRIGGER_RC — not firing retrain."
    notify "RenQuant 104 trigger check ERROR" "Conditional retrain trigger check failed rc=$TRIGGER_RC. Check $LOG"
    exit "$TRIGGER_RC"
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
