#!/usr/bin/env bash
# daily_retrain_alpha158_fund.sh — Production retrain pipeline for the
# alpha158+5fund XGBoost panel-LTR (promoted 2026-05-08, commit ca350c0).
#
# This is the chain that was MISSING from production after the promote.
# Pre-fix, daily_104.sh called scripts/train_104.py which retrained the
# OLD 21-feature panel-ltr.json — but the live config now reads the new
# 163-feature panel-ltr.alpha158_fund.json. Result: live model was a
# frozen 2026-05-08 snapshot with no daily refresh.
#
# Pipeline:
#   1. Rebuild alpha158 panel from latest OHLCV (148 features per ticker)
#   2. Merge with SEC 5-fund features → 163-feature production panel
#   3. Retrain XGB rank:pairwise (writes panel-ltr.alpha158_fund.json)
#   4. Refit calibrator on new model's predictions
#       (writes panel-rank-calibration.json)
#
# Output artifacts (all in backtesting/renquant_104/artifacts/):
#   panel-ltr.alpha158_fund.json
#   panel-rank-calibration.json
#
# Usage:
#   bash scripts/daily_retrain_alpha158_fund.sh [pipeline args]
#
# Designed to run BEFORE the live trading step in daily_104.sh. Lock
# file prevents concurrent invocations. Errors bubble up as non-zero
# exit so daily_104 can detect + skip the live trade if retrain fails.

set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/daily_retrain_alpha158_fund"
mkdir -p "$LOG_DIR"

notify_retrain_fallback_once() {
    local stamp="$LOG_DIR/.subrepo_fallback_alert_stamp"
    local cooldown="${RQ_RETRAIN_FALLBACK_ALERT_COOLDOWN_SEC:-86400}"
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
    body="RETRAIN_FALLBACK: renquant_orchestrator.retrain_alpha158_fund unavailable on $(hostname); using umbrella training_panel.daily_retrain_alpha158_fund. Set RQ_RETRAIN_STRICT=1 to fail closed."
    curl -sS -H "Title: RenQuant RETRAIN FALLBACK" \
        -H "Priority: low" \
        -d "$body" \
        "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
}

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

# Lock — prevent concurrent runs (e.g. retrain_panel104 firing simultaneously)
LOCK_FILE="/tmp/renquant_retrain_alpha158_fund.lock"
if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    # Check if PID is alive — clean stale locks
    if ! kill -0 "$EXISTING_PID" 2>/dev/null; then
        rm -f "$LOCK_FILE"
        echo "$$" > "$LOCK_FILE"
    else
        echo "Another retrain_alpha158_fund run is active (PID=$EXISTING_PID) — exiting." | tee -a "$LOG"
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

cd "$REPO_DIR/backtesting/renquant_104"

exec >> "$LOG" 2>&1
echo "═══ daily_retrain_alpha158_fund started $(date -u +'%Y-%m-%dT%H:%M:%SZ') ═══"

run_umbrella() {
    "$PYTHON" -m training_panel.daily_retrain_alpha158_fund "$@"
}

run_multirepo() {
    # Transitional multirepo path: orchestrator owns the weekly workflow and
    # delegates scorer training to renquant-model. The original umbrella module
    # remains available via RQ_RETRAIN_RUNNER=umbrella.
    cd "$REPO_DIR"
    "$PYTHON" -m renquant_orchestrator.retrain_alpha158_fund --repo-dir "$REPO_DIR" "$@"
}

# Delegate the actual work to a Python pipeline. Bash only handles lock, log
# redirect, runner selection, and fallback if the multirepo module is missing.
export RENQUANT_REPO_ROOT="$REPO_DIR"
GITHUB_DIR="$(dirname "$REPO_DIR")"
export PYTHONPATH="$GITHUB_DIR/renquant-orchestrator/src:$GITHUB_DIR/renquant-common/src:$GITHUB_DIR/renquant-base-data/src:$GITHUB_DIR/renquant-artifacts/src:$GITHUB_DIR/renquant-model/src:$GITHUB_DIR/renquant-pipeline/src:$GITHUB_DIR/renquant-execution/src:$GITHUB_DIR/renquant-strategy-104/src:$GITHUB_DIR/renquant-backtesting/src:${PYTHONPATH:-}"
RUNNER="${RQ_RETRAIN_RUNNER:-multirepo}"
if [ "$RUNNER" = "umbrella" ]; then
    CMD=run_umbrella
elif "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_orchestrator.retrain_alpha158_fund  # noqa: F401
PY
then
    "$PYTHON" - <<'PY' >&2
import renquant_orchestrator.retrain_alpha158_fund as m
print(f"renquant_orchestrator.retrain_alpha158_fund={m.__file__}")
PY
    CMD=run_multirepo
elif [ "${RQ_RETRAIN_STRICT:-0}" = "1" ]; then
    echo "ERROR: renquant_orchestrator.retrain_alpha158_fund unavailable and RQ_RETRAIN_STRICT=1"
    exit 1
else
    echo "WARN: renquant_orchestrator.retrain_alpha158_fund unavailable; falling back to umbrella retrain."
    notify_retrain_fallback_once
    CMD=run_umbrella
fi

if "$CMD" "$@"; then
    echo "═══ daily_retrain_alpha158_fund DONE ═══"
    exit 0
else
    echo "═══ daily_retrain_alpha158_fund FAILED ═══"
    exit 1
fi
