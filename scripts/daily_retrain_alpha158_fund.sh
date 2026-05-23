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

# Delegate the actual work to the Python pipeline (Task/Job per CLAUDE.md §1b).
# Bash only handles: lock, log redirect, ntfy on failure.
if "$PYTHON" -m training_panel.daily_retrain_alpha158_fund "$@"; then
    echo "═══ daily_retrain_alpha158_fund DONE ═══"
    exit 0
else
    echo "═══ daily_retrain_alpha158_fund FAILED ═══"
    exit 1
fi
