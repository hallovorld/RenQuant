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

# Overridable ONLY so the outcome classification can be driven by a hermetic
# test against a stub child; production passes nothing and gets the same path.
REPO_DIR="${RQ_RETRAIN_PANEL_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
# 2026-05-11 audit M-env: switched conda → .venv per feedback_python_env.md
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
# Shared with conditional_retrain_104.sh; two copies of this rule would drift.
source "$REPO_DIR/scripts/lib/wf_promote_outcome.sh"
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
LOCK_FILE="${RQ_RETRAIN_PANEL_LOCK_FILE:-/tmp/renquant_104_retrain_panel.lock}"
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
# EXIT CODE IS NOT THE OUTCOME (2026-08-23). This printed "PASS" whenever the
# child exited 0 — and a REFUSAL exits 0 deliberately. Measured today: the
# 2026-08-23 run logged "delegated weekly_wf_promote PASS" for a chain whose own
# verdict was `VERDICT: FAIL` (genuine_ic=+0.0000, aligned_real_ic == placebo_ic
# to four decimals) and which promoted nothing. That log line is also what the
# run-health scan reads to decide whether this job "acted", so the false PASS
# corrupted the scan as well as the reader.
#
# This wrapper still emits NO ntfy by design — weekly_wf_promote owns alerts.
# Only the log line is corrected.
RP_OUT=$(mktemp "${TMPDIR:-/tmp}/retrain_panel_chain.XXXXXX")
bash scripts/weekly_wf_promote.sh 2>&1 | tee "$RP_OUT"
RP_RC=${PIPESTATUS[0]}
RP_OUTCOME="$(classify_wf_promote_outcome "$RP_OUT" "$RP_RC")"
RP_WHY="$(describe_wf_promote_outcome "$RP_OUT")"
rm -f "$RP_OUT"

case "$RP_OUTCOME" in
    PROMOTED)
        echo "=== retrain_panel delegated weekly_wf_promote PROMOTED at $(date) — ${RP_WHY} ==="
        exit 0
        ;;
    NOTHING_PROMOTED)
        echo "=== retrain_panel delegated weekly_wf_promote RAN, NOTHING PROMOTED at $(date) — ${RP_WHY:-refused} ==="
        echo "Production preserved by weekly_wf_promote; check logs/weekly_wf_promote/$DATE.log."
        exit 0
        ;;
    FAILED)
        echo "=== retrain_panel delegated weekly_wf_promote FAILED at $(date) — rc=$RP_RC ==="
        echo "Production preserved by weekly_wf_promote; check logs/weekly_wf_promote/$DATE.log."
        exit 1
        ;;
    *)
        echo "=== retrain_panel delegated weekly_wf_promote OUTCOME UNVERIFIED at $(date) ==="
        echo "Child exited 0 but emitted neither a promotion marker nor a refusal."
        exit 0
        ;;
esac
