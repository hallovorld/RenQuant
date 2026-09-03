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

# Overridable ONLY so the promote-outcome classification below can be tested
# against a stub child. Production passes nothing and gets the same literal
# path it always had. A fix whose only evidence is "I read it carefully" is
# the kind that shipped the bug it is fixing.
REPO_DIR="${RQ_CONDITIONAL_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
GITHUB_DIR="$(cd "$REPO_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
# One definition of "what did the promote chain actually do", shared with
# retrain_panel.sh so the two wrappers cannot drift apart.
source "$REPO_DIR/scripts/lib/wf_promote_outcome.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/conditional_retrain_104"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    # Same testing seam weekly_wf_promote.sh already uses: when a log path is
    # set, record instead of paging, so a test can assert on WHAT the operator
    # would have been told.
    if [ -n "${RQ_CONDITIONAL_NOTIFY_LOG:-}" ]; then
        printf '%s: %s\n' "$title" "$body" >> "$RQ_CONDITIONAL_NOTIFY_LOG"
        return 0
    fi
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

    if [ "${RQ_CONDITIONAL_TRIGGER_RUNNER:-multirepo}" != "multirepo" ]; then
        echo "ERROR: unknown RQ_CONDITIONAL_TRIGGER_RUNNER=${RQ_CONDITIONAL_TRIGGER_RUNNER} (expected multirepo or legacy)"
        return 2
    fi

    echo "ERROR: renquant_orchestrator.anomaly_triggers unavailable; set RQ_CONDITIONAL_TRIGGER_RUNNER=legacy for explicit rollback."
    return 2
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

# EXIT CODE IS NOT THE OUTCOME (2026-08-21). This branched on the child's exit
# status alone, and `weekly_wf_promote.sh` exits 0 on a REFUSAL as well as on a
# promotion — deliberately: "Reject disposition: prod FRESH ... governance
# nominal, calm notify, exit 0" (weekly_wf_promote.sh:517). So on 2026-08-19
# and 2026-08-20 this wrapper printed "chain complete" and paged the operator
# "WF promote OK" while NOTHING had been promoted. The run-health scan caught
# it: "2 of them CLAIMED SUCCESS while weekly-wf-promote's own log for that
# date shows no promotion".
#
# A promotion is now established POSITIVELY, from the child's own terminal
# marker, of which it emits exactly two:
#     === weekly_wf_promote PASSED at ... ===
#     === weekly_wf_promote FALLBACK-PROMOTED (rfc210) at ... ===
# Anything else on a clean exit is "ran, decided not to promote" — a real and
# common outcome that deserves its own name, not silence and not "OK".
#
# The polarity matters more than the parsing: an outcome this wrapper cannot
# establish must NEVER read as success. If the markers are ever renamed, this
# reports UNVERIFIED and the operator finds out, instead of inheriting a
# permanent false OK.
#
# WHERE THE MARKERS ARE (2026-09-03). The child redirects its own stdout/stderr
# into logs/weekly_wf_promote/<date>.log before it prints anything that matters,
# so the tee below is not the evidence — the segment of that dated log written
# by THIS run is (see wf_promote_outcome.sh). Today's 13:10 chain refused
# correctly ("prod FRESH … exit 0") and this wrapper still said UNVERIFIED.
CHILD_LOG="$(wf_promote_child_log_path "$REPO_DIR")"
CHILD_LOG_MARK="$(wf_promote_child_log_mark "$CHILD_LOG")"
CHAIN_OUT=$(mktemp "${TMPDIR:-/tmp}/rq104_wf_chain.XXXXXX")
RENQUANT_WEEKLY_TRIGGER="$TRIGGER" bash scripts/weekly_wf_promote.sh 2>&1 | tee "$CHAIN_OUT"
CHAIN_RC=${PIPESTATUS[0]}
append_wf_promote_child_log_segment "$CHAIN_OUT" "$CHILD_LOG" "$CHILD_LOG_MARK"
CHAIN_OUTCOME="$(classify_wf_promote_outcome "$CHAIN_OUT" "$CHAIN_RC")"
CHAIN_WHY="$(describe_wf_promote_outcome "$CHAIN_OUT")"
rm -f "$CHAIN_OUT"

case "$CHAIN_OUTCOME" in
    PROMOTED)
        echo "=== Gated WF promote chain PROMOTED ($TRIGGER) at $(date) — ${CHAIN_WHY} ==="
        notify "RenQuant 104 WF promote PROMOTED" "$TRIGGER — ${CHAIN_WHY}"
        ;;
    NOTHING_PROMOTED)
        echo "=== Gated WF promote chain RAN, NOTHING PROMOTED ($TRIGGER) at $(date) — ${CHAIN_WHY:-refused} ==="
        # Deliberately not an alarm: a gate declining is the gate working. Also
        # deliberately not "OK" — the operator must be able to tell a promotion
        # from a refusal without opening the child's log.
        notify "RenQuant 104 WF promote: no change" "$TRIGGER — ${CHAIN_WHY:-gate declined; production unchanged}"
        ;;
    FAILED)
        echo "=== Gated WF promote chain FAILED ($TRIGGER) at $(date) — rc=$CHAIN_RC ==="
        notify "RenQuant 104 WF promote ERROR" "Anomaly-gated chain failed: $TRIGGER (rc=$CHAIN_RC)"
        exit 1
        ;;
    *)
        echo "=== Gated WF promote chain OUTCOME UNVERIFIED ($TRIGGER) at $(date) ==="
        notify "RenQuant 104 WF promote UNVERIFIED" \
            "$TRIGGER — exited 0 but emitted neither a promotion marker nor a refusal. Check logs/weekly_wf_promote."
        # Exit 2, NOT a fall-through to 0. The notification alone is not enough:
        # the job's exit status is what launchd and the run-health scan read,
        # and an unestablished outcome presenting as a successful job is the
        # same false-OK this file exists to remove. 2 rather than 1 so
        # automation can tell "the child failed" (1) from "the child's contract
        # drifted and we cannot say what it did" (2) — different repairs.
        exit 2
        ;;
esac
