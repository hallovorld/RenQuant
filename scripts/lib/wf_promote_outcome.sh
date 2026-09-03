#!/bin/bash
# Classify what weekly_wf_promote.sh ACTUALLY did — one definition, two callers.
#
# THE DEFECT (2026-08-21/23). Both wrappers branched on the child's EXIT CODE
# alone, and `weekly_wf_promote.sh` exits 0 on a refusal DELIBERATELY:
#
#     Reject disposition: prod FRESH ... governance nominal, calm notify, exit 0
#                                                    (weekly_wf_promote.sh:517)
#
# because a gate declining is the gate working. The consequences were measured,
# not hypothesised:
#
#   * conditional_retrain_104 on 2026-08-19 and 2026-08-20 printed "chain
#     complete" and PUSHED "RenQuant 104 WF promote OK" to the operator while
#     production was untouched — a positive report for a non-event, so there
#     was no reason to go look.
#   * retrain_panel on 2026-08-23 logged "delegated weekly_wf_promote PASS" for
#     a run whose own verdict was `VERDICT: FAIL` (genuine_ic=+0.0000) and which
#     promoted nothing. That log line is also what the run-health scan reads to
#     decide whether a job "acted", so the false PASS corrupts the scan too.
#
# ONE DEFINITION, because two copies of a rule like this drift and this repo has
# a documented history of exactly that.
#
# POLARITY IS THE POINT. A promotion is established POSITIVELY from the child's
# own terminal markers. An outcome that cannot be established is UNVERIFIED —
# never success. If the markers are ever renamed, callers say so out loud
# instead of inheriting a permanent false OK.

#: The child's two positive terminal markers [weekly_wf_promote.sh:362,695,699].
WF_PROMOTE_MARKERS='=== weekly_wf_promote (PASSED|FALLBACK-PROMOTED)'
#: Evidence that it ran and decided NOT to promote.
WF_REFUSAL_MARKERS='REFUSE|Reject disposition|VERDICT: FAIL|promote-staged REFUSED|WEEKLY-BLOCKED'

# classify_wf_promote_outcome <child_output_file> <child_exit_code>
#   -> echoes exactly one of: PROMOTED | NOTHING_PROMOTED | FAILED | UNVERIFIED
classify_wf_promote_outcome() {
    local out="$1" rc="${2:-0}"
    if [ "$rc" -ne 0 ]; then
        echo "FAILED"
        return 0
    fi
    if [ ! -r "$out" ]; then
        # Cannot read what the child said => cannot claim it succeeded.
        echo "UNVERIFIED"
        return 0
    fi
    if grep -qE "$WF_PROMOTE_MARKERS" "$out"; then
        echo "PROMOTED"
    elif grep -qE "$WF_REFUSAL_MARKERS" "$out"; then
        echo "NOTHING_PROMOTED"
    else
        echo "UNVERIFIED"
    fi
}

# describe_wf_promote_outcome <child_output_file> — a short human reason, or "".
describe_wf_promote_outcome() {
    local out="$1"
    [ -r "$out" ] || return 0
    grep -oE "=== weekly_wf_promote (PASSED|FALLBACK-PROMOTED)[^=]*|Reject disposition: [^.]*|RFC#210 fallback verdict: [A-Z]+|VERDICT: FAIL" "$out" | head -1
}

# ── The evidence file (2026-09-03) ───────────────────────────────────────────
# weekly_wf_promote.sh `exec >> "$LOG_DIR/$DATE.log" 2>&1`s (its line ~250)
# BEFORE it prints any terminal marker, so a wrapper that classifies the tee'd
# stdout classifies an (almost) EMPTY file: every clean run since 2026-08-21
# read UNVERIFIED and every alarm exit read FAILED without the markers ever
# being consulted — the guard validated the wrong object. The markers live in
# the child's own dated log. That log accumulates every run of the day (an
# operator --promote-staged the same morning leaves a FALLBACK-PROMOTED line in
# it), so THIS run's evidence is the segment written after the wrapper
# launched the child: record the line count before, take the tail after.
#: The child's dated log for today, under <repo_dir> (mirrors weekly_wf_promote.sh LOG_DIR/DATE).
wf_promote_child_log_path() {
    local repo_dir="$1"
    printf '%s\n' "$repo_dir/logs/weekly_wf_promote/$(date +%Y-%m-%d).log"
}

#: Line count of a file, 0 when absent — the pre-launch mark.
wf_promote_child_log_mark() {
    local child_log="$1"
    if [ -f "$child_log" ]; then wc -l < "$child_log" | tr -d ' '; else echo 0; fi
}

# append_wf_promote_child_log_segment <evidence_file> <child_log> <mark>
# Appends the child log's lines after <mark> to <evidence_file>; no-op when the
# child log does not exist (the classifier then sees only the tee'd stdout and
# reports UNVERIFIED — never success).
append_wf_promote_child_log_segment() {
    local out="$1" child_log="$2" mark="${3:-0}"
    [ -f "$child_log" ] || return 0
    tail -n +"$((mark + 1))" "$child_log" >> "$out"
}
