# WF-promote outcome classification reads the child's own log   (PR #634)

STATUS:    delivered — ops-truth fix (G-D) for the two wrappers that classify
           what `weekly_wf_promote.sh` did.
WHAT:      the two wrappers now OWN the child's redirect: they launch
           `weekly_wf_promote.sh` with `RQ_WEEKLY_PROMOTE_STDOUT=1` (a seam the
           child carries: unset ⇒ byte-identical `exec >> "$LOG" 2>&1` as
           before), tee its stdout+stderr into the SAME dated log
           (`logs/weekly_wf_promote/<date>.log` — the daily record and the
           run-health scan see exactly what they see today) and keep the
           private copy that only THAT child wrote as the evidence the shared
           classifier reads. `scripts/lib/wf_promote_outcome.sh` gains
           `wf_promote_child_log_path`. Markers, polarity (no evidence ⇒ never
           success) and exit codes unchanged. 10 new tests: stub children that
           redirect like the real one, a concurrent promotion landing in the
           dated log mid-run, a child that ignores the seam (still never
           success), the daily record still written, and source-shape guards
           on the seam and both wrappers.
WHY/DIR:   `weekly_wf_promote.sh` `exec >> logs/weekly_wf_promote/<date>.log
           2>&1`s (line ~250) before it prints any terminal marker, so the
           tee'd stdout the 2026-08-21 fix classified is (almost) empty: every
           clean run since then reported UNVERIFIED (rc 2) and every alarm
           exit FAILED without the markers ever being consulted — the guard
           validated the wrong object, and the 08-21 tests validated a stub
           that printed markers to stdout. Observed today: the 13:10 PT VIX
           anomaly chain refused correctly ("Reject disposition: prod FRESH
           (trained 2026-08-31, 3d <= 28d SLA) — governance nominal, calm
           notify, exit 0") and the wrapper still paged "OUTCOME UNVERIFIED".
           The dated log accumulates every run of the day (the operator's
           `--promote-staged` at 09:09 left a FALLBACK-PROMOTED line in it), so
           only the segment after the pre-launch mark is this run's evidence.
EVIDENCE:  artifact:      `logs/conditional_retrain_104/2026-09-03.log` ("OUTCOME UNVERIFIED … 13:16:28") vs `logs/weekly_wf_promote/2026-09-03.log` line 149+ (the 13:10 run's "Reject disposition: prod FRESH … exit 0") [VERIFIED — read 2026-09-03 13:16 PDT]
           prod or exp:   prod ops surface (classification + notify text; the child's output routing when wrapped — same dated log, same lines)
           existing data: `tests/test_wf_promote_outcome_claim.py` 20 passed (10 existing + 10 new); with `test_reject_notify_disposition.py` + `test_weekly_wf_promote_rfc210_fallback.py`: 62 passed [VERIFIED — 2026-09-03 13:29 PDT]
           best-known?:   n/a — ops classification, no model claim
           scope:         "this changes WHO performs the child's redirect when a wrapper needs the outcome, and what the wrappers read; it does not change what the child decides or when it alarms"
CORRECTIONS (review r1, codex HIGH): r0 recorded a line count of the shared
           dated log before launching the child and classified the segment
           after it. A line-count boundary is not a run boundary: the child
           writes its start line BEFORE taking the global lock, so a
           concurrent manual/scheduled invocation can append its own markers
           after the mark, and the classifier gives promotion markers
           priority — a refused wrapped run could have read PROMOTED. r1
           replaces the segment arithmetic with wrapper-owned redirection
           (evidence attributable to the launched child by construction) and
           adds the interleaving regression test.
NEXT:      after merge + live ff-only: the next anomaly-triggered chain should
           notify "no change" with the disposition, not UNVERIFIED; the
           run-health scan's "acted" reading follows. Consider a follow-up that
           makes the child print its terminal marker to BOTH stdout and its log
           so wrappers need no segment arithmetic.
