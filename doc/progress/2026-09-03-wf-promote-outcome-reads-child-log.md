# WF-promote outcome classification reads the child's own log   (PR TBD)

STATUS:    delivered — ops-truth fix (G-D) for the two wrappers that classify
           what `weekly_wf_promote.sh` did.
WHAT:      `scripts/lib/wf_promote_outcome.sh` gains three helpers
           (`wf_promote_child_log_path`, `wf_promote_child_log_mark`,
           `append_wf_promote_child_log_segment`); `conditional_retrain_104.sh`
           and `retrain_panel.sh` record the line count of the child's dated
           log BEFORE launching it and append the segment written AFTER that
           mark to the evidence file the shared classifier reads. Markers,
           polarity (never success on unestablished evidence) and exit codes
           are unchanged. 7 new tests use stub children that `exec >>` their
           dated log first — the shape production has always had.
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
           prod or exp:   prod ops surface (classification + notify text only; no trading path touched)
           existing data: `tests/test_wf_promote_outcome_claim.py` 17 passed (10 existing + 7 new) [VERIFIED]; the pre-fix wrapper against the redirected-refusal stub reports UNVERIFIED (the new tests fail on `origin/main`'s wrappers)
           best-known?:   n/a — ops classification, no model claim
           scope:         "this changes WHAT the wrappers read to classify a run; it does not change what the child does or when it alarms"
NEXT:      after merge + live ff-only: the next anomaly-triggered chain should
           notify "no change" with the disposition, not UNVERIFIED; the
           run-health scan's "acted" reading follows. Consider a follow-up that
           makes the child print its terminal marker to BOTH stdout and its log
           so wrappers need no segment arithmetic.
