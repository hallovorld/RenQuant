# 2026-08-04 — weekly wrapper threads the retrainer's session pins (manual-run bridge)

STATUS:    one-surface change + shape guards; scheduled runs byte-identical
WHAT:      scripts/weekly_wf_promote.sh Step 3 gains two env passthroughs —
           RENQUANT_RETRAIN_EXPECTED_SESSION → --expected-session and
           RENQUANT_RETRAIN_AS_OF → --as-of — the retrainer's OWN
           deterministic-replay pins, built for "freshness must not depend
           on when the job happens to run." Empty envs (every scheduled
           run) pass NOTHING: the invocation is byte-identical to before.
WHY/DIR:   this morning's operator-authorized manual promote failed CLEAN
           at the freshness guard: run mid-session, the wall-clock-derived
           expected session was 2026-08-03 while intraday bars already
           carried 2026-08-04 — 293/293 "future", fail-closed. The
           Saturday schedule never hits this; a manual weekday run always
           will. The fix pins the REFERENCE to a real completed session
           via the designed flags — it never loosens tolerances, never
           disables fail-on-stale (a shape test asserts the disable flag
           stays absent), and the guard still measures every ticker.
EVIDENCE:  bash -n clean; tests/test_weekly_wf_promote_rfc210_fallback.py
           9 passed (new shape test: pins ride the SAME staging-output
           invocation, empty-env = nothing added, no tolerance loosening).
NEXT:      merge → live pull → rerun the operator-authorized manual
           promote with RENQUANT_RETRAIN_EXPECTED_SESSION=2026-08-03 and
           RENQUANT_RETRAIN_AS_OF=2026-08-03T20:00:00-04:00 → RFC#210
           fallback ends the 44-day staleness today, pre-close.
