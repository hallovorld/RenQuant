# 2026-08-04 — weekly reject notification is state-aware (operator directive)

## The complaint, verbatim-adjacent

The operator received "Walk-forward gate rejected the staged model. Production
unchanged." after today's post-close run and read it as the system breaking the
model daily. Measured reality: that run was HEALTHY end-to-end — the gate
rejected the chronically placebo-dominated recipe (genuine_ic +0.0021), and the
RFC #210 decide correctly refused because the served model is 2d fresh. The
wrapper then notified with a failure tone and exited 1 anyway.

Under RFC #210 steady state this is the MAJORITY outcome (~4 of 5 Saturdays:
prod ages 2→9→16→23d under the 28d SLA before a fallback promote), so the
failure-toned message is scheduled to cry wolf weekly.

## Change

`scripts/reject_notify_disposition.py` (new): proves the fresh-refusal shape
from the fallback verdict JSON — decision exactly `REFUSE`, refused on exactly
`prod_stale`, that check's `ok` exactly the bool False, int `staleness_days`
within the 28d SLA, non-empty `prod_trained`. Prints one line
(`CALM_FRESH|age|trained` or `ALARM|reason`), always exits 0 — it is a
disposition, the wrapper maps it.

`scripts/weekly_wf_promote.sh` Step 4b reject branch: consults the disposition.
- CALM_FRESH → notify "WEEKLY-REJECT (prod fresh — no action)" with trained
  date + age, **exit 0** (the job's outcome IS success: governance did its job).
- Anything else (missing/malformed verdict, refusal on another check, prod
  actually stale, disarmed/unavailable paths that never wrote a verdict) →
  the existing alarm notify + **exit 1**, now with the specific reason in the
  body. Fail closed toward attention.

The sentinel's log-contract line ("WF gate REJECTED staged model — production
unchanged.") is emitted VERBATIM in both cases at its recorded source line —
the orchestrator emitter contract needs only line-number re-capture for the
lines below the insertion (companion orchestrator PR).

## Verification

- `tests/test_reject_notify_disposition.py` — 16 unit cases; every malformed
  twin (harness-stub `{"verdict":...}` schema, ok=0/None/absent, str/bool
  staleness, future-SLA inconsistency, empty trained, list top-level) ALARMs.
- `tests/test_weekly_wf_promote_rfc210_fallback.py` — shim template extended to
  emit a real-shaped verdict; two new end-to-end cases: fresh-refusal → calm
  title + rc 0 + artifacts untouched + lock released; stale/other-refusal →
  alarm title + reason + rc 1. Existing refuse/module-unavailable tests pass
  UNCHANGED (their legacy stub verdict schema lands in ALARM by design).
- File suites: 35 passed.

## Exit-code semantics note

`exit 0` on fresh-reject changes the launchd LastExitStatus signal for this
job. The ack-ledger rows that acknowledged the chronic exit-1 were cleared
earlier today (orch#789) because the RFC #210 promotion satisfied their
clears_when — so no ack row pins the old exit code. Future nonzero exits from
this wrapper are real alarms again.
