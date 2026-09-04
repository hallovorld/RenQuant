# A crashed WF simulation is not a reject: prove execution before the fallback   (PR #641)

STATUS:    delivered — ops-truth fix (G-D) in the weekly promote script's
           reject branch; companion to RenQuant#639 (which repairs the
           crash itself).
WHAT:      `scripts/wf_gate_sim_ran.py` (stdlib, hermetic): reads the staged
           artifact's stamped `metadata.wf_gate_metadata.cuts[*].returncode`
           and exits 0 iff every cut carries the int 0; anything else —
           non-zero, None, absent, bool, malformed, missing evidence,
           unreadable file — exits 1 with one line naming the cuts and the
           gate's own `wf_reason`. `scripts/weekly_wf_promote.sh` gains
           Step 4a: immediately after "WF gate REJECTED staged model" and
           BEFORE the RFC#210 fallback is consulted, it runs the helper on
           `$STAGING_ART`; on exit 1 it notifies
           `RenQuant 104 WEEKLY-FAIL (WF simulation crashed)` and exits 1.
           A crashed candidate is therefore neither reported calm nor
           eligible for fallback promotion. `tests/test_wf_gate_sim_ran.py`
           (7 tests + 6 parametrized cases): executed reject passes; the
           real 09-01 shape fails; one crashed cut suffices; only the int 0
           proves execution; missing/malformed evidence fails closed; CLI
           exit codes + one line; source-ordering guard on the promote
           script (call sits after REJECTED, before `FALLBACK_JSON=`, and
           the block alarms + exits 1). `job-outcome-contract.yml` now runs
           this test file and `bash -n` on the promote script.
WHY/DIR:   2026-09-01..03 every retrain's three WF cuts died inside the
           simulation (`ManifestUriResolutionError` — the resolver's digest
           window closed on the unstamped gate manifest; RenQuant#639).
           `run_wf_gate` stamped `wf_reason = "3/3 sim cuts failed
           execution"`, `cuts[*].returncode = 1`, exited non-zero, and the
           promote script's reject branch ran as designed for a REJECT:
           `reject_notify_disposition.py` proved the fallback verdict's
           prod-fresh shape (it never sees the cuts) and the run said
           "Reject disposition: prod FRESH (trained 2026-08-31, 3d <= 28d
           SLA) — governance nominal, calm notify, exit 0". Three daily
           candidates were discarded by an infrastructure crash reported as
           the gate declining, and nothing paged. Worse: had the served
           model been >28d stale, the RFC#210 fallback would have been
           consulted on a candidate whose simulation never ran (it checks
           freshness + stamped genuine_ic, not execution). The disposition
           helper's own contract is "an unproven healthy must alarm"; the
           thing it could not prove was that the gate had evaluated
           anything. Direction: G-D (a job must not report an outcome it
           did not have) and G-C (the refresh path must fail loudly where it
           is broken).
EVIDENCE:  artifact:      `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_2026090{1,2,3}T*.staging.json` `metadata.wf_gate_metadata` (`wf_reason`, `cuts[*].returncode = 1`, `error_tail`); `logs/weekly_wf_promote/2026-09-0{1,2,3}.log` ("Reject disposition: prod FRESH … exit 0") [VERIFIED — read 2026-09-03 18:14–18:20 PDT]
           prod or exp:   prod ops surface — the reject branch of the weekly/anomaly promote entry point (runs daily 13:10 PT via conditional_retrain_104); notify text + exit code only; the gate criterion, the fallback, and every promotion path are unchanged
           existing data: the helper run read-only against real staged artifacts: 09-03 candidate → exit 1 `WF-SIM-DID-NOT-RUN|3/3 cuts did not execute (… returncode=1 …) — gate said: 3/3 sim cuts failed execution`; 08-23 candidate (executed, rejected on benchmark/regime) → exit 0 `WF-SIM-RAN|all 3 cuts executed (returncode 0)`; the served 08-31 artifact (zero-trade reject) carries `returncode [0, 0, 0]` and would also pass [VERIFIED — 2026-09-03 between 18:24 and 18:28 PDT]; hermetic tests run in the sparse PR clone (carries `scripts/` + `tests/`): `test_wf_gate_sim_ran.py` (13 cases) + `test_weekly_wf_promote_rfc210_fallback.py` (19 existing, the reject stub now stamps executed cuts, + 2 new: crashed sim alarms before the fallback with the fallback armed to PROMOTE and the active artifact untouched; executed reject still reaches the fallback) + `test_weekly_wf_promote_snapshot_backstop.py` + `test_manual_promote_snapshot_backstop.py` + `test_wf_promote_outcome_claim.py` + `test_reject_notify_disposition.py` = 83 passed; `bash -n scripts/weekly_wf_promote.sh` ok [VERIFIED — 2026-09-03 between 18:30 and 18:32 PDT]
           best-known?:   n/a — ops truth; no model claim
           scope:         "this PR adds one pre-fallback execution check to the reject branch and its tests; it does not change what the gate decides, what the fallback decides, or any promotion"
NEXT:      after merge + live ff-only, the 13:10 PT retrain on the first day
           BEFORE #639 lands would page WEEKLY-FAIL (WF simulation crashed)
           — the correct signal for the state the system is in; after #639
           lands the check passes and the ordinary reject/promote flow
           resumes. CORRECTION to the WHY paragraph's "had the served model
           been stale" sentence: renquant-backtesting's
           `classify_gate_failures` already names a `"sim cuts failed"`
           `wf_reason` as the non-infra class `wf_sim_execution`, and
           `decide()` check 5 refuses any substance class (fail-closed) — so
           the fallback would have REFUSED a crashed candidate on
           `failure_classes`, not promoted it [VERIFIED — read
           `freshness_fallback.py` lines 358–371 and 562–578 at origin/main
           31b6d6e4, 2026-09-03 evening]. The hole this PR closes is the
           umbrella's CALM classification of that refusal, not a promotion
           path; no backtesting follow-up is needed.
