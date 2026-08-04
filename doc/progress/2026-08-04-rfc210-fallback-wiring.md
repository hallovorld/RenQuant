# weekly_wf_promote consumes the RFC#210 fallback (operator P0 wiring)

**Date:** 2026-08-04 · `RenQuant` (umbrella) · backtesting#101/#102

STATUS:    script wiring; ARMS ONLY after the backtesting runtime pin
           advances past #102 (until then Step 4b prints UNAVAILABLE and
           behaves exactly as today's REJECT — fail-closed, loud). Round-2
           review (2nd finding): arming also needs the orchestrator
           sentinel's FALLBACK-PROMOTED action-consumer contract
           (renquant-orchestrator#774, open, not yet merged) so a real
           fallback promotion is classified as an action rather than a
           silent-refusal incident. Do not advance the backtesting pin past
           #102 in production until #774 is merged — rollout order is
           #559 (this PR) -> orchestrator#774 -> the pin-advance PR that
           arms both.
WHAT:      Step 4b in scripts/weekly_wf_promote.sh: on gate REJECT, consult
           `renquant_backtesting.wf_gate.freshness_fallback --stamp`.
           REFUSE → today's behavior verbatim (REJECT ntfy, exit 1).
           FALLBACK_PROMOTE → pair-promote (same incoming/replace dance as
           Step 5) licensed by the promotion_basis stamp (and requiring the
           stamped passed=False — the gate-passed license check must not
           run on this path); Steps 6/7 run for both paths; final emitter
           line "weekly_wf_promote FALLBACK-PROMOTED (rfc210)" + its own
           ntfy title (paired orchestrator PR teaches the silent-refusal
           sentinel that this line is an ACTION).
           Round-2 review follow-up (this commit): (a) Step 5's GATE_SUMMARY
           read `$STAGING_ART`, which Step 4b's `_swap_into_active()`
           already unlinks on the fallback path — every fallback run logged
           "(metadata parse failed)"; it now reads back from `$ACTIVE_ART`
           when staging is gone (identical bytes, same copy). (b) Step 7's
           snapshot backstop ran AFTER the fallback promote and, on
           detecting the (expected) gate-verdict drift, `exit 1`'d BEFORE
           the FALLBACK-PROMOTED literal/notification ever printed — a
           genuine production mutation whose action-contract line the
           sentinel could never observe. The fallback path now treats a
           stale snapshot as a WARN follow-up (same notification, no hard
           fail) and always reaches the action-literal + notification with
           exit 0; the gate-passed path's existing hard-fail-on-stale
           behavior (pinned by test_weekly_wf_promote_snapshot_backstop.py)
           is untouched.
WHY/DIR:   Operator P0 (2026-08-03): the placebo-deadlocked gate starves
           prod (42d+ stale, 4 identical Sunday REJECTs). Policy decided on
           backtesting#101 (amended), implemented+merged as #102 with the
           real 08-02 reject dry-running to FALLBACK_PROMOTE.

EVIDENCE:
artifact:      scripts/weekly_wf_promote.sh (Step 4b/4c diff, this PR +
               the round-1-review follow-up commit)
prod or exp:   prod (production weekly promotion wrapper) — UNARMED: the
               live backtesting runtime pin (8f6700ab) predates #102, so
               Step 4b prints UNAVAILABLE and behaves exactly as today's
               REJECT until a separate pin-advance PR lands (see STATUS).
existing data: no IC/Sharpe/APY claim is made by this PR — it is a code
               path change, not a model result. `bash -n
               scripts/weekly_wf_promote.sh` clean; the REFUSE branch is
               byte-equivalent to the pre-PR REJECT branch plus two echo
               lines; the module-absent path was measured live (current
               pin → UNAVAILABLE + REFUSE, no observable change). Round-1
               review reproduced the BLOCKER
               (`ValueError: promote: refused — wf_gate_metadata.
               passed=False`) with a minimal stamped fallback artifact
               against `renquant_backtesting.forensics.model_acceptance.
               promote()`; the fix replaces that call, for the fallback
               path only, with an inline atomic swap that keeps this
               path's own promotion_basis-stamp license check but skips
               the shared helper's unconditional wf_gate_metadata.passed
               gate. Regression-pinned by
               `tests/test_weekly_wf_promote_rfc210_fallback.py` (fails
               against the pre-fix script with the exact reviewer-
               reproduced ValueError; passes after the fix) — run via
               `pytest tests/test_weekly_wf_promote_rfc210_fallback.py
               tests/test_weekly_wf_promote_snapshot_backstop.py
               tests/test_weekly_wf_promote_wrapper_guard.py -v`.
               Round-2: expanded `test_weekly_wf_promote_rfc210_fallback.py`
               from 1 to 7 cases (real subprocess runs of the production
               script against the fixture repo, per round-1 review) —
               successful fallback now asserted to exit 0 and emit both the
               FALLBACK-PROMOTED log literal and its ntfy title even with a
               stale snapshot doc; calibrator swap verified via a planted
               marker that the promote must overwrite; the
               `*.fallback_verdict.json` file is asserted present with the
               stamped verdict after the run; module-unavailable, REFUSE,
               missing-promotion_basis, and passed!=False all assert BOTH
               active artifacts byte-unchanged. All 7 pass; the pre-existing
               `test_layer3_cuts_match_candidate_artifact_recipe` failure in
               `test_weekly_wf_promote_wrapper_guard.py` also fails on clean
               `main` (38c4a34) and is unrelated to this PR (confirmed by
               round-1 review; re-confirmed this round).
best-known?:   n/a — first wiring of the RFC#210 fallback consumer; no
               prior variant of this code path exists to compare against.
scope:         "this is scripts/weekly_wf_promote.sh, prod (unarmed under
               the current pin), no IC/Sharpe/APY claim — behavioral/
               code-path evidence only, verified by bash -n +
               the new regression test above."

NEXT:      merge renquant-orchestrator#774 (sentinel action-consumer
           contract), then advance the backtesting runtime pin past #102
           (separate, reviewed pin PR + granted runtime sync) to arm the
           fallback; until both land this script's observable behavior is
           unchanged. A follow-up should also make the pin-advance
           verification assert both contracts are present before allowing
           the backtesting pin to cross #102 in the same lock-file change
           that would arm this path.

## Revert

git revert; the REJECT branch returns to unconditional exit 1.
