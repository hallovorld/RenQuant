# 2026-08-04 — --promote-staged: the operator promotion path becomes first-class

STATUS:    wrapper mode + ONE shared pair-promote implementation
WHAT:      this morning's operator-ordered promotion ("现在就promote到104
           和105！") had to replicate Step 4b's pair-swap by hand under a
           grant, because the reviewed mechanism existed only inside the
           scheduled retrain->gate->promote chain. Two changes:
           (1) scripts/fallback_pair_promote.py — the Step 4b swap dance
           extracted VERBATIM (stamp-license check, atomic model swap
           with .previous rollback, calibrator pairing with cleanup);
           Step 4b now calls it, so there is exactly ONE swap
           implementation (a guard asserts no def _swap_into_active
           remains in the wrapper and exactly two call sites).
           (2) weekly_wf_promote.sh --promote-staged <RUN_ID>: the SAME
           dual-contract arming check, the SAME fallback CLI --stamp as
           the decide gate (5 checks incl. no-downward-ratchet), the
           shared script, the VERBATIM sentinel emitter line, and ntfy —
           no training, no guard interaction, refuses a missing pair.
FOLLOW-UP REQUIRED (paired PR): the wrapper bytes changed, so the
           orchestrator emitter_contract must re-capture the wrapper sha
           + the new line numbers of the action templates (the #774
           dance). Until that merges+syncs, the drift-detection local
           test on the operator machine flags the contract mismatch —
           the designed reminder.
EVIDENCE:  rfc210 suite 15 passed (3 new mode guards; the two behavioral
           swap tests now exercise the SHARED script via the fixture
           repo, which gains it as a genuine copy); snapshot-backstop +
           daily-notify suites green; bash -n clean.
NEXT:      paired orch contract re-capture PR; then the mode is the
           documented answer to "promote an existing staged candidate".
