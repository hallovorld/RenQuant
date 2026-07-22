# Progress — `live.runner --preflight` dry-run mode (GOAL-5 AC5)

**Date:** 2026-07-22. **Type:** runner contract addition (read-only probe).
**Pairs with:** orchestrator PR #565 (dawn shell guard + attestation verifier).

## STATUS:
Implemented and tested here (12/12 passing). Not yet merged — `renquant-orchestrator`#565
already pushed its consumer side (`--preflight` shell invocation +
`dawn_preflight_attest.py`) expecting exactly this attestation contract, so
#565 is blocked on this PR merging first, not the other way around.

## WHAT:
`live/runner.py`: `--preflight` flag → threaded as `dry_run` →
`RunnerAdapter(preflight=True, preflight_guard=…)`. A process-wide
`PreflightGuard` records whether any mutation/notify boundary is hit.
- Adapter: opens NO runs DB (`self._db=None` ⇒ `ScoreDistributionJob` no-ops,
  no `data/runs_*.db` created); meta-label capture forced off; `commit()`
  refuses + flips the guard if ever entered (defense in depth).
- Runner: skips `commit()` and `_notify_decision()` in dry-run; `_post_ntfy_
  with_retries` suppresses+records any send while the guard is active; emits
  `preflight_attestation: {persisted,notified,promoted,ordered,reached_decision}`.
- `reached_decision:true` ONLY after `pipeline.run()` completes; any
  preflight-contract failure ⇒ attestation with `reached_decision:false`
  (the shell guard fails closed → the daily-killer is surfaced 8h early).

## WHY/DIR:
codex CR on orch #565: the dawn preflight ran `live.runner --once --broker
readonly-alpaca` as a "read-only probe". `readonly-alpaca` only constrains
BROKER writes — `--once` can STILL open/create the runs DB, allocate a run id,
persist `live_state`, run the score-distribution DB writer, and emit ntfy. Not a
safe operational probe.

Side-effect map (verified before implementing, not guessed):
- Orders (`place_order`), all `record_*` DB writers, `save_live_state_atomic`,
  L6 score-audit sidecar, trade-log `write_text` → ALL inside
  `RunnerAdapter.commit()` (the single write chokepoint). `make_context` and
  `LoadUniverseJob` write nothing.
- Only pre-commit DB write: `ScoreDistributionJob` (guarded on
  `score_db.enabled` AND `ctx._db is not None`).
- DB file creation: `get_connection` in adapter `__init__`.
- Meta-label parquet: gated on `meta_label_training.enabled`.
- Notifications: preflight-fail ntfy + `_notify_decision`, both via
  `_post_ntfy_with_retries` (single send chokepoint).

Guarding at these chokepoints (rather than skipping `commit()` at the
`live/runner.py` call site alone) closes the DB-open gap: if pre-commit setup
ever opened a runs DB, a call-site-only skip would miss it. The `PreflightGuard`
is defense-in-depth — `commit()` itself refuses and flips the guard if ever
entered, so a future regression that accidentally calls `commit()` in
preflight mode is caught by the attestation, not silently shipped.

## EVIDENCE:
`[VERIFIED]`
- `tests/test_runner_preflight_dry_run.py` (6): dry-run reaches decision, commit
  never called (isolated sentinel untouched), no ntfy, clean attestation; a
  stray notify flips `notified:true`; preflight-contract failure ⇒
  `reached_decision:false` + `SystemExit(2)`; active guard reset after run.
- `tests/test_runner_preflight_adapter.py` (3): real adapter under `--preflight`
  opens no `runs*.db` under an isolated path; non-preflight DOES create it
  (contrast); `commit()` refuses and flips persisted/ordered/promoted.
- Existing `tests/test_runner_preflight_fail_closed.py` (3) still green.
All 12 pass (with `renquant-pipeline/src` on `PYTHONPATH`, as the suite requires).

## NEXT:
- Merge this, then `renquant-orchestrator`#565's already-pushed
  `--preflight` + `dawn_preflight_attest.py` wiring is immediately consumable
  — no further consumer-side change needed, just re-verify against a real
  dawn run once both are live.
- Deploy: sync the live umbrella checkout to `main` after this + #565 land
  (launchd runs the live tree, not a pin).
