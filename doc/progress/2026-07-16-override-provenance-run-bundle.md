# Fix: persist diagnostic-only admission provenance in the run bundle

STATUS: delivered
WHAT: The adversarial re-review of pipeline#203 (orch issue #526) found a
MED audit gap: capital admitted under the governed diagnostic-only
operator override left no durable trail — the provenance existed only in
log/ntfy text (verified: run 2026-07-16-live-a24a8be1's run_bundle_json
has no override keys). The pipeline scoring path already records the
admission verdict on `ctx._regime_model_admission` and deliberately
preserves the override provenance across later admission stages;
`build_run_bundle()` now persists that record as
`bundle["regime_model_admission"]`. Fail-safe: non-dict/empty/absent
records are omitted (truthful absence, a malformed record can never break
a run); values pass through `_json_safe` for run_bundle_json fidelity.
WHY/DIR: GOAL-5 P0 (daily-run reliability) — every run must durably state
under which authorization buys were admitted; log text is not an audit
surface.
EVIDENCE: 4 new tests (provenance lands; absent stays absent;
malformed fail-safe; json round-trip) + existing suites =
tests/test_artifact_contract.py 16/16 pass (full kernel sandbox with the
pinned renquant-common on PYTHONPATH).
NEXT: none for this surface. (Preflight-side details remain in the
preflight check output as before; the scoring-path record is the
authoritative admission fact.)
