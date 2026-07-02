# Fix: monthly calibrator refresh gate missed the scorer/calibrator BINDING check

2026-07-01.

## Symptom
`monthly_calibrator_refresh.sh` (launchd, 1st-of-month 03:00) fits a new calibrator via
`fit_calibrator_alpha158_fund.py`, then only validates it with the pool_ic / n_unique_prob_y
quality-regression gate (Step 3). It never checked whether the newly-fit calibrator's stamped
`scorer_model_content_fingerprint` actually matches what the live runtime computes for the
active scorer — the exact contract `_assert_calibrator_matches_scorer` enforces in
`renquant-pipeline`'s `job_panel_scoring.py`. A calibrator that would fail-closed the live
daily-full at runtime could still pass this gate and get silently published. This is what
happened today (2026-07-01): 3 divergent `model_content_sha256` implementations
(renquant-model's fit-time allowlist vs renquant-pipeline's runtime-authoritative denylist)
meant a calibrator fit by the former could never bind to the latter, by construction.

## Root cause fix (in flight, separate repos)
- renquant-common#18 — extracts the runtime-authoritative `model_content_sha256` into a shared
  `renquant_common.model_fingerprint` module.
- renquant-pipeline#155 / renquant-model#40 — both consumers import the shared function instead
  of hand-copying it.

## This fix — defense-in-depth on the gate itself
The root-cause fix stops the 3 implementations from diverging again, but the monthly gate still
had no check for the failure mode itself — any future re-divergence, or any other cause of a
scorer/calibrator mismatch, would ship silently the same way. Added:

- `scripts/verify_calibrator_scorer_binding.py` — a new, independently testable Python module.
  `check_binding(scorer_path, calibrator_path)` loads the active scorer via the
  runtime-authoritative `PanelScorer.load` and runs the SAME match logic the runtime uses
  (`_any_fingerprints_match` / `_fingerprint_values`, imported from
  `renquant_pipeline.kernel.panel_pipeline.job_panel_scoring` — not reimplemented, so this
  gate can never independently drift from the runtime check the way the fit-time/runtime
  `model_content_sha256` copies did). Works regardless of whether renquant-common#18 /
  renquant-pipeline#155 have landed, because it calls the stable public loader entry point, not
  the internal hash function those PRs are unifying. Fails CLOSED (exit 2) if the
  runtime-authoritative modules aren't importable — never a silent skip, since a
  silently-skipping check is exactly the failure mode that let today's incident through.
- `scripts/monthly_calibrator_refresh.sh` Step 3b — runs the above between Step 3's quality gate
  and Step 4's dashboard refresh. On mismatch or fail-closed error: same atomic rollback path as
  the existing Step 3 gate (`$ROLLBACK_CAL` → smoke-test the rollback → `notify`), with an
  `ntfy` body that explicitly distinguishes "BINDING MISMATCH" from the pool_ic/n_unique quality
  gate so an operator reading the alert immediately knows which of the two gates fired.
- PYTHONPATH: added `renquant-pipeline` to the subrepo path list (previously only
  `renquant-model renquant-common renquant-base-data renquant-artifacts` — the same 4 repos
  `daily_104.sh`/`weekly_wf_promote.sh` already extend with `renquant-pipeline` for this exact
  reason).
- `tests/test_verify_calibrator_scorer_binding.py` — 9 unit tests against the extracted module
  (matching pair, mismatched pair, missing fingerprints, missing artifacts, loader exceptions,
  import failure fails closed) using fixture injection — no strategy venv needed. Also verified
  end-to-end against the real live `hf_patchtst` scorer + shadow calibrator artifacts (read-only,
  from a separate PYTHONPATH-injected invocation, never touching the live tree) for all three
  exit codes (0 pass / 1 mismatch / 2 import-failure).

## Verified
- `bash -n scripts/monthly_calibrator_refresh.sh` — OK.
- `pytest tests/test_verify_calibrator_scorer_binding.py tests/test_monthly_calibrator_acceptance.py`
  — 17 passed (existing Step 1-3 acceptance-gate tests untouched and still passing).
- `git diff --check` — clean.

## Scope discipline
Gate-only change. Steps 1/2/3 (smoke test, fit, pool_ic/n_unique regression) are untouched —
this is an additional gate, not a replacement.

## REVIEW FIX ROUND 2 (PR #425 CHANGES_REQUESTED, Codex)

Not actually a pre-publish gate: `fit_calibrator()` wrote the new candidate DIRECTLY to
`PROD_CAL` (the live production path) BEFORE Step 3/3b's validation ran. During the
fit-to-validation window the live runtime could read the new, unvalidated/mismatched
calibrator — a real exposure window, not merely a "roll back after the fact" bug. Separately:
if no prior calibrator existed (first-ever fit) and validation failed, the failure branches
alerted + exited but left the REJECTED artifact sitting at `PROD_CAL` — so a rejected artifact
remained published in the no-baseline case. Rollback-after-exposure ≠ atomic admission.

### Fix — stage, validate, atomically publish
- `scripts/monthly_calibrator_atomic_swap.py` (new) — extracted, independently-testable
  staging/publish module (same convention as `verify_calibrator_scorer_binding.py`):
  `sha256_file`, `atomic_publish` (re-verifies the candidate's digest against what was
  gate-checked — TOCTOU guard — then `os.replace()`s staging onto prod; both checks happen
  BEFORE any filesystem mutation), `quarantine_staging` (never touches prod — doesn't even
  accept a prod path — moves a rejected candidate to `_rejected_calibrators/`), and
  `build_receipt`/`write_receipt` (binds the checked scorer path/fingerprints + the exact
  candidate sha256 into an acceptance receipt).
- `scripts/monthly_calibrator_refresh.sh`:
  - Step 2: `fit_calibrator()` now writes to `$STAGING_CAL` (`${PROD_CAL}.staging-${RUN_ID}.json`
    — same directory as `PROD_CAL`, required for the same-filesystem atomic rename), never
    `PROD_CAL`. `$CANDIDATE_SHA256` captured right after fit.
  - Step 3: the old "rerun `smoke_test_model.py`" check would have been a no-op duplicate of
    Step 1 once fit no longer touches `PROD_CAL` (same scorer, same untouched calibrator) — it's
    replaced by an inline load+map check directly against `$STAGING_CAL` (two distinct synthetic
    scores, same collapse-detection invariant `smoke_test_model.py`'s scorer check uses). The
    pool_ic/n_unique quality gate now reads `$STAGING_CAL`.
  - Step 3b: binding check now runs against `$STAGING_CAL` instead of `$PROD_CAL`.
  - Step 3c (new): atomic publish — only after 3 AND 3b pass, calls `monthly_calibrator_atomic_swap.py
    publish` with the scorer identity/fingerprints from Step 3b's binding verdict + the captured
    digest. `PROD_CAL` is touched exactly once, by a single `os.replace` syscall.
  - Every failure branch (fit / staged-smoke / quality gate / binding gate / publish itself) now
    calls `monthly_calibrator_atomic_swap.py quarantine` instead of the old
    expose-then-`cp`-back-over-`PROD_CAL` rollback — there is nothing to roll back since
    `PROD_CAL` was never written. The pre-refit `ROLLBACK_CAL` backup is kept as an archival,
    operator-reference-only dated snapshot (same convention as `weekly_wf_promote.sh`'s Step 2
    backup), no longer wired into any automated restore path.
- `tests/test_monthly_calibrator_atomic_swap.py` (new, 19 tests) — unit tests for
  sha256/receipt/publish/quarantine, an INTEGRATION test with a concurrent background-thread
  read hook proving `PROD_CAL` bytes never change during a simulated failing run (both a
  gate-failure and a TOCTOU-publish-failure variant), the first-install (no baseline) failure
  case leaving no production artifact at all, the successful staging→prod swap case, explicit
  TOCTOU digest-mismatch protection, and CLI-level tests for all three subcommands
  (`sha256`/`publish`/`quarantine`).
- `tests/test_monthly_calibrator_acceptance.py` — rewritten: dropped the now-obsolete
  expose-then-rollback assertions, added regression guards that (a) the legacy
  `cp "$ROLLBACK_CAL" "$PROD_CAL.tmp" && mv ...` pattern is gone entirely, (b) every gate reads
  `$STAGING_CAL` not `$PROD_CAL`, (c) every failure branch quarantines staging and tells the
  operator production was untouched, (d) atomic publish runs strictly after both gates.

### Verified
- `bash -n scripts/monthly_calibrator_refresh.sh` — OK.
- `python3 -m py_compile scripts/monthly_calibrator_atomic_swap.py` — OK.
- `pytest tests/test_verify_calibrator_scorer_binding.py tests/test_monthly_calibrator_acceptance.py
  tests/test_monthly_calibrator_atomic_swap.py tests/test_monthly_jobs_multirepo_fail_closed.py
  tests/test_subrepo_ops_contract.py tests/test_p0_fixes_regression_guards.py::TestP0_16_AtomicCp`
  — 76 passed.
- `git diff --check` — clean.
- Manual end-to-end simulation of the `sha256` → `publish` CLI pair against fixture staging/prod
  paths (see session log) — confirms the wiring `monthly_calibrator_refresh.sh` depends on.
- NOT re-verified: `tests/test_smoke_test_model.py::TestSmokeRuns::test_smoke_test_passes_on_production_artifact`
  and the QP/PanelScorer P0 tests fail in this sandbox with `ModuleNotFoundError: No module named
  'xgboost'`/`'pydantic'` — pre-existing environment gaps (no project `.venv`, bare `python3.9`),
  unrelated to this change (neither file is touched by this diff).

### Scope discipline
Atomicity/publish-ordering fix only. The pool_ic/n_unique quality gate's thresholds, the binding
check's match logic, and Step 1's pre-flight smoke test are all unchanged — only WHERE/WHEN
`PROD_CAL` gets written changed.
