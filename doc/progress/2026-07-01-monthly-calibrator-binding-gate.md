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
