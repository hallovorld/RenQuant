# 2026-07-20 — GOAL-5 AC5 (D2): daily-entrypoint import check in the pin sweep

## Bottom line
Closes a verified gap in the GOAL-5 AC5 integration-preflight CI: the
`pin-import-integrity` sweep proved *aliased-kernel* imports resolve, but was
structurally blind to a **non-aliased cross-repo import** in the daily run's
own entrypoint closure — the exact class that broke the 2026-07-19 g5 pin
advance. This PR extends the sweep to also import the daily entrypoint
modules post-bootstrap, so that break fails at PR time instead of at 06:25 in
production. No workflow edit needed: `pin-import-integrity.yml` already runs the
sweep, so coverage extends automatically.

## The gap (verified)
- `pin_import_integrity_sweep.py` AST-walks the **pinned pipeline** source for
  imports in aliased namespaces (`kernel.*` / `renquant_pipeline.kernel.*`)
  only (`collect_aliased_imports`).
- The g5 break was an **orchestrator daily.py** top-level
  `from renquant_orchestrator.g4_admission import …` →
  `from renquant_pipeline.decision_schedule import …`, where
  `decision_schedule` was absent from the deployed pipeline pin. That import is
  (a) in the orchestrator, not the pipeline, and (b) in a plain namespace, not
  `kernel.*` — invisible to the aliased sweep.
- `make subrepo-daily-contract` caught it locally (it module-loads the daily
  entrypoint), but **no CI runs daily-contract** — so nothing would have failed
  the coordinated g5+g4 lock PR before it merged.

## The fix (surgical, one script + test)
- `check_daily_entrypoint(DAILY_ENTRYPOINT_MODULES)` imports
  `renquant_orchestrator.daily` + `renquant_orchestrator.cli` in-process after
  `bootstrap_multirepo`. Python's own import machinery walks the full
  module-level closure, so a missing cross-repo module surfaces as an
  ImportError finding (naming the module + fix side). Called from `sweep()`;
  result gains `n_entrypoint_modules`.
- `subrepo_daily_contract.py` was not reusable in CI directly: it resolves
  `SUBREPO_ROOT` to the deployed `.subrepo_runtime`, not the PR's candidate
  sibling checkouts. The sweep already bootstraps from `--siblings` at the
  candidate lock, so the entrypoint import there checks the *candidate* pins.

## Verification
- `py_compile` OK. Pure unit tests 7/7 (4 AST + 3 new `TestCheckDailyEntrypoint`).
- **No false positive on the current green lock**: `renquant_orchestrator.daily`
  and `.cli` both import cleanly at the current deployed pins (smoke-tested
  against `.subrepo_runtime` srcs).
- Would catch g5: on a lock PR advancing the orchestrator pin to the g5 daily.py
  while the pipeline pin lacks `decision_schedule`, importing
  `renquant_orchestrator.daily` raises ImportError → sweep FAIL → CI red.

## Scope
Extends the existing AC5 D1 sweep; no behaviour change to the daily run, no new
workflow. Blast radius = CI-time preflight only. Related lesson:
[[live-tree-mutation-preflight-required]] — "funnel-inert" must include the full
import tree, now enforced mechanically at PR time.
