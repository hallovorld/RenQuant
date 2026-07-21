# Progress — AC5 execution-funnel preflight (pin-bump gate closer)

**Date:** 2026-07-20
**Goal:** GOAL-5 P0 AC5 (month-1). **Type:** CI gate extension (shadow rollout).

## STATUS
AC5's acceptance VERIFY clause was already met by #519 (the #524-namespace-gap
fixture FAILS the import-integrity gate). This closes the remaining gap vs the
literal "full readonly inference FUNNEL" wording: the sweep proves imports
RESOLVE; this proves the decision Tasks EXECUTE.

## WHAT
Added one step to `.github/workflows/pin-import-integrity.yml`, after the import
sweep: run `scripts/subrepo_daily_contract.py --broker-type paper` (no
`--execute`) against the same candidate pin combination the sweep already
assembles under `siblings/`, via `RENQUANT_SUBREPO_ROOT`. This drives the real
synthetic scoring→selection→execution funnel through the actual
`renquant_pipeline` / `renquant_execution` / `renquant_strategy_104` package
contracts.

## WHY-DIR
#524 was a cross-repo lazy `renquant_pipeline.kernel.*` submodule that imported
fine per-repo but died mid-daily when the Task ran. #519's import sweep catches
the import-resolution class; a break that only manifests on Task *execution*
(a contract mismatch, a runtime signature drift) still passes imports. The
daily-contract funnel is the one existing piece of infra that runs a real
(no-network, synthetic-fixture) funnel end-to-end — reused here against the
candidate pins. No new machinery; env-redirect + one step.

## SHADOW ROLLOUT (protect production)
`continue-on-error: true` — the step runs and reports but does NOT block
pin-bumps yet. It gates nothing until it has run green across a few real
pin-bump PRs; a follow-up removes `continue-on-error` to make it a hard gate.
This avoids a spurious CI-environment issue freezing all pin-bumps
([[fix-wave-protect-production]], [[never-deploy-inert-scaffolding]] — shadow
here is a deliberate staged-enforce, not dark scaffolding).

## EVIDENCE
Validated the funnel locally before wiring: `subrepo_daily_contract.py
--broker-type paper` exits 0, emits a full run bundle (decision_trace /
stage_trace / order_intents) with a dry-run BUY, 0 tracebacks;
`resolve_subrepo_root` honors `RENQUANT_SUBREPO_ROOT` (scripts/subrepo_paths.py:55);
the workflow already installs full deps (requirements.lock + `--no-deps -e` each
subrepo) so the funnel has its runtime deps. `[VERIFIED — local funnel exit 0 +
env-redirect resolves; CI step observed via this PR's own run]`

## NEXT
- Watch this PR's `pin-import-integrity` run: the new step should pass against
  current pins (shadow, so a hiccup won't fail the PR — read the step log).
- After a few green pin-bumps, follow-up PR flips `continue-on-error` off → AC5
  full-funnel gate is hard-enforced. That closes GOAL-5 AC5 completely.
