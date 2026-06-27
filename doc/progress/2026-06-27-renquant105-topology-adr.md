# Progress: renquant105 cross-repo topology ADR

**Date:** 2026-06-27
**Status:** Proposed (PR open for Codex review)

## What

Authored ADR 0001 (`doc/arch/adr/0001-renquant105-intraday-topology.md`) as the
single authoritative record of the renquant105 intraday cross-repo topology change,
and registered the Proposed `renquant-strategy-105` repo in
`subrepos.lock.json` under `pending_subrepos`.

## Why

Codex's holistic review of orchestrator PR #198 (finding #8) flagged that the
cross-repo architecture (new `renquant-strategy-105` repo, forbidden imports,
artifact contracts across base-data/model/pipeline/execution/backtesting, lock/pin
migration) was being *defined* inside an orchestrator subrepo PR. Per the canonical
operating model (`doc/arch/subrepo-operating-model.md`), cross-repo architecture
must live ONCE under `RenQuant/doc/arch/` and be referenced, not copied into a
subrepo. PR #198 is now re-scoped to ORCHESTRATION and references this ADR; this
ADR supersedes #198's cross-repo scope and **must land first** — until then,
strategy-105 is not created and no pin order runs.

## Key decisions captured in the ADR

- New `renquant-strategy-105` repo: policy/config-only (config skeleton,
  point-in-time universe manifest, config fingerprint); no data/model/broker
  internals.
- Forbidden imports / dependency direction unchanged (orchestrator boundaries from
  CLAUDE.md hold; strategy-105 imports common contracts only; consume models by
  `artifact_path`).
- Artifact fingerprint/handoff chain model→pipeline→execution→backtesting mirrors
  104; pipeline preflight fail-closes on universe/config-fingerprint mismatch.
- Lock/pin migration: registered as Proposed in `pending_subrepos` (no
  `RENQUANT_REPOS.md` doctor drift — the generated map renders only `subrepos[]`);
  pin order base-data → pipeline → model → strategy-105 + orchestrator last; bridge
  routes by `--strategy` (no orchestrator code change).
- Rollback: un-pin to last-known-good 104-only assembly; default-OFF
  (`intraday_buys_enabled=false`); `# COMPAT-105-SHIM` retirement rule.
- CI gate: existing umbrella doctor/test/smoke + a cross-repo integration test that
  proves the 105 contract end-to-end before any 105 pin is production.

## Notes

- First ADR in the repo; establishes the `doc/arch/adr/NNNN-<slug>.md` convention
  (prior arch docs are flat descriptive files; there was no ADR convention).
- `RENQUANT_REPOS.md` is auto-generated from the lock and intentionally NOT
  hand-edited; the durable, machine-readable registration of the Proposed repo is
  the `pending_subrepos` lock entry the ADR references.
