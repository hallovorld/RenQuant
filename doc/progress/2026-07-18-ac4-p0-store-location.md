# AC4 migration P0: declared bundle-store location + index-proof store paths

Date: 2026-07-18
Spec: census `doc/design/2026-07-18-ac4-migration-census.md` §6 P0
("build, no live change") + RFC #492
`doc/design/2026-07-17-artifact-bundle-transactionality.md` §2.1 (store
alongside the flat pair) and §3 (phase commits revert cleanly, no
artifact surgery). Companion PR: renquant-artifacts (break-glass
operational against this declared location + idempotent store-init +
§2.4 alarm bound to the real drift-sentinel channel). Library layer
previously merged: artifacts#25/#26, pipeline#206, common#32, orch#547.

## Delivered (umbrella side of P0 — declaration only, nothing serves from it)

- `deploy/bundle_store_location.json` — the SINGLE declared real store
  root: `backtesting/renquant_104/artifacts/prod` (the flat pair's own
  directory; RFC §2.1 layout — `bundles/` + `ACTIVE` stand up alongside
  the pair). Consumed by `renquant_artifacts.bundle_store_location`
  (precedence: explicit `--store-root` > `RQ_BUNDLE_STORE_ROOT` >
  this file under `$RQ_ROOT`). Changing the value is a reviewed
  deployment change.
- `.gitignore` — `prod/bundles/`, `prod/ACTIVE`, `prod/ACTIVE.tmp` can
  never enter the git index. A git-tracked store would recreate census
  blocker B1 (git checkout/reset as an unmediated pair writer); keeping
  the store out of the index is the P0 down-payment on the P3 untrack.
- `tests/test_ac4_p0_store_declaration.py` — 3 pins: (1) declaration
  well-formed and pointing at the directory that still CONTAINS the flat
  pair (P0 keeps the pair authoritative); (2) the three ignore patterns
  present; (3) zero-serving-change scan — no `kernel/`, 104
  `kernel|adapters|training_panel`, `scripts/`, or `dagster_renquant/`
  file references `prod/bundles`, `prod/ACTIVE`, or any
  `renquant_artifacts.bundle*` module. File-based only (no git
  subprocess, no network, no store creation).

## Census P0 conformance

Census §6 P0 items (publisher + operation log + `bundle_breakglass`,
`validate_pair`, common fixture, kill-injection CI) were already merged
as libraries. The remaining P0 gap — a REAL declared store location the
break-glass tool can operate against, stood up with zero serving change —
is exactly this PR + the artifacts companion. No reader or writer is
redirected (that is P1+/P2 per census §6); serving keeps reading the
flat pair through the census §1 config keys, untouched here.

## Rollback invariant (RFC §3)

`git revert` of this PR removes the declaration, the ignore rules, and
the test. No serving module was touched (pinned by the scan test), so
serving behavior is identical before/after/reverted — no artifact
surgery. If the landing step had already initialized the store skeleton
on the machine, those files are inert (nothing in any census surface
reads them), become plain untracked paths after the revert, and are
removed with `rm -rf backtesting/renquant_104/artifacts/prod/bundles` —
the flat pair is never touched.

## Landing steps (NOT executed by this PR — ask-first, one grant per batch)

1. Sync the live umbrella + renquant-artifacts checkouts to the merged
   mains (merged-is-not-deployed).
2. Run `python -m renquant_artifacts.bundle_store_init` (resolves this
   declaration; idempotent; creates ONLY `bundles/` + `bundles/.lock`
   alongside the pair — no `ACTIVE`, no `OPERATIONS.jsonl`, no flat-file
   reads or writes).
3. Verify: sha256 of both flat pair members unchanged; `git status` in
   the live tree shows NOTHING (store paths ignored); daily run-surface
   drift scan stays green (store content is outside its launchd/pin
   surface; untracked would anyway be info-only).

## Follow-ups owned elsewhere (documented, not reached across)

- P1 seal (publish the current pair as generation 1, ACTIVE +
  OPERATIONS.jsonl live) — next phase, separate design-conformant PR.
- Orchestrator drift-sentinel defense-in-depth: teach the daily scan to
  surface breakglass/RECOVERY records from `OPERATIONS.jsonl` once the
  store is live (P1+); the P0 alarm already lands on the sentinel's ntfy
  channel via `renquant_common.notify` (artifacts companion PR).
