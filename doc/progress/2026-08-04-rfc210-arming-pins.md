# 2026-08-04 — RFC#210 arming: the held provider/consumer pins advance together

## Bottom line

This PR ARMS the RFC#210 freshness fallback by advancing the two pins that
PR #560 deliberately HELD, plus the two pins the fast-momentum Saturday lane
needs. After this merges and the live tree syncs, Step 4b in
`weekly_wf_promote.sh` finds both contracts satisfied (provider importable +
consumer vocabulary present) and stops refusing.

## Pin advances (4)

| subrepo | from | to | carries |
|---|---|---|---|
| renquant-backtesting | `8f6700ab` | `ea7b014a` | bt#102 — RFC#210 `freshness_fallback` provider (decide/stamp) |
| renquant-orchestrator | `ade07dd7` | `7e933c07` | orch#774 — sentinel action vocabulary + emitter contract for `FALLBACK-PROMOTED`; plus orch#775 fast-CLI wrapper step |
| renquant-model | `dec37193` | `96fe2d3d` | model#200+#201 — `params_v1_fast` + train CLI `--params-version` (Saturday fast step needs this pin or it exits 2 loudly) |
| renquant-strategy-104 | `320ed77c` | `e8fd07e9` | s104#84 — `momentum_fast_v1_shadow` dormant-by-declaration entry |

HELD unchanged: renquant-artifacts (F-7 canonical-snapshot constraint,
untested vs artifacts#31/#32), renquant-common, renquant-pipeline,
renquant-execution, renquant-base-data.

## Why together (pin-level arming order, from #560)

PR #560 advanced 4 pins and HELD backtesting + orchestrator **as a pair** so
the fallback could not half-arm: a provider without the consumer vocabulary
would fallback-promote silently past the sentinel; a consumer without the
provider would hard-refuse every Saturday. RQ#559's Step 4b enforces this as
a dual-contract check at run time; this PR is the pin-level half of the same
invariant. Both advance in this single commit.

## Prereqs, all merged before this PR

- bt#102 (provider), RQ#559 (Step 4b wiring, wrapper sha `f779f932fe352864`),
  RQ#560 (first pin wave), orch#774 (consumer vocabulary), orch#775 +
  model#200/#201 + s104#84 (fast lane), pipeline#259 (momentum primary
  surface, already pinned via #560).
- orch#776 (sentinel third watched lane for `momentum_fast_v1_shadow`) is
  in re-review; it is observability-only and NOT load-bearing for arming.
  Its exact-equality parity guard is already committed on the PR branch.

## After merge (deployment steps, logged in the grants trail)

1. Live umbrella `git pull --ff-only` (read-only preflight first).
2. Runtime sync of the four repos under `.subrepo_runtime/repos/` to the
   new pins.
3. Step 4b arming verification: provider import under the pinned runtime ✓,
   consumer emitter-contract line ✓, then a decide() dry-run against the
   real staging artifact expecting `FALLBACK_PROMOTE`.

## Review round 1 (codex): the candidate-pin artifact gate

The gate (`scripts/check_config_artifact_paths.py`) correctly failed closed
on the new s104 pin: its ledger-pointer admission was restricted to the v0
momentum contract, so `shadow_models[2]`'s fast ledger path was rejected
before deployment. Fix in this PR (umbrella-side, because this gate
validates the pinned cross-repo surface):

- `_MOMENTUM_FAST_LEDGER_REF` admitted ONLY while the entry carries its
  `*_pending_first_artifact` marker (the s104#84 dormant declaration).
- Marker removed (post-first-publish) → fail closed again, even for a
  valid resolvable fast ledger: widening to the full fast serving contract
  is a reviewed change (mirror of s104#78 for the v0 lane).
- Marker present but the ledger resolves anyway → full chain verification
  still runs (the marker never skips verification).
- Any other JSONL path: fail-closed, unchanged.
- 4 new tests; file suite 40 passed, 1 skipped (hermetic fake-contract
  skip on this machine).

## Rollback

Revert this commit; the lock returns to the #560 state where Step 4b
refuses loudly (the safe DISARMED state verified on 2026-08-03).
