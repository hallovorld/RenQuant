# Progress: pin bump — rawlabel sidecar single-writer unification

Date: 2026-07-18

## What

`subrepos.lock.json` advances two pins only (minimal diff, formatting
preserved):

- `renquant-base-data` `0678958e` → `021ca647` — Stage 1 of the
  single-writer amendment (base-data#48 §2): the canonical 179-col,
  sentiment-carrying, zero-bar-frontier-extension rawlabel builder becomes
  the SOLE writer of the served
  `alpha158_291_fundamental_dataset_rawlabel.parquet` (base-data#49).
- `renquant-orchestrator` `8c0acd5f` → `ade07dd7` — Stage 2 (base-data#48
  §2.1): `RefreshSigmaHeadRawLabelTask` is retired as a writer and becomes a
  fail-closed CONSUMER of the canonical file (orchestrator#553), plus today's
  other merged orchestrator work rolled into the same main HEAD.

`renquant-strategy-104` is UNCHANGED (`082dccd2`), so the production scorer
declaration is untouched. `doc/arch/strategy-104-snapshot.md` is regenerated
(see below) — the change is confined to the lock-derived lines.

## Why — clears the recurring Saturday weekly-retrain deadlock

The served sidecar had TWO weekly writers with contradictory recipes: the
base-data builder and the orchestrator's σ-head refresh. That writer war
deadlocked the weekly PatchTST corpus refresh (07-11 / 07-18 Saturday
alarms): whichever writer ran second could leave the file at a column shape
the other side's guard rejected, so the staged weekly build stalled.

- **base-data#49 (Stage 1)** makes the weekly staged build emit the canonical
  179-col file directly → the downstream 179/sentiment guard passes instead
  of tripping. This bump ALONE clears the deadlock.
- **orchestrator#553 (Stage 2)** is the durable single-writer fix: the σ-head
  task stops opening the served sidecar for write and instead verifies (exact
  `(ticker,date)` lockstep with the fresh panel), fail-closes on the 179
  contract at the consumption boundary (176-col → REFUSED; any bar-frontier
  extension row → REFUSED), then certifies provenance. One writer, one recipe.

## Preflight — GREEN (verified before this PR)

- **Live served sidecar is already logically-canonical 179-col** — the two
  writers had converged on the 179 shape in practice, so the bumps normalize
  the CONTRACT, not the live bytes.
- **base-data#49 bump alone clears the deadlock** (staged weekly build →
  179-col → guard passes).
- **Full-funnel daily-contract sim: exit 0, 1 BUY** (AAPL, dry-run;
  `submitted_orders.json` / `run_bundle.json`) — daily buy path unaffected, no
  zero-buy regression.
- **Subrepo suites green:** base-data 458 tests, orchestrator 4095 tests.
- **Both pin advances are fast-forward-safe:** old pin is a strict ancestor of
  new (base-data ahead 24 / behind 0; orchestrator ahead 6 / behind 0). Both
  target commits are the real #49 / #553 merge commits at their repos'
  `origin/main` HEAD with all check-runs `success`.

## Snapshot regeneration

`doc/arch/strategy-104-snapshot.md` was regenerated with
`scripts/render_strategy_104_snapshot.py` against a pin-consistent assembly
(strategy-104 checked out at the lock pin `082dccd2`, umbrella-committed
artifacts, new lock). Because strategy-104 is unchanged, the diff is confined
to the lock-derived lines ONLY:

- Subrepo-pins table: base-data `0678958ec2f5` → `021ca6474102`,
  orchestrator `8c0acd5f58ce` → `ade07dd797b0`.
- `subrepos.lock.json` source fingerprint `bd5e251ad484` → `04851d22d8c5`
  (in the fingerprints list and the machine block).
- Top-level `Source fingerprint:` recomputed (`ad55b653…` → `a0a66b80…`).

No scorer/calibrator/policy/artifact fingerprint moved. `--check` (byte-exact
idempotence) and `--verify-pinned-declaration` (the CI gate on lock bumps)
both pass; the machine block's `strategy_104_pin` stays `082dccd2`.

## DEPLOYMENT NOTE — AC-D migration is a benign post-landing run

The one-time supervised migration
`scripts/migrate_rawlabel_sidecar_to_canonical.py --execute` (shipped in
umbrella #509) is a **provenance-normalization** run to be executed
ask-first AFTER this lock lands on the machine. There is **no ordering
hazard**: the live served sidecar is already logically-canonical 179-col, so
the migration only normalizes the stamped provenance/contract, not the label
bytes; and orchestrator#553's fail-closed consumer preflight already passes
against the live file today. The migration is therefore a cleanup, not a
prerequisite for the σ-head consumer to run correctly.

## Landing protocol + revert (separate from this PR)

Merging this PR changes nothing on the machine (merged-is-not-deployed). The
operator performs the machine FF-land ask-first:

- `git merge --ff-only origin/main` on the live umbrella tree, then
  re-point the pinned subrepo runtime roots (`scripts/promote_pin.py bump
  --apply`, which syncs `.subrepo_runtime/repos/{renquant-base-data,
  renquant-orchestrator}` to the new commits and regenerate-and-compares the
  snapshot), then the ask-first AC-D migration + Saturday-chain dry-run.

**Revert:** re-point the two commits in `subrepos.lock.json` back to
`0678958e` / `8c0acd5f` and re-sync the subrepo-runtime roots
(`scripts/promote_pin.py revert --apply`). Because the target advances are
fast-forward, revert is a clean pin rollback with no history rewrite.
