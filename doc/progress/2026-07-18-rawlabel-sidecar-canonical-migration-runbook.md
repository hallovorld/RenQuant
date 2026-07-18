# Progress: rawlabel sidecar canonical migration runbook (AC-D)

Date: 2026-07-18
Branch: `feat/rawlabel-sidecar-canonical-migration-runbook`
Author: hallovorld · Reviewer: haorensjtu-dev

## What shipped (script + tests only — no migration executed)

The umbrella runbook for AC-D of the sidecar single-writer amendment
(`renquant-base-data/doc/design/2026-07-18-sidecar-single-writer-amendment.md`).
Companion to Stage 1 (base-data#49, exposes `RAWLABEL_SIDECAR_COLUMNS`, 179-col)
and Stage 2 (orchestrator#553, σ-head writer → fail-closed consumer, MERGED).

- `scripts/migrate_rawlabel_sidecar_to_canonical.py` — one-time supervised
  regeneration of the served sidecar to the canonical contract (179-col,
  sentiment-carrying, zero bar-frontier extension rows) via the Stage-1
  canonical builder. AC-D integrity: BEFORE/AFTER snapshots (builder revision,
  input fingerprints, sha256 + schema digest, row count, PK/date coverage,
  retained-column checksum); asserts the diff is ONLY the intended contract
  change; atomic `.bak` swap; hash-verified `--rollback`; `--dry-run`; and a
  `--preflight` deployment-ordering gate. NEVER runs automatically — exactly one
  mutating `--execute` mode, which writes a containment record (CLAUDE.md §5).
- `scripts/ac_c_sidecar_dryrun_harness.py` — AC-C Saturday-chain dry-run
  (refresh → REAL refresh guard → non-promoting retrain prep) against a SANDBOX
  copy; asserts the guard passes and the former `dropped columns` rejection no
  longer fires.
- Tests (all temp/sandbox, never the live file): `tests/test_migrate_rawlabel_
  sidecar_to_canonical.py` + `tests/test_ac_c_sidecar_dryrun_harness.py` —
  **31 tests, all green**.

## Ordered landing sequence (the §2 hazard)

The deployed #553 consumer fails closed with no canonical file, so:
**(a)** run the migration (`--dry-run` → `--execute`; `--preflight` passes) →
**(b)** pin-bump base-data#49 + orchestrator#553 (gated on `--preflight`) →
**(c)** AC-C sandbox dry-run + served-file digest watch →
**(d)** retire the `weekly-retrain-patchtst` sentinel ack after the first green
Saturday (amendment AC-E).

Full detail: `doc/ops/2026-07-18-rawlabel-sidecar-canonical-migration-runbook.md`.

## Not in this PR (operator-gated)

The destructive migration is NOT executed here. Running
`--execute` against the live served file is a later ask-first landing action
(live-tree mutation preflight). The live served sidecar was hashed read-only for
reference only.

## Failure signature this closes (AC-E)

`weekly_retrain_patchtst` Saturday failure `staged corpus dropped columns
(recipe/schema drift): ['mean_sentiment', 'n_articles_log',
'sentiment_pos_share']`. The sentinel ack is retired after the first green
Saturday post-landing.
