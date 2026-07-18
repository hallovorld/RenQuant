# Runbook: rawlabel sidecar → canonical single-writer migration (AC-D)

Date: 2026-07-18
Owner: hallovorld (executes ask-first with the operator)
Status: SCRIPTED — the destructive migration is NOT executed by this PR. This PR
ships the runbook + scripts + tests only. Running the migration against the live
served sidecar is a later, operator-gated landing action.

Amendment: `renquant-base-data` `doc/design/2026-07-18-sidecar-single-writer-amendment.md`
(AC-C / AC-D). Stage 1 builder: base-data#49 (exposes `RAWLABEL_SIDECAR_COLUMNS`,
179-col, sentiment-carrying, extension-free default). Stage 2 σ-head cessation:
orchestrator#553 (MERGED — the σ-head becomes a fail-closed CONSUMER).

## 1. Why

The served sidecar `data/alpha158_291_fundamental_dataset_rawlabel.parquet` had
TWO active weekly writers with contradictory recipes — the base-data builder and
the orchestrator σ-head refresh. That writer war deadlocked the weekly PatchTST
corpus refresh (07-11 / 07-18: `staged corpus dropped columns … sentiment`). The
amendment resolves it to ONE file, ONE writer: the base-data builder is the SOLE
producer; the canonical contract CARRIES the three sentiment columns (**179
cols**) and, for this artifact, DROPS the bar-frontier extension rows.

This migration is the one-time supervised regeneration that makes the live
served file provably canonical, so the weekly guard stops rejecting and the
σ-head consumer (#553) finds a canonical file present.

## 2. Deployment ordering — the §2 hazard (READ THIS)

orchestrator#553's deployed consumer FAILS CLOSED if the canonical file is
absent. Therefore the migration MUST produce the canonical file BEFORE any
pin-bump deploys #553. Landing sequence, in order:

1. **(a) Run the migration** — `--dry-run` then `--execute`. The canonical file
   is now present and passes `--preflight`.
2. **(b) Pin-bump** deploying base-data#49 (179-col builder) + orchestrator#553
   (σ-head consumer). Gate this step on `--preflight` returning 0.
3. **(c) AC-C Saturday-chain dry-run** against a SANDBOX copy of the migrated
   file (`scripts/ac_c_sidecar_dryrun_harness.py`) + a served-file digest watch:
   the digest may change ONLY at the canonical builder's swap step.
4. **(d) Retire the `weekly-retrain-patchtst` sentinel ack** (amendment AC-E)
   after the first green Saturday.

> The migration in (a) uses the canonical 179-col builder. If the operator's
> environment still resolves the stale 176-col builder (base-data pin predates
> #49), the migration's **builder-contract preflight REFUSES** — it will not
> write a 176-col file that #553 would then reject. Ensure the canonical builder
> is resolvable before (a) (it is merged on base-data `main`); (b) bumps the live
> PIN afterward.

## 3. Execution steps (operator-gated, ask-first)

The live served sidecar is a production input. Per CLAUDE.md §2/§5 and the
live-tree-mutation-preflight rule, this is an ASK-FIRST landing action with a
full-funnel preflight. Do NOT run it from a scheduled job.

Paths (defaults; override with `--served-path` / `--fund-panel` / `--ohlcv-dir`):

- served: `/Users/renhao/git/github/RenQuant/data/alpha158_291_fundamental_dataset_rawlabel.parquet`
- fund panel: `/Users/renhao/git/github/RenQuant/data/alpha158_291_fundamental_dataset.parquet`
- ohlcv dir: `/Users/renhao/git/github/RenQuant/data/ohlcv`

**Step 1 — dry run (no mutation).** Verifies the AC-D integrity contract; leaves
the served file untouched.

```
scripts/migrate_rawlabel_sidecar_to_canonical.py --dry-run \
    --report-out doc/ops/artifacts/rawlabel-migration-dryrun-<date>.json
```

Confirm the report shows: `after.columns == RAWLABEL_SIDECAR_COLUMNS` (179 in
order), `after.n_extension_rows == 0`, `diff.retained_columns_checksum_equal ==
true`, and only the intended `added_columns` (sentiment, if migrating from a
176-col file) / reorder.

**Step 2 — execute (atomic swap + containment).** Only after the dry run is
clean and the operator approves the landing:

```
scripts/migrate_rawlabel_sidecar_to_canonical.py --execute \
    --task-ref "<tracked task/issue>" \
    --owner "hallovorld" \
    --restore-condition "until first green Saturday retrain confirms the unified contract" \
    --report-out doc/ops/artifacts/rawlabel-migration-<date>.json \
    --containment-out doc/ops/artifacts/rawlabel-migration-containment-<date>.json
```

`--execute` backs the served file up to a timestamped
`.pre-canonical-migration-<runid>.bak`, records its sha256, `os.replace`s the
verified candidate into place, re-verifies the post-swap digest, and writes the
containment record. It REFUSES without `--task-ref` / `--owner` /
`--restore-condition`.

**Step 3 — ordering gate before the #553 pin-bump.**

```
scripts/migrate_rawlabel_sidecar_to_canonical.py --preflight
```

Exit 0 ⇒ canonical file present + canonical ⇒ the #553 pin-bump may proceed.
Non-zero ⇒ do NOT pin-bump.

## 4. Rollback (hash-verified)

Read `backup_path` + `backup_sha256` from the containment record, then:

```
scripts/migrate_rawlabel_sidecar_to_canonical.py --rollback \
    --backup-path "<backup_path>" \
    --expected-backup-sha256 "<backup_sha256>"
```

Rollback verifies the backup bytes hash to `expected-backup-sha256` BEFORE
restoring (a filename alone is not trusted), restores via `os.replace`, and
re-verifies the restored file's digest. A digest mismatch REFUSES and leaves the
served file untouched.

## 5. AC-D / AC-C traceability

| Requirement | Where |
|---|---|
| AC-D BEFORE/AFTER snapshot (builder revision, input fingerprints, sha256 + schema digest, row count, PK/date coverage, retained-column checksum) | `frame_snapshot`, `run_migration` |
| AC-D "only intended diff" (canonical columns ordered, zero extension rows, no fabricated rows, only extension rows dropped, retained checksums equal) | `assert_only_intended_diff` |
| AC-D atomic swap + `.bak` | `run_migration` (`--execute`) |
| AC-D hash-verified rollback | `_rollback` (`--rollback`) |
| §2 deployment-ordering gate | `ordering_preflight` (`--preflight`) |
| Containment record | `build_containment_record` (`--execute`) |
| AC-C refresh→guard→non-promoting-retrain-prep on a SANDBOX copy | `scripts/ac_c_sidecar_dryrun_harness.py` |

Tests: `tests/test_migrate_rawlabel_sidecar_to_canonical.py`,
`tests/test_ac_c_sidecar_dryrun_harness.py` (all against temp/sandbox files —
never the live served sidecar).

## 6. Safety invariants

- The live served sidecar is READ-ONLY in this PR; it is hashed for reference,
  never written.
- The script NEVER runs automatically and NEVER defaults to a mutating mode
  (exactly one of `--dry-run` / `--execute` / `--rollback` / `--preflight`).
- `--execute` is the only mutating mode; it writes a containment record in the
  same action batch (CLAUDE.md §5).
- The ongoing weekly writer remains `refresh_transformer_corpus.py`
  `RebuildRawLabelSidecarTask` (which calls the same canonical builder) — no
  launchd/manifest change is part of this migration.
