# Progress: AC4 migration census (bundle-transactionality RFC §3 P2 entry criterion)

Date: 2026-07-18
PR: design-only — adds `doc/design/2026-07-18-ac4-migration-census.md`
Related: `doc/design/2026-07-17-artifact-bundle-transactionality.md` (GOAL-5 AC4 RFC, r4)

## What

Committed the reader/writer census of the flat serving-pair paths
(`backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` +
`panel-rank-calibration.json` + their side files) required by RFC §3 before
migration phase P2 may start. Read-only sweep across all ten sibling
checkouts (umbrella, orchestrator, pipeline, backtesting, model,
strategy-104, common, base-data, execution, artifacts) plus the deployed
`-run` variants; every hit classified {migrates to bundle API | keeps flat
VIEW | test-only n/a}.

## Key findings (details in the design doc)

- All four RFC §1 incident writers are confirmed in code with exact
  write sites; no fifth AUTOMATED writer of the pair was found.
- Two surprise writer classes surfaced that the RFC's §2.4 authorization
  model does not yet cover: (1) git working-tree operations — the pair is
  git-TRACKED in the umbrella repo, so checkout/reset/pull is an
  unmediated pair writer (the 2026-07-08 incident class); (2) ad-hoc
  session edits evidenced by orphan side files
  (`*.pre-v1-restamp-*`, `*.pre-binding-fix-*`) with no committing tool in
  any checkout — the exact break-glass gap §2.4 is designed to close.
- Additional census-time blockers: both daily-retrain runners (orchestrator
  primary + umbrella rollback twin) DEFAULT to writing the live prod pair
  when invoked bare (B4), and the orchestrator `prune-artifacts --execute`
  CLI deletes staging/rollback side files (maps to bundle-store GC).
- Writer inventory, per-file classification tables, migration blockers,
  and the P0-P3 phase mapping with this census attached to P2 are in the
  design doc.

## Verification

- Sweep method: filename/path-fragment/config-key grep across all sibling
  checkouts (read-only; no git commands in any live tree), then manual
  read of every writer call site; config indirection followed through
  `ranking.panel_scoring.artifact_path`,
  `ranking.panel_scoring.global_calibration.artifact_path`,
  `panel_ltr.artifact_path`, and
  `configs/xgb_prod_artifact_manifest.json` in pinned renquant-strategy-104.
- Cross-checked against the four incident writers named in RFC §1 and the
  live `artifacts/prod/` directory contents (side-file conventions).
