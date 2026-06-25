# FMP one-month harvest — plan + harvester

2026-06-25.

## What & why
The FMP Starter plan was upgraded today (full-symbol + 5-year history, 300 calls/min,
20 GB/30d). Operator directive: harvest **all** retrievable FMP data for the training
universe now, store it locally, max out the paid month — then cancel and let the free
Finnhub cron carry daily deltas. Deep history doesn't change, so this is a **one-time**
pull. This PR is the plan to discuss before the broad pull, plus the resumable harvester.

## Deliverables
- `doc/research/2026-06-25-fmp-harvest-plan.md` — budget, storage layout, endpoint list
  (A analyst · B fundamentals · C earnings/events · D ownership · E sentiment/news ·
  F macro), and the discipline note (storing ≠ using).
- `scripts/fmp_harvest.py` — **auditable, fail-closed, manifest-resumable** harvester. One
  tidy parquet + sidecar `<endpoint>_291.manifest.json` per endpoint under `data/fmp_harvest/`,
  every row stamped `ticker`/`fetched_at`/`source`. Atomic writes (tmp→replace). Skips an
  endpoint only when its parquet AND an `status: ok` manifest both exist (a partial/errored
  run re-pulls). Bounded retry/backoff on 429/5xx/timeout. Exits non-zero on any http/fetch
  error unless `--allow-errors`. Key from `FMP_API_KEY` (`.env`, never committed).
- `tests/test_fmp_harvest.py` — 11 unit tests (classify list/dict/empty/http/fetch, atomic
  write+manifest, no_data≠failure, errors→non-ok status, skip/rerun gate, bounded retry).

## Status (as of this PR)
- **Analyst (A) already harvested** full 291: `grades_historical` 283/291 (23,931 rows,
  2018→2026), `analyst_estimates` 282/291, `price_target_consensus`/`price_target_summary`
  283/291. These fed the retrain go/no-go ablation (separate work) — verdict **regime-split,
  no global retrain** (BULL_CALM +0.011 adds / BULL_VOLATILE −0.034 hurts).
- **First-pass B–D + treasury already pulled** (18/19 endpoints, ~13 MB). `institutional-
  ownership` is **plan-locked above Starter** (402) — dropped. That first pass is local-only
  / gitignored and **NOT experiment-ready** — it sits behind this review gate.
- The **canonical, auditable** harvest (with manifests) is the re-run produced by the
  hardened script in this PR. Nothing feeds a model until a feature-eng PR → placebo-clean WF.

## Review fixes (Codex CHANGES_REQUESTED → addressed)
Robust manifest-based resumability + atomic writes; per-endpoint audit manifest (counts,
error samples, URL, timestamps, sha256); fail-closed exit code; real bounded retry/backoff
(docs no longer over-claim); unit tests; and this honest execution-state reconciliation
(first-pass output is gated raw inventory, not experiment-ready).

## Scope discipline
`/data/` is already gitignored (`.gitignore:41`) → the parquet inventory stays local;
nothing large enters git. The harvest is **raw inventory** — no feature/retrain decision
rides on B–F. Anything that later becomes a model feature still goes through a
feature-engineering PR → placebo-clean WF validation → promote. Storing is not deploying.

## Follow-ups
- Run B–F to completion this month; verify coverage per endpoint.
- After harvest: cancel the paid plan; the free Finnhub cron (#408) accumulates deltas.
- Analyst-feature retrain decided by its own ablation (gated on WF placebo-clean), not here.
