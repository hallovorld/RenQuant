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
- `scripts/fmp_harvest.py` — resumable harvester. One tidy parquet per endpoint under
  `data/fmp_harvest/<endpoint>_291.parquet`, every row stamped `ticker`/`fetched_at`/
  `source`. Skips an endpoint whose output already exists (safe re-runs). Throttle 0.2s
  (≈300/min). Key from `FMP_API_KEY` (`.env`, never committed).

## Status (as of this PR)
- **Analyst (A) already harvested** for the full 291 universe: `grades_historical`
  283/291 (23,931 rows, 2018→2026), `analyst_estimates` 282/291, `price_target_consensus`
  283/291. These feed the immediate retrain go/no-go ablation (separate work).
- **Broad pull (B–F) running** now via this script (auto-skips the 3 analyst files).

## Scope discipline
`/data/` is already gitignored (`.gitignore:41`) → the parquet inventory stays local;
nothing large enters git. The harvest is **raw inventory** — no feature/retrain decision
rides on B–F. Anything that later becomes a model feature still goes through a
feature-engineering PR → placebo-clean WF validation → promote. Storing is not deploying.

## Follow-ups
- Run B–F to completion this month; verify coverage per endpoint.
- After harvest: cancel the paid plan; the free Finnhub cron (#408) accumulates deltas.
- Analyst-feature retrain decided by its own ablation (gated on WF placebo-clean), not here.
