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
- `scripts/fmp_harvest.py` — **auditable, fail-closed, manifest-resumable** harvester over
  **20 endpoints** (18 per-ticker + `treasury-rates` + `economic-indicators`, the latter an
  8-name list call). One tidy parquet + sidecar `<endpoint>_291.manifest.json` per endpoint
  under `data/fmp_harvest/`, every row stamped `ticker`/`fetched_at`/`source`. Atomic writes
  (tmp→replace). **Content/config-aware skip**: an endpoint is skipped only when its manifest
  `status: ok` AND its recorded `path_template` matches the current endpoint AND its recorded
  `universe_hash` matches the current target list AND **either** the parquet exists with a
  matching sha256 (data completion) **or** it is a valid ZERO-DATA record (`output: null`,
  `rows: 0`) — which skips without needing a parquet (manifest is the completion record).
  A tampered/stale/missing parquet, a changed endpoint/config, or a changed universe re-pull.
  A re-pull returning zero rows atomically **retires** any older parquet (→ `.parquet.retired`)
  so a later run can't skip on a stale parquet. Error samples record the **HTTP code / error
  type**, not just the ticker. Bounded retry/backoff on 429/5xx/timeout. Exits non-zero on any
  http/fetch error unless `--allow-errors`. Key from `FMP_API_KEY` (`.env`, never committed).
- `tests/test_fmp_harvest.py` — 18 unit tests (classify list/dict/empty/http/fetch; atomic
  write+manifest; no_data≠failure + valid zero-data completion; errors→non-ok status; error
  samples carry http code / err type; content/config-aware skip — sha256 match/mismatch,
  changed-template & changed-universe invalidation; zero-data rerun retires stale parquet;
  end-to-end loop skips a zero-data endpoint without re-pulling; bounded retry).

## Status (as of this PR)
- **Analyst (A) already harvested** full 291: `grades_historical` 283/291 (23,931 rows,
  2018→2026), `analyst_estimates` 282/291, `price_target_consensus`/`price_target_summary`
  283/291. These fed the retrain go/no-go ablation (separate work) — verdict **regime-split,
  no global retrain** (BULL_CALM +0.011 adds / BULL_VOLATILE −0.034 hurts).
- **First-pass B–D + treasury already pulled** (~13 MB), before `economic-indicators` was
  added; the script now ships **20 endpoints** (18 per-ticker + treasury + economic-indicators).
  `institutional-ownership` is **plan-locked above Starter** (402) — dropped. That first pass
  is local-only / gitignored and **NOT experiment-ready** — it sits behind this review gate.
- The **canonical, auditable** harvest (with manifests) is the re-run produced by the
  hardened script in this PR. Nothing feeds a model until a feature-eng PR → placebo-clean WF.

## Review fixes (Codex CHANGES_REQUESTED → addressed)
Round 1: robust manifest-based resumability + atomic writes; per-endpoint audit manifest
(counts, error samples, URL, timestamps, sha256); fail-closed exit code; real bounded
retry/backoff (docs no longer over-claim); unit tests; and an honest execution-state
reconciliation (first-pass output is gated raw inventory, not experiment-ready).

Round 2 (this revision):
1. **Skip is content/config aware** — `_manifest_ok` now requires `status: ok` AND a
   matching `path_template` AND a matching `universe_hash` AND a verified parquet sha256
   (data completion). A stale/tampered/missing parquet or a changed endpoint/universe
   re-pulls instead of being silently accepted.
2. **Valid zero-data completion** — `output: null` + `rows: 0` is now a recognized
   completed state; the manifest *is* the completion record, so a zero-row endpoint skips
   on rerun without needing a parquet (no more re-pull-forever).
3. **Atomic stale-output retirement** — a re-pull that returns zero rows while an older
   parquet exists atomically moves it to `.parquet.retired`, so a later run can't skip on a
   stale parquet paired with an `output: null` manifest.
4. **Richer error samples** — each sample now records the HTTP code (`http`) or error type
   (`err`) alongside the ticker, threaded from `_get`'s `{"_http"}/{"_err"}` sentinels.
5. **Counts reconciled** — script has **20 endpoints** (18 per-ticker + treasury +
   economic-indicators); tests are **18** (matches `test_fmp_harvest.py`); docs updated
   to match the code.

## Scope discipline
`/data/` is already gitignored (`.gitignore:41`) → the parquet inventory stays local;
nothing large enters git. The harvest is **raw inventory** — no feature/retrain decision
rides on B–F. Anything that later becomes a model feature still goes through a
feature-engineering PR → placebo-clean WF validation → promote. Storing is not deploying.

## Follow-ups
- Run B–F to completion this month; verify coverage per endpoint.
- After harvest: cancel the paid plan; the free Finnhub cron (#408) accumulates deltas.
- Analyst-feature retrain decided by its own ablation (gated on WF placebo-clean), not here.
