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
  `status: ok` AND its `manifest_version` matches AND its recorded `path_template` matches the
  current endpoint AND its recorded `universe_hash` matches the current target list AND
  **either** the parquet exists with a matching sha256 (data completion) **or** it is a valid
  ALLOWED zero-data record satisfying the full invariant (`output`/`sha256` null,
  `with_data`/`errors`/`tickers`/`rows` all 0, `requested == no_data`, `allow_zero_data: true`)
  — which skips without needing a parquet. A tampered/stale/missing parquet, a changed
  endpoint/config, a changed universe, or an inconsistent/forged zero-data record re-pull.
  **All-target zero data FAILS CLOSED by default** (`zero_data_unexpected`, non-zero exit) —
  per-endpoint `allow_zero_data` policy gates the only legitimate empty-completion path, and
  every shipped endpoint is `False`. A **suspicious refresh preserves the last verified
  parquet/manifest** (no retire on `zero_data_unexpected`/errors); only an accepted allowed
  zero-data completion atomically **retires** an older parquet (→ `.parquet.retired`). Error
  samples record the **HTTP code / error type**, not just the ticker. Bounded retry/backoff on
  429/5xx/timeout. Exits non-zero on any http/fetch error **or** unexpected all-target zero
  data unless `--allow-errors`. Key from `FMP_API_KEY` (`.env`, never committed).
- `tests/test_fmp_harvest.py` — 35 unit tests (classify list/dict/empty/http/fetch; atomic
  write+manifest; partial no_data≠failure; **all-target zero-data default fail-closed** +
  **explicit allow_zero_data success**; errors→non-ok status; error samples carry http code /
  err type; content/config-aware skip — sha256 match/mismatch, changed-template &
  changed-universe invalidation; **full zero-data manifest-invariant rejection** of
  inconsistent/forged empties; **preservation of the last verified parquet on a rejected
  empty refresh** vs. retire-on-allowed-empty; **HTTP-200 app-error fail-closed** (bare
  dict, list-of-errors, AND mixed real+error lists → fail closed, real row never written);
  **partial-error refresh preserves the canonical parquet** (partial rows quarantined to
  `.parquet.staging`, errored manifest not skippable); **app_error in the zero-data
  invariant**; end-to-end loop — systemic-empty fails closed & re-pulls, verified-data
  endpoint skips without re-pull; bounded retry).

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
   economic-indicators); docs updated to match the code.

Round 3 (this revision) — *new fail-open blocker: all-target zero data must not be silently
accepted (missingness is data; measure & gate it per endpoint)*:
1. **Per-endpoint completion policy** — each endpoint carries an `allow_zero_data` flag;
   **default `False`**. An all-target-empty pull (no errors) on a `False` endpoint is now
   `zero_data_unexpected` → counts toward the non-zero exit (fail closed). Every shipped
   endpoint is `False` (none is known to legitimately return all-empty across the 291
   universe); a future genuinely-empty endpoint can opt in with a justification.
2. **Last verified artifact preserved** — a suspicious refresh (`zero_data_unexpected`, or
   any http/fetch error) **no longer retires** the existing good parquet/manifest. The prior
   verified state stands; only an *accepted allowed* zero-data completion retires the old
   parquet (→ `.parquet.retired`). This stops a suspicious empty from becoming canonical.
3. **Full zero-data invariant + schema version** — added `manifest_version` (=2) and an
   `allow_zero_data` field to every manifest. A zero-data record is honored as a skip ONLY
   when the whole invariant holds: `manifest_version` matches, `allow_zero_data: true`,
   `status: ok`, `with_data == http_error == fetch_error == tickers == rows == 0`,
   `requested == no_data`, and `output`/`sha256` both null, with matching template/universe.
   Any inconsistency (forged/stale/corrupt) re-pulls instead of being trusted.
4. **Tests** — the 4 required cases: systemic-empty default failure (+ loop re-pull),
   explicit allowed-empty success (+ skip-on-rerun), preservation of the last verified
   parquet on a rejected empty refresh, and invalid/inconsistent zero-data manifest
   rejection. Suite is now **23 tests** (was 18); all pass.

Round 4 (this revision) — *HTTP-200 app-error bodies + partial-error preservation*:
1. **Partial-error refresh preserves the canonical parquet** — `harvest_endpoint` no longer
   replaces `<key>_291.parquet` when ANY target errors (`bad > 0`), even if other targets
   returned real rows. Partial rows are quarantined to `<key>_291.parquet.staging`; the
   manifest records `output:null, status:"errors"` + a `staging`/`staged_rows` audit pointer.
   The last-verified canonical parquet+manifest only advance on a CLEAN full replacement, so
   an AAA-data + BBB-402 refresh can't clobber the prior good artifact and can't be skipped.
2. **`classify()` fails closed on mixed real+error lists** — a list containing ANY FMP
   top-level error object is `app_error` (not `with_data`); the real sibling row is NOT
   written to the parquet, and the offending error dict is returned as the error sample.
3. **`app_error` added to the zero-data invariant** — `_is_valid_zero_data_completion`
   now requires `app_error == 0`, so a crafted `status:ok, allow_zero_data:true, output:null,
   app_error:1` manifest can no longer be accepted by `_manifest_ok`.
4. **Tests** — updated the mixed-list classify test to expect fail-closed; added a
   harvest-level mixed-list fail-closed case, a partial-error canonical-preservation
   regression (verified sha/rows UNCHANGED, manifest not skippable, staging quarantine),
   and an `app_error:1` zero-data-invariant mutation. Suite is now **35 tests**; all pass.

## Scope discipline
`/data/` is already gitignored (`.gitignore:41`) → the parquet inventory stays local;
nothing large enters git. The harvest is **raw inventory** — no feature/retrain decision
rides on B–F. Anything that later becomes a model feature still goes through a
feature-engineering PR → placebo-clean WF validation → promote. Storing is not deploying.

## Follow-ups
- Run B–F to completion this month; verify coverage per endpoint.
- After harvest: cancel the paid plan; the free Finnhub cron (#408) accumulates deltas.
- Analyst-feature retrain decided by its own ablation (gated on WF placebo-clean), not here.
