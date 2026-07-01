# shadow/PatchTST training-corpus refresh — full-universe OHLCV + partial-freeze guard

STATUS: code + tests (this PR). Mocks/fixtures only — no real OHLCV fetch, no real
        transformer rebuild, no retrain executed, no production data file written.
BRANCH: feat/transformer-patchtst-data-refresh → main (`hallovorld/RenQuant`).

ROOT CAUSE (training-data-freeze investigation, fix #4):
  The PatchTST SHADOW model trains on `data/transformer_v4_wl200_clean.parquet`
  (the `--dataset` default of `scripts/train_walkforward_patchtst.py` and every
  PatchTST/xgb baseline). That corpus was frozen at 2026-02-10 for two reasons:
    1. Its builder (`scripts/transformer_dataset_builder.py`, sourcing the
       transformer universe from `data/transformer_universe_inventory.json`
       tier_A + tier_B) is on NO refresh cadence — nothing rebuilds it.
    2. It inherits the SAME full-universe OHLCV coverage gap fixed for the
       alpha158 panel in orchestrator PR #217/#210: only the ~142-ticker live
       watchlist gets fresh daily `data/ohlcv/<ticker>/1d.parquet` bars (a
       live-path side effect). The ~150 extra research tickers in the transformer
       universe have no refresh cadence, so half the corpus froze; after the
       correct fwd_60d label clip that surfaced as a 2026-02-10 corpus.

WHAT: New self-contained module `scripts/refresh_transformer_corpus.py` (ordered
      pipeline), wired into the PatchTST retrain cadence
      (`scripts/weekly_retrain_patchtst.sh`) BEFORE the WF build + shadow promote:
      1. `RefreshTransformerUniverseOhlcvTask` — iterate the FULL transformer
         universe (tier_A + tier_B, sourced exactly where the builder reads it,
         NOT just the watchlist) and call the incremental (append-merge,
         non-destructive, timeout-protected) OHLCV fetch per ticker. Resilient: a
         single ticker's failure / delisting NEVER aborts. Records n_refreshed /
         n_stale / n_delisted / n_failed.
      2. `TransformerUniverseFreshnessGuardTask` — after the refresh, compute each
         ticker's RAW OHLCV bar max date; if > `freshness_max_stale_fraction`
         (default 10%) of the universe lags the universe frontier by >
         `freshness_stale_after_days` (default 10 trading days), emit a LOUD ntfy
         alert and — per `freshness_fail_on_stale` (default fail-closed) — fail or
         proceed. This would have caught the freeze the watchlist-only scan passed.
      3. `RebuildTransformerCorpusTask` — rebuild the transformer panel to a
         STAGING path, then swap it in only if it advances the corpus date
         frontier + keeps >= `min_row_ratio` (default 95%) of the prior rows.
         Non-destructive: the prior corpus is moved to `<corpus>.bak` before the
         staged corpus takes its place; a regressed rebuild (older / materially
         smaller) NEVER clobbers the served corpus (fail-closed by default, or
         warn + keep prior via `--no-swap-fail-on-regression`).

FWD-60D: The guard reads RAW OHLCV bars, whose frontier is ~today-1 — NOT the
         built corpus, which legitimately ends ~today-60 trading days after the
         (correct) fwd_60d label clip. So the expected fwd_60d frontier is
         distinguished from genuine input staleness; an on-frontier universe never
         trips the guard. The model architecture and the label clip are UNCHANGED.

NON-DESTRUCTIVE: uses ONLY the incremental append-merge OHLCV primitive; never
         overwrites/deletes `data/ohlcv/`. The corpus swap is staging + `.bak`,
         sanity-gated, and skipped entirely under `--dry-run`.

RUNTIME WIRING (important — every external seam is dependency-injected so the
         module is unit-testable without a network / rebuild / production write):
  - OHLCV fetch: `fetch_ohlcv_incremental` is a base-data primitive
    (`renquant_base_data.loaders.data.fetch_ohlcv_incremental`), import-resolved
    via the subrepo PYTHONPATH `weekly_retrain_patchtst.sh` already sets up. It is
    injected via `CorpusRefreshContext.fetch_fn`; when None it resolves lazily
    through `_default_fetch_fn()` at call time. Tests inject a fake.
  - Corpus builder: injected via `CorpusRefreshContext.builder_fn`; when None
    `_default_build_corpus` invokes `scripts/transformer_dataset_builder.py`
    (`--inventory` tier_A+tier_B over `--ohlcv-dir data/ohlcv`) to the staging
    path. The served corpus is the wl200-clean transformer_v4 corpus; if the
    operator's exact wl200-clean recipe diverges from the canonical builder,
    point `builder_fn` at it — the injection seam makes that a one-line change
    with no code churn here. This is the one place to confirm before the first
    real scheduled run.
  - On-disk readers (`ohlcv_max_date_fn`, `corpus_stats_fn`) are likewise
    injectable, defaulting to reading the parquet.
  - Shell wiring: the refresh runs before the WF-manifest delegation, gated by
    `RQ_PATCHTST_REFRESH_CORPUS` (default 1); `set -e` aborts the retrain on
    fail-close. `RQ_PATCHTST_REFRESH_ARGS` lets ops relax to warn-and-proceed
    (`--no-freshness-fail-on-stale --no-swap-fail-on-regression`).

CONFIG: CLI flags on `refresh_transformer_corpus.py`: `--repo-dir`,
         `--inventory-path`, `--corpus-path`, `--staging-path`,
         `--transformer-universe-file` (list or inventory),
         `--refresh-ohlcv/--no-`, `--ohlcv-timeout-sec`, `--rebuild-corpus/--no-`,
         `--freshness-stale-after-days`, `--freshness-max-stale-fraction`,
         `--freshness-fail-on-stale/--no-`, `--require-date-advance/--no-`,
         `--min-row-ratio`, `--swap-fail-on-regression/--no-`, `--ntfy-topic`,
         `--dry-run`, `--quiet`. ntfy honors the suite-wide `RENQUANT_NO_NOTIFY`.

TESTS: `tests/test_transformer_corpus_refresh.py` (25 tests, mocks/fixtures):
         full transformer universe refreshed (not just watchlist), universe
         sourced from inventory tier_A+tier_B (tier_C excluded), delisted + failed
         tickers do not abort, guard fires past threshold (fail-closed raises +
         loud ntfy) and proceeds-with-warning when configured, guard stays QUIET at
         the expected fwd_60d frontier and below threshold, injected reader +
         runtime default-fetch seam, non-destructive sanity-gated swap (advance →
         swap + `.bak`; regression → keep prior, no clobber, alert, fail-closed),
         first-time build with no prior, dry-run builds/swaps nothing, and
         end-to-end refresh→guard catches the partial freeze.
         Run: `python3 -m pytest tests/test_transformer_corpus_refresh.py -o addopts="" -q`
         → 25 passed (system Python has no pytest-xdist/-env; CI runs them under
         the repo's `-n auto` + `RENQUANT_NO_NOTIFY=1`).

SCOPE: umbrella code + tests only. Does NOT run the refresh/rebuild/retrain, does
       NOT touch the live umbrella tree, does NOT write production data. Follow-up
       (out of scope): confirm `builder_fn` matches the exact wl200-clean recipe,
       and retire any skip-existing fetch behavior for the research tail in favor
       of this incremental path.

NEXT: land the PR, confirm the builder recipe wiring, then let the scheduled
      `weekly_retrain_patchtst.sh` run the refresh + WF build + shadow promote on
      the fresh corpus.
