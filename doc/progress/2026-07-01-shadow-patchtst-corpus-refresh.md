# shadow/PatchTST training-corpus refresh — full-universe OHLCV + partial-freeze guard

STATUS: code + tests (this PR). Mocks/fixtures only — no real OHLCV fetch, no real
        transformer rebuild, no retrain executed, no production data file written.
BRANCH: feat/transformer-patchtst-data-refresh → main (`hallovorld/RenQuant`).

CODEX ROUND-2 (PR #424): every load-bearing input now FAILS CLOSED on an
        unassessable state instead of silently degrading —
          1. UNIVERSE PROVENANCE — a missing / corrupt / empty-tiers inventory (or a
             digest that does not match a bound `--expected-inventory-digest`) raises;
             the refresh/rebuild never runs on an empty universe (`--no-require-universe`
             is the explicit ops escape hatch back to a safe no-op).
          2. GLOBAL FREEZE — the raw-bar frontier is compared to an INDEPENDENT
             expected completed market session (`--freshness-as-of`, default last
             business day), so a uniformly-stale universe (zero *relative* staleness)
             trips; no resolvable dates fail closed rather than skip.
          3. STRICT ADVANCE — the staged frontier must strictly advance; an equal
             (non-advanced) corpus is rejected (matches the `require_date_advance` name).
          4. ATOMIC SWAP — prior corpus backed up by COPY + a single `os.replace`
             (atomic rename on one fs) + fsync; the served corpus is never moved out of
             the way, so an interrupted swap can never leave it missing.
          5. RECIPE BINDING — the default builder invocation now passes
             `--integrity-report` + `--labels` (the same seams the builder reads), and
             the staged output is validated against the served corpus's schema / feature
             columns / label horizons / ticker coverage. A wrong recipe that changes
             features / universe / schema while still producing a plausible row count
             FAILS CLOSED — no "confirm after merge" deferral. REFRESH/REBUILD IS
             PLUMBING, NOT A PROMOTION: whether the freshly-trained shadow is promoted is
             still decided downstream (WF gate + PR #419 shadow replay).

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
         ticker's RAW OHLCV bar max date; trip on either a PARTIAL freeze (>
         `freshness_max_stale_fraction`, default 10%, of the universe lags the
         universe frontier by > `freshness_stale_after_days`, default 10 trading
         days) OR a GLOBAL freeze (the frontier itself lags the independent expected
         market as-of by > `freshness_stale_after_days`, so a uniformly-stale
         universe with zero relative staleness is still caught). Unassessable input
         (no resolvable dates) FAILS CLOSED. On a trip: LOUD ntfy alert and — per
         `freshness_fail_on_stale` (default fail-closed) — fail or proceed. This
         would have caught the freeze the watchlist-only scan passed.
      3. `RebuildTransformerCorpusTask` — rebuild the transformer panel to a
         STAGING path, then swap it in only if it STRICTLY advances the corpus date
         frontier (equal is rejected) + keeps >= `min_row_ratio` (default 95%) of
         the prior rows + keeps the prior schema / feature columns / label horizons
         + >= `min_ticker_coverage_ratio` (default 90%) of prior distinct tickers.
         Non-destructive & ATOMIC: the prior corpus is backed up by COPY to
         `<corpus>.bak` and replaced with a single `os.replace` (atomic rename on one
         filesystem) + fsync, so the served corpus is never moved out of the way and
         an interrupted swap can never leave it missing. A regressed / divergent-recipe
         rebuild NEVER clobbers the served corpus (fail-closed by default, or warn +
         keep prior via `--no-swap-fail-on-regression`).

FWD-60D: The guard reads RAW OHLCV bars, whose frontier is ~today-1 — NOT the
         built corpus, which legitimately ends ~today-60 trading days after the
         (correct) fwd_60d label clip. So the expected fwd_60d frontier is
         distinguished from genuine input staleness; an on-frontier universe never
         trips the guard. The model architecture and the label clip are UNCHANGED.

NON-DESTRUCTIVE: uses ONLY the incremental append-merge OHLCV primitive; never
         overwrites/deletes `data/ohlcv/`. The corpus swap is staging + COPY-to-`.bak`
         + atomic `os.replace`, sanity/schema-gated, and skipped entirely under
         `--dry-run`.

RUNTIME WIRING (important — every external seam is dependency-injected so the
         module is unit-testable without a network / rebuild / production write):
  - OHLCV fetch: `fetch_ohlcv_incremental` is a base-data primitive
    (`renquant_base_data.loaders.data.fetch_ohlcv_incremental`), import-resolved
    via the subrepo PYTHONPATH `weekly_retrain_patchtst.sh` already sets up. It is
    injected via `CorpusRefreshContext.fetch_fn`; when None it resolves lazily
    through `_default_fetch_fn()` at call time. Tests inject a fake.
  - Corpus builder: injected via `CorpusRefreshContext.builder_fn`; when None
    `_default_build_corpus` invokes `scripts/transformer_dataset_builder.py` with
    `--inventory` + `--integrity-report` + `--labels` + `--ohlcv-dir data/ohlcv`
    (the exact seams the builder reads) to the staging path. The served corpus is
    the wl200-clean transformer_v4 corpus; the staged OUTPUT is validated against
    the served corpus's schema / feature columns / label horizons / ticker coverage
    (the swap gate), so a divergent recipe FAILS CLOSED rather than silently serving
    a changed corpus — recipe binding is enforced pre-merge, not deferred to a
    post-merge "confirm". Point `builder_fn` at the operator's exact recipe to also
    match the build side; that is a one-line injection change with no code churn.
  - On-disk readers (`ohlcv_max_date_fn`, `corpus_stats_fn`, `corpus_schema_fn`,
    `expected_as_of_fn`) are likewise injectable, defaulting to reading the parquet /
    the last business day.
  - Shell wiring: the refresh runs before the WF-manifest delegation, gated by
    `RQ_PATCHTST_REFRESH_CORPUS` (default 1); `set -e` aborts the retrain on
    fail-close. `RQ_PATCHTST_REFRESH_ARGS` lets ops relax to warn-and-proceed
    (`--no-freshness-fail-on-stale --no-swap-fail-on-regression`).

CONFIG: CLI flags on `refresh_transformer_corpus.py`: `--repo-dir`,
         `--inventory-path`, `--labels-path`, `--integrity-report-path`,
         `--corpus-path`, `--staging-path`, `--transformer-universe-file` (list or
         inventory), `--require-universe/--no-`, `--expected-inventory-digest`,
         `--refresh-ohlcv/--no-`, `--ohlcv-timeout-sec`, `--rebuild-corpus/--no-`,
         `--freshness-stale-after-days`, `--freshness-max-stale-fraction`,
         `--freshness-as-of`, `--freshness-fail-on-stale/--no-`,
         `--require-date-advance/--no-`, `--min-row-ratio`,
         `--min-ticker-coverage-ratio`, `--validate-schema/--no-`,
         `--swap-fail-on-regression/--no-`, `--ntfy-topic`, `--dry-run`, `--quiet`.
         ntfy honors the suite-wide `RENQUANT_NO_NOTIFY`.

TESTS: `tests/test_transformer_corpus_refresh.py` (37 tests, mocks/fixtures):
         full transformer universe refreshed (not just watchlist), universe
         sourced from inventory tier_A+tier_B (tier_C excluded), delisted + failed
         tickers do not abort; PROVENANCE fail-closed on missing / corrupt / empty
         inventory + digest-binding mismatch (and safe no-op under
         `--no-require-universe`); guard fires on a PARTIAL freeze and on a GLOBAL
         freeze (uniform-stale bars, zero relative staleness, caught by the
         independent as-of), fails closed when no dates are resolvable, stays QUIET
         at the expected fwd_60d frontier and below threshold; swap gate rejects a
         non-advancing (equal) frontier, a wrong-recipe schema / label-horizon drift,
         and a ticker-coverage drop; ATOMIC swap preserves the served corpus under an
         injected `os.replace` failure; injected readers + runtime default-fetch seam,
         first-time build with no prior, dry-run builds/swaps nothing, and end-to-end
         refresh→guard catches the partial freeze.
         Run: `python3 -m pytest tests/test_transformer_corpus_refresh.py -o addopts="" -q`
         → 37 passed (system Python has no pytest-xdist/-env; CI runs them under
         the repo's `-n auto` + `RENQUANT_NO_NOTIFY=1`).

SCOPE: umbrella code + tests only. Does NOT run the refresh/rebuild/retrain, does
       NOT touch the live umbrella tree, does NOT write production data. The recipe
       is now bound by the output-contract swap gate (not a post-merge TODO);
       follow-up (out of scope): point `builder_fn`/`--expected-inventory-digest` at
       the exact operator recipe + inventory to also bind the build side, and retire
       any skip-existing fetch behavior for the research tail in favor of this
       incremental path.

NEXT: land the PR, then let the scheduled `weekly_retrain_patchtst.sh` run the
      refresh + WF build. REFRESH/REBUILD IS PLUMBING, NOT A PROMOTION — before the
      weekly job promotes anything, run a FROZEN shadow replay (identical model
      code/seeds, PIT checks, regime IC, turnover/cost, coverage diagnostics on old
      vs new corpus) per the PR #419 shadow gate.
