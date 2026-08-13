# 2026-08-13 — G-J daily-full speedup: memoize the redundant SPY `rolling_hurst`

STATUS:   PROPOSED (2026-08-13). Speed-only change: dedup the 63-day SPY
          rolling-Hurst recomputation in the daily-full inference feature
          prep. Output is byte-identical per ticker — the neutralized feature
          frames AND the z-scored factor frames match the un-memoized baseline
          exactly `[VERIFIED — tests/test_hurst_dedup_invariance.py, 3 passed;
          assert_frame_equal(check_exact=True) BEFORE (memo off) vs AFTER
          (memo on) on a 12-ticker input]`.

WHAT:     `build_training_features` (`training/features.py:119`) reindexes SPY
          returns to each ticker's `common_idx` and then runs
          `rolling_hurst(spy_rets, window=63)` — a SPY-only regime feature.
          `prepare_inference_panel_frames` calls it once per ticker across the
          per-ticker chain. Within one run SPY OHLCV is a single shared frame,
          so every ticker that shares a `common_idx` feeds an IDENTICAL
          `spy_rets` into `rolling_hurst`.

          This memoizes the EXACT existing computation, keyed by the content
          of `spy_rets` (SHA-1 over its index bytes + float64 value bytes). On
          a cache hit the identical Series is reused and the subsequent
          `.reindex(common_idx)` still runs per ticker. The memoizer
          (`SpyHurstMemo`) is created once in
          `prepare_inference_panel_frames` and threaded down via
          `TickerPanelContext.hurst_cache` -> `TickerPanelFeatureJob` ->
          `build_training_features(hurst_cache=...)`. The new parameter
          defaults `None`, so the training path (`PanelFeatureJob`) and every
          other caller of `build_training_features` are byte-for-byte
          unchanged.

WHY/DIR:  The `rolling_hurst` call is ~99.3% of per-ticker feature-prep cost
          (~4.5 s x 145 tickers ~= 10 min per daily-full run, per the G-J
          finding) `[GUESS — production wall-clock from the finding; not
          re-measured here]`. The SPY hurst is (near-)identical across tickers,
          so N recomputations collapse to one per DISTINCT `spy_rets`.

          The math is NOT restructured. A naive "compute rolling_hurst once on
          the full SPY series then reindex per ticker" is NOT safe: the current
          code reindexes `spy_rets` to each ticker's `common_idx` BEFORE the
          rolling, so rolling-then-reindex != reindex-then-rolling whenever
          date ranges differ. Memoizing the exact reindex-then-rolling call is
          output-invariant by construction (identical input -> identical
          output, computed once).

EVIDENCE:
  artifact:      `training/features.py` (`_spy_rets_fingerprint`, `SpyHurstMemo`,
                 `build_training_features(..., hurst_cache=None)`),
                 `training_panel/context.py` (`TickerPanelContext.hurst_cache`
                 field, default None),
                 `training_panel/pp_panel_training.py`
                 (`TickerPanelFeatureJob` passes `tc.hurst_cache`),
                 `training_panel/pipeline.py` (`_new_spy_hurst_memo` factory +
                 one memo shared across the per-ticker chain in
                 `prepare_inference_panel_frames`),
                 `tests/test_hurst_dedup_invariance.py` (new oracle),
                 + this record.
  oracle:        the new test distinguishes the bug from the fix rather than
                 exercising the path. It runs `prepare_inference_panel_frames`
                 BEFORE (memoizer disabled by monkeypatching
                 `_new_spy_hurst_memo` -> None -> the original per-ticker
                 computation) and AFTER (real memoizer), then asserts the
                 neutralized feature frames AND factor frames are byte-identical
                 per ticker via `assert_frame_equal(check_exact=True)`. It FAILS
                 if a single feature value changes. A second test proves the
                 same at the `build_training_features` level (the frame that
                 carries `hurst_proxy`), and a third proves the memo primitive
                 returns exactly what `rolling_hurst` returns and hits the cache
                 on a repeated identical input
                 `[VERIFIED — .venv/bin/python -m pytest -n 0 -q
                 tests/test_hurst_dedup_invariance.py -> 3 passed]`.
  prod or exp:   prod — this is the daily-full inference feature prep that feeds
                 the live model. Behaviour is byte-identical for every produced
                 feature value; only the NUMBER of `rolling_hurst`
                 evaluations changes.
  existing data: yes — the test builds synthetic OHLCV; no production path is
                 read or written; all work is in an isolated git worktree; the
                 live umbrella tree was not touched and the daily job was not
                 run.
  best-known?:   yes — memoizing the exact computation is the only
                 output-invariant option. The naive "hoist onto the full SPY
                 series" is unsafe (reindex-then-rolling != rolling-then-reindex
                 across differing date ranges) and is deliberately NOT used. A
                 module-global cache would leak across runs/dates; this is
                 run-scoped (a fresh memo per call).
  scope:         "this is the SPY 63-day `rolling_hurst` inside
                 `build_training_features` (prod feature prep), vs existing best
                 = recompute it once per ticker. Key = content of `spy_rets`
                 (index + float64 values); identical input -> identical key ->
                 identical cached Series; the per-ticker `.reindex(common_idx)`
                 is unchanged. Thread-safe via double-checked locking under a
                 `threading.Lock` (the chain runs under a ThreadPoolExecutor).
                 Measured call-count: 12 tickers -> 12 `rolling_hurst` calls
                 baseline -> 2 with the memo (one per distinct `common_idx`
                 group in the fixture); a production run where every inference
                 ticker shares one `common_idx` collapses to exactly 1
                 `[VERIFIED — instrumented count, patching
                 renquant_common.hurst.rolling_hurst]`."
  TESTS:         curated touched-module set (training / training_panel):
                 `.venv/bin/python -m pytest -n auto` over
                 `test_hurst_dedup_invariance test_train_inference_symmetry
                 test_panel_bugfixes test_panel_inference test_panel_frame
                 test_panel_frame_unpack_arity test_panel_neutralization
                 test_panel_factors test_inference_frame_cache
                 test_panel_alignment test_panel_ltr_audit_2026_04_24
                 test_panel_hourly_wiring test_inference_no_autofetch
                 test_train_infer_feature_parity test_notebook_lean_feature_parity
                 test_lookahead_propagation test_momentum_features_no_leakage
                 test_panel_training_pipeline` -> **205 passed, 6 failed,
                 2 skipped**. The 6 failures are PRE-EXISTING / environmental,
                 NOT introduced here: 4 are `ModuleNotFoundError: No module
                 named 'renquant_pipeline'` (the subrepo runtime is not
                 assembled in a bare worktree) and 2 are a cwd/snapshot
                 resolution + an architectural-import invariant. The identical 6
                 fail on a clean `origin/main` worktree with none of these
                 changes `[VERIFIED — same 6 failed, 34 passed on baseline]`.

NEXT: merge is not deploy. The daily run consumes local sibling checkouts /
pinned assembly, so this speedup reaches the live daily-full only when the
operator syncs the pin/tree — the same operator-gated cutover as any umbrella
change. No behaviour changes at that point (byte-identical features); the only
observable effect is the reduced feature-prep wall-clock.
