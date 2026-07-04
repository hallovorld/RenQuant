# Refresh the rawlabel sidecar in the weekly chain (S12 rawlabel gap, B1 pattern)

Date: 2026-07-03
Scope: `scripts/refresh_transformer_corpus.py` (new `RebuildRawLabelSidecarTask`
+ ctx/CLI wiring), `tests/test_rawlabel_refresh.py`,
`tests/test_transformer_corpus_refresh.py` (pipeline-order),
`scripts/weekly_retrain_patchtst.sh` (comment only — no invocation change).
Companion PR: renquant-base-data `feat/rawlabel-recipe` (#33, the committed
recipe — MERGE FIRST + pin bump).

## Problem

2026-07-02 S12 shadow promote refused:
`source[fast] rawlabel: cutoff=2026-02-11 age=142d sla=28d OFF-SLA`.

`data/alpha158_291_fundamental_dataset_rawlabel.parquet` — the UN-standardized
forward-label sidecar both calibrator fits read
(`renquant_model_patchtst/fit_calibrator.py`,
`renquant_model_gbdt/fit_calibrator_alpha158_fund.py`) — was a one-off Track A
research build (`scripts/build_raw_fwd60d_label.py`) with NO committed recipe
and NO refresh mechanism. The B1 fix (base-data #31 + RQ #434) put the
transformer panel on a committed recipe + weekly staged/gated/swap refresh;
the rawlabel sidecar was left behind and froze at the 2026-02-11 panel
vintage. The promote gate is CORRECT to hold it to the raw 28d fast-axis SLA:
its documented semantics is "keeps unlabeled rows — its max(date) IS the bar
frontier" (only `transformer_panel` is the label-clipped source).

## Change (mirrors B1 exactly)

- The TRUE recipe is committed in base-data
  (`renquant_base_data.rawlabel_sidecar`): prod fund panel, FULL universe, NO
  label dropna, sentiment columns dropped, `fwd_60d_excess_raw` recomputed
  point-in-time from OHLCV closes vs SPY (the original derivation), and the
  (ticker, date) axis EXTENDED per ticker to its OHLCV bar frontier —
  extension rows carry NaN features/split (honest, never fabricated; the
  upstream panel is label-clipped at ~frontier−60td so the frontier rows
  cannot exist otherwise). Exact served 176-column schema; fail-closed drift
  gate.
- New `RebuildRawLabelSidecarTask` runs as the 4th task of the SAME weekly
  `refresh_transformer_corpus.py` pipeline (refresh OHLCV → freshness guard →
  corpus rebuild/swap → rawlabel rebuild/swap), so one weekly
  `weekly_retrain_patchtst.sh` run refreshes BOTH fund-panel-derived training
  products. Identical discipline: staged build (never touches the served
  file), the SAME `_sanity_reasons` schema/row/date/coverage gate (shared
  knobs), atomic `.bak`-copy + `os.replace` swap, keep-prior-on-reject,
  fail-closed default builder seam (base-data pin predating the recipe →
  `CorpusRefreshError`, never a silent freeze).
- New ctx fields / CLI flags: `--rawlabel-path`, `--rawlabel-staging-path`,
  `--rebuild-rawlabel/--no-rebuild-rawlabel` (default ON). Wrapper invocation
  unchanged.

## Verification

- Ground-truth rebuild (read-only prod inputs, scratch output): today's prod
  panel + OHLCV → 740,151 rows / 292 tickers (722,343 panel + 17,808
  extension), parquet schema IDENTICAL to the served sidecar (names + order +
  arrow types), cutoff **2026-07-02 = the bar frontier → age 1d** vs the 28d
  SLA (was 142d), labeled frontier 2026-04-07 (the correct ~60td structural
  lag). Raw-label parity on the 715,629 overlapping labeled rows: 99.3%
  within 1e-9, corr 0.9992 (residual = re-adjusted OHLCV re-downloads on
  FDX/KLAC/CRWD since the Jun-17 vintage — input revision, not recipe
  divergence).
- Projected promote-gate rawlabel age post-fix: ≈1 trading day right after a
  weekly run; worst case ≈8 calendar days at the weekly cadence — comfortably
  inside the 28d SLA.
- `tests/test_rawlabel_refresh.py`: 13 new (advancing swap, first-time build,
  reject matrix keep-prior, interrupted-swap preservation, warn-and-proceed,
  dry-run/disabled, fail-closed builder seam, CLI/defaults);
  `test_transformer_corpus_refresh.py` 43 green (pipeline-order updated).
- base-data suite: 227 passed (21 new recipe tests).

## Landing order (NOT done here)

1. Merge base-data #33; advance the base-data pin in `subrepos.lock.json`.
2. Merge this PR.
3. Umbrella-ops runs `bash scripts/weekly_retrain_patchtst.sh` on the live
   tree (the actual refresh RUN is a landing action, not an agent action).
   The first run swaps the frozen 2026-02-11 sidecar for a bar-frontier build
   and the S12 promote's rawlabel source goes back ON-SLA.
