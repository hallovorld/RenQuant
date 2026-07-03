# Point the corpus refresh at the TRUE transformer-panel recipe (S12 B1)

Date: 2026-07-02
Scope: `scripts/refresh_transformer_corpus.py` default `builder_fn` +
`tests/test_transformer_corpus_refresh.py`. Companion PR:
renquant-base-data `feat/transformer-panel-true-recipe` (the committed recipe).

## Problem (S12 diagnosis §4-B1)

`doc/research/2026-07-02-s12-panel-refresh-diagnosis.md` measured the served
shadow corpus `transformer_v4_wl200_clean.parquet` as an alpha158+fund-family
panel (178 columns, 142-ticker live watchlist, forward labels dropna'd) with NO
committed builder — while the #424 refresh chain's default `builder_fn` invokes
`scripts/transformer_dataset_builder.py` (raw 5-channel OHLCV, 292 tickers).
That output can never satisfy the swap gate's schema/coverage parity, so the
(correct) gate fail-closes forever and the served corpus stays frozen at
2026-02-10.

## Change

- The TRUE recipe is now committed in base-data
  (`renquant_base_data.transformer_corpus`): derive the corpus from the
  daily-refreshed prod fund panel `alpha158_291_fundamental_dataset.parquet` —
  watchlist subset (pinned `renquant-strategy-104` `strategy_config.json`),
  label dropna, exact served 178-column schema, deterministic ordering.
- `_default_build_corpus` now resolves that recipe lazily via the subrepo
  PYTHONPATH (same seam as `_default_fetch_fn`) and FAILS CLOSED when
  unresolvable — never a silent fallback to the divergent legacy recipe. The
  inventory universe still drives the OHLCV refresh + freshness guard; corpus
  rows come from the watchlist (S12 §1).
- New ctx fields / CLI flags: `--fund-panel-path`, `--strategy-config-path`
  (defaults: `data/alpha158_291_fundamental_dataset.parquet`;
  `$RENQUANT_SUBREPO_ROOT/renquant-strategy-104/configs/strategy_config.json`,
  which `weekly_retrain_patchtst.sh` already exports). Wrapper unchanged.

## Verification

- Ground-truth rebuild (read-only inputs, scratch output): today's prod panel
  + pinned watchlist -> 351,134 rows / 142 tickers / frontier 2026-04-02 (the
  achievable ~today-60td labeled frontier), parquet schema IDENTICAL to the
  served file (names + order + dtypes), and the module's own
  `_sanity_reasons` swap gate returns ZERO reasons (would swap).
- `tests/test_transformer_corpus_refresh.py`: 42 passed (36 existing + 6 new
  covering recipe resolution, watchlist-not-universe row selection, fail-closed
  unresolvable import, config resolution precedence, defaults, CLI).
- base-data suite: 184 passed (13 new recipe tests).

## Landing order (NOT done here)

1. Merge the base-data recipe PR; advance the base-data pin.
2. Merge this PR.
3. Umbrella-ops runs `bash scripts/weekly_retrain_patchtst.sh` on the live
   tree (per #212 §4 the actual refresh RUN is a landing action, not an agent
   action). B2 (promote 28d SLA horizon-adjust) and B3 (static retrain cutoff)
   remain open follow-ups per the diagnosis §5.2/§5.3.
