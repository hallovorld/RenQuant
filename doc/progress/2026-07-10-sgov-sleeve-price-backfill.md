# SGOV sleeve-leg price backfill — landing-step tool for pipeline #185 (RS-1 SGOV floor)

**Date:** 2026-07-10 · **Branch:** `feat/sgov-sleeve-price-backfill` · **Repo:** RenQuant (umbrella)

## Bottom line

Parking-sleeve `mode="live"` (renquant-pipeline#185, RS-1 SGOV floor variant)
fail-closes on a missing SGOV price, and SGOV daily bars exist in **neither**
OHLCV store `[VERIFIED 2026-07-10]`. This PR ships the missing dependency as
**code only**: an operator-run, dry-run-by-default warm-up backfill
(`scripts/backfill_sleeve_prices.py`) plus the pre-enable leg resolver it
needs (`parking_sleeve_leg_tickers`). No data is written by this PR; the
backfill run itself is a separate operator-granted landing step.

## Owning-repo / mechanism determination `[VERIFIED]`

* The daily OHLCV fetch universe is assembled in the **umbrella** runner
  (`backtesting/renquant_104/adapters/runner.py`: watchlist + benchmark +
  sector ETFs + held + extra symbols) — strategy-104's `sleeve` config comment
  pins "daily price fetch is umbrella-owned". Umbrella is the owning repo.
* The sleeve coverage mechanism **already exists** (commit `4ec7ac9a`,
  st104#39 follow-up): `adapters/sleeve_prices.parking_sleeve_price_tickers`
  wired into all four price paths (runner live fetch, `adapters/lean.py`
  History batch, `main.py` AddEquity, `adapters/sim_price.py`), gated on
  `sleeve.enabled` per st104#39's pinned decision.
* **The gap:** `sleeve.enabled=false` today ⇒ SGOV is never fetched, and it
  *cannot* be warmed before the flip. Flip day would depend on a cold-start
  ~10y remote fetch mid-live-run; one vendor hiccup ⇒ day-1 live fail-close.
  The RS-1 §4 pre-registered cash/SGOV/SPY comparison also needs SGOV history
  *before* the flip.
* Note: `data/ohlcv/BIL` + `SHV` are **stale research leftovers** (last write
  2026-05-15), not members of any daily-fetched group — there is no static
  "auxiliary symbol list" to append SGOV to; the conditional sleeve coverage
  IS the sanctioned mechanism, so the warm-up reuses its exact normalization.

## Changes

1. `backtesting/renquant_104/adapters/sleeve_prices.py` — new
   `parking_sleeve_leg_tickers()` (legs ignoring `enabled`, same
   normalization); `parking_sleeve_price_tickers()` refactored on top of it,
   **behavior unchanged** (pinned by pre-existing tests + a new
   gate-unchanged regression test).
2. `scripts/backfill_sleeve_prices.py` — operator landing-step tool:
   dry-run by default; `--write` backfills via canonical
   `kernel.data.fetch_ohlcv_incremental` into the repo-root-anchored
   `data/ohlcv/{SYMBOL}/1d.parquet` store the runner reads; NYSE-session
   freshness verification after write; exit 1 on a still-missing leg;
   exit 2 (refusal) if the SGOV leg is found in the watchlist.
3. Tests: `tests/test_backfill_sleeve_prices.py` (8, tmp-store only, no
   network) + `tests/test_sleeve_prices.py` extended (leg resolver ×4,
   fingerprint non-coupling ×1).

## Watchlist / panel-fingerprint coupling check `[VERIFIED]`

The fetch list and the watchlist are **not coupled**. The panel config
fingerprint (`renquant_common.config_consistency._model_relevant_fields`)
hashes only `watchlist`, `panel_ltr` flags (`lookahead_days`, `objective`,
`asset_embeddings`, `training_resolution`, `hourly`/`minute` enabled),
`sector_map` (watchlist-projected), `sector_etf_map` (used sectors). Neither
the `sleeve` section nor the runner's price-fetch list enters the hash.
SGOV joins price coverage only — never the watchlist, panel scoring, or
admission stats. New test
`TestPanelFingerprintNonCoupling::test_sleeve_section_and_enable_flip_do_not_move_fingerprint`
pins this against the real `fingerprint_config` impl (ran, not skipped).
The backfill additionally refuses to run if `sgov_symbol` ever appears in
the watchlist (fail-closed, don't mask an st104#39 violation).

## Evidence

* `pytest tests/test_sleeve_prices.py tests/test_backfill_sleeve_prices.py`
  → **28 passed** (umbrella .venv).
* `pytest tests/test_universe_alignment.py tests/test_benchmark_sleeve.py`
  → **42 passed** (adapter-parity + sibling-sleeve regression).
* Dry-run smoke against a clean checkout: resolves store cwd-independently,
  reports `SPY`/`SGOV` `[MISSING]`, performs no fetch/write, exit 0.

## Operator landing step (NOT part of this PR)

```bash
cd /Users/renhao/git/github/RenQuant
.venv/bin/python scripts/backfill_sleeve_prices.py            # inspect
.venv/bin/python scripts/backfill_sleeve_prices.py --write    # backfill
```

Expected outcome: `data/ohlcv/SGOV/1d.parquet` (~10y daily bars) +
refreshed `SPY`, both `[VERIFIED fresh]`, exit 0. After the future
`sleeve.enabled=true` flip, the existing conditional daily coverage keeps
SGOV fresh; this script is one-shot, idempotent, and safe to re-run.

## Rollback

Revert the commit. The script is standalone tooling; the
`sleeve_prices.py` refactor is behavior-preserving (regression-pinned).
