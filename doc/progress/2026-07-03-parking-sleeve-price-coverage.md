# Parking-sleeve SPY/SGOV daily price coverage (st104 #39 follow-up)

STATUS: open PR (feat/sleeve-symbol-coverage). Flag-off byte-inert — the
`sleeve` config section defaults to `enabled=false` (and is not yet present in
the umbrella's shipped config at all), so this PR changes nothing live.

## Why

The S7 parking sleeve (renquant-pipeline #157, `ParkingSleeveShadowTask`;
config keys defined inert by renquant-strategy-104 #39) prices two legs:
`sleeve.spy_symbol` (default SPY) and `sleeve.sgov_symbol` (default SGOV).
Strategy-104 #39 pinned the coverage decision: SGOV must NOT join the
watchlist (a T-bill ETF would enter panel scoring and cross-sectional
admission stats), so daily price coverage is umbrella-owned — SPY is already
covered everywhere as benchmark/watchlist member, SGOV was fetched nowhere.
Until this lands, #157's shadow tolerates the missing SGOV price (logs
`qty: null`, SGOV leg tracked at cost); this PR is the declared follow-up
that must land before or with any sleeve enable.

## What changed

Mirrors the existing `benchmark_sleeve` conditional-coverage precedent
(main.py `AddEquity`s the sleeve ticker only when that sleeve is enabled;
lean/runner/sim append it to their fetch universes the same way):

* `backtesting/renquant_104/adapters/sleeve_prices.py` (NEW) —
  `parking_sleeve_price_tickers(config)`: `[spy_symbol, sgov_symbol]`
  (upcased, deduped) when `sleeve.enabled`, else `[]`. Single umbrella
  implementation used by every price path (the calibrator-fingerprint
  triple-impl incident is why this is one function, not four inline reads).
  Key names, defaults, and strip/upper normalization mirror the pipeline
  task's reads exactly. Lives in `adapters/` (not `kernel/pipeline/`)
  because the multirepo bridge aliases all `kernel.*` modules to the pinned
  renquant-pipeline checkout — an umbrella-only helper under `kernel/`
  would be shadowed on the live path; `adapters/*` is umbrella-owned on
  every path.
* `backtesting/renquant_104/main.py` — after the benchmark-sleeve
  `AddEquity` block: conditionally `AddEquity` each parking leg with the
  same dedup guards (skip if == benchmark / in watchlist / already a
  sector-ETF symbol); subscribed symbols land in `_sector_etf_symbols`,
  same as the benchmark-sleeve ticker.
* `backtesting/renquant_104/adapters/lean.py` — `make_context` appends the
  legs to the batch-History ticker list (right after the benchmark-sleeve
  append; downstream dedup unchanged).
* `backtesting/renquant_104/adapters/runner.py` — live `make_context`
  appends the legs to `extra_symbols` feeding the `fetch_ohlcv` loop
  (immediately after the benchmark-sleeve `extra_symbols` append).
* `backtesting/renquant_104/adapters/sim_price.py` —
  `context_price_tickers` appends the legs so sim pricing stays aligned
  with live/LEAN (same rationale as the benchmark sleeve: optional-sleeve
  logic must not silently no-op only in research).

Deliberately NOT done: no watchlist changes (st104 #39 pin), no
`managed_symbols` change in runner.py (shadow mode creates no broker
positions), no duplication of #157's missing-SGOV tolerance (owned by the
pipeline task), no port of `ParkingSleeveShadowTask` into the umbrella
kernel mirror (the live daily runs the pipeline's copy via the bridge
aliases; the umbrella-kernel rollback path has no parking-sleeve task to
feed, and coverage there is inert until one exists).

## Tests

New `tests/test_sleeve_prices.py` (16 tests across 3 classes):

* helper unit tests — flag absent/off/malformed ⇒ `[]`; flag-on ⇒ both
  legs with defaults, custom-symbol normalization, blank-symbol fallback,
  identical-leg dedup, ctx-like `.config` objects;
* `context_price_tickers` behavior — flag-off output is byte-identical to
  the no-section output; flag-on appends both legs, still deduped against
  benchmark/watchlist;
* source pins for main.py / lean.py / runner.py / sim_price.py wiring
  (same convention as `TestAdapterParity` in test_universe_alignment.py —
  LEAN main.py is not importable under plain pytest).

Ran (renquant .venv py3.10, scratchpad clone): `pytest
tests/test_sleeve_prices.py tests/test_sim_price.py
tests/test_benchmark_sleeve.py` → 40 passed; adjacent suites
`test_universe_alignment test_lean_backend test_lean_policies
test_lean_price_resolution test_lean_preflight_wiring
test_lean_data_normalization test_lean_execution_model test_runner_trace
test_runner_ext_sell test_runner_sell_attribution test_partial_sell
test_bug22_rs_score_keyerror test_live_multirepo_entrypoints` → 326 passed;
`test_lean_guard` → 13 passed (needs sibling renquant-pipeline on
PYTHONPATH, pre-existing). `py_compile` clean on all touched files. Smoke:
shipped strategy config (no `sleeve` section) ⇒ helper returns `[]` and
the sim universe is unchanged; injected `{"sleeve": {"enabled": true}}` ⇒
appends exactly `SGOV` (SPY deduped against benchmark).
