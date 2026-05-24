# Sim/Live Parity Audit — 2026-05-24

## Answer

Sim and live do **not** run byte-for-byte identical code.

They are intended to share the same decision kernel:

- Sim: `backtesting/renquant_104/sim/runner.py` builds `SimAdapter`, then runs
  `InferencePipeline().run(ctx)`, then `SimAdapter.commit(ctx)`.
- Live: `live/runner.py` builds `RunnerAdapter`, then runs
  `InferencePipeline()` or `SellOnlyPipeline()`, then
  `RunnerAdapter.commit(ctx)`.
- LEAN: `backtesting/renquant_104/main.py` builds `LeanAdapter`, then runs the
  same `InferencePipeline`, then `LeanAdapter.commit(ctx)`.

The shared kernel includes regime, drawdown gates, sell tasks, candidate
scoring, panel scoring, ranking, joint QP, selection, top-up/trim, benchmark
sleeve, and monitor tasks. That is the part that should define the decision
tree.

The adapters are different by design:

- Data: sim slices historical OHLCV to each bar; live reads current parquet
  cache and optional intraday overlay; LEAN uses `History`.
- State: sim keeps state in-process; live round-trips through
  `live_state.<broker>.json` and DB snapshots; LEAN keeps state in the
  algorithm object.
- Execution: sim fills at simulated prices and models settlement; live sends
  broker orders and handles pending-order/cash/fill failures; LEAN uses QC
  execution primitives.
- Persistence: all three write the same DB contract. BUY and SELL row
  construction now both go through `kernel.trade_events` across
  sim/live/LEAN. Sim still owns adapter-local tax-lot disposal, fees, cash
  debit mode, and settlement mutation before calling the shared SELL builder,
  so execution accounting remains adapter-specific while audit row shape is
  shared.

Therefore the honest status is: **core decision code is shared, adapter
plumbing is not fully unified, and adapter parity needs active tests.**

## Bug Fixed Here

Found a concrete sim/live divergence while auditing:

- `SimAdapter` carried `skip_buys` across bars with `self._skip_buys`.
- `RunnerAdapter` always constructed `InferenceContext(skip_buys=False)`.
- Production config uses drawdown hysteresis via `drawdown_resume_pct`; if live
  was previously halted and current drawdown was between resume and halt,
  live could re-enable buys earlier than sim.

Fix:

- `RunnerAdapter` now reads `skip_buys` from live state via
  `persisted_skip_buys(state)`.
- `RunnerAdapter.commit()` writes `skip_buys` back into live state.
- Regression coverage:
  `tests/test_runner_hwm_guard.py`, `tests/test_pipeline.py::TestDrawdownCircuitTaskResets`,
  `tests/test_joint_qp_task.py`, `tests/test_runner_state_fixes.py`,
  `tests/test_no_trade_monitor.py`, `tests/test_live_state_db_canonical.py`.
- Result: `139 passed`.
- Commit: `e262783 fix(live): persist drawdown buy halt state`.

## Related WF Diagnostic

The sigma-cap WF diagnostic was rerun after fixing the config-builder issue
that silently dropped experiment overrides.

Baseline strict WF trace:

- 56 closed trades.
- Gross `+11238.72`, estimated tax `+10370.53`, net `+868.19`.
- Mean Sharpe `+0.133`, SPY mean Sharpe `+1.081`, delta `-0.948`.

True `BULL_CALM max_sigma=0.38` diagnostic:

- 31 closed trades.
- Gross `+5181.30`, estimated tax `+4477.16`, net `+704.14`.
- Mean Sharpe `+0.255`, SPY mean Sharpe `+1.081`, delta `-0.825`.
- Still fails sanity: real IC `+0.0385`, shuffled `+0.0024`, placebo `+0.0460`.

Conclusion: simple high-sigma admission cap reduces stop-loss count but also
cuts participation and does not solve benchmark-relative alpha or placebo
sanity. It is diagnostic evidence, not production design.

## Prior Bug Fixed In Same Thread

`scripts/wf_config_builder.py` previously derived a production-semantic WF
config from prod and silently dropped side-config experiment overrides such as
`rotation.joint_actions.qp_admission_gate.max_sigma_by_regime`.

Fix:

- Builder now fails closed when a semantic experiment override would be
  dropped.
- `--preserve-experiment-overrides` explicitly carries whitelisted diagnostic
  overrides and marks the generated config as non-production-equivalent.
- Commit: `d45c38b fix(wf): fail closed on dropped experiment overrides`.

## Remaining Parity Risks

1. Context construction is duplicated.
   Live, sim, and LEAN each build `InferenceContext` by hand. Critical fields
   can drift: `skip_buys`, `last_sell_pls`, `last_stop_exit_dates`,
   `pending_broker_tickers`, `_db`, panel frames, calibrators, and state
   carryover. Fixed concrete drift: LEAN now passes `last_sell_pls` and stamps
   it on full exits, so cost-aware wash-sale handling matches sim/live. LEAN
   also now attaches `ctx._db` before the pipeline runs, so DB-aware pipeline
   tasks have the same context surface as sim/live. LEAN now also syncs
   holding share counts and tax lots from `Portfolio`, and sell P&L/tax uses
   the same FIFO/HIFO disposed-basis primitive as sim instead of pro-rating
   aggregate `UnrealizedProfit`.

2. Decision trace writing is mostly shared now.
   `kernel.decision_trace` builds candidate pools, QP maps, selected-buy
   tickers, model types, and `ticker_daily_state` rows for sim/live/LEAN. Keep
   adapter-specific DB calls thin; new decision fields should be added to the
   shared helper first.

3. Exit duplicate resolution is shared now.
   `kernel.pipeline.task_execution.dedupe_exit_signals` resolves duplicate
   same-ticker exits with held-quantity-aware full-liquidation priority. This
   prevents an earlier partial trim from swallowing a later full exit expressed
   as `quantity >= held`.

4. BUY trade-event construction is shared now.
   `kernel.trade_events.build_buy_trade_event` normalizes sim/live/LEAN buy
   rows so score snapshots and decision inputs do not drift across adapters.

5. SELL trade-event construction is shared now.
   `kernel.trade_events.build_sell_trade_event` normalizes sim/live/LEAN sell
   rows, including source attribution, shares, tax/net P&L, score snapshots,
   and applied exit params. Sim still computes lot disposal, fees, settlement,
   and tax-cash debit before calling the builder, then adds those adapter
   accounting fields to the shared audit row.

6. Execution semantics intentionally differ.
   This is not a bug, but every performance report must label whether it is
   simulated close fill, live market fill, LEAN backtest fill, annual-net tax,
   or event-level tax.

7. Reconciliation exists but should become a routine gate.
   `scripts/reconcile_live_sim.py` can compare live fills to sim decisions, but
   it should be promoted from optional report to scheduled/acceptance evidence
   after any live run.

8. Training feature input is now canonicalized.
   The full suite exposed a duplicate-date SPY axis during parallel feature
   construction. `training.features` now coerces OHLCV to the required schema,
   sorts by datetime, keeps the latest duplicate date, and gives each ticker
   build an isolated SPY copy before concurrent execution. This protects
   SPY-relative features and forward labels from ambiguous `reindex` behavior.

## Next Engineering Moves

1. Add an adapter context contract test that checks sim/live/LEAN all populate
   required `InferenceContext` fields for buy/full mode. Completed with
   `tests/test_adapter_context_contract.py`: actual sim, runner, and LEAN
   `make_context()` paths must now expose the shared fields that historically
   drifted (`last_sell_pls`, `last_stop_exit_dates`, DB handle, run id,
   prices, cash/NAV, holdings, and model/data payloads).
2. Keep migrating execution post-processing into shared kernel helpers:
   broker-state mutation is still adapter-heavy. The next useful extraction is
   a lot-disposal/accounting helper that returns disposed basis, gross P&L, tax,
   net proceeds, and settlement/cash-debit fields before adapter mutation.
3. Make live-vs-sim reconciliation run after daily/live cycles and write a
   small divergence report.
4. Continue model-side repair separately: current alpha is still
   placebo-dominated in BULL_CALM and does not beat SPY after the decision
   tree.
