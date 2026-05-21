# RenQuant 104 Daily E2E Fix Report — 2026-05-21

Scope: fixes applied after the daily e2e decision-tree audit for the 2026-05-20
PDT live run. This is written as an operator/agent handoff note, not as a
claim of model alpha.

## What Was Fixed

### 1. Decision trace tables are now run-scoped

Pre-fix, `ticker_daily_state`, `score_distribution`, and
`score_percentiles_daily` were keyed by date. A later same-day smoke/e2e run
could overwrite the real daily decision tree.

Fix:

- `ticker_daily_state` primary key changed to `(run_id, ticker)`.
- `score_distribution` primary key changed to `(run_id, ticker)`.
- `score_percentiles_daily` primary key changed to `run_id`, with `date` kept
  as a query field.
- Existing date-keyed rows are preserved under synthetic `legacy-YYYY-MM-DD`
  run ids.
- Runner and sim adapters stamp `ctx.run_id` and pass it through persistence.

Files:

- `backtesting/renquant_104/kernel/persistence.py`
- `backtesting/renquant_104/kernel/pipeline/task_score_distribution.py`
- `backtesting/renquant_104/adapters/runner.py`
- `backtesting/renquant_104/adapters/sim.py`

### 2. Buy-blocked runs still build an auditable candidate set

Pre-fix, when `ctx.buy_blocked=True`, the pipeline skipped the buy scan. That
made the DB look like there were no candidates rather than showing that buys
were deliberately suppressed by a regime gate.

Fix:

- Full inference now scans candidates when `score_db.scan_when_buy_blocked`
  is unset or true.
- Order emission remains gated; this does not re-enable real buys.
- `SizeAndEmitTask` also stamps selected-but-gated names as `buy_blocked` or
  `skip_buys`.

Files:

- `backtesting/renquant_104/kernel/pipeline/pp_inference.py`
- `backtesting/renquant_104/kernel/pipeline/task_selection.py`

### 3. QP no-trade reasons are now ticker-level telemetry

Pre-fix, QP only logged aggregate skip counts. `ticker_daily_state.blocked_by`
could be blank for top candidates that QP did not allocate.

Fix:

- QP stamps per-candidate reasons into `ctx._blocked_by_ticker`, including:
  `qp_no_trade_band`, `qp_delta_below_min_dw`, `qp_zero_shares`,
  `qp_no_buy_delta`, `qp_nonfinite_delta`, `buy_blocked`, `skip_buys`,
  `earnings`, and `qp_not_selected`.
- Live no-trade summary now prefers QP reasons over the old `tier_threshold`
  fallback.

Files:

- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
- `live/runner.py`

### 4. LIVE Alpaca account guard now fails closed

Pre-fix, missing `RENQUANT_EXPECTED_LIVE_ACCOUNT` only logged a warning.

Fix:

- LIVE Alpaca connections now require `RENQUANT_EXPECTED_LIVE_ACCOUNT`.
- A mismatch raises before trading.
- Connect logs both settled cash and `non_marginable_buying_power`.
- `.env` now includes the expected live account pin.

Files:

- `live/alpaca_broker.py`
- `.env`

### 5. Daily audit cash uses non-margin buying power

Pre-fix, the daily audit snapshot logged settled `cash`, while the broker
sizing path uses `non_marginable_buying_power` to include T+2-available cash
without using margin leverage.

Fix:

- `scripts/daily_104.sh` audit row now records `cash` from
  `non_marginable_buying_power` and `settled_cash` separately.

### 6. Run-level buy-gate state is now structured telemetry

The 2026-05-20 decision tree exposed a gap: logs said
`no trade (buy_blocked)`, while `pipeline_runs` did not persist the root
`buy_blocked`, `skip_buys`, or `bear_only` flags and ticker rows only showed
downstream Kelly/QP reasons.

Fix:

- `pipeline_runs` now stores `buy_blocked`, `skip_buys`, `bear_only`, and
  `counters_json`.
- Live and sim adapters write those run-level fields.
- Live no-trade ntfy only reports a global buy gate as the cause when QP
  actually reached and suppressed a buy-sized delta. When QP produced no
  actionable buy delta, the notification keeps the QP reason.

### 7. Shadow trade notifications are explicitly hypothetical

The post-run shadow path uses a read-only Alpaca wrapper, but the notification
title still used the same `TRADE` tag as live orders. That was operationally
confusing.

Fix:

- Shadow action notifications now use `SHADOW-ACTION`.
- Shadow bodies start with `SHADOW/HYPOTHETICAL (no live orders)`.
- Shadow actions use default ntfy priority, not high-priority live-trade
  priority.

Files:

- `live/runner.py`
- `tests/test_runner_trade_ntfy.py`

## Validation

### Regression tests

Command:

```bash
./.venv/bin/pytest \
  tests/test_score_distribution.py \
  tests/test_ticker_daily_state.py \
  tests/test_p0_fixes_regression_guards.py \
  tests/test_no_trade_monitor.py \
  tests/test_regime_alpha_gate.py \
  tests/test_joint_qp_task.py \
  tests/test_runner_ranking.py \
  tests/test_buy_emit_contract.py \
  tests/acceptance/jobs/test_split_jobs.py -q
```

Result:

- `121 passed, 1 xfailed`

Targeted post-override regression command:

```bash
./.venv/bin/pytest \
  tests/test_runner_trade_ntfy.py \
  tests/test_persistence.py \
  tests/test_regime_alpha_gate.py \
  tests/test_config_consistency.py \
  tests/test_preflight.py -q
```

Result:

- `113 passed`

### Daily e2e primary run

Command:

```bash
bash scripts/daily_104.sh
```

Primary live run:

- Local time: 2026-05-20 20:21-20:41 PDT
- DB: `data/runs.alpaca.db`
- `run_id`: `2026-05-20-live-2858f63b`
- Regime: `BULL_CALM`, confidence `0.583338923108727`
- Account guard: passed
- Cash field persisted: `7760.95`
- Portfolio value: `10741.49`
- Candidates persisted: `70`
- Exits: `0`
- Buys: `0`
- `score_distribution`: `76` rows (`70` candidates + `6` holdings)
- `ticker_daily_state`: `142` rows
- Notification reason: `no trade (buy_blocked)`

Important: at the time of this run buys were blocked by existing config:

```json
regime_params.BULL_CALM.disable_new_buys = true
```

The configured reason was the 2026-05-20 truly-OOS evaluation:
`PROD top-10 alpha = -0.045` in `BULL_CALM`, `IC = +0.005 ≈ 0`, over
`234` BULL_CALM OOS days. The operator override below changes that gate
after this audited run.

Primary run blocked-by distribution:

- `qp_delta_below_min_dw`: 45
- blank/no blocker: 44
- `universe_floor`: 28
- `kelly_zero:mu_le_min_edge`: 20
- `qp_no_trade_band`: 5

### Shadow e2e run

Shadow run:

- DB: `data/runs.alpaca_shadow.db`
- `run_id`: `2026-05-20-live-bce1fbe8`
- Broker: `readonly-alpaca`
- Account guard: passed
- Candidates persisted: `87`
- Exits: `1` hypothetical shadow exit (`FTNT`, `qp_sell`)
- Buys: `0`
- `ticker_daily_state`: `142` rows

The shadow exit is not a real broker order. `ReadOnlyBrokerWrapper` forwards
read-side calls to Alpaca but swallows `place_order`, `place_stop_order`, and
`cancel_order`, returning synthetic filled responses for downstream accounting.

## Follow-up Operator Override

After the decision-tree audit, `BULL_CALM` new buys were re-enabled in both
the live and golden 104 configs:

```json
regime_params.BULL_CALM.disable_new_buys = false
```

This opens new-stock buys in the normal-risk accumulation regime without
removing downstream QP, Kelly, min-dw, cash, wash-sale, earnings, and quality
constraints. The truly-OOS warning remains a known model-risk signal; a
graduated risk policy is still the preferred next design over a whole-regime
kill switch.

### Post-Override Daily Rerun

Command:

```bash
bash scripts/daily_104.sh
```

Primary live run after enabling BULL_CALM buys:

- Local time: 2026-05-20 21:24-21:45 PDT
- DB: `data/runs.alpaca.db`
- `run_id`: `2026-05-20-live-9694f79f`
- Regime: `BULL_CALM`, confidence `0.583338923108727`
- Account guard: passed
- Candidates persisted: `70`
- Exits: `0`
- Buys: `0`
- `pipeline_runs.buy_blocked`: `0`
- `pipeline_runs.skip_buys`: `0`
- `pipeline_runs.bear_only`: `0`
- `score_distribution`: `76` rows (`70` candidates + `6` holdings)
- `ticker_daily_state`: `142` rows
- `trades`: `0`
- Notification reason: `no trade (qp_delta_below_min_dw(70))`

Interpretation: buys are now open. The rerun still placed no new live buys
because the joint QP/minimum-adjustment layer found no candidate above the
2% minimum delta threshold, not because the regime gate disabled buying.

Primary run candidate block distribution:

- `qp_delta_below_min_dw`: 45
- `kelly_zero:mu_le_min_edge`: 20
- blank/holding rows: 6
- `qp_no_trade_band`: 5

Shadow run after the primary daily rerun:

- DB: `data/runs.alpaca_shadow.db`
- `run_id`: `2026-05-20-live-fbe66f7e`
- Candidates persisted: `87`
- Exits: `1` hypothetical shadow exit (`FTNT`, `qp_sell`)
- Buys: `0`
- `trades`: `1` in shadow DB only
- `score_distribution`: `93` rows (`87` candidates + `6` holdings)
- `ticker_daily_state`: `142` rows

Shadow interpretation: the HF PatchTST primary-shadow path saturated the
calibrator and abstained from new buys; it emitted one hypothetical FTNT trim
inside the shadow ledger only.

## Residual Issues / Next Work

1. The truly-OOS BULL_CALM warning should be handled with graduated
   de-risking: raise buy floors, reduce size, require shadow agreement, or
   require regime-local IC recovery rather than hard-disabling the whole regime.

2. `kernel/panel_pipeline/feature_matrix.py` emits many pandas fragmentation
   warnings during daily/shadow e2e. This is a real performance smell: the
   frame should be assembled with batched `pd.concat(axis=1)` instead of repeated
   column insertion.

3. Shadow notifications currently say `[SHADOW] ... TRADE | EXIT ...`. They are
   safe because the broker is read-only, but the wording should be sharpened to
   `HYPOTHETICAL` or `SHADOW-ONLY` to prevent operator confusion.

4. Same-day legacy rows remain under `legacy-YYYY-MM-DD` run ids. That preserves
   history, but analytics should prefer explicit `pipeline_runs.run_id` joins.

## Agent Guidance

Do not mark this area fixed from logs alone. For every future daily e2e claim,
verify all three layers:

1. Log evidence: account guard, gate reason, candidate scan, QP summary, ntfy.
2. DB evidence: `pipeline_runs`, `score_distribution`, `ticker_daily_state`,
   and `trades`, keyed by the same `run_id`.
3. Broker safety evidence: real broker for primary, `readonly-alpaca` for shadow.
