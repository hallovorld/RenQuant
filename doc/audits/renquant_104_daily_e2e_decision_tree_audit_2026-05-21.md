# RenQuant 104 Daily/E2E Decision Tree Audit

Date: 2026-05-21

Scope: recent `renquant_104` daily/live/e2e decision path around 2026-05-20.
I treated docs as pointers only. Evidence comes from live logs, SQLite run DB,
current code, git diffs, and targeted regression tests.

## Executive Verdict

The two Claude-reported P0 bugs are not equally fixed:

1. **P0-9 Alpaca cash semantics: fixed in code and visibly active.**
   `live/alpaca_broker.py::get_cash()` now uses
   `account.non_marginable_buying_power` before falling back to `account.cash`.
   DB evidence also shows the cash value changed from settled cash around
   `$5,690` to buying-power cash around `$7,740-$7,757` on later 2026-05-20
   live rows.

2. **P0-10 LIVE account assertion: only partially operational.**
   The code now raises if `RENQUANT_EXPECTED_LIVE_ACCOUNT` is set and mismatches.
   But the latest e2e log says the env var is **not set**, so production still has
   no positive live-account pin. That is a warning-only guard, not a hard safety
   invariant.

A third bug surfaced in the same decision tree:

3. **`no_trade_streak` false inflation: patched in working tree, but not fully
   committed in the current worktree.**
   The runner now overrides the stateful counter with broker-filled-order truth,
   and `task_monitor.py` increments only once per trading day. Tests pass. But
   these files are still modified in the working tree, so the fix is not a clean
   reproducible baseline unless committed.

## What Actually Happened On 2026-05-20

There are two different paths that must not be confused:

### Full daily path, 14:17 / 15:34 PT

Logs:

- `logs/daily_104/2026-05-20.log`
- Full panel path ran.
- `ApplyScoresTask` scored 76 tickers.
- 70 candidates and 6 holdings were calibrated.
- Kelly: 50 candidates non-zero target, 20 candidates blocked by
  `kelly_zero:mu_le_min_edge`.
- QP emitted 0 buys and 0 sells.
- QP skipped 6 trades by no-trade band.
- Notification said `no trade (tier_threshold)`.

DB evidence:

- `pipeline_runs` rows:
  - `2026-05-20-live-56e526a1`: `n_candidates=70`, `n_buys=0`
  - `2026-05-20-live-a543eb33`: `n_candidates=70`, `n_buys=0`
- Top candidates in `candidate_scores` for `2026-05-20-live-a543eb33`:
  `CRWD`, `AFRM`, `MCD`, `HPE`, `TXN`.
- These top candidates had `selected=0` but mostly blank `blocked_by`.

Interpretation: this was **not** a no-candidate failure. It was a portfolio
construction / QP / no-trade-band / current-holdings decision, but the operator
message collapsed it to `tier_threshold`, which is misleading.

### Post-gate live e2e path, 18:57 PT

Log:

- `logs/live_e2e/post_gate_20260520-173840.log`
- `RegimeAlphaGateTask`: `BULL_CALM` with `disable_new_buys=True`.
- Candidate scan was skipped.
- Only 6 holdings were scored.
- QP emitted 0 buys and 0 sells.
- Notification said `no trade (buy_blocked)`.

Interpretation: this is a **gate-blocked e2e run**, not a complete buy-decision
run. It validates that the hard gate suppresses buys; it does not validate that
the full buy funnel still behaves well.

## Decision Tree Problems Found

### 1. `ticker_daily_state` destroys same-day decision history

Schema primary key:

```sql
PRIMARY KEY (date, ticker)
```

The 14:17/15:34 full daily run wrote 142 rows with 70 candidates. The 18:57
post-gate e2e run later wrote 142 rows for the same date with 0 candidates,
overwriting the full daily decision tree.

Current DB query for 2026-05-20 now says:

- rows: 142
- candidates: 0
- held: 6
- blocked: 28

That is not the true full daily decision tree. It is the later gate-test snapshot.

Fix: add `run_id` to `ticker_daily_state` and make the key
`PRIMARY KEY (run_id, ticker)`, or keep a separate latest-view table and preserve
append-only run history.

### 2. `score_distribution` is date-keyed and can become a mixed/stale table

`score_distribution` also uses:

```sql
PRIMARY KEY (date, ticker)
```

`RecordScoreDistributionTask` does `INSERT OR REPLACE` for tickers seen in the
current run, but it does not delete same-date rows from prior runs. A holdings-only
e2e run can update the 6 holdings while leaving the previous 70 candidate rows
behind. `score_percentiles_daily` also only updates when `cand_scores` exists.

Result: date-level score distribution can silently mix rows from different runs.

Fix: same as above: make score distribution run-scoped. At minimum, delete same
date rows inside a transaction before inserting a new date snapshot, but run_id is
the cleaner design.

### 3. No-trade explanation is wrong for QP path

`live/runner.py::_why_no_trade()` falls through to `tier_threshold` whenever:

- not `buy_blocked`
- not `bear_only`
- not drawdown/transition/wash/correlation counters
- ranked list exists
- no orders

For the 15:34 full daily run, the logs show QP/no-trade-band behavior, not a
tier threshold failure. The message `no trade (tier_threshold)` is therefore not
trustworthy for QP-enabled runs.

Fix: QP should write structured counters such as `qp_skipped_band`,
`qp_zero_delta_candidates`, `qp_candidate_buy_delta_lt_min_share`,
`qp_buy_blocked_by_budget`, and `_why_no_trade()` should prefer those before
falling back to `tier_threshold`.

### 4. Candidate-level final block reason is incomplete

For `2026-05-20-live-a543eb33`, top candidates like `CRWD`, `AFRM`, `MCD`,
`HPE`, `TXN` had good rank/expected-return fields, `selected=0`, and blank
`blocked_by`.

This makes post-hoc diagnosis impossible: we cannot tell from DB whether each
candidate lost due to QP optimizer preference, no-trade band, integer-share floor,
budget, sector/correlation constraints, or something else.

Fix: after QP solve, stamp each non-selected candidate with final reason.

### 5. BULL_CALM hard buy block is high-impact and uncommitted

Current working config sets:

```json
"regime_params": {
  "BULL_CALM": {
    "disable_new_buys": true
  }
}
```

Local artifact support:

- `backtesting/renquant_104/artifacts/prod/truly_oos_eval/eval_truly_oos.json`
- BULL_CALM `n=234`
- BULL_CALM `ic_mean=0.0053`
- BULL_CALM `top10_alpha=-0.0448`
- Overall `pbo=0.5907`

The BULL_CALM concern is real, but this gate changes strategy behavior radically:
BULL_CALM is a large share of OOS days, and the gate prevents even candidate
generation. This should be treated as a new policy variant requiring A/B validation,
not as a small patch.

Fix: if the goal is safety, keep the gate, but still run candidate generation and
scoring in audit/shadow mode, then suppress orders at execution. That preserves
evidence without taking risk.

### 6. Account guard still depends on an env var that production did not set

Latest e2e log:

```text
RENQUANT_EXPECTED_LIVE_ACCOUNT not set in env — no positive verification of LIVE account identity.
```

Fix: for `paper=False`, missing `RENQUANT_EXPECTED_LIVE_ACCOUNT` should be fatal,
or the deployment checklist must set it and CI/preflight must prove it exists.

### 7. Broker connect log still prints settled `cash`

`Alpaca connected ... cash=$5690.2` logs `account.cash`, while actual dispatchable
cash in DB was about `$7757`. This is now a logging inconsistency after the P0-9
fix and can mislead operators.

Fix: log both `settled_cash` and `non_marginable_buying_power`.

## Regression Tests Run

Command:

```bash
./.venv/bin/pytest tests/test_p0_fixes_regression_guards.py tests/test_no_trade_monitor.py tests/test_regime_alpha_gate.py tests/test_score_distribution.py tests/test_ticker_daily_state.py -q
```

Result:

```text
61 passed in 6.13s
```

This verifies the existing guards, but the tests do **not** yet catch the
run-history overwrite problem because the current schemas are date-scoped by
design.

## Recommended Fix Order

1. Make `ticker_daily_state`, `score_distribution`, and
   `score_percentiles_daily` run-scoped. This is the biggest auditability bug.
2. Make missing `RENQUANT_EXPECTED_LIVE_ACCOUNT` fatal for live mode, or enforce
   it in preflight.
3. Commit or otherwise stabilize the no-trade counter/broker-truth fix.
4. Add QP per-candidate final reason stamping.
5. Change `_why_no_trade()` so QP/no-trade-band reasons beat `tier_threshold`.
6. Re-run BULL_CALM `disable_new_buys` as an explicit policy A/B, because it is
   a strategy change, not just a bug fix.
