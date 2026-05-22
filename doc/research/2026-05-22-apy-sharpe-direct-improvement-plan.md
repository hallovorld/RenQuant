# APY / Sharpe Direct Improvement Plan

Date: 2026-05-22

## Problem Statement

The prior audit work made the pipeline more trustworthy, but reliability
patches do not directly improve APY or Sharpe. The next design loop must start
from the performance identity:

```text
net_return = signal_edge * capital_deployed
             - turnover_cost
             - tax_drag
             - drawdown / volatility drag
```

Any proposed change must state which term it improves and how it will be
measured. Otherwise it is engineering hygiene, not a performance proposal.

## CLAUDE.md Constraints

- New decision logic must enter through Task -> Job -> Pipeline.
- Reported APY / Sharpe / IC cannot be a single unverified number.
- Parameter bounds must come from empirical distributions, not round-number
  intuition.
- Promotion follows the documented 3-tier methodology: reject if mean APY and
  Sharpe both deteriorate; screen only if APY improves and Sharpe is non-worse;
  live promotion requires DSR/PBO or sufficient sample evidence.
- Code is source of truth. Docs explain, but cannot override Job/Task bodies.

## First Forensic Evidence

Implemented:

- `scripts/analyze_trade_decision_attribution.py`
- `tests/test_trade_decision_attribution.py`

The analyzer pairs executed buys and sells into FIFO long round trips, preserves
entry attribution payloads when available, and groups P&L by:

- entry order type
- entry source job
- entry regime
- exit reason
- hold-time bucket
- ticker
- entry rank-score quantile

### Live Rows: Small Sample, Not Decisive

Command:

```bash
.venv/bin/python scripts/analyze_trade_decision_attribution.py \
  --db data/runs.alpaca.db \
  --run-type live \
  --min-n 20
```

Result:

- 110 live trade events
- 18 matched round trips
- win rate 55.6%
- gross/net P&L -$175
- profit factor 0.65

This sample is too small for model conclusions, but it is enough to show that
live attribution must keep accumulating with the new DB decision payload.

### Current Sim DB: Edge Exists, Net Edge Is Mostly Eaten

Command:

```bash
.venv/bin/python scripts/analyze_trade_decision_attribution.py \
  --db data/sim_runs.db \
  --run-type sim \
  --min-n 20
```

Result:

- 865 matched round trips
- win rate 66.5%
- gross P&L +$142,399
- tax +$133,252
- net P&L +$9,147
- profit factor 1.08

Interpretation: this is not yet a "no signal" diagnosis. It is a weak net-edge
diagnosis: gross edge is present, but tax / churn / exit policy consumes almost
all of it.

Major negative groups:

- `stop_loss`: 90 trades, net -$76,198, win rate 0%
- `00-05d` holds: 30 trades, net -$17,186
- `06-20d` holds: 256 trades, net -$17,053
- `CHOPPY`: 28 trades, net -$3,486
- worst tickers in this trace: SNOW, OKTA, MCD, LLY, ZM

Positive groups:

- `model_sell`: net +$33,575
- `qp_sell`: net +$44,863
- `trailing_stop`: net +$9,937
- `061-180d`, `181-365d`, and `366d+` hold buckets are positive

## Direct Improvement Hypotheses

### 1. Tax / Turnover Must Become a First-Class Objective

Target term: reduce `tax_drag` and churn.

Evidence: current sim gross +$142k becomes net +$9k after tax. A strategy with
positive gross edge can still have poor APY / Sharpe if it realizes short-term
gains too often.

Design direction:

- Add a pre-trade hurdle: expected edge must exceed estimated tax + spread +
  slippage + volatility risk.
- Make turnover penalty explicit in QP objective, not only post-hoc.
- Keep HIFO / tax-aware lot logic, but verify it is active in all sim/live
  paths.

Reference pattern: cvxportfolio separates expected return, risk, transaction
cost, holding cost, and constraints.

### 2. Exit Policy Needs Counterfactual Replay Before Another Tuning Pass

Target terms: increase average trade return, reduce realized volatility drag.

Evidence: stop-loss exits are deeply negative, while longer holds are positive.
This does not mean "disable stop-loss." It means the current stop policy may be
realizing losses without proving that it prevents worse tail outcomes.

Required next step:

- Build a counterfactual exit replay: for every stopped trade, compute what
  would have happened under hold-to-20d, hold-to-60d, ATR stop, trailing-only,
  and regime-specific stop variants.
- Derive stop thresholds from the empirical distribution of adverse excursion,
  not from round numbers.

Reference pattern: triple-barrier labeling and meta-labeling separate entry
signal from path-dependent exit outcome.

### 3. Early Holding Buckets Are Hurting Net Performance

Target term: improve `signal_edge * capital_deployed` by letting valid entries
develop and avoiding churn.

Evidence: 0-5d and 6-20d hold buckets are negative in current sim, while 61d+
holding buckets are positive.

Design direction:

- Test a post-entry no-touch window for model/QP exits, while keeping true
  catastrophic risk exits active.
- Test separate handling for top-up/trim versus full liquidation.
- Do not change this globally until counterfactual replay proves the trade-off
  versus drawdown.

### 4. CHOPPY Deployment Should Be Cut or Reframed

Target terms: reduce volatility drag and bad capital deployment.

Evidence: CHOPPY round trips are negative in the current trace.

Design direction:

- CHOPPY should not use the same entry/exit/sizing contract as BULL_CALM.
- Either lower exposure until CHOPPY has positive OOS edge, or require a
  stronger relative-value / mean-reversion-specific meta-label.

### 5. Ticker Admission Should Use Shrunk Trade-Level Evidence

Target term: reduce recurring negative contributors without overfitting.

Evidence: some tickers are repeatedly negative in current trace, but naive
blacklists overfit.

Design direction:

- Use Bayesian/shrunk estimates of per-ticker expectancy and profit factor.
- Require minimum sample count and out-of-window confirmation.
- Feed this as a universe floor or sizing multiplier, not a hard permanent
  blacklist.

### 6. Rank Score Needs Regime-Aware Calibration, Not Blind Averaging

Target term: improve capital allocation to higher expected return trades.

Evidence:

- Older large sim rows in `runs.alpaca.db` show rank quantiles are monotonic
  positive.
- Current `sim_runs.db` rank quantiles are not cleanly monotonic.

Design direction:

- Refit rank-to-realized-return calibration by regime and source.
- Use isotonic or Platt-style calibration only where sample size supports it.
- Size by calibrated expected net return, not raw rank alone.

## Next Implementation Order

1. Land trade attribution analyzer and keep it read-only.
2. Build exit counterfactual replay with tests.
3. Convert counterfactual results into empirical stop / no-touch bounds.
4. Add cost/tax hurdle into QP objective or pre-QP admissibility, using one
   shared function.
5. Add regime-specific deployment rules only after replay/WF confirms they
   improve APY and do not worsen Sharpe.
6. Run acceptance as paired A/B across fixed windows, with SPY and current prod
   as references. Report mean +/- std, DSR/PBO where applicable.

## Current Status

Done in this slice:

- Read-only trade decision attribution analyzer.
- Unit tests for legacy schema, decision payload preservation, FIFO pairing,
  win rate, and profit factor.
- Read-only exit counterfactual replay:
  - `scripts/analyze_exit_counterfactuals.py`
  - `tests/test_exit_counterfactuals.py`
- First local diagnostics written to the paths below. These are intentionally
  local generated artifacts because they can contain real trade history:
  - `artifacts/trade_decision_attribution_alpaca_all.json`
  - `artifacts/trade_decision_attribution_alpaca_since_2026-05-01.json`
  - `artifacts/trade_decision_attribution_alpaca_db_sim_rows.json`
  - `artifacts/trade_decision_attribution_sim_runs_current.json`
  - `artifacts/exit_counterfactuals_sim_runs_current.json`

## Exit Counterfactual Replay Result

Command:

```bash
.venv/bin/python scripts/analyze_exit_counterfactuals.py \
  --db data/sim_runs.db \
  --run-type sim \
  --horizons 20,60,120 \
  --barrier-window 20 \
  --min-n 20
```

Result summary:

| Group | Actual net P&L | Hold-20d delta | Hold-60d delta | Hold-120d delta | Interpretation |
|---|---:|---:|---:|---:|---|
| all exits | +$9,147 | -$37,090 | +$46,377 | +$276,612 | Not all exits are too early; 20d blanket hold hurts. |
| `stop_loss` | -$76,198 | +$12,004 | +$6,847 | +$50,204 | Stop-loss is the main false-positive suspect. |
| `single_day_loss` | -$7,591 | +$849 | +$23,147 | +$37,733 | SDL also looks too aggressive in some paths. |
| `model_sell` | +$33,575 | -$31,866 | -$13,617 | +$59,048 | Model sell is useful at 20d/60d horizon. |
| `qp_sell` | +$44,863 | -$18,978 | +$10,159 | +$68,318 | QP sell is useful at 20d, mixed longer. |
| `CHOPPY` | -$3,486 | -$1,353 | +$5,481 | +$14,034 | CHOPPY needs separate deployment logic. |

Interpretation:

- The result does **not** support a blanket "hold longer" rule.
- It does support targeted work on path-rule exits, especially `stop_loss` and
  `single_day_loss`.
- `model_sell` and `qp_sell` should not be casually weakened; they improve
  short-horizon net P&L in this trace.
- Longer 120d counterfactual gains are interesting but not automatically
  promotable; they may increase drawdown, capital lockup, and benchmark risk.

Not done yet:

- Tax/cost hurdle integration.
- Any production behavior change.
