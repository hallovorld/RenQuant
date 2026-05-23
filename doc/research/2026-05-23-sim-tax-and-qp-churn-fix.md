# 2026-05-23 Sim Tax Cash Mode and QP Churn Fix

## Problem

Two separate issues were coupled in prior simulation diagnostics:

1. `SimAdapter` always debited estimated capital-gains tax from cash on each
   sell. That is useful as an event-level stress path, but it is not broker-like
   cash accounting and it can change later sizing/QP decisions. LEAN/live do not
   remove capital-gains tax from broker cash at trade time.
2. After hidden QP tax/LT sell suppression was disabled, the QP revealed a
   turnover problem: the objective underpriced trading cost (`qp_cost_kappa`
   was 0.0001 while round-trip fee+slippage is 0.002), so marginal signals could
   produce too much churn.

These are different problems. The fix must not reintroduce tax-driven hold
logic.

## Fix

### Tax cash mode

`tax.cash_debit_mode` now controls how per-trade tax estimates affect sim cash:

- `event_level`: legacy stress mode; profitable realized sells debit estimated
  tax immediately.
- `reporting_only`: live-like mode; trade rows still record the tax estimate,
  but simulated broker cash is not reduced. Annual-net reporting subtracts the
  estimated calendar-year net tax at year end.

Production `renquant_104` config uses `reporting_only`. Reports now expose both:

- `event_level_tax_estimate`: per-trade tax estimate.
- `tax_cash_debited`: amount actually removed from simulated cash.

The annual-net curve adds back only tax that was actually debited, avoiding the
previous double-count/inflation risk when tax is reporting-only.

### QP churn control

The QP now floors its L1 turnover penalty at round-trip fee+slippage when
`qp_cost_kappa_floor_round_trip=true`.

Current config:

- `fee_pct=0.0005`
- `slippage_pct=0.0005`
- `qp_cost_kappa=0.002`
- `qp_cost_kappa_floor_round_trip=true`
- global `qp_turnover_max=0.20`
- `regime_params.BULL_CALM.qp_turnover_max=0.15`

This keeps BULL_CALM buys enabled while blocking high-turnover small-edge
rebalances. QP knobs including cost, min-delta, no-trade band, and turnover now
support regime overrides through `_qp_cfg`.

## Scientific Basis

- IRS capital-gains tax is an annual reporting/payment overlay, not an Alpaca
  cash debit on each sell. Immediate debit is a stress scenario, not the live
  account path.
- Gârleanu and Pedersen (2013), "Dynamic Trading with Predictable Returns and
  Transaction Costs": proportional transaction costs imply gradual trading and
  no-trade regions.
- Davis and Norman (1990) and Constantinides (1986): proportional costs create
  inaction bands around target allocations.
- cvxportfolio/Boyd single-period optimization practice: transaction costs are
  represented as convex turnover penalties plus hard risk/budget constraints.

## Regression Tests

Targeted:

```bash
.venv/bin/python -m pytest \
  tests/test_long_pnl_vectorbt_validator.py \
  tests/test_sim_result_perf_triple.py \
  tests/test_sim_trade_ledger.py \
  tests/test_qp_contracts.py \
  tests/test_qp_cfg_per_regime_override.py \
  tests/test_joint_qp_task.py -q
```

Result: 70 passed.

Broader:

```bash
.venv/bin/python -m pytest tests/test_golden_preservation.py \
  tests/test_qp_integration.py tests/test_portfolio_qp_solver.py \
  tests/test_short_cover_tax.py -q
```

Result: 70 passed, 6 skipped.

Full:

```bash
.venv/bin/python -m pytest -q
```

Result: 12598 passed, 8791 skipped, 1 xfailed.
