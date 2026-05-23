# 2026-05-23 QP Tax-Gate Resim Results

## Question

After fixing the hidden QP tax soft-sell gate, did APY and Sharpe improve?

Answer: no. The contract fix made the decision tree more honest, but it exposed
QP churn. The prior hidden tax gate was acting as an undocumented turnover
brake.

## Comparable Runs

Both runs use XGB true-OOS diagnostic config:

- Sim window: 2024-07-02 to 2026-02-10
- Panel artifact: `artifacts/walkforward_truly_oos_2024-07-01_embargo60_20260522/panel-ltr.json`
- Historical, single-run diagnostic only; not a promotion claim.

| Run | Final | APY | Sharpe | MaxDD | Event tax | Annual-net APY | Annual-net Sharpe | Buys | Sells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QP horizon fix only | 110364.57 | 6.36% | 0.625 | 10.46% | 10958.02 | 10.70% | 1.082 | 85 | 53 |
| QP horizon + tax soft-sell contract | 96768.94 | -2.03% | -0.137 | 10.32% | 18242.01 | 6.31% | 0.656 | 231 | 172 |

## What Changed

Exit reasons shifted hard toward QP churn:

| Exit reason | Horizon-only sells | After tax-gate fix sells |
|---|---:|---:|
| stop_loss | 24 | 37 |
| trailing_stop | 20 | 24 |
| panel_conviction | 5 | 4 |
| model_sell | 1 | 1 |
| single_day_loss | 0 | 1 |
| qp_sell | 2 | 64 |
| qp_close | 1 | 41 |

The fix removed hidden tax suppression from QP sells, and QP immediately started
issuing many more short-horizon trims/closes. Those QP exits were gross-positive
in aggregate, but they created heavy event-level tax drag and churned the
portfolio enough to lower event-level and annual-net performance.

## Interpretation

Do not restore the hidden tax gate as the solution. That would violate the
`qp_tax_aware=false` contract and would hide a portfolio-construction problem
behind tax logic.

The scientific fix should be explicit and testable:

- A stronger transaction-cost / turnover objective inside the optimizer.
- A no-trade or soft-sell edge gate based on expected-return advantage, not tax.
- Regime-conditional QP sell aggressiveness, because BULL_CALM and CHOPPY have
  different thesis half-lives.
- Report QP proposed sells, emitted sells, and suppressed sells separately so
  the decision tree cannot hide churn.

## Remaining Issue

`panel_conviction` tax-adjusted suppression still fires in the full pipeline.
That is outside the QP tax contract fixed here. It needs a separate A/B:

- tax-adjusted panel exit on
- tax-adjusted panel exit off
- explicit non-tax replacement based on transaction cost and expected edge

Acceptance should be regime-stratified and should report APY, Sharpe, MaxDD,
turnover, median hold, gross PnL, event-level tax, and annual-net tax.
