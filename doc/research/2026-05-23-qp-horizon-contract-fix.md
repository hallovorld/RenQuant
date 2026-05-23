# 2026-05-23 QP Horizon Contract Fix

## Summary

This change fixes a portfolio-optimization unit mismatch in renquant_104.

The panel calibrator emits `expected_return` on the label horizon
(`panel_ltr.lookahead_days`, currently 60 trading days). The QP risk fallback
was using realized volatility in annualized units. A single-period Markowitz QP
requires expected return and covariance to describe the same period. Before
this fix, the optimizer was penalizing 60-day expected return with annualized
risk, which made risk too expensive and distorted deployment, exits, and cash
behavior.

## Invariant

For every QP solve:

- `mu` must be an expected-return-like value, not a rank score.
- `sigma` and `Sigma` must be expressed on the same horizon as `mu`.
- A minimum invested constraint may encourage deployment only when the best
  candidate has positive expected edge after a round-trip cost hurdle.

Strict mode blocks the QP when the horizon contract cannot be proven.

## Implementation

Code path:

- `kernel/portfolio_qp/job_qp.py` now wires
  `AlignQPHorizonUnitsTask` between `ValidateQPMuContractTask` and covariance
  construction.
- `kernel/portfolio_qp/tasks.py` scales annualized sigma by
  `sqrt(qp_mu_horizon_days / 252)` when `qp_sigma_unit = "annualized"`.
- `SolveMarkowitzQPTask` now calls an effective minimum-invested helper. If
  `qp_min_invested_requires_positive_edge = true` and best finite `mu` does not
  clear `qp_min_invested_edge_floor`, effective `qp_min_invested_pct` becomes
  zero for that solve.
- `strategy_config.json` and `strategy_config.golden.json` now stamp the QP
  contract:
  - `qp_mu_horizon_days = 60`
  - `qp_sigma_unit = "annualized"`
  - `qp_sigma_horizon_mode = "match_mu"`
  - `qp_horizon_contract = "strict"`
  - `qp_min_invested_requires_positive_edge = true`
  - `qp_min_invested_edge_floor = 0.002`

## Literature Support

- Markowitz (1952), "Portfolio Selection": mean-variance portfolio selection is
  a tradeoff between expected return and variance of portfolio return on the
  same decision period.
  Reference: https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
- Boyd, Busseti, Diamond et al. (2017), "Multi-Period Trading via Convex
  Optimization": single-period optimization trades off expected return, risk,
  transaction cost, and holding cost using period-consistent forecasts.
  Reference: https://arxiv.org/abs/1705.00109
- Magill and Constantinides (1976), "Portfolio selection with transaction
  costs": transaction costs create a no-trade/inaction region; deployment
  should not be forced when expected edge does not cover trading friction.
  Reference: https://doi.org/10.1016/0022-0531(76)90018-1
- cvxportfolio cost model documentation: mature open-source implementation for
  optimization policies with explicit risk and transaction cost terms.
  Reference: https://www.cvxportfolio.com/en/stable/costs.html

## Regression Tests

Added coverage in `tests/test_qp_grinold_kahn_transform.py`:

- Annualized sigma is scaled to the mu horizon.
- Horizon-native sigma is not rescaled.
- Strict missing horizon blocks the QP.
- The horizon task is wired before covariance construction.
- Min-invested cash deployment is dropped when no candidate clears the
  positive-edge hurdle.
- Min-invested survives when best mu clears the hurdle.

Verification snapshot:

- `pytest tests/test_qp_grinold_kahn_transform.py tests/test_qp_cfg_per_regime_override.py -q`
  -> 25 passed
- `pytest tests/test_qp_integration.py tests/test_qp_contracts.py tests/test_qp_multi_bar_ramp.py tests/test_qp_cvxpy_fallback.py tests/test_qp_backend_switch.py -q`
  -> 33 passed
- `pytest tests/test_wf_config_parity.py tests/test_lookahead_propagation.py tests/test_config_consistency.py -q`
  -> 59 passed
- `pytest tests -q`
  -> 12592 passed, 8791 skipped, 1 xfailed

## Diagnostic Sim Evidence

This is not a promotion claim because CLAUDE.md requires multi-run evidence for
performance numbers. It is mechanism evidence that the unit contract mattered.

XGB true-OOS diagnostic sim, 2024-07-02 to 2026-02-10:

- Final value: 110364.57
- Total return: 10.36%
- APY: 6.36%
- Sharpe: 0.625
- Max drawdown: 10.46%
- Event-level tax: 10958.02
- Annual-net tax estimate: 3668.89
- Gross closed round-trip PnL: +4161.30
- Event-level after-tax closed PnL: -6796.72

The prior comparable report before this fix showed APY around 2.88% and Sharpe
around 0.39. The horizon contract improves behavior, but it does not solve all
session-level issues.

PatchTST historical APY/Sharpe was intentionally not forced through this sim:
the leakage guard blocked the current shadow checkpoint because its training
label window extends beyond the historical sim start. That is correct behavior.
A rigorous PatchTST APY/Sharpe needs a walk-forward PatchTST manifest, not a
guard bypass.

## Remaining Issues

The sim logs still show QP sell intents being suppressed by tax-adjusted exit
logic in cases where the QP target wants to reduce or close a position. This
needs a separate audit because the user asked that tax be treated as reporting
or as a scientifically justified decision cost, not as an opaque alpha-killer.

Next fix target:

- Trace `tax_adjusted_exit` suppression in sell/rotation/QP paths.
- Prove whether it still affects live/sim decision flow when `qp_tax_aware` is
  false.
- If it is active unintentionally, move it behind an explicit Task-level
  contract and add a regression test through the production pipeline path.
