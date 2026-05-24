# QP Sigma Admission Diagnosis — 2026-05-23

## Question

Why does a model with positive-looking IC fail to convert into benchmark-beating
APY/Sharpe after the decision tree?

## Latest Evidence

Trace:
`backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/codex_featspace_20260523-211211_wf_after_tracefix_20260523-221438`

Forensic report:
`artifacts/wf_trade_forensics_featspace_tracefix_prodsemantic_20260523.md`

Key numbers from the prod-semantic HIFO replay:

- Closed round trips: 42.
- Gross P/L: +$10.45k.
- Estimated tax: +$8.52k.
- Net after estimated tax: +$1.93k.
- Stop-loss exits: 9, gross/net -$5.91k.
- All entries came from `JointPortfolioQPJob`.
- Entry rank_score vs net P/L Spearman: -0.007.
- Entry mu vs net P/L Spearman: +0.016.
- Entry panel_score vs net P/L Spearman: -0.042.
- Highest entry-sigma quartile net: -$0.49k.
- Entries with rank_score >= 0.63 net: -$1.01k.

Conclusion: tax reporting is clean in this trace. The stronger failure is that
the QP-admitted traded subset is not score-monotone, and high-uncertainty names
are over-represented in the path-loss bucket.

## Theory

This diagnosis is consistent with the standard risk-adjusted allocation
framework rather than a new ad-hoc rule:

- Markowitz, "Portfolio Selection" (1952): expected return must be traded
  against variance/covariance, not considered alone.
- Kelly, "A New Interpretation of Information Rate" (1956): optimal exposure
  scales with edge relative to risk; high uncertainty should reduce or block
  marginal bets.
- Davis and Norman, "Portfolio Selection with Transaction Costs" (1990):
  proportional costs imply a no-trade region; marginal rebalances with weak
  risk-adjusted edge should not fire.
- Grinold/Kahn active management framing: alpha needs to be scaled by risk and
  breadth; raw rank is not enough for sizing.

## Implemented Hook

`rotation.joint_actions.qp_admission_gate` now accepts:

- `max_sigma`
- `max_sigma_by_regime`
- `topup_max_sigma`
- `topup_max_sigma_by_regime`

When configured, new QP buys or top-ups fail closed with
`qp_admission_sigma` if the candidate sigma is missing, non-finite, or above
the cap. Defaults are absent, so current production behavior is unchanged.

## Next A/B

Run a strict paired WF with the same feature-space manifest:

- Control: current prod-semantic config.
- Candidate: BULL_CALM `max_sigma_by_regime.BULL_CALM = 0.39` as a first
  range-finding cap, because the highest sigma quartile starts around 0.387
  and is net-negative in the latest trace.

Promotion requirement remains unchanged: no production flip unless WF,
SPY-relative Sharpe/APY, per-regime diagnostics, sanity, and decision-trace
contracts pass.

## A/B Result

The BULL_CALM `max_sigma_by_regime.BULL_CALM = 0.39` diagnostic A/B completed.

- Trace:
  `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/qpsigma_bullcalm039_20260523`
- Forensics:
  `artifacts/wf_trade_forensics_qpsigma_bullcalm039_20260523.md`
- Mean annual-net Sharpe: `+0.523` versus prior strict feature-space
  `+0.400`.
- Annual-net Sharpe cuts: `+0.640`, `+0.523`, `+0.407`.
- Mean annual-net APY: `+1.45%`.
- SPY mean Sharpe: `+1.081`; strategy-minus-SPY Sharpe: `-0.558`.
- SPY mean APY: `+16.94%`; strategy-minus-SPY APY: `-15.49%`.
- Closed round trips: `34`; win rate: `73.5%`; stop-loss exits: `3`.
- Trade monotonicity passed in BULL_CALM, but benchmark gate failed in all
  three cuts.

Conclusion: the sigma cap is a useful diagnostic and risk-control direction,
but it is not a production promotion. It reduces bad high-uncertainty trades
and improves annual-net Sharpe, yet the strategy still under-participates in
SPY-dominated bull/calm periods. The next design change needs to target
benchmark-relative exposure/alpha conversion, not only suppress high-risk
names.
