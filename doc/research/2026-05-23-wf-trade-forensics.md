# RenQuant 104 WF Trade Forensics — 2026-05-23

Source trace:
`backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/codex_current_contract_20260523-190033`.

This report uses `scripts/analyze_wf_trade_forensics.py`, which rebuilds
round trips from raw `*.trades.json` with the strategy config's tax-lot method.
For current 104 that method is `hifo`. The older sidecar CSVs were generated
with a FIFO forensic replay and could create misleading per-lot rows where
allocated tax exceeded the row's gross P/L. That was a forensic attribution bug,
not a broker-cash debit regression.

## Tax Integrity

- Configured tax cash mode: `reporting_only`.
- HIFO-aligned closed lots: `158`.
- Gross P/L: `+$32.78k`.
- Event tax estimate: `+$27.16k`.
- Net after event-tax estimate: `+$5.63k`.
- Tax cash debited: `$0`.
- Positive rows with `tax > gross`: `0`.
- Losing rows with positive tax: `0`.

Conclusion: the old "gross smaller than tax" symptom was caused by forensic
lot-method mismatch. Sim cash is not being reduced by estimated capital-gains
tax in current semantics.

## Main Failure Buckets

| bucket | n | gross P/L | tax | net after tax | win rate | median hold |
|---|---:|---:|---:|---:|---:|---:|
| `stop_loss` | 36 | `-$17.59k` | `$0` | `-$17.59k` | `0.0%` | `31d` |
| `trailing_stop` | 59 | `+$39.23k` | `+$20.15k` | `+$19.08k` | `89.8%` | `147d` |
| `qp_sell` | 39 | `+$8.79k` | `+$5.15k` | `+$3.65k` | `87.2%` | `24d` |
| `single_day_loss` | 10 | `+$1.58k` | `+$1.19k` | `+$0.39k` | `50.0%` | `61d` |
| `qp_close` | 5 | `+$1.06k` | `+$0.62k` | `+$0.44k` | `40.0%` | `18d` |
| `panel_conviction` | 9 | `-$0.29k` | `+$0.04k` | `-$0.34k` | `44.4%` | `36d` |

Stop-loss is the direct gross-loss bucket. Tax drag is a separate overlay on
profitable exits, mostly `trailing_stop` and `qp_sell`.

## Contract Findings

- `TickerSellJob.stop_loss` is behaving as currently coded: exit parameters are
  taken from the current regime; only `max_hold_days` is anchored to the entry
  regime. This is policy, not an accidental code path.
- Most stop losses are current-regime hard stops on positions entered under a
  BULL thesis. That may be correct risk control, but it must be A/B tested by
  regime instead of changed globally.
- QP is not buying totally unscored names, but it is still serving as selector
  over the broad candidate pool. That violates the stricter contract that alpha
  admission must happen before portfolio optimization.
- TopUp is looser than new-buy admission and should be tested under the same
  qualification contract.

## Immediate A/B Plan

1. Selector isolation:
   compare `joint_actions.enabled=true` versus `false` with TopUp disabled in
   both arms. This measures whether QP-as-selector is harming alpha conversion.
2. Qualified-QP gate:
   keep QP, but disable forced cash deployment and require positive panel/rank
   qualification before QP/TopUp can add risk. This tests whether marginal
   allocations are the drag.
3. Stop-loss policy:
   test non-BULL volatility-aware stops or earlier panel/mu soft exits for the
   subset where model deterioration precedes hard stop. Acceptance must be
   regime-conditional and benchmark-relative, not pooled only.

Do not promote a model or decision-tree change from this report alone. It is a
root-cause map; acceptance still requires strict WF with SPY comparison, regime
cuts, calibration/sanity gates, and full trade-ledger contract.

## Implemented Repair

Commit pending at time of writing:

- Added `rotation.joint_actions.qp_admission_gate` to production and golden
  configs.
- QP buy/top-up emission now fails closed unless the ticker has finite,
  pre-qualified alpha evidence:
  - calibrated `rank_score >= 0.55`;
  - raw cross-sectional `panel_score >= 0`;
  - for new names, available position capacity under `max_concurrent_positions`.
- Disabled forced QP cash deployment:
  - `qp_min_invested_pct = 0`;
  - `qp_cash_drag_lambda = 0`.
- Enabled QP conviction caps with the existing panel-score sizing primitive.
- Raised standalone TopUp admission from `0.20` to `0.55`.

This does not assert performance improvement by itself. It closes the
identified contract hole so the next WF run can test whether the model signal
survives without QP/TopUp turning marginal scores into trades.
