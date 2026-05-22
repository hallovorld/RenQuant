# 2026-05-22 Exit A/B Strict-WF Results

## Setup

- Window: 2024-04-01 to 2026-03-26.
- Baseline: current production `strategy_config.json` semantics routed through the leakage-safe WF manifest.
- WF manifest: `artifacts/sim/walkforward_manifest_merged.json`.
- Static placeholder panel artifact: `artifacts/sim/walkforward_retrains/2024-01-01/panel-ltr.json`.
- All A/B configs use `parallel_workers=3`; this changes only scheduling, not trading logic.
- Quality gates passed before running:
  - QP strict contract OK for all four configs.
  - Static preflight ACTIVE / 0 DEAD_PATH for changed knobs.
  - Meta-label preflight mapping fixed and covered by `tests/test_validate_sim_config_active.py`.

## Configs

| Label | Config | Difference vs baseline |
|---|---|---|
| baseline | `strategy_config.sim_exit_strict_wf_baseline.json` | Current prod strict decision/QP config, WF artifact routing |
| meta_t050 | `strategy_config.sim_exit_strict_wf_meta_t050.json` | Enables AFML-style path-exit meta-label veto at threshold 0.50 |
| soft_path | `strategy_config.sim_exit_strict_wf_soft_path.json` | Softens stop/SDL path exits |
| combo | `strategy_config.sim_exit_strict_wf_combo.json` | Meta-label + softened path exits |

## Portfolio Results

| label | APY % | Sharpe | MaxDD % | Final value | Closed RT | Net win % | Gross P&L | Tax | Net P&L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 4.520 | 0.613 | 5.929 | 109128.81 | 343 | 54.81 | 35733.00 | 26784.61 | 8948.39 |
| meta_t050 | 4.045 | 0.547 | 6.972 | 108152.17 | 275 | 51.64 | 25283.35 | 21579.98 | 3703.36 |
| soft_path | 3.001 | 0.407 | 6.501 | 106016.42 | 340 | 56.18 | 32445.10 | 26643.04 | 5802.05 |
| combo | 1.861 | 0.272 | 6.556 | 103711.94 | 300 | 52.33 | 26231.29 | 22576.54 | 3654.75 |

## Verdict

Do **not** promote the tested exit changes.

The strict-WF baseline wins on APY, Sharpe, final value, max drawdown, and net P&L. Meta-labeling and softer path exits both reduce gross P&L more than they reduce tax. The combo is worst, which suggests these two changes remove or delay useful exits rather than rescuing alpha.

## Exit Attribution

Baseline positive contributors:

| Exit reason | Net P&L | Notes |
|---|---:|---|
| `max_hold` | +8767.86 | Positive, high tax, but still largest net contributor |
| `qp_sell` | +8720.89 | Positive and central to realized gains |
| `trailing_stop` | +3825.74 | Few trades, high win rate |

Baseline negative contributors:

| Exit reason | Net P&L | Notes |
|---|---:|---|
| `panel_conviction` | -6015.29 | Large, persistent drag; worsens under meta/soft/combo |
| `stop_loss` | -5960.25 | Always losing by construction, but low count |
| `single_day_loss` | -558.09 | Small count and small total impact |

Important implication: the earlier hypothesis that stop/SDL softness would directly improve APY is not supported in true-OOS WF. The bigger, more direct APY lever is `panel_conviction` exit quality.

## Follow-Up

1. Audit `panel_conviction` exit as the next primary APY/Sharpe target.
2. Require paired WF A/B for any `panel_conviction` change, with exit-reason P&L as the first acceptance metric.
3. Keep meta-label exit veto in shadow only until it proves positive on WF; this run says threshold 0.50 is not promotable.
4. Keep stop/SDL production settings unchanged for now. Softening stop/SDL reduced path-exit count but did not improve portfolio metrics.

## Local Artifacts

- Summary: `artifacts/exit_ab_strict_wf_summary.md`
- Machine-readable summary: `artifacts/exit_ab_strict_wf_summary.json`
- Equity curves: `artifacts/exit_ab_strict_wf_*_equity.json`
- Matched round trips: `artifacts/exit_ab_strict_wf_*_round_trips.csv`
- Raw trade logs: `artifacts/exit_ab_strict_wf_*_trade_log.json`
