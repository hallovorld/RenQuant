# 2026-06-06 — QP Step 4 fwd60 decision-grade replay verdict

**Agent-Origin**: Codex

This evidence refresh re-runs the QP Step 4 allocator A/B replay at the
production label horizon (`fwd_60d`) after the multirepo pin refresh and
runtime backfill fixes. It supersedes any lingering "no verdict" status
from [`2026-06-04-qp-step4-replay-blocked-no-verdict.md`](../../2026-06-04-qp-step4-replay-blocked-no-verdict.md)
for the current local sim DB: the replay now loads bars and emits a verdict.

## Command

```bash
PYTHONPATH=backtesting/renquant_104 \
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
.venv/bin/python -m kernel.portfolio_qp.run_ab_replay \
  --wf-artifact-root data/sim_runs.db \
  --start-cut 2024-01-02 --end-cut 2026-03-27 \
  --out artifacts/qp_step4_replay/verdict_fwd60_sector_20260606.json \
  --allocators equal_weight_top_k,inverse_vol_top_k,fractional_kelly_top_k,hybrid_option_f_allocator,hard_only_qp_allocator \
  --incumbent fractional_kelly_top_k \
  --fwd-horizon-days 60 \
  --strategy-config .subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json
```

## Result

- `n_bars`: 483
- `constraint_fidelity.decision_grade`: `true`
- `verdict.promotion_candidate`: `null`
- `verdict.next_action`: `iterate`
- Blocking gates: `pbo_below_0_5=false`, `pbo_plus_se_below_0_55=false`,
  `win_rate_z_score_above_2=false`
- `hybrid_option_f_allocator` has the highest raw Sharpe but does not pass
  the promotion gates.
- `hard_only_qp_allocator` is rejected for promotion because it produced
  29 hard-constraint violations (`cash_budget=3`, `dw_max=26`).

## Per-Allocator Summary

| Allocator | Sharpe raw | Mean daily return | Violations | BULL_CALM Sharpe |
|---|--:|--:|--:|--:|
| hybrid_option_f_allocator | 9.13 | 0.0788 | 0 | 9.67 |
| hard_only_qp_allocator | 9.08 | 0.0764 | 29 | 9.61 |
| inverse_vol_top_k | 8.94 | 0.0542 | 0 | 9.58 |
| fractional_kelly_top_k | 8.93 | 0.0573 | 0 | 9.54 |
| equal_weight_top_k | 8.55 | 0.0612 | 0 | 9.12 |

## Operator Read

The QP Step 4 track is no longer blocked on "zero bars" for this DB, but it
still does not authorize any allocator switch. Keep the incumbent until a
candidate passes the PBO/win-rate gates with zero hard-constraint
regressions at the production horizon.

Raw artifact: [`verdict_fwd60_sector.json`](./verdict_fwd60_sector.json).
