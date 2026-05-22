# RenQuant 104 P0/P1 Guard Implementation

Date: 2026-05-22

## Why This Patch Exists

The previous audit found a dangerous split:

- raw XGB top-K signal is positive versus shuffle/reverse controls
- full WF is negative
- order sources were only partially inferable after the fact
- WF configs could drift from production decision semantics

That means future debugging must first prove two invariants:

1. Every emitted buy order has a named pipeline owner and score snapshot.
2. Every WF acceptance run uses production-equivalent decision semantics.

This follows `CLAUDE.md`: code is source of truth, every fix names the
invariant, and every acceptance number must be protected from experimental
drift.

## Literature / Mature Scheme Support

- **Qlib record/evaluation discipline**:
  https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst
  Qlib separates prediction score, strategy construction, and evaluation. The
  new order-attribution contract preserves that separation in RenQuant logs:
  score state is captured when the strategy turns scores into orders.

- **cvxportfolio / Boyd-style decomposition**:
  https://www.cvxportfolio.com/en/stable/index.html
  Forecasts, costs, risks, and constraints are separate model terms. The new
  `decision_inputs` field records which terms the order path used.

- **Bailey and Lopez de Prado 2014 DSR/PBO**:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
  Backtest evidence is fragile under multiple trials and configuration drift.
  The config-parity guard blocks non-production-equivalent WF runs before they
  generate a misleading Sharpe.

## P0: Order Attribution Contract

Added:

- `backtesting/renquant_104/kernel/pipeline/order_attribution.py`
- `tests/test_order_attribution_contract.py`

Every buy order emitted into `ctx.orders` now includes:

- `attribution_version`
- `source_job`
- `source_task`
- `order_source`
- `score_snapshot`
- `decision_inputs.acceptance_reason`

Emit paths now stamped:

- `SelectionJob.SizeAndEmitTask` for `NEW_BUY`
- `TopUpJob.TopUpHeldTask` for `TOP_UP`
- `RotationJob.EmitRotationsTask` for rotation buys
- `JointActionJob.JointActionTask` for joint buys/rotations
- `JointPortfolioQPJob.JointPortfolioQPTask` for `QP_BUY`

Invariant: no buy order can be appended by a known emitter without attribution.

## P1: WF Config Parity Guard

Added:

- `scripts/wf_config_parity.py`
- `tests/test_wf_config_parity.py`
- `scripts/run_wf_gate.py` now calls the parity guard by default

The guard allows expected WF differences:

- static artifact path vs walk-forward manifest path
- NGBoost artifact path
- comment/provenance keys beginning with `_`

The guard blocks decision-semantic drift:

- scorer kind
- buy floor mode
- Kelly sizing knobs
- QP / joint action knobs
- regime params
- tax settings
- tiered thresholds
- sector/defensive maps
- feature recipe mismatch between candidate and WF artifacts

Invariant: acceptance/WF cannot spend compute or stamp metadata on a config
that is not production-equivalent unless explicitly run with
`--skip-config-parity` for exploratory work.

## Current WF Config Check

Command:

```bash
.venv/bin/python scripts/wf_config_parity.py \
  --wf-config strategy_config.sim_wl200_172_sentiment.json \
  --candidate-artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json
```

Result: **FAIL**, as intended. The current local WF config is not production
equivalent. Major mismatches:

- `ranking.panel_scoring.buy_floor`: prod `adaptive_mean_std`, WF
  `adaptive_mean_std_cap`
- `ranking.kelly_sizing`: WF lacks prod realized-vol floor/ceiling
- `rotation.joint_actions.qp_tax_lot_method`: prod `hifo`, WF `fifo`
- `regime_params`: multiple prod regime/sentiment/exit settings absent or
  different in WF

This is exactly the bug class that previously let stale WF configs produce
overconfident conclusions.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/test_order_attribution_contract.py \
  tests/test_buy_emit_contract.py \
  tests/test_joint_actions.py \
  tests/test_rotation_atomic.py \
  tests/test_wf_config_parity.py \
  -q
```

Result at implementation time:

- order attribution / buy emit / joint / rotation: 38 passed
- WF config parity: 3 passed
