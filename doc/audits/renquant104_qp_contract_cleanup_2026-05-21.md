# renquant_104 QP Contract Cleanup - 2026-05-21

## What Was Fixed

The failure class was system-level, not model-specific. XGB, PatchTST, and
NGBoost all feed the same live/sim path:

`model score -> panel/global calibration -> rank_score/mu/sigma -> QP/Kelly -> order emit -> execution/tax ledger`

Three protections are now enforced:

1. QP must have an explicit strict mu contract.
   - `rotation.joint_actions.qp_mu_contract` is now `strict` in prod, golden,
     and WL200 sim configs.
   - Sim/WF entrypoints fail before running if QP is enabled without a legal
     expected-return-like mu source.
   - Valid mu sources are calibrator `expected_return`, NGBoost `mu`, or an
     explicit `alpha_to_mu` transform.

2. Kelly/QP sigma fallback must be present when NGBoost is off.
   - WL200 sim now matches prod: `use_calibrator_mu=true` and
     `use_realized_vol_fallback=true`.
   - This prevents stale side configs from treating raw rank scores as
     optimizer means or running Kelly with no sigma path.

3. QP emitted buys must fit executable cash.
   - `EmitOrdersFromQPSolutionTask` now caps buy shares to available cash after
     cash reserve and execution buffer.
   - If one-share minimum cash is unavailable, the buy is skipped and stamped
     `qp_cash_exhausted`.
   - This closes the mismatch where the solver could output target weights
     that looked valid but downstream execution rejected due to T+2 cash.

## Acceptance Gate Added

WF now includes a trade-level monotonicity gate when trade traces are enabled.
It reads persisted round-trip ledgers and checks, per active regime:

- Spearman correlation between `entry_rank_score` and realized `pnl_pct`
- top-vs-bottom quintile return spread
- pass-open only for regimes with insufficient closed-trade sample

This is intentionally regime-first. A model that looks acceptable on pooled
Sharpe but has anti-predictive buy ranking inside `BULL_CALM` must not pass
promotion.

## Tests Added

- `tests/test_qp_contracts.py`
- `tests/test_trade_monotonicity_gate.py`
- `tests/test_joint_qp_task.py::TestQPCashBudget`
- `tests/test_wf_gate_cli_contract.py` contract guards

Targeted result:

`76 passed, 1 xfailed`

## Design Notes

This does not declare XGB, PatchTST, or NGBoost good. It makes future
experiments harder to fool:

- a stale side config now fails early;
- a raw rank score cannot silently become optimizer mu;
- a high score must show trade-level monotonicity in its actual regime;
- emitted buys are constrained by executable cash, not just optimizer NAV.

If the next WF still loses money after these fixes, the evidence is cleaner:
it points to alpha/label/regime quality rather than broken plumbing.
