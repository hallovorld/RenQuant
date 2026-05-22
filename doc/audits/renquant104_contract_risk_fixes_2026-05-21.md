# renquant_104 Contract + Risk Fixes — 2026-05-21

## Executive Summary

This pass found four production-path issues that previous docs did not fully
close:

1. **QP buy orders dropped model provenance.** `_emit_qp_buy()` emitted only
   `rank_score` and `source`, so the sim trade ledger could not audit the
   actual `mu`, `sigma`, `panel_score`, `rs_score`, regime, or confidence used
   when the optimizer bought. This explains old WF round trips with blank
   `entry_mu` / `entry_sigma`.
2. **WF acceptance could pass unauditable trades.** The gate checked score
   monotonicity but did not require ledger-level model/policy fields.
3. **`adaptive_mean_std_cap` became a no-op.** Capping the calibrated buy floor
   at 0.30 made sense when scores sat near the 0.27 base rate. After current
   calibration clustered candidates around 0.55-0.65, `floor=0.30` admitted
   almost everything.
4. **QP sector cap was count-derived only.** With `max_positions_per_sector=6`
   and 15% per-name caps, the QP allowed up to 90% weight in one sector. That
   is not the direct group-weight cap used by mature portfolio optimizers.

## Implemented Fixes

| Area | Fix | Regression guard |
|---|---|---|
| QP order contract | QP BUY orders now carry `mu`, `sigma`, `panel_score`, `rs_score`, `regime`, `confidence`, `kelly_target_pct`, `order_type` | `tests/test_qp_integration.py`, `tests/test_bug22_rs_score_keyerror.py` |
| Trade ledger | Sim sell events now stamp `exit_regime`, exit confidence, signal reason, and active stop/trailing/SDL/max-hold thresholds | `tests/test_sim_trade_ledger.py` |
| WF acceptance | New `trade_contract` gate fails if strict QP/Kelly ledgers lack finite `entry_mu` / `entry_sigma` or exit policy fields | `tests/test_trade_contract_gate.py`, `tests/test_wf_gate_cli_contract.py` |
| Buy floor | New production mode `adaptive_mean_std` = `max(buy_floor_min, mean + k*std)` with no 0.30 cap | `tests/test_veto_weak_buys_p0_fix.py` |
| Sector concentration | QP sector cap now supports `regime_params.<REGIME>.max_sector_weight_pct` and uses `min(count_cap, direct_weight_cap)` | `tests/test_qp_sector_constraint.py` |
| Tax reporting | Forensic reports now show event-level tax debited plus annual-net tax estimate, without changing sim cash | `tests/test_sim_trade_ledger.py` |

## Reference Basis

- IRS Publication 550 says short-term gains/losses are combined into net
  short-term capital gain/loss, long-term gains/losses into net long-term, then
  those are combined for total net gain/loss. That supports reporting an
  annual-net tax stress lens in addition to the sim's conservative event-level
  cash debit: <https://www.irs.gov/publications/p550>.
- IRS Publication 550 also states wash-sale replacement holding period includes
  the old stock's holding period, which is why wash-sale treatment belongs in
  lot/basis logic rather than a simplistic permanent penalty:
  <https://www.irs.gov/publications/p550>.
- PyPortfolioOpt exposes `add_sector_constraints` as sum-of-weights group
  constraints, e.g. tech exposure less than a configured upper bound:
  <https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html>.
- cvxportfolio constraints accept scalar/vector/dataframe limits on post-trade
  weights/holdings, matching the direct-weight-cap approach:
  <https://www.cvxportfolio.com/en/stable/constraints.html>.
- Qlib uses cross-sectional z-score / rank processors (`CSZScoreNorm`,
  `CSRankNorm`) for panel workflows, supporting distribution-relative gating
  over fixed magic thresholds:
  <https://qlib.readthedocs.io/en/latest/component/data.html?highlight=alpha158>.

## Config Changes

- `strategy_config.json`, `strategy_config.golden.json`,
  `strategy_config.sim_wl200.json`:
  - `ranking.panel_scoring.buy_floor = "adaptive_mean_std"`
  - `ranking.panel_scoring.buy_floor_std_mult = 1.0`
  - `regime_params.BULL_CALM.max_sector_weight_pct = 0.35`
  - `regime_params.BULL_VOLATILE.max_sector_weight_pct = 0.30`
  - `regime_params.CHOPPY.max_sector_weight_pct = 0.30`
  - `regime_params.BEAR.max_sector_weight_pct = 0.20`
- `strategy_config.sim_wl200.json` also now matches prod on:
  - `rotation.joint_actions.qp_tax_lot_method = "hifo"`
  - `ranking.panel_scoring.kind = "xgb"`

## Validation Snapshot

Passed:

```bash
.venv/bin/python -m json.tool backtesting/renquant_104/strategy_config.json
.venv/bin/python -m json.tool backtesting/renquant_104/strategy_config.golden.json
.venv/bin/python -m json.tool backtesting/renquant_104/strategy_config.sim_wl200.json
.venv/bin/python -m py_compile scripts/trade_contracts.py scripts/sim_trade_ledger.py scripts/run_wf_gate.py
.venv/bin/python -m pytest tests/test_trade_contract_gate.py tests/test_sim_trade_ledger.py tests/test_qp_integration.py tests/test_qp_sector_constraint.py tests/test_veto_weak_buys_p0_fix.py tests/test_wf_gate_cli_contract.py tests/test_bug22_rs_score_keyerror.py tests/test_buy_sell_audit_fixes.py -q
.venv/bin/python -m pytest tests/test_golden_preservation.py tests/test_side_config_artifact_paths.py tests/test_qp_contracts.py tests/test_sigma_aware_stop_integration.py tests/test_walkforward_eval_config.py tests/test_config_consistency.py -q
```

## Remaining Required Evidence

This patch makes the pipeline more auditable and closes several contract bugs.
It does **not** by itself prove the model is good. The next acceptance run must
retrain/stamp with the updated contract, then rerun WF with trade traces. A
passing promotion now requires both:

- positive performance gates vs SPY/regime context, and
- `trade_contract.passed = true` plus `trade_monotonicity.passed = true`.
