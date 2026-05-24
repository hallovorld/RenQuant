# 2026-05-24 RenQuant 104 Model Contract Hardening

## Scope

This patch answers the audit requirement:

- Daily/full must validate WF Sharpe, SPY comparison, regime-layered IC/monotonicity, calibration health, sector map coverage, and config fingerprint before buy/QP.
- Missing metadata or weaker-score fallback must fail closed in buy mode.
- Alpha admission and portfolio QP are separated: the model/decision tree decides buy eligibility; QP can size/rebalance only the eligible surface.
- Per-ticker audit rows must include blocker, model type, sector, score, QP delta, and sell P&L/tax/net.

## Implemented

1. `P-WF-GATE` is now a strict full/buy contract.
   - Missing `wf_gate_metadata`, missing numeric WF Sharpe, missing SPY Sharpe, missing strategy-minus-SPY Sharpe, or missing `n_cuts_beat_spy_sharpe` hard-fails full/buy.
   - Sell-only remains soft so exits/risk reductions can run.

2. New `P-REGIME-IC` preflight.
   - Reads `metadata.wf_gate_metadata.trade_monotonicity`.
   - Requires at least one eligible regime and all eligible regimes to pass rank monotonicity / rank-IC style evidence.
   - Supports both list and dict regime metadata shapes.
   - Hard-fails full/buy on missing or failed evidence; sell-only remains soft.

3. `P-CALIBRATOR-HEALTH` and `P-CALIBRATOR-FLAT-REGION` are strict for enabled global calibration.
   - Missing calibrator, missing `n_unique_prob_y`, low unique probability count, non-positive `pool_ic`, malformed probability curve, or wide flat region hard-fails full/buy.
   - Sell-only remains soft.

4. `P-CONFIG-FP` is strict for full/buy.
   - Missing fingerprint, non-diffable `config_fingerprint_fields`, and legacy missing sector fingerprint fields no longer soft-pass buy mode.
   - Sell-only remains soft.

5. Runtime calibrator fallback is fail-closed.
   - If `global_calibration.enabled=true` but no calibrator is loadable, `ApplyGlobalCalibrationTask` clears buy candidates, sets `buy_blocked=True`, `skip_buys=True`, stamps `blocked_by`, and sets `_calibrator_contract_failed=True`.
   - `JointPortfolioQPJob` skips when `_calibrator_contract_failed=True`.

6. DB audit now captures QP output.
   - Added `qp_delta_w`, `qp_target_w`, and `qp_status` to `candidate_scores` and `ticker_daily_state`.
   - Runner and sim adapters persist those fields from `ctx._qp_solution`.
   - Existing `trades` table already records sell `gross_pnl`, `tax`, `net_pnl_after_tax`, and decision payload JSON.

7. `daily_104.sh` sell-only fallback recognizes all buy-side preflight blockers.
   - Includes WF, regime IC, config fingerprint, sector map, panel contract, calibrator health/flat-region, feature coverage, watchlist, and model artifact gates.

## Current Prod Dry-Run State

As of this patch, active prod full preflight correctly fails closed:

- `P-WF-GATE`: active artifact carries failed WF evidence.
  - `wf_3cut_sharpe_mean = -1.3233`
  - `spy_sharpe_mean = +1.0808`
  - `strategy_minus_spy_sharpe_mean = -2.4042`
- `P-CONFIG-FP`: active artifact lacks the newly required sector/config fingerprint stamp for current `sector_map` / `sector_etf_map`.

Sell-only dry-run passes the risk-exit path while keeping buys blocked.

## Verification

- `pytest tests/test_preflight.py -q` -> 65 passed.
- `pytest tests/test_persistence.py tests/test_ticker_daily_state.py -q` -> 29 passed.
- `pytest tests/test_calibrator_lean_no_prod_default.py tests/test_calibrator_no_flat_region.py -q` -> 21 passed.
- `pytest tests/test_joint_qp_task.py tests/test_qp_sector_constraint.py tests/test_qp_integration.py -q` -> 70 passed.
- Targeted `git diff --check` on touched files -> clean.

Repository-wide `git diff --check` still reports trailing whitespace in dirty `doc/dashboard.md`, which was pre-existing/unrelated in the current worktree and was not changed by this patch.

## Remaining Requirement Before Live Buys

Promote only an artifact that passes strict WF and stamps the full config fingerprint under the current sector/config metadata. Until then, full daily should intentionally stop before buy/QP and use the sell-only fallback.
