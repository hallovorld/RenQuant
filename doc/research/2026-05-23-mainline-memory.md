# RenQuant 104 Mainline Memory — 2026-05-23

This is the first file to read when continuing the current repair campaign.
It exists because multiple result streams were mixed together during the
session. Do not infer current state from a single stale metric.

## Mission

Make RenQuant 104 scientifically trustworthy end to end:

1. A model is accepted only with leak-safe WF evidence, SPY comparison,
   per-regime IC/Sharpe, calibration health, and config/sector fingerprint.
2. IC must convert into tradeable alpha after the decision tree, QP, exits,
   turnover, and tax reporting.
3. Every buy/sell decision must be explainable from persisted decision-tree
   fields, not reconstructed from scattered logs.
4. XGB remains primary unless a challenger passes the same strict acceptance.
   PatchTST stays shadow/router research until it has a true WF manifest.
5. No silent fallback: missing metadata, weaker score fallback, missing sector
   metadata, or failed artifact evidence blocks buy/full paths.

## Current Truth

- The active production artifact still carries an old failed WF stamp:
  APY `+0.63%`, Sharpe `-1.3233`, SPY Sharpe `+1.0808`, `passed=false`.
  This is stale failed metadata, and strict preflight now blocks full/buy on it.
- That `-1.3233` stamp is not the whole repaired research state. It must not be
  quoted as "current prod performance" without the above context.
- Tax cash corruption is fixed in current production semantics:
  `tax.cash_debit_mode=reporting_only`. Sim reports estimated tax separately
  and does not debit broker cash for estimated capital-gains tax.
- QP/strict-contract safety improved, but alpha conversion is not solved. The
  remaining main problem is whether signal survives the final decision tree
  into realized APY/Sharpe after turnover, exits, and annual-net tax.
- Manifest-OOS sanity now supersedes the earlier static-artifact sanity number:
  the manifest-scoped point-in-time diagnostic gives real IC `+0.0269`,
  shuffled IC `-0.0019`, and time-shift placebo IC `+0.0282`. This fails
  because placebo is not below half of real IC. Treat the older real IC
  `+0.0750` as an invalid static/full-artifact diagnostic, not acceptance
  evidence.
- The best current short-window style evidence says PatchTST and XGB differ:
  PatchTST bought more names and outperformed over 2026-05-06 to 2026-05-22,
  but that 13-trading-day, zero-sell window is not promotion evidence.
- Current-contract WF has now completed and failed acceptance. This replaces
  the earlier "running" state below.
- A new feature-space-aligned staged model now exists from run
  `codex_featspace_20260523-211211`: 172 features, CV OOS IC `+0.0473`,
  train IC `+0.1190`. Its paired calibrator fit reports pool IC `+0.1152`
  and per-date IC `+0.1193`. These are training/calibration diagnostics only,
  not acceptance evidence.
- The feature-space-aligned staged model correctly fails strict WF admission
  against the old manifest because the feature-source contract changed. A
  same-recipe 40-row manifest was then generated and evaluated; it also fails
  acceptance on SPY-relative Sharpe/APY, BULL_CALM trade monotonicity, and
  manifest-OOS placebo sanity. Do not promote it.

## Pushed Progress

- `81bd338 fix(renquant104): enforce strict model contracts`
  - Hard-fails buy/full preflight on bad or missing WF/SPY/regime
    IC/calibration/config evidence.
  - Stops global calibrator fallback to raw score.
  - Persists QP target/delta/status fields.
  - Full test suite passed at that checkpoint:
    `12654 passed, 8791 skipped, 1 xfailed`.
- `6e68f09 docs(renquant104): record current evaluation state`
  - Adds `doc/research/2026-05-23-current-state-ledger.md`.
  - Adds a WF-config regression test proving prod `tax.cash_debit_mode` wins
    over stale side-config event-cash debit.
  - Targeted test passed: `tests/test_wf_config_parity.py`.
- `00fdf70 fix(renquant104): fail closed on unavailable sanity gates`
  - `run_wf_gate.py` no longer skip-passes when the rawlabel panel is missing,
    the scorer kind is unsupported, or sanity prediction fails.
  - Existing-model sanity now stamps `sanity_method` and a multi-shift placebo
    profile into `wf_gate_metadata`.
  - Targeted tests passed:
    `tests/test_wf_gate_cli_contract.py tests/test_promote_wf_gate.py`.
- Latest forensic update:
  - `scripts/sim_trade_ledger.py` now rebuilds forensic round trips using the
    configured tax-lot method (`hifo` for current 104) instead of hard-coded
    FIFO.
  - New `scripts/analyze_wf_trade_forensics.py` gives a repeatable WF trace
    report for exit/source/regime/score/tax attribution.
  - Targeted tests passed:
    `tests/test_sim_trade_ledger.py tests/test_wf_trade_forensics.py`.
- Latest decision-tree repair:
  - QP buy/top-up emission now has a strict alpha-admission gate:
    finite `rank_score >= 0.55`, finite raw `panel_score >= 0`, and available
    slot capacity for new tickers.
  - Production/golden configs disable forced QP cash deployment
    (`qp_min_invested_pct=0`, `qp_cash_drag_lambda=0`) and enable conviction
    caps.
  - Standalone TopUp floor raised from `0.20` to `0.55`.
  - Targeted tests passed:
    `tests/test_qp_admission_gate.py tests/test_qp_conviction_cap.py
    tests/test_buy_quality_gates.py tests/acceptance/jobs/test_split_jobs_e2e.py
    tests/test_qp_grinold_kahn_transform.py tests/test_p0_fixes_regression_guards.py
    tests/test_wf_config_parity.py`.
- `2bb1f8f fix(renquant104): enforce single panel exit owner`
  - Production/golden config now disables legacy per-ticker
    `PanelConvictionExitTask`; raw panel/NGBoost exit ownership belongs to the
    cross-sectional panel-exit job.
  - `wf_config_parity.py` now compares `risk.panel_exit`.
- `f410ed8 fix(renquant104): require explicit decision trace reasons`
  - Sim/live adapters now stamp terminal blocked reasons for every non-selected
    watchlist ticker.
  - `decision_trace_integrity_report()` fails on missing non-selected
    `blocked_by`.
- `807e97e fix(renquant104): make stop-loss regime anchoring explicit`
  - Production/golden config declares
    `risk.stop_loss_anchor_policy.mode=current_regime`.
  - A/B-only `max_entry_current` mode exists to test BULL_CALM entry-stop
    anchoring without changing production semantics.
- `fb3c69a fix(renquant104): gate qp soft sells by disposed lot age`
  - QP soft-sell horizon checks the actual tax lot that would be disposed under
    the configured lot method, not just the aggregate position `entry_date`.
  - This fixes the HIFO churn bug where a position looked old while QP sold a
    recently-added high-cost lot.
- `cd1be00 fix(renquant104): harden training ticker concurrency`
  - Per-ticker training parallelism now waits with a real wall-clock timeout,
    logs completed work, and raises `ParallelTimeoutError` instead of silently
    returning partial results.
  - Targeted tests passed:
    `tests/test_training_parallel_timeout.py tests/test_pipeline_parallel_timeout.py`.
- `e2f233b fix(renquant104): make wf sanity manifest-oos safe`
  - WF sanity now validates every walk-forward manifest row, rejects static
    artifacts without an explicit safe cutoff, and runs sanity diagnostics
    through the manifest's point-in-time artifacts rather than a full-trained
    staging artifact.
  - Skipped/unavailable placebo diagnostics fail closed.
  - Targeted tests passed:
    `tests/test_wf_gate_recipe_scope.py tests/test_wf_gate_cli_contract.py
    tests/test_promote_wf_gate.py`.
- `9a5ea1f fix(renquant104): harden decision trace integrity`
  - Decision trace integrity now fails on fallback trade attribution,
    missing sell shares, missing QP `delta_w`/`target_w`/`solver_status`, and
    in-universe rows without `model_type`.
  - Realized-vol and concentration risk gates now stamp terminal
    `blocked_by` reasons.
  - Targeted tests passed:
    `tests/test_persistence.py::TestTrades tests/test_risk_gates.py` plus
    the broader sell/QP/state repair neighborhood.
- Latest QP solver-universe hardening:
  - QP buy admission now happens before vector construction, not only at order
    emission. New long candidates that fail `qp_admission_gate`, have no open
    slot, or arrive while buys are globally gated are removed from
    `_qp_tickers` and `_qp_mu_source_map` before the optimizer sees them.
  - Held names remain in the QP universe so the optimizer can still trim/sell;
    held top-ups remain blocked at order emission unless they pass the stricter
    top-up floors. Short candidates bypass buy admission and still override a
    same-ticker long candidate in the long-short path.
  - Invariant: model/gates decide buy eligibility; QP may only size/rebalance
    the admitted universe. This prevents weak new candidates from consuming QP
    risk/cash budget even if the final order emitter would later suppress them.
  - Targeted tests passed:
    `tests/test_qp_admission_gate.py tests/test_joint_qp_task.py
    tests/test_qp_long_short_phase2a.py tests/test_short_candidate_selection.py
    tests/test_runner_sell_attribution.py
    tests/test_repair_decision_trace_invariants.py` (`77 passed`).
- Post-prefilter WF validation completed and still failed:
  - Annual-net Sharpe by cut: `+1.037`, `+0.191`, `-0.310`.
  - Mean Sharpe `+0.306`; SPY mean Sharpe `+1.081`; delta `-0.775`.
  - Beat SPY Sharpe/APY: `0/3` and `0/3`.
  - Trade ledger contract passed, but BULL_CALM score monotonicity failed.
  - Manifest sanity remained weak: real IC `+0.0269`, shuffled `-0.0019`,
    placebo `+0.0282` versus required `< +0.0135`.
  - Conclusion: QP solver prefilter is necessary architecture hardening, but
    it is not the alpha-conversion fix.
- Latest repair bundle after sidecar audits:
  - `run_wf_gate.py` now marks any run with skipped WF/sanity/trade/parity
    gates or disabled trade traces as `diagnostic_only`; skipped gates can no
    longer stamp a promotable PASS.
  - `TopUpHeldTask` is disabled when joint QP is active (`solver=qp`), because
    held-position adds must be sized by QP and pass the same panel/rank/slot/
    turnover/cash/correlation contracts. Potential standalone top-ups are
    stamped `topup_owned_by_qp`.
  - Feature-space transform is centralized in
    `kernel.panel_pipeline.feature_transform`: runtime raw rows apply all
    artifact mean/std stats; prebuilt panel rows apply only columns declared
    raw in the panel, currently robust-z fundamental columns. Training,
    calibrator fitting, WF sanity, and runtime scoring now use this contract.
  - Targeted tests passed:
    `tests/test_panel_feature_transform.py tests/test_wf_gate_cli_contract.py
    tests/test_kelly_sizing.py tests/test_buy_quality_gates.py
    tests/test_qp_admission_gate.py tests/test_joint_qp_task.py
    tests/test_lookahead_propagation.py` (`159 passed`).
- Latest WF recipe-contract hardening:
  - `run_wf_gate.py` now includes `feature_norm_kind` and
    `feature_source_contract` in the recipe fingerprint. This prevents a
    scorer trained with one feature-space contract from reusing a manifest
    generated under another contract.
  - `daily_retrain_alpha158_fund.py` resolves CLI output overrides relative
    to the repo root, fixing the staged retrain crash where a relative output
    path could not be reported with `relative_to(ctx.repo_dir)`.
  - Targeted tests passed:
    `tests/test_wf_gate_cli_contract.py
    tests/test_daily_retrain_alpha158_fund.py
    tests/test_panel_feature_transform.py tests/test_kelly_sizing.py
    tests/test_buy_quality_gates.py tests/test_qp_admission_gate.py
    tests/test_joint_qp_task.py tests/test_lookahead_propagation.py`
    (`183 passed`).
- Latest WF manifest cutoff fix:
  - Walk-forward manifests now preserve `effective_train_cutoff_date` from
    each scorer artifact.
  - `WalkForwardModelLoader` uses `effective_train_cutoff_date +
    lookahead_days` for label-safety when available, instead of applying
    lookahead to the selection cutoff a second time.
  - Invariant: the model may become eligible after the last label it could
    have seen, not after an extra redundant 60-business-day delay.
  - Targeted tests passed:
    `tests/test_walkforward_loader.py tests/test_walkforward_manifest.py
    tests/test_walkforward_artifact_isolation.py tests/test_sim_walkforward.py
    tests/test_walkforward_eval_config.py` (`65 passed`).
- Latest LEAN/QP target parity fix:
  - QP buy orders now set executable `target_pct` from the actual emitted
    share count after integer rounding and cash caps.
  - The optimizer's desired `target_w` remains in decision inputs as
    `target_w`; `actual_target_w` records the post-cap execution target.
  - This prevents LEAN `SetHoldings` from re-expanding a cash-capped QP buy
    back to the unconstrained optimizer weight.
  - Targeted tests passed:
    `tests/test_joint_qp_task.py tests/test_qp_admission_gate.py
    tests/test_lean_backend.py tests/test_bug22_rs_score_keyerror.py
    tests/test_emit_orders_helpers.py` (`86 passed`).
- `d45c38b fix(wf): fail closed on dropped experiment overrides`
  - `scripts/wf_config_builder.py` now refuses to silently drop semantic
    experiment overrides such as
    `rotation.joint_actions.qp_admission_gate.max_sigma_by_regime` while
    deriving production-semantic WF configs.
  - `--preserve-experiment-overrides` exists for diagnostic A/B runs only;
    these are non-promotable unless production parity also passes.
  - Targeted tests passed:
    `tests/test_wf_gate_cli_contract.py tests/test_wf_config_parity.py
    tests/test_qp_contracts.py tests/test_audit_2026_05_04_fixes.py::TestQPTaxAwareDisabledByDefault`
    (`44 passed`).
- `e262783 fix(live): persist drawdown buy halt state`
  - Sim carried drawdown `skip_buys` hysteresis across bars; live always
    rebuilt `InferenceContext(skip_buys=False)`. With
    `drawdown_resume_pct`, live could re-enable buys earlier than sim while
    still in the recovery band.
  - RunnerAdapter now reads `skip_buys` from `live_state.<broker>.json` and
    writes it back on commit.
  - Targeted tests passed:
    `tests/test_runner_hwm_guard.py tests/test_runner_state_fixes.py
    tests/test_no_trade_monitor.py tests/test_live_state_db_canonical.py
    tests/test_pipeline.py::TestDrawdownCircuitTaskResets
    tests/test_joint_qp_task.py` (`139 passed`).
- Sim/live parity audit:
  - Sim/live/LEAN share `InferencePipeline` / `SellOnlyPipeline` and the core
    decision kernel, but adapters are not byte-identical. Context construction,
    execution, and DB row construction remain separate code paths.
  - New handoff doc:
    `doc/research/2026-05-24-sim-live-parity-audit.md`.
  - Key remaining risk: duplicated adapter decision-trace writers and
    manually built `InferenceContext` fields can drift again.
- Sigma-cap diagnostic after preserving the actual override:
  - Baseline strict trace: 56 closed trades, gross `+11238.72`, tax
    `+10370.53`, net `+868.19`, mean Sharpe `+0.133`, SPY mean `+1.081`.
  - True `BULL_CALM max_sigma=0.38` diagnostic: 31 closed trades, gross
    `+5181.30`, tax `+4477.16`, net `+704.14`, mean Sharpe `+0.255`, SPY mean
    `+1.081`.
  - It still fails benchmark-relative WF and sanity (`real_ic=+0.0385`,
    `shuffled_ic=+0.0024`, `placebo_ic=+0.0460`). Simple sigma cap is not a
    production fix.

## Active Validation

Current-contract WF gate completed:

```bash
.venv/bin/python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.foreground_20260523-094050.staging.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json \
  --derive-config-from-prod \
  --jobs 3 \
  --trace-dir artifacts/diagnostics/wf_trade_traces/codex_current_contract_20260523-190033
```

Log:
`logs/wf_gate_104/current_contract_20260523-190033.log`.

Generated WF config checked:
`tax.cash_debit_mode=reporting_only`.

Verdict: `FAIL`.

- Annual-net WF Sharpe mean: `+0.816`.
- Annual-net WF APY mean: `+7.55%`.
- SPY Sharpe mean: `+1.081`.
- Strategy minus SPY Sharpe: `-0.265`.
- SPY APY mean: `+16.94%`.
- Strategy minus SPY APY: `-9.39pt`.
- Positive Sharpe cuts: `3/3`.
- Beat SPY Sharpe: `1/3`.
- Beat SPY APY: `0/3`.
- Benchmark-lag regimes: `HIGH_CALM`, `LOW_SPIKED`.
- Trade ledger contract: passed.
- Trade monotonicity gate: passed only because BULL_CALM was eligible and
  passed; pooled Spearman was `-0.002`, so single-trade score monotonicity is
  still weak.
- Sanity battery: failed. Real IC `+0.0750`, shuffled IC `-0.0020`, placebo IC
  `+0.0462`; placebo must be `< +0.0375`.
- Follow-up diagnostic: shuffled labels are clean across 10 seeds (max |IC|
  about `0.0047`), but future-shift labels remain correlated at many horizons:
  shift 5d `+0.0734`, 20d `+0.0670`, 60d `+0.0462`, 120d `+0.0835`,
  252d `+0.0741`. Treat this as unresolved slow-factor/placebo methodology
  risk, not proof of clean alpha.

Do not promote this candidate. Do not run production buy/full from this
evidence. The immediate research question is why the time-shift placebo keeps
too much signal, and why positive event-level returns still fail benchmark and
annual-net acceptance.

Current-contract cut-level metrics:

| cut | event APY | event Sharpe | annual-net APY | annual-net Sharpe | SPY Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | `+15.71%` | `+1.497` | `+9.54%` | `+0.848` | `+1.778` | `-0.931` |
| 2024-07-01 to 2025-06-30 | `+11.51%` | `+1.152` | `+4.67%` | `+0.462` | `+0.715` | `-0.253` |
| 2025-04-01 to 2026-03-28 | `+13.36%` | `+2.140` | `+8.44%` | `+1.139` | `+0.749` | `+0.390` |

Current-contract trade forensics, after rebuilding from raw trade events with
the current config's `hifo` tax-lot method:

- Round trips: `191` total, `158` closed, `33` open.
- Closed gross P/L: `+$32.78k`.
- Closed tax estimate: `+$27.16k`.
- Closed tax-estimated net P/L: `+$5.63k`.
- Closed win rate: `62.0%`.
- Median hold: `60d`; average hold: `86.2d`.
- Tax integrity is clean under current semantics: `tax_cash_debited=$0`, no
  positive closed row has `tax > gross_pnl`, and no losing row has positive tax.
- Exit buckets:
  - `stop_loss`: 36 exits, gross/net `-$17.59k`, win rate `0%`.
  - `trailing_stop`: 59 exits, gross `+$39.23k`, net `+$19.08k`.
  - `qp_sell`: 39 exits, gross `+$8.79k`, net `+$3.65k`.
  - `panel_conviction`: 9 exits, gross `-$0.29k`, net `-$0.34k`.
- Entry source:
  - QP buys: 89 closed, gross `+$21.53k`, net `+$2.46k`, win `57.3%`.
  - Top-ups: 69 closed, gross `+$11.26k`, net `+$3.17k`, win `68.1%`.
- Entry rank-score versus realized `pnl_pct` Spearman: `-0.0028`.
- Score deciles are not cleanly monotonic; the 8th decile lost money and the
  6th/9th deciles carried much of the gross P/L.

Important correction: the earlier "tax greater than gross" forensic symptom
was caused by replaying HIFO-configured sim trades with a FIFO round-trip
matcher. It was an attribution bug, not a broker-cash debit regression.

## 2026-05-23 Panel-Exit Ownership Fix

- Found one more decision-tree ownership issue while the QP-gated WF rerun was
  running: `TickerSellJob` still executed legacy `PanelConvictionExitTask`,
  while `InferencePipeline` also runs `CrossSectionalPanelExitTask` immediately
  after `PanelScoringJob`.
- Fix: production/golden config now sets
  `risk.panel_exit.legacy_enabled=false`. The legacy task still defaults on for
  backward-compatible unit tests and explicit A/B configs, but prod 104 has one
  owner for raw panel/NGBoost exits: `CrossSectionalPanelExitTask`.
- Added a regression test that proves the legacy task is a no-op when
  `legacy_enabled=false`.
- Hardened `scripts/wf_config_parity.py` to compare `risk.panel_exit` as a
  semantic path. A WF side config can no longer drift on panel-exit ownership
  or sell thresholds without failing before simulation.
- Also corrected stale exit docs: trailing stop, stop loss, and single-day loss
  are regime-configured, not BULL_CALM-only.

## 2026-05-23 Decision-Trace Reason Completeness

- Tightened the daily decision-tree DB contract. For every non-selected
  watchlist ticker, sim/live adapters now write an explicit terminal reason:
  `universe_floor`, `broker_pending`, `held_no_new_buy`, `no_model_signal`, or
  `not_selected` when no earlier gate populated `blocked_by`.
- `decision_trace_integrity_report()` now reports `decision_reason_gaps` and
  fails `ok` when any non-selected row has NULL `blocked_by`.
- Added tests so the live and sim adapters keep these reason labels wired.

## 2026-05-23 Stop-Loss Regime Contract Hardening

- Made cumulative stop-loss regime ownership explicit. Production/golden
  configs now declare `risk.stop_loss_anchor_policy.mode=current_regime`, which
  preserves current behavior: cumulative stops use the current regime's
  `stop_loss_pct`.
- Added an explicit A/B-only mode, `max_entry_current`, for the BULL_CALM thesis:
  when a position entered under BULL_CALM and the market later relabels to a
  tighter-stop regime, the cumulative stop may be kept no tighter than the
  entry-regime stop. This is not promoted; it is only a paired experiment hook.
- Live and sim sell logs now persist applied stop-anchor fields in
  `decision_inputs`, so future trade forensics can tell whether a stop came
  from the current regime or entry-regime anchoring.
- WF config parity now checks `risk.stop_loss_anchor_policy`, preventing a
  side config from silently changing risk semantics.
- Targeted tests passed:
  `tests/test_exit_param_wiring.py tests/test_sim_sell_attribution.py
  tests/test_runner_sell_attribution.py tests/test_wf_config_parity.py` and
  the broader exit neighborhood through panel-exit/sell-gate tests.

## 2026-05-23 QP Soft-Sell Lot-Age Guard

- WF forensics separated order-level holds from HIFO lot-level round trips:
  QP sells looked old at the aggregate position level, but HIFO could dispose
  recently-added high-cost lots with very short lot ages.
- Fixed QP soft-sell horizon gating to check the minimum age of the actual lot
  that would be disposed under the configured lot method (`hifo`/`fifo`), not
  only the position's aggregate `entry_date`.
- QP now stamps held-ticker suppression reasons into `_blocked_by_ticker`, so
  daily decision traces can explain a held ticker's blocked QP trim.
- Regression tests cover HIFO blocking a 4-day top-up lot while allowing FIFO
  when the disposed lot is old enough.
- Targeted tests passed:
  `tests/test_joint_qp_task.py tests/test_hifo_lot_selection.py
  tests/test_tax_lots_g7.py tests/test_qp_contracts.py
  tests/test_sim_trade_ledger.py`, plus the broader QP suite through
  `tests/test_portfolio_qp_solver.py tests/test_qp_refactor_2026_04_29.py
  tests/test_qp_integration.py tests/test_qp_admission_gate.py`.

## 2026-05-23 Manifest-OOS Sanity Fix

The prior WF sanity battery used a full-trained staging artifact for the
diagnostic score path. That was not point-in-time OOS-safe. The repaired sanity
path now scores through the walk-forward manifest artifacts with
`WalkForwardModelLoader`, validates every manifest row, requires an explicit
cutoff/lookahead contract for static artifacts, and fail-closes skipped placebo
diagnostics.

Direct manifest-scoped sanity result:

- Real IC: `+0.0269`.
- Shuffled-label IC: `-0.0019`.
- Time-shift placebo IC: `+0.0282`.
- Eval range: 2024-02-02 to 2026-02-11.
- OOS dates: `508`.
- Manifest artifacts used: `37`.
- Verdict: `FAIL`, because placebo must be available and below half of real IC
  (`+0.0135`).

Interpretation: current model evidence is much weaker than the old static
sanity suggested. The main scientific issue is now explicit: the signal is not
cleanly separated from slow persistence/placebo structure in the true manifest
OOS path.

## 2026-05-23 WF Validation After Lot-Age Fix

Two exploratory WF runs completed. Both skipped strict promotion sanity/config
gates by design and both failed. They are diagnostics only; promote nothing.

### Robust QP Mean Penalty + Lot-Age Guard

- Config:
  `artifacts/diagnostics/wf_eval_configs/codex_robust_mu_k015_lotage_20260523.json`.
- Trace:
  `artifacts/diagnostics/wf_trade_traces/codex_robust_mu_k015_lotage_20260523`.
- Log:
  `logs/wf_gate_104/robust_mu_k015_lotage_20260523.log`.
- WF result: `FAIL`.
- Mean annual-net Sharpe: `+0.719`.
- Positive Sharpe cuts: `3/3`.
- SPY mean Sharpe: `+1.081`.
- Strategy minus SPY Sharpe: `-0.362`.
- Beat SPY Sharpe: `1/3`.
- Beat SPY APY: `0/3`.
- Lag regimes: `HIGH_CALM`, `LOW_SPIKED`.
- Closed round trips: `66`.
- Closed gross P/L: `+$6.09k`.
- Tax estimate: `+$8.21k`.
- Tax-estimated net P/L: `-$2.12k`.
- Win rate: `53.0%`; median hold `33.5d`.
- Tax integrity: clean. `tax_cash_debited=0`; tax is reporting-only.
- Score monotonicity still bad: rank-score vs net Spearman `-0.1396`;
  `mu` vs net `-0.0644`; raw panel score vs net `-0.1233`.

Interpretation: the lot-age guard is still a valid protective fix, but it does
not create tradable alpha or benchmark-relative acceptance by itself.

### Robust QP Mean Penalty + Lot-Age Guard + BULL_CALM Stop Anchor A/B

- Config:
  `artifacts/diagnostics/wf_eval_configs/codex_robust_mu_k015_stop_anchor_lotage_20260523.json`.
- Trace:
  `artifacts/diagnostics/wf_trade_traces/codex_robust_mu_k015_stop_anchor_lotage_20260523`.
- Log:
  `logs/wf_gate_104/robust_mu_k015_stop_anchor_lotage_20260523.log`.
- WF result: `FAIL`.
- Mean annual-net Sharpe: `+0.396`.
- Positive Sharpe cuts: `3/3`.
- SPY mean Sharpe: `+1.081`.
- Strategy minus SPY Sharpe: `-0.685`.
- Beat SPY Sharpe: `0/3`.
- Beat SPY APY: `0/3`.
- Closed round trips: `66`.
- Closed gross P/L: `+$8.01k`.
- Tax estimate: `+$9.05k`.
- Tax-estimated net P/L: `-$1.04k`.
- Win rate: `63.6%`; median hold `48.5d`.
- Stop losses dropped to 5 exits, but those were much larger: stop-loss
  gross/net `-$6.67k`, average P/L `-22.4%`.
- Score monotonicity remained near zero: rank-score vs net `-0.0339`,
  `mu` vs net `-0.0333`, raw panel score vs net `+0.0039`.

Interpretation: reject this stop-anchor A/B. It reduces stop count but lets
concentrated losses grow, weakening Sharpe/APY and SPY-relative performance.

Decision rule remains unchanged: promote nothing unless a strict rerun without
`--skip-sanity` and without side-config drift passes WF, SPY-relative, regime,
calibration, config, and decision-trace gates.

## 2026-05-23 Feature-Space Retrain Status

Run `codex_featspace_20260523-211211` trained a new 172-feature
feature-space-aligned panel scorer and staged calibrator.

- Staged scorer:
  `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json`.
- Staged calibrator:
  `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.codex_featspace_20260523-211211.staging.json`.
- Feature contract: `global_z=158`, `robust_z=5`, `identity=9`.
- Model diagnostics: CV OOS IC `+0.0473`; train IC `+0.1190`.
- Calibrator diagnostics: pool IC `+0.1152`; per-date IC `+0.1193`;
  base rate about `0.5005`; flat-gate passed.
- Strict WF gate result: `FAIL`, intentionally fail-closed because the old
  manifest recipe fingerprint (`sha256:ccc412d08c0f3463`) does not match the
  candidate recipe fingerprint (`sha256:f4596e333baf90a8`).

Interpretation: the feature-space fix produced a better-looking training
diagnostic, but there is no valid WF Sharpe/APY yet. The next acceptance-grade
step is regenerating a walk-forward manifest under this exact feature contract.

## 2026-05-23 Universe Fail-Closed Fix

Universe admission now fails closed for missing or invalid model evidence on
offensive new-buy names.

- Missing `trained_date` under `model_staleness_days > 0` rejects the ticker as
  `trained_date_missing`.
- Invalid `trained_date` rejects the ticker as `trained_date_invalid`.
- Missing universe-floor metrics reject as `{floor_type}_missing`.
- Unknown `ranking.universe_floor.type` raises `ValueError` instead of
  admitting all.
- Held tickers remain exempt from staleness/floor rejection so the sell path
  stays armed and existing positions cannot become structurally unsellable.

Verification:

- `.venv/bin/python -m pytest tests/test_universe_alignment.py tests/test_universe_held_exemption.py tests/test_daily_104_e2e.py -q`
  -> `38 passed`.

## 2026-05-23 Calibrator Metric-Scope Fix

`scripts/fit_calibrator_alpha158_fund.py` no longer writes calibrator-fit IC
into the numeric `scorer_oos_mean_ic` field. That field was misleading: the
script scores the same panel window used to fit the calibrator, so the metric
is a fit-window diagnostic even when the caller bounds the window with
`--data-start/--data-end`.

New metadata:

- `scorer_ic_scope="calibrator_fit_window"`.
- `scorer_ic_window` is `cli_bounded_panel` or `full_available_panel`.
- `scorer_fit_window_mean_ic`, median, and `n_dates` carry the diagnostic.
- `scorer_oos_mean_ic` and `scorer_oos_mean_ic_vs_er_label` are deliberately
  `null`; true OOS IC must come from WF manifests/evaluators.

Verification:

- `.venv/bin/python -m pytest tests/test_fit_calibrator_raw_label_contract.py tests/test_calibrator_no_flat_region.py tests/test_calibrator_saturation_guards.py -q`
  -> `38 passed`.

Operational note: the currently running 172-feature WF job was started before
this fix. It may finish as a useful diagnostic, but its per-fold calibrators
must be re-stamped/refit with the fixed script before being used as
acceptance-grade evidence.

## 2026-05-23 ntfy Duplicate-Success Fix

`live.runner` remains the single source of success/trade decision ntfy. The
shell wrappers no longer send a second raw success ntfy after a successful
runner cycle.

- `scripts/daily_104.sh` now keeps failure alerts and the explicit
  `BUY-BLOCKED` fallback alert, but suppresses normal wrapper success ntfy.
- `scripts/live_only_104.sh` now keeps failure alerts only on wrapper failure;
  open/preclose success decisions come from `live.runner`.
- Removed the stale wrapper trade-log parser that looked for legacy
  `signal/order.qty` fields while current 104 logs use `action/shares/qty`.

Verification:

- `.venv/bin/python -m pytest tests/test_smoke_test_model.py tests/test_runner_trade_ntfy.py tests/test_alerts.py -q`
  -> `66 passed`.

## 2026-05-23 Preopen Cancel Alert Fix

The preopen severe-gap cancel gate no longer reports a misleading
`cancelled` action when every Alpaca cancel request fails.

- Full success -> `action="cancelled"` and taxonomy `PREOPEN_CANCEL`.
- Mixed success/failure -> `action="partial_cancelled"` and taxonomy
  `PREOPEN_CANCEL_PARTIAL`.
- All failures -> `action="cancel_failed"` and taxonomy
  `PREOPEN_CANCEL_FAILED`.
- Only successful cancels are written to the preopen cancel ledger.

Verification:

- `.venv/bin/python -m pytest tests/test_preopen_cancel_gate.py tests/test_alerts.py -q`
  -> `12 passed`.

## 2026-05-23 Correlation Metadata Fail-Closed Fix

Correlation artifacts without `as_of_date` are no longer accepted silently by
strict sim/LEAN/QP paths.

- `assert_correlation_no_leakage()` raises on missing `as_of_date` by default.
- Explicit migration override:
  `regime.allow_legacy_correlation_without_as_of=true`.
- SimAdapter, LEAN `main.py`, and QP full-sigma fallback pass that override
  explicitly and otherwise fail closed.
- New preflight check `P-CORR-METADATA` hard-fails full/buy runs on missing,
  unreadable, or unstamped correlation artifacts; sell-only soft-passes so
  held-position risk exits stay armed.
- Current prod correlation artifact is stamped with `as_of_date=2026-05-22`.

Verification:

- `.venv/bin/python -m pytest tests/test_correlation_guard.py tests/test_preflight.py tests/test_strategy_artifact_contracts.py tests/test_p0_fixes_regression_guards.py tests/test_sim_walkforward.py tests/test_sim_pipeline_smoke.py tests/test_persistence.py::TestSimAdapterIntegration -q`
  -> `142 passed`.

## 2026-05-23 SEC Fundamentals Point-In-Time Fix

SEC fundamentals now use actual filing availability instead of pretending
every quarterly row becomes available exactly `end + 45 days`.

- `scripts/fetch_sec_fundamentals.py` sets each quarterly row's
  `available_date` to the max actual SEC `filed` date among contributing
  concepts.
- `scripts/build_extended_fundamentals.py` uses the same rule for extended
  raw-concept features.
- If old/raw fixtures lack `filed`, the scripts still fall back to the
  conservative `end + 45 days` rule.
- Daily forward-fill begins on `available_date`, so pre-filing dates remain
  NaN rather than leaking future filings.

Verification:

- `.venv/bin/python -m pytest tests/test_sec_fundamentals_pit.py tests/test_panel_training_cutoff.py tests/test_panel_bugfixes.py tests/test_panel_factors.py -q`
  -> `38 passed`.

## 2026-05-23 Feature-Space WF Gate Rerun

A same-recipe 40-row walk-forward manifest now covers the feature-space staged
candidate `codex_featspace_20260523-211211`.

- Manifest:
  `artifacts/sim/walkforward_manifest_172_featspace_20260523.scopefixed.covered.json`.
- Validation trace:
  `artifacts/diagnostics/wf_trade_traces/codex_featspace_20260523-211211_wf_featspace_scopefixed_covered`.
- Recipe fingerprint: `sha256:f4596e333baf90a8`.
- Verdict: `FAIL`.

Annual-net gate metrics:

| cut | annual-net APY | annual-net Sharpe | SPY Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | `+3.13%` | `+0.701` | `+1.778` | `-1.077` |
| 2024-07-01 to 2025-06-30 | `+2.19%` | `+0.707` | `+0.715` | `-0.008` |
| 2025-04-01 to 2026-03-28 | `-0.56%` | `-0.208` | `+0.749` | `-0.958` |

Summary:

- Mean annual-net Sharpe: `+0.400`.
- Positive Sharpe cuts: `2/3`.
- Beat SPY Sharpe/APY: `0/3` and `0/3`.
- Benchmark-lag regimes: `HIGH_CALM`, `LOW_SPIKED`.
- Trade ledger contract: passed.
- Trade monotonicity: failed in active regime `BULL_CALM`.

The first full run exposed a WF gate bug: manifest sanity used
`cutoff_date + lookahead_days` even when a row stamped
`effective_train_cutoff_date`, double-embargoing valid folds and failing as
`prediction failed`. Fixed in `scripts/run_wf_gate.py` by using the same
safe-label convention as `WalkForwardModelLoader`.

After the gate fix, sanity computes real diagnostics instead of crashing:

- Real IC: `+0.0218`.
- Shuffled IC: `+0.0012` (clean).
- Time-shift placebo IC: `+0.0263`, above the required `< +0.0109`.
- Verdict remains `FAIL`.

Interpretation: the feature-space retrain improved train/CV diagnostics but
does not yet prove tradable, benchmark-beating alpha. The current failure is
not tax-cash corruption: `event_level_tax_debited=0` in all three traces, and
the gate uses annual-net tax economics.

## 2026-05-23 Decision Trace / QP Reason Hardening

Decision-trace opacity found during sidecar audits is partially fixed:

- Strict QP μ contract failures now stamp affected tickers with
  `qp_mu_contract_block`.
- Non-optimal global QP status now stamps QP tickers with
  `qp_global:<status>` or `qp_no_signal`, and stores
  `ctx._qp_status`, `ctx._qp_failure_reason`, and `ctx._qp_diagnostics`.
- Empty cached feature slices now stamp `empty_cached_features`.
- Non-selected `candidate_scores` rows now default to
  `candidate_not_selected` instead of NULL.
- `decision_trace_integrity_report()` now fails on candidate reason gaps and
  selected candidate rows carrying a blocker.
- Walk-forward forensic reports now label WF scoring as
  `walkforward_manifest_per_bar`; the config artifact path is reported only as
  a seed, not as the per-bar model actually used.
- Sim/live `ticker_daily_state.blocked_by` now preserves exact universe
  rejection reasons as `universe:<reason>` instead of collapsing
  `ic_missing`, `trained_date_missing`, stale models, and auto-drop into a
  generic `universe_floor` label.
- Decision-trace integrity now fails when a sell row lacks realized economic
  attribution (`gross_pnl`, `tax`, `net_pnl_after_tax`). Shares alone are not
  enough to explain a loss bucket.
- LEAN now writes the same sidecar SQLite decision trace as sim/live:
  `pipeline_runs`, `candidate_scores`, `trades`, and full-watchlist
  `ticker_daily_state`. Universe rejection reasons are preserved as
  `universe:<reason>` in LEAN too, so LEAN Sharpe/APY can be tied back to
  the per-ticker decision tree rather than only runtime logs.

Verification:

- `.venv/bin/python -m pytest tests/test_qp_integration.py tests/test_joint_qp_task.py tests/test_candidate_blocked_by.py tests/test_persistence.py tests/test_sim_trade_ledger.py tests/test_wf_gate_recipe_scope.py tests/test_wf_gate_cli_contract.py tests/test_promote_wf_gate.py -q`
  -> `145 passed`.
- `.venv/bin/python -m pytest tests/test_runner_state_fixes.py tests/test_universe_alignment.py tests/test_ticker_daily_state.py tests/test_persistence.py::TestTrades -q`
  -> `92 passed`.
- `.venv/bin/python -m pytest tests/test_persistence.py tests/test_sim_sell_attribution.py tests/test_runner_sell_attribution.py tests/test_sim_trade_ledger.py -q`
  -> `40 passed`.
- `.venv/bin/python -m pytest tests/test_lean_trace_persistence.py tests/test_persistence.py tests/test_universe_alignment.py tests/test_audit_2026_04_24_fixes.py::TestLeanAdapterPartialAndTopup tests/test_audit_2026_05_04_fixes.py::TestLeanAdapterPrevClosesNaNGuard tests/test_audit_2026_05_04_fixes.py::TestLeanAdapterTaxNaNGuard -q`
  -> `60 passed`.

Still pending: run an actual LEAN backtest smoke with persistence enabled and
verify row counts/artifact paths in Docker output before treating LEAN traces
as operationally proven.

## 2026-05-23 Feature-Space WF Trade-Quality Diagnosis

Latest strict feature-space WF trace was re-analyzed with the prod-semantic
config and HIFO tax lots:

- Report:
  `artifacts/wf_trade_forensics_featspace_tracefix_prodsemantic_20260523.md`.
- Closed round trips: `42`.
- Gross P/L: `+$10.45k`; tax estimate `+$8.52k`; net after estimated tax
  `+$1.93k`.
- Tax integrity is clean: `tax_cash_debited=0`, no positive row has tax above
  gross, and no losing row has positive tax.
- All entries came from `JointPortfolioQPJob`; no greedy/top-up mix in this
  trace.
- Entry score monotonicity is effectively absent:
  `rank_score` vs net Spearman `-0.007`, `mu` vs net `+0.016`,
  raw `panel_score` vs net `-0.042`.
- Stop-loss exits are the dominant gross loss bucket:
  9 exits, gross/net `-$5.91k`, win rate `0%`.
- High-score and high-uncertainty names are not safer in the traded subset:
  entries with `rank_score >= 0.63` had net `-$1.01k`, while lower-rank
  entries had net `+$2.95k`; the highest `entry_sigma` quartile had net
  `-$0.49k`.

Interpretation: the current failure is no longer a tax-cash bug. The model
still has weak point-in-time IC, and after QP admission the traded subset is
not score-monotone. QP is selecting high-μ/high-σ names that later become
stop-loss losses. This supports testing an uncertainty-aware admission cap,
but only as a regime-conditional A/B, not a silent production flip.

Implementation hook added:

- `rotation.joint_actions.qp_admission_gate.max_sigma` and
  `max_sigma_by_regime` can now block new QP buys whose candidate sigma is
  above the configured cap.
- `topup_max_sigma` and `topup_max_sigma_by_regime` provide the same hook for
  held top-ups.
- Default remains off, preserving production behavior until a paired WF A/B
  passes.

Theory support: Markowitz mean-variance allocation requires expected return to
pay for risk; Kelly sizing scales exposure by edge over variance; and
transaction-cost/no-trade literature warns against trading marginal edges once
cost and risk bands are considered. In this project, the empirical symptom is
that high sigma is associated with stop-loss realizations after QP admission.

Verification:

- `.venv/bin/python -m pytest tests/test_qp_admission_gate.py tests/test_joint_qp_task.py tests/test_qp_integration.py -q`
  -> `68 passed`.

Follow-up A/B result:

- Diagnostic config:
  `backtesting/renquant_104/artifacts/diagnostics/wf_eval_configs/qpsigma_bullcalm039_20260523.json`.
- Trace:
  `backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/qpsigma_bullcalm039_20260523`.
- Forensics:
  `artifacts/wf_trade_forensics_qpsigma_bullcalm039_20260523.md`.
- Result: diagnostic-only FAIL because it still loses to SPY in all 3 cuts.
  Mean annual-net Sharpe improved from the prior strict feature-space run
  (`+0.400`) to `+0.523`, all 3 cuts became positive, and trade
  monotonicity passed, but mean SPY Sharpe was `+1.081` and
  strategy-minus-SPY Sharpe remained `-0.558`.
- Annual-net APY by cut: `+2.49%`, `+0.80%`, `+1.07%`; mean `+1.45%`
  versus SPY mean APY `+16.94%`.
- Trade forensics: closed round trips fell from `42` to `34`; win rate rose
  to `73.5%`; net after estimated tax improved to `+$3.15k`; stop-loss exits
  fell from `9` to `3` but remain pure losses.

Interpretation: sigma admission is directionally useful as a risk-control
filter, not sufficient alpha conversion. Do not promote it directly. The next
work item is to combine this with an explicit benchmark/exposure objective or
model-side improvement so BULL_CALM does not under-participate versus SPY.

## 2026-05-23 Trace / Rotation Hardening

Sidecar audits found additional silent-fallback holes. Fixed and tested:

- QP slot accounting now budgets already-admitted/emitted new candidates, so
  one open slot cannot admit multiple new names.
- `thesis_primary` and `thesis_symmetric` rotation modes now exclude holdings
  that already have same-bar exits; `EmitRotationsTask` also suppresses any
  duplicate sell if a prior exit exists.
- `candidate_scores` now persists missing raw/rank/RS scores as SQL `NULL`,
  not `0.0`.
- LEAN contexts now stamp a run id before score-distribution tasks execute, so
  score distribution rows are not orphaned from `pipeline_runs`.
- LEAN panel-frame preparation is fail-closed when panel scoring is enabled.
- Sim decision trace now extracts `model_type` from dict artifacts with
  `_metadata`, matching live/LEAN helpers.
- `TickerInferenceContext` score snapshots for model-signal hold/sell rows are
  propagated into `ticker_daily_state`, so non-buy model decisions retain
  rank/expected-return evidence.
- Live `ticker_daily_state` write failures default to strict re-raise via
  `persistence.strict_ticker_daily_state=true` unless explicitly disabled.
- `RegimeFinalizeTask` now stamps `_regime_evidence` and `build_run_bundle()`
  persists it in `pipeline_runs.run_bundle_json`. The evidence includes the
  branch source, Hurst state, GMM/HMM probabilities, hard-bear flag, 5d
  vol/return, transition state, and SPY MA50/MA200 proof fields. This closes
  the audit gap where a BEAR flip had to be reconstructed from logs.

Verification:

- `.venv/bin/python -m pytest tests/test_qp_admission_gate.py tests/test_joint_qp_task.py tests/test_qp_integration.py tests/test_thesis_primary_rotation.py tests/test_session_silent_bugs.py::TestThesisSymmetricReachable tests/test_rotation_atomic.py tests/test_persistence.py tests/test_lean_trace_persistence.py tests/test_runner_state_fixes.py tests/test_ticker_daily_state.py -q`
  -> `173 passed`.
- `.venv/bin/python -m pytest tests/test_artifact_contract.py tests/test_regime_detector_5day_and_chop.py tests/test_trend_overlay.py tests/test_regime_confidence_fix.py tests/test_wf_config_parity.py -q`
  -> `62 passed`.
- `.venv/bin/python -m py_compile backtesting/renquant_104/kernel/portfolio_qp/job_qp.py backtesting/renquant_104/kernel/portfolio_qp/tasks.py backtesting/renquant_104/kernel/pipeline/task_rotation.py backtesting/renquant_104/kernel/pipeline/pp_inference.py backtesting/renquant_104/kernel/persistence.py backtesting/renquant_104/adapters/lean.py backtesting/renquant_104/adapters/sim.py backtesting/renquant_104/adapters/runner.py`
  -> passed.

## 2026-05-23 Metadata Fail-Closed Hardening

Selection, rotation, joint-action, and QP now fail closed when required
sector/correlation metadata is missing.

- `passes_sector_guard()` no longer maps missing sectors to `"other"`.
  Missing candidate sector, or missing non-defensive held sector, blocks the
  new buy.
- `passes_correlation_guard()` no longer treats `corr_matrix=None` or a
  missing pair as diversification evidence. If there is an existing/selected
  holding and correlation cannot be verified, the buy is blocked.
- `JointActionTask` now calls the same correlation guard unconditionally
  instead of bypassing it when `ctx.corr_matrix is None`.
- `ApplySectorMetadataGuardTask` now caps every unmapped QP ticker at current
  weight even when `sector_map` is entirely empty. New candidates get a zero
  upper bound and `blocked_by=missing_sector_map`.
- `BuildCorrelationGroupConstraintTask` now caps tickers with missing
  correlation matrix/pairs at current weight. New candidates get a zero upper
  bound and `blocked_by=missing_correlation_matrix` or
  `missing_correlation_pair`.

Scientific reason: sector and correlation controls are risk constraints, not
optional features. If the risk model is incomplete, the system cannot infer
diversification from missing data. This follows conservative robust portfolio
construction practice: unverifiable covariance/metadata should reduce allowed
risk, not increase it.

Verification:

- `.venv/bin/python -m pytest tests/test_kernel_units.py tests/test_joint_actions.py tests/test_qp_sector_constraint.py tests/test_qp_correlation_constraint.py tests/test_selection_wash_sale_cost_aware.py tests/test_rotation_atomic.py tests/test_thesis_primary_rotation.py tests/test_session_silent_bugs.py::TestThesisSymmetricReachable tests/test_qp_admission_gate.py tests/test_joint_qp_task.py tests/test_qp_integration.py -q`
  -> `306 passed`.
- `.venv/bin/python -m pytest tests/test_policy_alignment.py tests/test_candidate_sector_map_gate.py tests/test_lean_policies.py tests/test_runner_state_fixes.py tests/test_ticker_daily_state.py tests/test_lean_trace_persistence.py tests/test_persistence.py -q`
  -> `515 passed`.
- `.venv/bin/python -m py_compile backtesting/renquant_104/kernel/selection.py backtesting/renquant_104/kernel/pipeline/task_joint_actions.py backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
  -> passed.

Post-fix WF diagnostic:

- Command:
  `.venv/bin/python scripts/run_wf_gate.py --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json --strategy-config artifacts/diagnostics/wf_eval_configs/base_featspace_scopefixed_covered_20260523.prod_semantic.json --strict --jobs 3 --skip-sanity --trace-dir artifacts/diagnostics/wf_trade_traces/post_metadata_failclosed_20260523`
- Verdict: FAIL because benchmark/regime gates still fail. This was
  diagnostic-only because sanity was skipped.
- Annual-net cuts:
  - 2024-01-02 to 2024-12-31: APY `+3.04%`, Sharpe `+0.671`,
    SPY Sharpe `+1.778`, ΔSharpe `-1.107`.
  - 2024-07-01 to 2025-06-30: APY `+3.34%`, Sharpe `+0.694`,
    SPY Sharpe `+0.715`, ΔSharpe `-0.021`.
  - 2025-04-01 to 2026-03-28: APY `+0.33%`, Sharpe `+0.154`,
    SPY Sharpe `+0.749`, ΔSharpe `-0.595`.
- Mean annual-net Sharpe `+0.506`; `3/3` positive, `0/3` beat SPY Sharpe/APY.
  Benchmark-lag regimes: `HIGH_CALM`, `LOW_SPIKED`.
- Forensics: `artifacts/wf_trade_forensics_post_metadata_failclosed_20260523.md`.
  Closed round trips `37`; gross `+$13.30k`; annual/event tax integrity clean
  (`tax_cash_debited=0`, reporting-only); net after event-level tax `+$4.09k`;
  win rate `62.2%`; median hold `30d`.
- Remaining structural issue: entries are now score-monotone enough to pass the
  trade gate, but APY still lags SPY because the book is low-exposure /
  under-participating in bull/calm market regimes. Stop-loss/single-day-loss
  exits are still pure loss buckets: `9` risk exits, `-$4.15k` gross.

## 2026-05-23 PatchTST / XGB Experiment Audit

PatchTST experiments did complete, but they are not promotion evidence.

- HF Trainer 5-cut x 5-seed: mean best-val IC `+0.0467`, std `0.0816`, min
  `-0.0607`, max `+0.1878`.
- HF FiLM 5-cut x 5-seed: mean best-val IC `+0.0477`, std `0.0767`, min
  `-0.0502`, max `+0.1718`.
- HF cross-stock 5-cut x 5-seed: mean best-val IC `+0.0507`, std `0.0878`,
  min `-0.0594`, max `+0.2035`.
- All three families have negative Fed/unwind cuts.

Current shadow is the strict seed44 baseline:

- Checkpoint:
  `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`.
- Summary IC: `best_val_ic=+0.030657`.
- Per-regime IC: `BULL_VOLATILE +0.0524`, `BEAR +0.1916`,
  `CHOPPY +0.0307`.
- It does not use the HF DOE point 1 `weight_decay=0.01` winner, and it does
  not use FiLM/cross-stock variants.

PatchTST vs XGB diagnostic sim is not acceptance-grade:

- Window 2024-07-02 to 2026-02-10:
  - XGB strict-cutoff: APY `+1.17%`, Sharpe `+0.20`.
  - PatchTST clean diagnostic: APY `+1.49%`, Sharpe `+0.23`.
  - SPY: APY `+15.59%`, Sharpe `+0.91`.
- Short-window 2026-05-06 to 2026-05-22 remains style-only, zero-sell
  evidence: PatchTST is more aggressive; it is not promotable.

Promotion requirement: build a true PatchTST walk-forward manifest with
per-cut artifacts, causal calibrators, 60BD embargo, train-only preprocessing,
fingerprints, per-regime/per-seed IC, PBO/DSR, and full decision-tree
APY/Sharpe/tax/turnover versus XGB and SPY.

## 2026-05-24 Benchmark Sleeve Audit / A-B

Implemented a default-off benchmark core sleeve to separate market beta from
alpha selection.

- Solver: `BenchmarkSleeveTask` uses third-party SciPy `scipy.optimize.linprog`
  with HiGHS (`solver=scipy_linprog_highs`). It is not a hand-rolled optimizer.
- The sleeve ticker is excluded from alpha buy scan, alpha QP source maps,
  cross-sectional panel exits, legacy panel exits, and the per-ticker sell
  chain. The sleeve can only be bought/sold by `BenchmarkSleeveTask` when the
  feature is enabled.
- Decision traces include the sleeve ticker when enabled. Trade contracts now
  require entry `mu/sigma` only for alpha entries; properly attributed
  `BENCHMARK_SLEEVE_BUY` rows are still audited through source/exit fields.
- Targeted tests: benchmark sleeve, panel exit, trade contract, QP integration,
  and joint-QP suites pass. The default xdist QP 200ms perf check was flaky at
  201-211ms; single-process rerun passed at ~120ms.

WF diagnostic with the same model artifact
`panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json`:

| Variant | Sleeve target | Mean APY | Mean Sharpe | SPY mean Sharpe delta | Beat SPY Sharpe | Beat SPY APY | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| regime sleeve | BULL_CALM 100%, BULL_VOL 50%, CHOPPY/BEAR 0% | +5.21% | +0.463 | -0.618 | 0/3 | 0/3 | pass |
| core100 | 100% all regimes | +16.59% | +1.187 | +0.106 | 2/3 | 0/3 | pass |
| core85 | 85% max sleeve, 15% alpha budget | +14.79% | +1.122 | +0.042 | 1/3 | 0/3 | pass |

Interpretation:

- The earlier low-exposure diagnosis is confirmed. Removing CHOPPY/BEAR
  market-timing from the benchmark sleeve lifted mean Sharpe from `+0.463` to
  `+1.187`.
- The sleeve fix is not alpha proof. `core100` mostly behaves like SPY plus
  tiny alpha and therefore improves Sharpe by restoring beta participation.
- `core85` gives alpha budget, but current alpha/QP trades do not add enough:
  alpha closed trades were gross `+$3.95k` but net `-$0.67k` after tax estimate
  across the three WF cuts. The active sleeve is still not ready for promotion
  as an SPY-beating strategy.
- The old regime sleeve was effectively using the regime detector as a market
  timer. That was the wrong default for a benchmark core. Core-satellite theory
  supports separating benchmark exposure from active bets; tactical de-risking
  needs a separate accepted overlay, not CHOPPY=0 by default.

Pending design work:

1. Decide whether the benchmark sleeve is a pure benchmark core (`core100`) or a
   core-satellite allocator with an explicit alpha budget (`core85`-style).
2. If alpha budget is kept, QP must be evaluated on marginal contribution after
   tax and turnover versus the displaced benchmark sleeve, not just raw trade
   gross P/L.
3. Do not promote the benchmark-sleeve config until strict WF passes benchmark
   APY/Sharpe gates or the acceptance policy explicitly changes to
   benchmark-relative Sharpe with documented APY trade-off.

## 2026-05-24 Follow-Up: Sleeve Funding Bug Fixed, Alpha Still Small

After the first benchmark-sleeve A/B, a structural funding bug was found in the
core-satellite implementation:

- `core85` nominally reserved 15% NAV for alpha, but QP still saw only actual
  cash after the benchmark sleeve was filled.
- Existing QP `cash_reserve_pct` could also double-count reserve against the
  SPY sleeve, so nominal alpha budget translated into only ~4-5% realized
  alpha exposure in earlier traces.
- Live runner had a separate parity bug: even when a sell was submitted before
  buys, its local buy-cash ledger used stale `ctx.cash`, so same-bar rotation
  or benchmark-sleeve funding could be locally rejected in live while sim
  accepted it.

Code fixes now in the working tree:

- `BenchmarkSleeveTask` exposes explicit alpha-funding capacity through
  `fund_alpha_from_sleeve` + `alpha_funding_budget_pct`.
- QP can treat that sleeve capacity as liquidity only when explicitly
  configured, and can offset configured cash reserve via
  `sleeve_counts_as_cash_reserve`.
- `BenchmarkSleeveTask` emits a real SPY sell whenever pending alpha buys need
  sleeve funding, even if the normal rebalance band or LP cash cap would
  otherwise no-op.
- Funding sells round share count up so buy cash is actually covered.
- Live runner credits broker-confirmed same-bar sell proceeds into its local
  buy budget (`LIVE-SAME-BAR-SELL-CREDIT`) so live/sim do not diverge.

TDD:

- `tests/test_benchmark_sleeve.py`: 14 benchmark-sleeve/funding tests.
- `tests/test_runner_state_fixes.py::TestRunnerCashBudgetGuard`: same-bar
  sell-credit contract.
- Targeted suite: 118 passed
  (benchmark sleeve + runner cash guard + QP/contract/panel-exit suites).

WF A/B with explicit `core100_fund15` diagnostic config:

| Cut | Annual-net APY | Annual-net Sharpe | Avg SPY exposure | Avg alpha exposure | Avg gross exposure |
|---|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | +22.03% | +1.571 | 94.1% | 3.3% | 97.3% |
| 2024-07-01 to 2025-06-30 | +14.70% | +0.843 | 92.1% | 2.1% | 94.1% |
| 2025-04-01 to 2026-03-28 | +10.30% | +0.896 | 86.7% | 1.2% | 87.9% |

WF gate result:

- Absolute gate: pass (`3/3` cuts positive).
- Benchmark gate: fail.
- Mean annual-net Sharpe: `+1.103` versus SPY `+1.081`, delta `+0.023`.
- Beat SPY Sharpe: `2/3`; beat SPY APY: `1/3`.
- Remaining benchmark-lag regime: `HIGH_CALM`.

Alpha trade read:

- 16 closed alpha trades across the three cuts.
- Gross P/L `+$6.93k`; after-tax net `+$2.12k`.
- Same-capital same-period SPY P/L `+$1.55k`.
- Active after-tax net versus SPY `+$0.57k`.
- Gross win rate `68.8%`; active win rate `37.5%`; median hold `24.5d`.
- Good bucket: `qp_close` active net `+$2.68k`.
- Bad buckets: `stop_loss` active net `-$1.42k`,
  `single_day_loss` active net `-$1.06k`.

Interpretation:

- The funding bug was real and is fixed under default-off flags.
- The fix improves active contribution from negative to slightly positive, but
  it does not make the strategy promotable.
- The alpha sleeve is still too small to materially improve APY/Sharpe; the
  next bottleneck is not just cash starvation. QP/admission emits very little
  active risk, and hard loss exits still dominate the active drag.
- Do not enable `fund_alpha_from_sleeve` in production until a stricter
  marginal-alpha gate shows the alpha sleeve can beat displaced SPY after tax,
  turnover, and stop-loss drag.

## 2026-05-24 PatchTST/PatchTXT Status Rechecked

PatchTST is runnable and scientifically interesting, but it remains shadow
only:

- 5-cut x 5-seed HF PatchTST families completed with positive pooled IC
  (`+0.0467` to `+0.0507`), but `cut2_fed` and `cut5_unwind` are negative
  across all checked families.
- Current shadow artifact is the stricter
  `pt07_strict_trainfit_embargo60_20260522/seed_44` model, not the older
  higher-IC canonical seed. Its validation evidence is
  `best_val_ic=+0.030657`, with positive BULL_VOLATILE/BEAR/CHOPPY IC.
- DOE best point has bull IC `+0.0580` and PBO `0.33`, but DSR is `-0.702`;
  not promotion-grade after multiple-testing correction.
- Static PatchTST long-window sim (`2024-07-02` to `2026-02-10`) reports APY
  `+1.49%` and Sharpe `+0.23`; it is not true OOS because the artifact was
  selected with validation labels reaching into the later period.
- Raw signal control is weak: the 5-date diagnostic has pooled IC `-0.016`,
  after-tax Sharpe `+0.08`, and shuffle control Sharpe `+1.17`.

Conclusion: keep PatchTST as shadow/router research. Do not promote until a
PatchTST-specific walk-forward manifest exists with causal per-cut artifacts,
calibrators, per-regime IC, PBO/DSR, shuffle/time-shift controls, and full
portfolio WF against XGB and SPY.

## 2026-05-24 Mainline Forensics Upgrade: Alpha vs Benchmark

`scripts/analyze_wf_trade_forensics.py` now reports two missing diagnostics:

- reconstructed exposure by cut: average alpha weight, benchmark weight,
  gross weight, cash weight, alpha position count, and max alpha weight;
- same-capital alpha vs benchmark P/L: for every non-benchmark closed alpha
  trade, compare after-tax alpha P/L to buying `portfolio.benchmark_sleeve.ticker`
  over the same entry/exit dates with the same entry notional.

Regression tests:

- `tests/test_wf_trade_forensics.py::test_alpha_vs_benchmark_measures_same_capital_active_pnl`
- `tests/test_wf_trade_forensics.py::test_cut_exposure_summary_separates_alpha_and_benchmark`

Targeted test result: `3 passed`.

Applied to the latest `benchmark_sleeve_core100_fund15_fundingceil_20260524`
trace:

- average alpha weight by cut: `3.25%`, `2.08%`, `1.17%`;
- average benchmark weight by cut: `94.10%`, `92.72%`, `87.08%`;
- average cash weight by cut: `2.65%`, `5.20%`, `11.75%`;
- alpha closed trades: `16`;
- alpha gross P/L: `+$6.93k`;
- alpha after-tax net P/L: `+$2.12k`;
- same-capital SPY P/L: `+$1.55k`;
- alpha active after-tax net versus SPY: `+$0.57k`;
- gross win rate: `68.8%`;
- active win rate: `37.5%`.

Exit bucket active net versus SPY:

- `qp_close`: `+$2.68k`;
- `trailing_stop`: `+$0.41k`;
- `qp_sell`: `-$0.04k`;
- `single_day_loss`: `-$1.06k`;
- `stop_loss`: `-$1.42k`.

Interpretation:

- Tax is not the current main explanation; `tax_cash_debited=0` and no losing
  rows have positive tax.
- The main APY/Sharpe ceiling is tiny active exposure plus stop/single-day-loss
  drag. The benchmark sleeve restored market participation, but the alpha
  sleeve is still too small and too inconsistent to move headline metrics.
- PatchTST must be evaluated through this same lens. IC-only and static
  long-window APY/Sharpe are not enough; promotion needs strict WF traces with
  alpha-vs-SPY active P/L, exposure, score monotonicity, and regime buckets.

## 2026-05-24 Settlement / Buying-Power Parity Fix

The benchmark-sleeve alpha-funding probe exposed two execution/parity bugs
after the `core100_fund15` A/B:

- Sim still defaulted sell-proceeds settlement to legacy T+2, while US equity
  settlement has been T+1 for most broker-dealer transactions since
  2024-05-28 under SEC Rule 15c6-1 amendments.
- Sim decision contexts exposed only settled cash to the decision tree. Live
  Alpaca uses `account.non_marginable_buying_power`, so executed-but-unsettled
  sell proceeds can fund new equity buys without using 2x/4x margin buying
  power. Result: sim/live cash semantics diverged exactly where the benchmark
  sleeve sold SPY to fund alpha.
- `BenchmarkSleeveTask` sized initial SPY buys and alpha-funding sells from
  mid price only, so a "valid" order could still be rejected once sim applied
  buy slippage, sell slippage, SEC/TAF fees, and cash buffer.

Fixes now in the working tree:

- `T2CashQueue` keeps its historical class name for import compatibility, but
  its default lag is now T+1. Tests can still request `settlement_days=2` for
  explicit legacy stress cases.
- `SimAdapter` now separates `settled_cash`, `pending_settle_cash`, and
  decision-tree `cash`. Default `execution.buying_power_mode` is
  `non_marginable_buying_power`, mirroring the live Alpaca path. A
  `settled_cash` mode exists for conservative cash-account stress tests and
  intentionally blocks same-bar unsettled reinvestment.
- `BenchmarkSleeveTask` only treats sleeve sale proceeds as alpha funding when
  the execution mode allows unsettled buying power. In `settled_cash` mode the
  alpha-funding capacity is zero.
- Benchmark sleeve buy/sell share counts now reserve slippage, fees, and the
  configured cash buffer before emitting executable orders.
- `strategy_config.json` and `strategy_config.golden.json` explicitly stamp
  the execution contract; `scripts/wf_config_parity.py` compares it as a
  semantic path so side configs cannot drift silently.

Verification:

- Targeted regression suite:
  `tests/test_t2_settlement.py tests/test_sim_execution_integration.py
  tests/test_benchmark_sleeve.py tests/test_wf_config_parity.py
  tests/test_runner_state_fixes.py::TestRunnerCashBudgetGuard
  tests/test_p0_fixes_regression_guards.py::TestP0_9_BugDSettledCash`
  -> `52 passed`.
- Focused rerun:
  `tests/test_benchmark_sleeve.py tests/test_sim_execution_integration.py
  tests/test_t2_settlement.py tests/test_wf_config_parity.py`
  -> `44 passed`.
- Py-compile passed for the changed sim, benchmark sleeve, execution backend,
  Alpaca broker, and WF parity modules.

- Probe:
  `backtesting/renquant_104/artifacts/diagnostics/alpha_exposure_probe_20260524_t1_nmbp_costcap`.
  Log scan found no `insufficient cash`, `insufficient buying power`,
  `Traceback`, or `ERROR` matches.

Probe result:

- Window: 2025-12-01 to 2026-01-31.
- Final value: `$100,069`; return `+0.1%`; APY `+0.4%`; Sharpe `+0.10`;
  max drawdown `2.9%`.
- Tax reporting remains clean: event-estimated tax `$439`, cash-debited `$0`,
  `mode=reporting_only`.
- Trades: 7 buys, 10 sells; win rate `60%`; average hold `34d`.
- Closed gross P/L: `-$1,044`; tax-estimated net P/L `-$1,483`.
- SPY sleeve round trips were positive overall: gross `+$335`, net `+$151`.
- The loss bucket is alpha, not tax: APP, COIN, and DELL entered in
  `BULL_CALM` and were sold on 2026-01-30 after the regime flipped to BEAR.
  The three stop-loss exits lost `-$1,889` gross/net. ANET was the one strong
  alpha winner (`+10.2%`, trailing stop).

Interpretation:

- This is an execution/parity fix, not alpha proof.
- The previous rejected-buy symptom is gone, so bad trade economics are now
  visible instead of hidden by cash rejection.
- The next alpha-conversion work should focus on BULL_CALM admission quality
  near regime transitions, volatility/uncertainty caps, and earlier thesis
  deterioration exits before hard stop-loss. Tax is not the current main
  blocker.

## 2026-05-24 QP Slot-Budgeting And Rejected Kelly-Priority Probe

Problem found:

- QP source-map admission was still using candidate iteration order while open
  slots were scarce. That made the admission stage partly order-dependent:
  high-rank/high-sigma candidates could consume scarce open slots before lower
  rank but better risk-adjusted candidates were even considered.
- This violates the main contract: model/gates decide whether a ticker is
  allowed to buy; QP may only size/rebalance the admitted alpha universe.

Implemented locally:

- `_BuildSourceMapTask` now prefilters new long candidates while ignoring slot
  capacity, then allocates scarce open slots in one deterministic pass.
- Held names remain in the QP universe for trim/sell; short candidates still
  bypass buy admission and override same-ticker long candidates.
- The code supports optional `qp_admission_gate.slot_priority` modes such as
  `rank_score`, `kelly_target_pct`, `mu_over_sigma`, and `panel_score`.
- Production/golden configs do **not** enable Kelly slot priority. The default
  remains `rank_score` because the diagnostic below failed acceptance.

Evidence:

- Base T+1/non-marginable buying-power short probe:
  `alpha_exposure_probe_20260524_t1_nmbp_costcap`.
  Final `$100,069`; APY `+0.4%`; Sharpe `+0.10`; closed gross P/L `-$1,044`.
  APP/COIN/DELL were the main alpha loss bucket.
- Kelly-priority probe:
  `alpha_exposure_probe_20260524_t1_nmbp_costcap_qpkelly`.
  Final `$98,859`; APY `-6.8%`; Sharpe `-0.50`; closed gross P/L `-$2,272`.
  It correctly blocked APP/COIN by `qp_admission_no_slot`, but admitted earlier
  ORCL/RBLX/NEM/SNOW losses. This means Kelly slot priority is **rejected** as
  a production parameter, even though the slot-budgeting contract fix is still
  valid.
- Sigma-cap diagnostic:
  `alpha_exposure_probe_20260524_t1_nmbp_costcap_qpsigma039`.
  Final `$101,659`; APY `+10.6%`; Sharpe `+1.16`; only 2 buys / 1 sell.
  This proves high realized-vol candidates were dangerous in the probe window,
  but it over-suppresses alpha and is not a promotion-ready rule.
- Candidate-level forensic slice:
  on 2026-01-21/22, APP and COIN had higher rank but much worse forward excess
  returns than alternatives such as MPWR/EOG. Over the 4,353 candidate rows in
  the probe, rank/mu were locally anti-monotonic to 10d/20d future excess
  returns, while sigma was especially harmful. Treat this as a red flag for
  BULL_CALM admission quality, not as a tuned rule.

Theory:

- Markowitz/Kelly sizing assumes expected returns and covariance/volatility are
  trustworthy enough that edge divided by variance is meaningful. Here the
  capped Kelly target tied across many candidates and noisy mu/sigma did not
  improve realized active P/L.
- The correct architectural lesson is narrower: slot capacity must be handled
  before vector construction so QP cannot optimize an order-dependent universe.
  The alpha-quality lesson still needs acceptance-grade WF evidence.

Verification:

- `tests/test_qp_admission_gate.py` adds a scarce-slot regression proving that
  optional Kelly priority selects the better Kelly candidate when configured.
- Targeted suite after the fix:
  `tests/test_qp_admission_gate.py tests/test_joint_qp_task.py
  tests/test_qp_integration.py tests/test_wf_config_parity.py
  tests/test_persistence.py::TestTrades tests/test_ticker_daily_state.py`
  -> `97 passed`.

Next implication:

- Do not chase a single volatility cap or Kelly priority knob. The remaining
  APY/Sharpe blocker is the upstream alpha admission layer: regime-specific
  trade-domain monotonicity, model-vs-benchmark active P/L, sigma/RS/recent-vol
  evidence, and fail-closed metadata must decide buy eligibility before QP.

## PatchTST WF Contract Progress 2026-05-24

Problem:

- PatchTST/HF sequence artifacts are `.pt` files, but
  `WalkForwardModelLoader._scorer_fingerprint_for_entry()` only derived scorer
  identity from local `.json` artifacts. A walk-forward manifest pointing at a
  PatchTST `.pt` scorer therefore could not enforce the per-fold
  scorer/calibrator fingerprint contract.

Fix:

- Local non-JSON scorer artifacts now use the exact file-byte SHA256 as
  `sha256:<hex>` scorer identity. JSON artifacts still prefer a stamped
  artifact fingerprint and fall back to file hash. Missing or non-local scorer
  URIs still return no fingerprint and fail closed in `calibrator_as_of()`.

Verification:

- `tests/test_walkforward_loader.py` now includes a PatchTST-style `.pt`
  regression: a calibrator stamped with the exact `.pt` file hash is accepted.
- Targeted WF loader/manifest suite:
  `tests/test_walkforward_loader.py tests/test_walkforward_manifest.py`
  -> `25 passed`.

Next implication:

- PatchTST is still not production-ready. This makes the strict WF contract
  capable of covering `.pt` scorer artifacts. The training script now also
  supports point-in-time `--train-cutoff` / `--data-end` windows and emits a
  `*.pt.metadata.json` sidecar with the file-byte artifact fingerprint.
- One leakage fix was important: because HF Trainer selects the best checkpoint
  using validation labels, the artifact's `effective_train_cutoff_date` now
  covers train + validation labels, not only the raw train split.
- The HF calibrator now refuses to treat `config_fingerprint` as scorer
  identity. It binds to artifact/file identity, matching the WF loader
  contract.
- `scripts/train_walkforward_patchtst.py` now provides the HF PatchTST WF
  manifest driver. It invokes `patchtst_hf.py` per cutoff, fits the matching
  `fit_hf_patchtst_calibrator.py` per-fold calibrator with causal
  `data_end=cutoff-label_lookahead`, and writes the standard
  `kernel.walk_forward` manifest. It supports cutoff-level concurrency via
  `--jobs` and refuses partial manifests unless explicitly allowed.
- SimAdapter now recognizes `.pt` artifacts inside a walk-forward manifest as
  history-requiring scorers and probes the PatchTST sidecar for `seq_len`.
  This avoids falling back to per-bar lazy parquet loads and prevents a config
  default sequence length from undersupplying the active PatchTST fold.
- `scripts/run_wf_gate.py` now supports PatchTST acceptance inputs: it can load
  `.pt.metadata.json` sidecars for recipe validation, and manifest sanity uses
  `score_with_history()` with strictly prior panel history for history-requiring
  scorers. This keeps PatchTST on the same fail-closed WF gate path as XGB
  instead of inventing a separate acceptance shortcut.
- Critical protection: WF gate metadata for non-JSON sequence checkpoints is
  now written to the JSON sidecar, never over the `.pt` artifact itself.
- PatchTST WF smoke exposed and fixed a native calibrator crash: the HF
  calibrator had raised torch intra-op threads up to 14, conflicting with the
  repo's Apple-Silicon OMP=1 stability rule. It now defaults to
  `RENQUANT_TORCH_THREADS=1`, and the WF driver passes an explicit calibrator
  batch size. The smoke calibrator completed on 297,600 rows and the
  `WalkForwardModelLoader.calibrator_as_of()` fingerprint check passed.
- The WF PatchTST driver now has `--reuse-existing`, so if a long fold trains
  successfully but a later calibrator/manifest step fails, reruns can reuse the
  completed `.pt`/sidecar/calibrator instead of repeating training.
- Additional gate hardening after the first pilot:
  - recipe fingerprints ignore execution/window-size counters such as
    `total_steps` and `warmup_steps`; these vary by cut but do not change the
    model architecture or feature contract;
  - manifest sanity skips validation dates before the first covered manifest
    entry and records `n_skipped_pre_manifest_dates`;
  - sanity panel loading keeps labels from
    `data/alpha158_291_fundamental_dataset_rawlabel.parquet` and merges missing
    PatchTST features from `data/transformer_v4_wl200_clean.parquet`, fixing
    the real data-flow bug where sentiment/transformer columns were requested
    from the rawlabel panel.
- Two-cut pilot command:
  `.venv/bin/python scripts/train_walkforward_patchtst.py --start-date 2025-01-02 --end-date 2025-01-23 --cadence-days 21 --artifact-root walkforward_patchtst_pilot_20260524 --manifest-output backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524.json --epochs 2 --seq-len 16 --patch-length 4 --d-model 32 --n-heads 4 --n-layers 1 --device cpu --seed 44 --jobs 2 --calibrator-batch-size 512`.
- Pilot result: both cuts trained and calibrated, but this is not promotion
  evidence. Sidecar `best_val_ic` was `-0.01599` for cut `2025-01-02` and
  `-0.03424` for cut `2025-01-23`. The matching calibrator fit-window pooled
  ICs were positive (`+0.02428`, `+0.01959`), but those are fit diagnostics,
  not OOS acceptance evidence.
- WF gate sanity on the pilot used the manifest path, covered 277 OOS dates
  from `2025-01-02` to `2026-02-10`, skipped 231 pre-manifest dates, used 2
  manifest artifacts, and merged the three PatchTST sentiment features from
  the transformer panel. Verdict: `FAIL`; real IC `+0.0049`, shuffled-label IC
  `+0.0036`, and 60d time-shift placebo IC `+0.0240`. The placebo is larger
  than real IC, so the gate correctly blocks promotion.
- Latest verification:
  `tests/test_wf_gate_recipe_scope.py tests/test_wf_gate_cli_contract.py::test_wf_gate_sanity_reindexes_missing_optional_features`
  -> `15 passed`; `py_compile` and `git diff --check` passed for
  `scripts/run_wf_gate.py` and `tests/test_wf_gate_recipe_scope.py`.
- Remaining required work: run the driver for acceptance-grade folds and score
  PatchTST through the same decision-tree / benchmark-sleeve / active P&L
  acceptance lenses used for XGB and SPY.

## 2026-05-24 ntfy Alert Noise Fix

Two noisy alert paths were found and fixed:

- `live/alerts.py` now resolves the alert state path before logging
  `RENQUANT_NO_NOTIFY` suppressions, and pytest/mock alert logs write to
  per-test `pytest-*.jsonl` files instead of the production
  `logs/alerts/alert_log.jsonl`. This prevents local regression tests from
  looking like real TRADE/DECISION alert spam in the operator ledger.
- `scripts/retrain_panel.sh` no longer runs the obsolete
  `sunday_panel_sweep.py -> train_104.py` path. That path is intentionally
  refused for the current 172-feature alpha158_fund production artifact, so
  the launchd agent was producing a stale Sunday "panel ERROR" alert. The
  wrapper now no-ops when `weekly_wf_promote.sh` already ran today; otherwise
  it delegates to `weekly_wf_promote.sh` and does not emit a second wrapper
  ntfy.

Verification:

- `tests/test_alerts.py tests/test_runner_trade_ntfy.py
  tests/test_smoke_test_model.py tests/test_daily_104_shadow_notify.py`
  -> `70 passed`.
- `py_compile` passed for `live/alerts.py`, `tests/test_alerts.py`, and
  `tests/test_smoke_test_model.py`.
- Manual `bash scripts/retrain_panel.sh` on 2026-05-24 exited 0 as a no-op
  because the weekly WF log already existed, and emitted no ntfy.

## 2026-05-24 WF Sanity / Placebo Decomposition

Added `scripts/analyze_manifest_sanity_placebo.py` to make the sanity failure
reproducible instead of arguing from one scalar IC. It scores a WF manifest
through the same `run_wf_gate.py` manifest contract, then reports:

- real per-date cross-sectional IC;
- time-shift placebo IC across 5/10/20/40/60/80/120/180/252 trading days;
- raw label autocorrelation at the same shifts, so overlapping labels and
  regime persistence are not confused with model alpha;
- production regime-task labels and regime-sliced IC / placebo diagnostics.

Regression tests:

- `tests/test_manifest_sanity_placebo_analysis.py` verifies cross-sectional
  IC aggregation, the label-persistence confounder, and the markdown failure
  marker.

Verification:

- `tests/test_manifest_sanity_placebo_analysis.py` -> `3 passed`.
- `py_compile` passed for the diagnostic script and tests.

XGB 172-feature WF manifest diagnostic:

- Command:
  `.venv/bin/python scripts/analyze_manifest_sanity_placebo.py --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json --manifest backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_featspace_20260523.scopefixed.covered.json --output-dir backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_20260524_xgb`
- Report:
  `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_20260524_xgb/panel-ltr_alpha158_fund_codex_featspace_20260523-211211_staging.md`.
- Validation window: 2024-02-01 to 2026-02-10, 508 OOS dates, 71,840 rows,
  36 WF artifacts.
- Real IC `+0.0385`; 60d model-placebo IC `+0.0460`;
  60d label autocorr IC `-0.0008`; promotion evidence remains `False`.
- Regime split: BEAR `+0.2565` IC over 50 dates; BULL_CALM `+0.0152` over
  400 dates; BULL_VOLATILE `-0.0296`; CHOPPY `+0.0315`.
- Interpretation: the model has real cross-sectional signal, but most of the
  strong signal lives in BEAR, where the current decision tree usually blocks
  offensive buys. The tradeable BULL_CALM sleeve has only weak IC. This is a
  direct mechanism for "IC does not convert to APY/Sharpe": alpha is strongest
  in a branch with little/no buy capacity, while the branch that buys has weak
  rank evidence.

PatchTST pilot WF diagnostic:

- Command:
  `.venv/bin/python scripts/analyze_manifest_sanity_placebo.py --artifact backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524/2025-01-23/hf_patchtst_all_seed44_model.pt --manifest backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524.json --output-dir backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_20260524_patchtst_pilot`
- Report:
  `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_20260524_patchtst_pilot/hf_patchtst_all_seed44_model.md`.
- Validation window: 2025-01-02 to 2026-02-10, 277 OOS dates, 39,038 rows.
- Real IC `+0.0049`; 60d model-placebo IC `+0.0240`;
  60d label autocorr IC `+0.1110`; promotion evidence remains `False`.
- Regime split: BEAR `+0.0367`, BULL_CALM `+0.0030`,
  BULL_VOLATILE `-0.1164`, CHOPPY `+0.0126`.
- Interpretation: this two-cut PatchTST pilot is structurally valid as a WF
  scoring path, but not an alpha candidate. It mostly tracks persistent label
  structure and is too weak in BULL_CALM. PatchTST should remain shadow-only
  until acceptance-grade folds show regime-specific trade-domain edge.

Next implication:

- Fixing APY/Sharpe should not start with QP. The immediate target is alpha
  admission by regime: BULL_CALM needs a stronger threshold / monotonicity
  contract, BEAR needs a separate defensive/short/hedged thesis before any
  offensive signal is allowed, and BULL_VOLATILE should be blocked or routed
  away for this scorer until it has positive regime evidence.

## 2026-05-24 Regime Model Admission Runtime Gate

Problem:

- QP/selection must not be able to transform weak or unsupported model output
  into trades. The model must first prove that the current regime is an
  admissible buy regime. This is especially important after the sanity
  decomposition showed XGB's strongest IC in BEAR while BULL_CALM was weak.

Fix:

- Added `RegimeModelAdmissionTask` to `PanelScoringJob` after global
  calibration and before `VetoWeakBuysTask`, realized-vol fallback, Kelly, and
  QP-facing quality floors.
- The task reads scorer metadata
  `metadata.wf_gate_metadata.trade_monotonicity.regimes` for the current
  `ctx.regime`.
- If the current regime is missing, ineligible, or failed, all buy candidates
  are cleared before QP can see them. Each ticker is stamped in
  `ctx._blocked_by_ticker` with reasons such as
  `regime_admission:no_trade_stats:BULL_CALM` or
  `regime_admission:ineligible:BULL_VOLATILE`.
- The task can also require future
  `metadata.wf_gate_metadata.sanity_regime_ic` evidence via
  `ranking.panel_scoring.regime_admission.require_sanity_regime_ic=true`; this
  is wired but not yet stamped by `run_wf_gate.py`.

Verification:

- `tests/test_regime_model_admission.py` covers pass, missing-regime block,
  ineligible-regime block, optional sanity-IC requirement, and experiment
  disable.
- Targeted panel scoring suite:
  `tests/test_panel_scoring_job.py tests/test_veto_weak_buys_p0_fix.py tests/test_regime_model_admission.py`
  -> `62 passed`.

Next implication:

- The next gate hardening should stamp `sanity_regime_ic` directly from
  `run_wf_gate.py`, then enable `require_sanity_regime_ic` for production.
  That will make weak BULL_CALM IC a runtime blocker instead of a diagnostic
  note.

Update:

- `scripts/run_wf_gate.py::run_sanity_battery()` now stamps
  `sanity_regime_ic` into the returned WF gate metadata. It uses the same
  production regime task chain as the diagnostic script, reports per-regime
  mean IC and 60d placebo/label-autocorr evidence, and marks regimes eligible
  when they have at least 30 OOS dates.
- Default regime sanity pass threshold is `mean_ic >= 0.02` and 60d placebo
  not larger than the regime's real IC. This is intentionally a metadata
  contract first; production can hard-require it via
  `ranking.panel_scoring.regime_admission.require_sanity_regime_ic=true` once
  the next strict WF artifact is stamped.
- Regression:
  `tests/test_wf_gate_regime_sanity_metadata.py` proves
  `run_sanity_battery()` emits `sanity_regime_ic`.

Strict production requirement:

- `kernel.preflight._check_regime_layered_ic()` now hard-fails full/buy runs
  when `sanity_regime_ic` is absent or failed; sell-only remains soft-pass so
  risk exits are not blocked.
- `RegimeModelAdmissionTask` now requires `sanity_regime_ic` by default as a
  runtime backstop if preflight is skipped. Experiments can opt out with
  `ranking.panel_scoring.regime_admission.enabled=false`.
- Current active production artifact check:
  `P-REGIME-IC hard False regime sanity IC evidence absent from WF metadata`.
  That is expected until a new strict WF gate run stamps the artifact.
- Regression:
  `tests/test_preflight_regime_sanity.py` covers missing/failed/passed sanity
  metadata and sell-only behavior.

## 2026-05-24 Strict WF Rerun After Admission-Cycle Fix

Root-cause fix before rerun:

- Strict WF was self-blocking to zero trades because
  `RegimeModelAdmissionTask` required WF/sanity metadata that the WF gate itself
  was supposed to produce. The production-semantic WF config builder now
  disables runtime `ranking.panel_scoring.regime_admission` only inside WF
  evaluation. Live/preflight still fail closed on missing or failed evidence.
- Regression:
  `tests/test_wf_config_parity.py tests/test_wf_gate_cli_contract.py
  tests/test_regime_model_admission.py tests/test_preflight_regime_sanity.py`
  passed after the fix.

Strict WF rerun:

- Command trace:
  `artifacts/diagnostics/wf_trade_traces/strict_prod_semantic_20260524_admissionfix`.
- Validation scope: walk-forward manifest recipe matched candidate recipe;
  config parity PASS; QP contract OK; trade ledger contract OK; trade score
  monotonicity passed in active regime.
- Verdict: FAIL.
- Annual-net acceptance metrics:
  mean Sharpe `+0.133`, mean APY `+1.42%`; SPY mean Sharpe `+1.081`,
  SPY APY `+16.94%`; delta Sharpe `-0.948`, delta APY `-15.52%`.
  Beat SPY Sharpe `0/3`; beat SPY APY `0/3`.
- Per-cut annual-net:
  2024-01-02..2024-12-31 Sharpe `+0.695`, APY `+3.42%`, SPY Sharpe `+1.778`;
  2024-07-01..2025-06-30 Sharpe `+0.669`, APY `+3.62%`, SPY Sharpe `+0.715`;
  2025-04-01..2026-03-28 Sharpe `-0.966`, APY `-2.78%`, SPY Sharpe `+0.749`.
- Regime benchmark lag:
  `HIGH_CALM` two cuts, mean Sharpe `-0.135` vs SPY `+1.264`;
  `LOW_SPIKED` one cut, Sharpe `+0.669` vs SPY `+0.715`.
- Sanity battery: FAIL. Real IC `+0.0385`, shuffled IC `+0.0024`, placebo IC
  `+0.0460`. The placebo being stronger than real IC means the reported IC is
  not acceptable alpha evidence; it is likely dominated by time/regime
  persistence or label autocorrelation.
- Tax/metric interpretation:
  event-level sim numbers look much better (`+6.8%/+7.2%` APY in two cuts),
  but acceptance correctly uses annual-net tax. Tax is a major drag on the
  positive cuts, while the 2025-04..2026-03 cut is negative before tax because
  stop-loss exits dominate.
- Trade anatomy:
  all closed entries were `QP_BUY` in `BULL_CALM`; stop-loss bucket in the
  failing 2025-04..2026-03 cut was `8` trades, gross `-$5,069`, win rate `0%`.
  Winners still exist (`trailing_stop` and QP sells positive), so this is not
  a pure tax bug; it is an entry/exit/regime conversion problem.

Code hardening after rerun:

- `scripts/run_wf_gate.py` now stamps `benchmark_by_dominant_regime`,
  `regime_benchmark_failures`, and `performance_tax_basis_counts` into artifact
  metadata. Previous code calculated these but omitted them from the metadata
  payload, violating regime-first auditability.
- `scripts/run_wf_gate.py` also now stamps `sanity_regime_ic` into artifact
  metadata. Previous code returned it from `run_sanity_battery()` but failed to
  copy it into `wf_gate_metadata`, causing preflight to see absent regime
  sanity evidence even after the diagnostic ran.
- `scripts/daily_104.sh` no longer sends phone `SHADOW-FAIL` alerts for
  expected shadow buy-side preflight blocks. True shadow crashes/timeouts still
  alert.
- Regression suites:
  `tests/test_daily_104_shadow_notify.py tests/test_smoke_test_model.py
  tests/test_runner_trade_ntfy.py tests/test_alerts.py`,
  `tests/test_qp_admission_gate.py tests/test_joint_qp_task.py
  tests/test_qp_contracts.py tests/test_benchmark_sleeve.py
  tests/test_qp_cvxpy_fallback.py`, and
  `tests/test_wf_gate_cli_contract.py tests/test_wf_trade_forensics.py
  tests/test_trade_monotonicity_gate.py`.

Operational conclusion:

- Do not promote this artifact.
- Do not call the model trustworthy for live buys until the placebo IC and
  benchmark-relative WF failures are fixed.
- Next work should target label/split/sanity causality and BULL_CALM
  entry/stop-loss behavior, not just QP sizing.

## Mainline Queue

1. Convert the sanity decomposition into an alpha-admission fix: regime-specific
   monotonicity/IC gates must decide whether a model can buy in the current
   regime before QP sees candidates. XGB currently has strong BEAR IC but weak
   BULL_CALM IC; PatchTST pilot is too weak for promotion. Check whether BEAR
   alpha belongs in a defensive/short/hedged sleeve, not an offensive long-only
   buy path.
2. Continue benchmark/annual-net work from the benchmark-sleeve A/B:
   cash drag is confirmed, CHOPPY/BEAR sleeve de-risking is rejected as a
   default benchmark-core design, and `core100` restores risk-adjusted
   participation. Remaining blocker is alpha budget/marginal contribution:
   QP alpha must beat the displaced benchmark sleeve after tax and turnover
   before it deserves capital.
3. Re-run strict WF only after the model/sanity issue has a theory-backed fix.
   Compare event-level, annual-net, SPY-relative, regime cuts, score
   monotonicity, stop-loss bucket, and QP/TopUp source buckets.
4. Evaluate stop-loss changes only through paired A/B acceptance. The current
   BULL_CALM entry-regime stop-anchor A/B (`max_entry_current`) is rejected.
   Other candidates remain non-BULL volatility-aware stops and earlier
   panel/mu soft exits for positions whose model thesis deteriorates before
   hard stop.
5. Fold PatchTST into the same mainline acceptance path before quoting
   PatchTST portfolio APY/Sharpe as OOS. Static PatchTST full-window sims are
   style diagnostics only. Completed infrastructure: `.pt` scorer fingerprint
   support, sidecar metadata instead of checkpoint mutation, rolling
   `patchtst_hf.py --train-cutoff/--data-end`, and file-identity HF calibrator
   fingerprinting, HF PatchTST WF manifest driver, and causal per-fold
   calibrator orchestration. Remaining: run acceptance folds and compare with
   the same decision-tree / benchmark sleeve / active P&L lens used for XGB and
   SPY.
6. Continue after-tax/no-trade-region and stop-loss research per regime, using
   literature-backed hypotheses and paired A/B sims.
7. Fix remaining audit findings before promotion: run an actual LEAN trace
   smoke after the new DB wiring. The WF `effective_train_cutoff_date`
   double-embargo bug, SEC fundamentals point-in-time filed-date bug, LEAN/QP
   cash-capped target parity bug, universe metadata fail-closed bug,
   calibrator metric-scope bug, selection/rotation/QP sector-correlation
   metadata fail-closed semantics, QP global status reason stamping, candidate
   reason-gap contract, and exact sim/live/LEAN universe-rejection reason
   preservation plus LEAN sidecar trace wiring are fixed. Correlation artifacts
   without `as_of_date` now require an explicit legacy override, while
   sell-only risk exits remain soft-passed.

## Known Failure Modes To Keep Front And Center

- Signal IC does not automatically become alpha. Trade-domain monotonicity must
  be measured after the full decision tree.
- Placebo IC must be compared against same-row aligned real IC. The
  2026-05-24 audit found a WF gate reporting bug where full real IC used 508
  dates while 60d placebo used only 448 shift-evaluable dates. Corrected
  numbers for the active XGB manifest are aligned real `+0.0548` vs placebo
  `+0.0460` (ratio `0.84`), so the original `placebo > real` headline was a
  sample-mismatch bug but the model still fails sanity. BULL_CALM remains
  placebo-dominated: aligned real `+0.0323` vs placebo `+0.0312`.
- Runtime metadata must be read from the same layer where artifact promotion
  stamps it. A 2026-05-24 follow-up found `PanelScorer.load()` left
  `metadata.wf_gate_metadata` nested under `scorer.metadata["metadata"]`, while
  `RegimeModelAdmissionTask` read `scorer.metadata["wf_gate_metadata"]`.
  Flattening is now enforced by loader tests.
- Per-regime placebo admission must use the same `0.5 x aligned_real` ratio as
  the top-level WF gate. The earlier runtime rule only blocked placebo above
  `1.0 x aligned_real`, which would have let BULL_CALM's `0.97` ratio through.
- Panel scoring is the alpha surface for 104. If the scorer artifact, config
  consistency check, feature matrix, or per-ticker panel score is missing, the
  buy/QP path must fail closed and write `blocked_by`; it must not continue on
  Phase-2 per-ticker tournament scores. A 2026-05-24 repair added runtime
  fail-closed guards and regression tests for scorer load failure, preloaded
  scorer config mismatch, missing matrix, missing per-ticker panel score, and
  scorer runtime exceptions.
- Regime-router scoring must be contract-strict. A configured route like
  `BEAR -> hf_patchtst` must not fall back to the default scorer when the
  routed scorer is missing, and missing routed-scorer feature columns must not
  be zero-filled. Both cases are now hard errors covered by regression tests;
  `ApplyScoresTask` converts runtime scorer errors into buy/QP fail-closed
  state on the shared sim/live/LEAN path.
- Sim, live runner, and LEAN must share panel frame preparation through
  `adapters.panel_runtime.prepare_panel_runtime_frames`. The direct adapter
  calls were consolidated on 2026-05-24 so tuple arity, benchmark injection,
  and sector-map filtering cannot drift independently across validation and
  live trading.
- Sim, live runner, and LEAN must also share decision-trace row construction
  through `kernel.decision_trace`. The 2026-05-24 refactor centralized model
  type extraction, full candidate snapshots, QP delta/target/status maps, and
  `ticker_daily_state` blocked-by precedence (`universe:*`, `broker_pending`,
  `held_no_new_buy`, `no_model_signal`, `not_selected`) so execution surfaces
  explain decisions with the same schema and semantics.
- Execution-layer migration is not complete. The `ExecutionPipeline` exists
  but adapters still run legacy execution monoliths for tax lots, trade-log
  attribution, and live broker state. First safe convergence step completed:
  `kernel.pipeline.task_execution.is_full_liquidate_signal` is now the shared
  partial/full sell predicate, and `dedupe_exit_signals` is now the shared
  duplicate-exit resolver used by SimAdapter, RunnerAdapter, LeanAdapter, and
  ExecutionPipeline. This fixes the same-bar bug where an earlier partial trim
  could swallow a later full exit expressed as `quantity >= held`.
- BUY trade-event rows are now built through `kernel.trade_events` across
  SimAdapter, RunnerAdapter, and LeanAdapter. This keeps shares/price/invest,
  score snapshots, decision inputs, attribution version, regime, and
  confidence aligned for post-run audit.
- SELL trade-event rows are now built through `kernel.trade_events` for
  RunnerAdapter and LeanAdapter. This preserves source_job/source_task,
  tax/net P&L fields, applied exit params, score snapshots, and decision
  inputs with one helper instead of two adapter-specific implementations.
  Sim sell rows still contain extra tax-lot disposal semantics and should be
  migrated only after the lot-attribution helper is shared too.
- LEAN now carries `last_sell_pls` into `InferenceContext` and stamps realized
  P/L on full exits, matching sim/live cost-aware wash-sale semantics. Without
  this, LEAN treated recent-sale P/L as unknown and binary-blocked gain-sale
  re-entries that sim/live would allow.
- LEAN now attaches `ctx._db` before the pipeline runs, matching sim/live.
  Pipeline tasks that need DB context, such as score-distribution gates and
  thesis-symmetric rotation lookup, no longer silently no-op in LEAN while
  working in sim/live.
- QP must size/rebalance qualified alpha; it must not turn weak candidates into
  trades.
- Bull markets punish low exposure. Low beta can look safe while failing to
  participate.
- Event-level tax stress and annual-net economic tax are different metrics.
  Current headline should use reporting-only cash plus annual-net tax estimate.
- Stop-loss exits have been the main gross loss bucket in multiple traces.
  Do not change thresholds blindly; split by regime, score decile, hold age,
  volatility, and drawdown path first.
- PatchTST evidence is positive but weaker and not yet strict-WF accepted.
  Treat it as shadow/router candidate, not replacement primary.
- Noisy ntfy/reopen-cancel alerts are partly fixed. Wrapper success duplicate
  alerts are fixed; remaining watch item is shadow/reopen-cancel alert policy.

## Stop Conditions

Stop and fix before reporting performance if any of these happen:

- WF config loses `tax.cash_debit_mode=reporting_only`.
- A calibrator/scorer fingerprint mismatch is detected.
- Sector metadata is missing for a buyable ticker.
- A buy/full path silently falls back to raw score or a weaker score.
- Sim/live/LEAN construct panel inference frames through different code paths.
- Trade logs lack `blocked_by`, model type, sector, score snapshot, QP
  target/delta/status, or sell P/L/tax/net for emitted orders.
- A metric is not labeled as event-level, annual-net, short-window style, or
  acceptance-grade WF.

## Companion Docs

- Detailed stream separation:
  `doc/research/2026-05-23-current-state-ledger.md`.
- PatchTST/XGB style handoff:
  `doc/research/2026-05-23-pending-research-and-patchtst-xgb-style.md`.
- Strict WF rerun claim needing artifact reconciliation:
  `doc/research/2026-05-23-strict-wf-xgb-patchtst-rerun.md`.
- Decision-tree and sim audit:
  `doc/research/2026-05-23-decision-tree-and-sim-audit.md`.
- Decision-tree contract:
  `doc/research/2026-05-23-decision-tree-contract.md`.
- HIFO-aligned WF trade forensics:
  `doc/research/2026-05-23-wf-trade-forensics.md`.
- Placebo IC alignment/root-cause debug:
  `doc/research/2026-05-24-placebo-ic-debug.md`.
