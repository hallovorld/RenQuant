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
  against the old manifest because the feature-source contract changed. Old
  walk-forward artifacts are no longer recipe-comparable; a new WF manifest
  must be regenerated under the new feature contract before quoting WF
  Sharpe/APY for this candidate.

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

## Mainline Queue

1. Regenerate the 172-feature walk-forward manifest under the current
   feature-space contract before running acceptance WF on
   `codex_featspace_20260523-211211`. Old manifest evidence is invalid for
   this candidate.
2. Diagnose the manifest-OOS sanity failure: real IC `+0.0269` is weak and the
   time-shift placebo IC `+0.0282` is slightly higher. Check splitter/label
   horizon, feature timestamping, regime persistence, slow beta/momentum
   persistence, and calibrator scope before trusting any reported IC.
3. Diagnose benchmark/annual-net failure: event Sharpe is positive, but
   annual-net and SPY-relative metrics fail. Split by regime, exposure, tax,
   stop-loss bucket, and QP/top-up source.
4. Re-run strict WF only after the model/sanity issue has a theory-backed fix.
   Compare event-level, annual-net, SPY-relative, regime cuts, score
   monotonicity, stop-loss bucket, and QP/TopUp source buckets.
5. Evaluate stop-loss changes only through paired A/B acceptance. The current
   BULL_CALM entry-regime stop-anchor A/B (`max_entry_current`) is rejected.
   Other candidates remain non-BULL volatility-aware stops and earlier
   panel/mu soft exits for positions whose model thesis deteriorates before
   hard stop.
6. Build PatchTST true WF manifest before quoting PatchTST portfolio APY/Sharpe
   as OOS. Static PatchTST full-window sims are style diagnostics only.
7. Continue after-tax/no-trade-region and stop-loss research per regime, using
   literature-backed hypotheses and paired A/B sims.
8. Fix remaining audit findings before promotion: calibrator metric scope
   labels for in-sample versus OOF IC, point-in-time SEC filed-date handling,
   and per-ticker trace stamping for global panel/QP failures. The WF
   `effective_train_cutoff_date` double-embargo bug, LEAN/QP cash-capped target
   parity bug, universe metadata fail-closed bug, calibrator metric-scope bug,
   and correlation metadata fail-closed semantics are fixed. Correlation
   artifacts without `as_of_date` now require an explicit legacy override,
   while sell-only risk exits remain soft-passed.

## Known Failure Modes To Keep Front And Center

- Signal IC does not automatically become alpha. Trade-domain monotonicity must
  be measured after the full decision tree.
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
- Noisy ntfy/reopen-cancel alerts are partly fixed but wrapper success alerts
  may still duplicate runner alerts.

## Stop Conditions

Stop and fix before reporting performance if any of these happen:

- WF config loses `tax.cash_debit_mode=reporting_only`.
- A calibrator/scorer fingerprint mismatch is detected.
- Sector metadata is missing for a buyable ticker.
- A buy/full path silently falls back to raw score or a weaker score.
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
