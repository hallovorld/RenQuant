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
- 2026-05-24 correction to the current experiment ledger: any experiment that
  is not regime-stratified is diagnostic/smoke only, not acceptance evidence.
  This does **not** mean the model cannot be regime-based; it means promotion
  evidence must report regime first and pooled metrics second.
- The BULL_CALM confidence-veto diagnostic
  `erfloor_bullcalm040_confveto060_20260524` is rejected. It reduced some
  transition damage but worsened all three WF cuts versus the matching
  ER-floor baseline and created long no-trade stretches. Do not ship.
- The ER-floor WF trace did use the manifest per-fold raw-return calibrator,
  not the stale static sim calibrator. Example checked: 2024-01-02 MO
  `panel_score=0.25059849` produced `mu=0.04877512`, exactly matching
  `WalkForwardModelLoader.calibrator_as_of("2024-01-02")`; the stale static
  calibrator would have produced `0.02547308`.
- New safety invariant: when `ranking.kelly_sizing.use_calibrator_mu=true`,
  preflight and runtime buy/QP paths require calibrator metadata
  `expected_return_label_contract="raw_return_units_required"`. Missing or
  non-return contracts fail closed for full/buy paths while sell-only remains
  armed.
- Remaining main issue after that audit: within BULL_CALM, traded scores are
  still not monotonic with realized trade P/L. In the rejected confidence-veto
  trace, BULL_CALM entry-rank Spearman versus net P/L was about `-0.30`; the
  raw panel score and μ were also negative. This is an alpha-conversion/scoring
  semantics problem, not merely a regime gate problem.
- Follow-up on the accepted ER-floor baseline trace
  `erfloor_bullcalm040_20260524`: after adding forward-return alignment to
  the forensic tool, BULL_CALM entry events show score/μ are negative versus
  20d/60d SPY-relative forward returns but positive at 120d:
  `entry_mu` Spearman is about `-0.217` at 20d, `-0.123` at 60d, and `+0.227`
  at 120d. Mean hold is far shorter than 120d. This points to a horizon
  mismatch: the model may be selecting longer-horizon relative winners while
  the current stop/QP/rotation path realizes them on a shorter horizon.
- New diagnostic hook: `scripts/analyze_wf_trade_forensics.py --ohlcv-root`
  now reports entry-score versus forward excess-return alignment. Use this
  before declaring a decision-tree change solved alpha conversion.
- New A/B hook: QP alpha admission now supports
  `min_expected_return_over_sigma[_by_regime]` and top-up aliases. This is
  not enabled in production; it is for testing risk-adjusted admission instead
  of raw μ floors.
- 2026-05-24 exit-path audit added to
  `scripts/analyze_wf_trade_forensics.py`: after each closed sell it applies
  an AFML-style 60-business-day triple-barrier label using realized daily
  volatility known at the exit bar. In the accepted ER-floor diagnostic trace
  `erfloor_bullcalm040_20260524`, BULL_CALM exits are mostly not path-correct:
  overall barrier-correct exit rate is `43.3%` and false-positive rate is
  `56.7%`. `stop_loss` is worse: `14` exits, `0%` win rate, net `-$7.78k`,
  only `35.7%` barrier-correct, `64.3%` false-positive, with mean post-exit
  excess returns versus SPY of `+4.0%` at 20d, `+10.8%` at 60d, and `+21.7%`
  at 120d. Examples include PLTR/NVDA/INTC/COHR stop exits that later
  recovered strongly. This pins the current alpha-conversion thesis:
  the model may be selecting longer-horizon BULL_CALM winners, while current
  stop/QP/rotation exits realize them on a shorter/noisier path.
- Do not respond by globally disabling or widening stops. The same audit shows
  some stop-loss exits were correct, so the next fix must be regime- and
  path-conditional: either add short-horizon confirmation at entry, train/use a
  meta-label exit veto, or align BULL_CALM exit barriers with the model's
  target horizon. Any rule change needs WF evidence reported per regime first.
- Horizon-contract repair implemented 2026-05-24: production/golden 104 now
  aligns `rotation.target_horizon_days`,
  `rotation.joint_actions.qp_mu_horizon_days`, and
  `panel_ltr.lookahead_days` at `60` trading days. The global panel calibrator
  now accepts an explicit `horizon_days`, pipeline scoring stamps both
  `expected_return_horizon_days` and `mu_horizon_days`, and BULL_CALM
  optimizer-driven QP soft sells wait at least the 60d panel thesis horizon.
  Hard risk exits remain armed. This is a structural alpha-conversion fix, not
  final WF acceptance evidence.
- Post-repair diagnostic WF
  `horizon60_erfloor_bullcalm040_diag_20260524-190959` still failed. It was
  explicitly diagnostic-only (`--skip-sanity --skip-config-parity`, ER-floor
  override preserved) and must not be used for promotion. Mean annual-net
  Sharpe was `+0.459` vs SPY `+1.081`; cut Sharpes were `+0.782`, `+0.696`,
  and `-0.101`. Trade count dropped from `67` to `35`, median hold rose from
  `31d` to `60d`, and QP sells/closes became less churny, but BULL_CALM score
  monotonicity remained negative (`entry_mu` Spearman vs net P/L `-0.268`).
  Therefore the next main issue is not only premature exit/QP churn; BULL_CALM
  entry-score ordering itself is still wrong for realized active P/L.
- Decision-trace horizon observability is now part of the contract:
  `candidate_scores` and `ticker_daily_state` persist
  `expected_return_horizon_days` and `mu_horizon_days`. Future audits should
  never report a `mu`/expected-return number without its horizon.
- Meta-label exit veto is not promoted. Current production config has
  `ranking.meta_label.enabled=false`, and the old 2026-05-11 artifact has only
  `146` events with CV AUC about `0.554`. New preflight hardening now requires
  a usable meta-label artifact whenever that veto is enabled for buy/full:
  valid kind, feature columns, booster payload, threshold, CV AUC, training
  sample count, class balance, and label horizon. Missing/corrupt/weak
  artifacts fail closed for buy/full while sell-only remains armed. This
  prevents a future path-veto experiment from silently degrading back to the
  un-vetoed stop path.
- Meta-label training contamination found and fixed. Old snapshot labels marked
  `any_trigger=1` for model/QP exits, but runtime `MetaLabelVetoTask` only
  applies to path-rule exits (`stop_loss`, `trailing_stop`,
  `single_day_loss`, `max_hold`). The old W1 artifact therefore trained on
  `146` labeled rows, but only `37` were path-rule rows; `109` rows were a
  different decision problem. `SnapshotHoldingsTask`, `label_snapshots`, and
  `_meta_label_train.py` now filter to the same path-rule trigger surface as
  runtime inference. A diagnostic W1/W2/W3 path-rule-only retrain had only
  `108` events and weak CV AUC `0.491 ± 0.104`; do not promote meta-label until
  more clean path-rule samples exist and per-regime WF A/B passes.

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
- Latest horizon-contract repair:
  - `GlobalPanelCalibration.expected_return()` and its vectorized helper now
    support an explicit `horizon_days` argument, scaling from the artifact's
    native ER lookahead when needed.
  - `ApplyGlobalCalibrationTask` computes rotation expected return and QP
    Kelly/optimizer `mu` at separate explicit horizons and stamps both fields
    into candidate/holding decision records.
  - Production/golden 104 require the panel label horizon, rotation horizon,
    and QP μ horizon to match at 60 trading days; BULL_CALM QP soft sells now
    use a 60d thesis-age guard while hard risk exits remain unchanged.
  - Targeted tests passed:
    `tests/test_wf_config_parity.py tests/test_order_attribution_contract.py
    tests/test_persistence.py::TestTrades tests/test_qp_grinold_kahn_transform.py
    tests/test_joint_qp_task.py tests/test_phase3_mu_sigma_wiring.py
    tests/test_global_calibrator.py tests/test_horizon_contracts.py`
    (`115 passed`).
- Latest decision-trace horizon observability:
  - `candidate_scores` and `ticker_daily_state` now persist
    `expected_return_horizon_days` and `mu_horizon_days` for candidates and
    holdings.
  - `build_ticker_daily_state_rows()` carries those fields through the shared
    sim/live/LEAN trace builder.
  - Targeted tests passed:
    `tests/test_persistence.py::TestCandidateScores
    tests/test_persistence.py::TestTrades tests/test_ticker_daily_state.py
    tests/test_decision_trace_horizon.py tests/test_order_attribution_contract.py
    tests/test_phase3_mu_sigma_wiring.py tests/test_global_calibrator.py
    tests/test_horizon_contracts.py tests/test_wf_config_parity.py`
    (`72 passed`).
- Post-prefilter WF validation completed and still failed:
  - Annual-net Sharpe by cut: `+1.037`, `+0.191`, `-0.310`.
  - Mean Sharpe `+0.306`; SPY mean Sharpe `+1.081`; delta `-0.775`.
  - Beat SPY Sharpe/APY: `0/3` and `0/3`.
  - Trade ledger contract passed, but BULL_CALM score monotonicity failed.
  - Manifest sanity remained weak: real IC `+0.0269`, shuffled `-0.0019`,
    placebo `+0.0282` versus required `< +0.0135`.
  - Conclusion: QP solver prefilter is necessary architecture hardening, but
    it is not the alpha-conversion fix.
- QP admission now supports an optional calibrated expected-return floor:
  `min_expected_return` / `min_expected_return_by_regime` for new buys and
  `topup_min_expected_return` for held top-ups. It checks
  `expected_return` first and falls back to `mu` only when the explicit field is
  missing. Missing or below-floor values stamp
  `qp_admission_expected_return`.
  - This is not enabled in production yet. It is an A/B hook for the observed
    failure where BULL_CALM candidates clear rank/panel floors but do not carry
    enough expected excess return after turnover, stops, and tax.
  - Targeted tests passed:
    `tests/test_qp_admission_gate.py tests/test_qp_integration.py
    tests/test_joint_qp_task.py` (`74 passed`).
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
- Do not confuse the 13-day PatchTST style win with acceptance. The
  `2026-05-06` to `2026-05-22` run had `+3.21%` return and Sharpe `+6.61`,
  but it had only 7 buys, 0 sells, no tax realization, and all P/L was
  mark-to-market open exposure. It is useful for style only.
- One older shadow path really was contaminated: `artifacts/shadow/
  panel-rank-calibration.shadow.json` is byte-identical to the production XGB
  calibrator (`sha256:1baaf489cae3175fa03a13336d35b4875da57bc4ea2186d8cd1bcd2c02ae9990`).
  Any PatchTST result using that calibrator is invalid because PatchTST raw
  scores were interpreted on an XGB score scale. Current shadow configs must
  use the PatchTST-specific calibrator and strict scorer identity checks.

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
- 2026-05-24 follow-up: the earlier "PatchTST looked great" stream is explained
  by evidence mixing, not by a single clean acceptance number. The `+3.21%`,
  Sharpe `+6.61` result was a 13-trading-day, zero-sell style sim. The older
  2024-07 to 2026-02 PatchTST APY `+1.49%`, Sharpe `+0.23` was a static
  diagnostic, not a point-in-time PatchTST WF. The old generic shadow
  calibrator is byte-identical to the prod XGB calibrator, so any run using
  `artifacts/shadow/panel-rank-calibration.shadow.json` for PatchTST score
  interpretation is invalid. Current shadow config uses the PatchTST-specific
  calibrator with strict scorer match.
- `scripts/eval_xgb_5cut_5seed.py` now fills the missing same-window XGB arm
  for the PatchTST architecture screen. It writes the same
  `cut,seed,regime,ic` aggregate schema as the HF PatchTST drivers, so
  `compare_arch_5cut_5seed.py` can compare PatchTST variants against XGB
  instead of only comparing PatchTST against itself. Targeted smoke tests:
  `tests/test_eval_drivers_smoke.py` -> `26 passed`.
- Same-window comparator result:
  - HF cross-stock PatchTST mean min-regime IC `+0.0507`.
  - HF FiLM PatchTST `+0.0477`.
  - HF baseline PatchTST `+0.0467`.
  - XGB same-window baseline `+0.0283` with high variance (`std=0.2191`).
  - XGB dominates cut1/cut3/cut4 but collapses in cut2_fed (`-0.3752` mean
    min-regime IC). PatchTST is weaker in several stress cuts but much less
    broken in cut2_fed. This supports regime/router research, not PatchTST
    promotion as a single primary.
- Expected-return QP-admission A/B (`BULL_CALM` new/top-up floor `0.04`) was
  run as diagnostic-only with config parity intentionally skipped. Verdict:
  FAIL. Annual-net cut metrics were:
  - 2024-01-02 to 2024-12-31: APY `+6.08%`, Sharpe `+0.768`, SPY Sharpe
    `+1.778`.
  - 2024-07-01 to 2025-06-30: APY `+5.12%`, Sharpe `+0.692`, SPY Sharpe
    `+0.715`.
  - 2025-04-01 to 2026-03-28: APY `+0.47%`, Sharpe `+0.126`, SPY Sharpe
    `+0.749`.
  Mean Sharpe `+0.529`, all 3 cuts positive, but `0/3` beat SPY Sharpe/APY.
  Trade ledger contract passed; BULL_CALM score monotonicity still failed;
  sanity failed with real IC `+0.0385` and placebo IC `+0.0460`. Conclusion:
  expected-return floor is useful defensive wiring but not the APY/Sharpe
  root-cause fix.

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
- Calibrator/scorer binding must use scorer-file identity, not shared config
  identity. `fit_panel_calibrator.py` now stamps `scorer_artifact_fingerprint`
  from artifact/model/file hashes only; `config_fingerprint` is intentionally
  ignored because it can be common across multiple scorer artifacts.
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
- SELL trade-event rows are now built through `kernel.trade_events` across
  SimAdapter, RunnerAdapter, and LeanAdapter. This preserves
  source_job/source_task, tax/net P&L fields, applied exit params, score
  snapshots, and decision inputs with one helper. Sim still computes
  lot-disposal/accounting fields locally before calling the shared builder,
  then records tax cash debit mode/amount on the same audit row.
- LEAN now carries `last_sell_pls` into `InferenceContext` and stamps realized
  P/L on full exits, matching sim/live cost-aware wash-sale semantics. Without
  this, LEAN treated recent-sale P/L as unknown and binary-blocked gain-sale
  re-entries that sim/live would allow.
- LEAN now attaches `ctx._db` before the pipeline runs, matching sim/live.
  Pipeline tasks that need DB context, such as score-distribution gates and
  thesis-symmetric rotation lookup, no longer silently no-op in LEAN while
  working in sim/live.
- LEAN now synchronizes holding share counts from `Portfolio`, maintains
  `HoldingState.lots` on buys/top-ups, and computes sell tax/P&L from the
  same FIFO/HIFO disposed-basis primitive used by sim. This fixes the prior
  avg-cost/pro-rated `UnrealizedProfit` path that could misstate partial-sell
  tax, `gross_pnl`, `pnl_pct`, and wash-sale P/L.
- Training feature construction now canonicalizes OHLCV inputs before
  indicators, SPY-relative alignment, and forward labels: required columns
  only, datetime index coercion, sort, duplicate-date removal with latest row
  kept. Parallel feature builds also get per-ticker frame copies so shared SPY
  data cannot create reindex failures or ambiguous labels under concurrency.
- Added an executable adapter-context contract covering actual
  `SimAdapter.make_context`, `RunnerAdapter.make_context`, and
  `LeanAdapter.make_context`. All three must now expose the critical shared
  pipeline surface: run id, OHLCV, SPY returns, models, holdings, prices,
  cash/NAV, `last_sell_dates`, `last_sell_pls`, `last_stop_exit_dates`, DB
  handle when configured, and regime state counters.
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

## 2026-05-24 WF Gate And Repo Hygiene Update

- Weekly WF promotion exposed an evidence-path bug: `--derive-config-from-prod`
  could inherit a stale base manifest whose recipe fingerprint did not match
  the staged scorer, even when a same-recipe manifest with broader coverage
  already existed. `scripts/run_wf_gate.py` now scans the strategy sim
  manifest directory, picks the recipe-compatible manifest with the widest
  retrain coverage, and fails closed on the preferred manifest only when no
  compatible manifest exists.
- Current concrete case: the stale `172_sentiment` base manifest mismatched the
  weekly staged scorer, while
  `artifacts/sim/walkforward_manifest_172_featspace_20260523.scopefixed.covered.json`
  matches the staged feature contract and checks 40 manifest rows.
- Repo hygiene is now a mainline workstream. `scripts/audit_repo_hygiene.py`
  provides a read-only dirty-tree classifier with an inventory-only policy:
  no delete, no move, archive requires review. Current snapshot: 368 dirty
  entries, dominated by experiment/diagnostic artifacts, per-ticker model
  metadata, production/sim artifacts, strategy configs, and scratch training
  scripts.
- `.gitignore` now blocks obvious local runtime scratch stores, local DBs, and
  backup snapshots from adding future noise. It intentionally does not hide
  production artifacts or strategy configs; those must remain visible until
  reviewed.

## 2026-05-24 Model Artifact / Compute Discipline

- Latest PatchTST checkpoints are saved locally, not just logs. The strict
  shadow artifact is
  `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`
  with a `.metadata.json` sidecar. The three 5-cut/5-seed architecture screens
  have 75 `.pt` model files, 75 summaries, and 75 validation-prediction
  parquet files across:
  `artifacts/hf_trainer_5cut_5seed_pt07_clean`,
  `artifacts/hf_film_5cut_5seed_pt07_clean`, and
  `artifacts/hf_cross_stock_5cut_5seed_pt07`.
- The PatchTST WF pilot is also saved under
  `backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524`
  with per-cut `.pt` and `.pt.metadata.json` files plus the manifest
  `backtesting/renquant_104/artifacts/walkforward_patchtst_pilot_20260524.json`.
- The expensive PatchTST WF driver has `--reuse-existing`; reruns should use it
  unless intentionally changing the recipe. It reuses completed checkpoint,
  sidecar, and calibrator outputs after downstream manifest/gate failures.
- Found and fixed one compute-discipline gap in the new XGB comparator:
  `scripts/eval_xgb_5cut_5seed.py` originally saved summaries and validation
  predictions but not the trained XGBoost booster. Future runs now save
  `xgb_*_model.json` and stamp `model_saved/model_path` into each summary.
  The already-run XGB comparator can reproduce its IC from saved predictions,
  but exact booster reuse would require rerunning that cheap XGB arm.
- Policy: before launching any expensive model job, check for an existing
  checkpoint + sidecar + summary + predictions under the intended artifact
  root, run with `--reuse-existing` where available, and record whether the
  artifact is promotion-grade, diagnostic-only, or cache-only.

## 2026-05-24 Entry Score Alpha-Conversion Ladder

- `scripts/analyze_wf_trade_forensics.py` now includes a regime-first
  `entry_score_ladder`: within each entry regime and each score column
  (`entry_rank_score`, `entry_panel_score`, `entry_mu`), it buckets closed
  alpha trades by score and reports same-capital benchmark active P&L, active
  return, win rate, hold time, tax, and exit mix. This prevents pooled score
  IC from hiding a bad regime-local entry ladder.
- Applied to diagnostic trace
  `horizon60_erfloor_bullcalm040_diag_20260524-190959` with SPY same-capital
  benchmark. Result: all 35 closed alpha trades entered in BULL_CALM; overall
  active net after tax was only `+$30.68` versus SPY same capital, with active
  win rate `34.3%`.
- The new ladder confirms the root problem is entry-score semantics, not only
  tax/QP churn. In BULL_CALM, `entry_panel_score` Q1 produced active
  `+$6,069`, while Q4/Q5 produced `-$2,294` and `-$2,620`. `entry_mu` is
  similarly inverted: Q1 `+$6,472`, Q4 `-$3,304`, Q5 `-$2,449`. `rank_score`
  also deteriorates in Q4/Q5. This is consistent with negative BULL_CALM
  Spearman (`entry_mu` vs net P/L about `-0.268`) and negative 60d forward
  excess alignment (`entry_mu` about `-0.219`).
- Current next thesis: the model/calibrator is scoring a target that is not
  the realized BULL_CALM trade objective. The next repair should not be a
  global score threshold tweak. It should audit and, if confirmed, replace or
  regime-condition the BULL_CALM entry label/calibrator objective against
  active forward return and/or realized trade-compatible path risk, then pass
  leak-safe WF with regime-first score ladders.
- A/B hook added after this diagnosis: `panel_ltr.label_target` now supports
  `benchmark_relative`, which Gaussianizes the exact stock/SPY forward active
  return `(1+r_stock)/(1+r_spy)-1` before LTR training. Default remains
  `residual` so production behavior is unchanged until a WF side run passes.
  This candidate target is aligned with the calibrator and same-capital SPY
  forensic lens; it must be retrained and accepted regime-first before any
  promotion.

## 2026-05-24 Production Label + BULL_CALM RS Finding

- Correction: active 104 alpha158/fund production training does not use the
  older `PanelTrainingPipeline` label hook above. The live retrain path is
  `build_alpha158_qlib.py -> build_alpha158_fund_panel.py ->
  train_production_model.py -> fit_calibrator_alpha158_fund.py`. It trains
  XGB rank:pairwise on `fwd_60d_excess`, a per-date cross-sectionally
  standardized stock-minus-SPY 60d excess-return label; the calibrator then
  fits expected-return μ on raw `fwd_60d_excess_raw`.
- Extra diagnostic on the same `horizon60_erfloor_bullcalm040_diag_20260524-190959`
  trace joined entries to production labels. In BULL_CALM selected trades,
  `entry_panel_score` is inverted not only versus realized active P&L but also
  versus the model's own 60d label: Spearman vs `fwd_60d_excess` is about
  `-0.304`, and vs raw `fwd_60d_excess_raw` about `-0.301`. This is a selected
  decision-tree subset failure, not just a tax/reporting artifact.
- In the same closed-trade subset, `entry_rs_score` is the only score with the
  right sign: Spearman vs raw 60d label about `+0.335`, vs normalized 60d
  label about `+0.351`, and vs active net P&L about `+0.259`. RS quintiles
  were monotone in net P&L: bottom RS quintile about `-$1.39k`, top quintile
  about `+$4.26k`. This supports a BULL_CALM-specific momentum/relative-strength
  thesis, not a global RS promotion.
- Found and fixed an engineering blocker before any RS A/B: `BlendScoresTask`
  could sort by blended `rank_score/rs_score`, but `SortCandidatesTask`
  immediately re-sorted by original `rank_score`, discarding the blend. The
  new explicit knob is `ranking.regime_blend_weights`; default remains
  panel-rank-only. Tests now prove a BULL_CALM rs-only blend changes order
  while tier/QP admission still sees the original calibrated rank score.
- Next caveat: JointActionJob/QP still sorts and sizes by calibrated expected
  return and raw panel tiebreaks. Therefore `regime_blend_weights` repairs the
  ranking contract but may not be sufficient to improve APY/Sharpe if QP's
  action menu continues to use the anti-predictive BULL_CALM panel μ as the
  economic owner. If the RS-blend A/B is neutral, the next structural repair is
  a regime-conditional signal combiner/expected-return head, not another
  threshold tweak.

## 2026-05-24 QP Ranking Bypass Root Cause

- The first BULL_CALM rs-only WF diagnostic produced byte-identical
  `round_trips.csv` files to the panel-rank baseline. That falsified the
  hypothesis that the repaired `SortCandidatesTask` order was reaching the
  active production execution path.
- Root cause: active 104 uses `rotation.joint_actions.solver="qp"`.
  `JointPortfolioQPJob` built its source map and one-slot candidate admission
  from raw `ctx.candidates`, then defaulted slot priority to raw
  `rank_score`. It bypassed `ctx.ranked`, which is the output of
  `BlendScoresTask -> SortCandidatesTask`. This violated the architecture
  principle that model/ranking decides eligibility and QP only sizes.
- Fix: `score_candidates` now stamps `_ranking_composite`, normalized
  components, and `_ranking_order_index` onto candidates for auditability.
  QP source-map construction now consumes `ctx.ranked` when available, and
  QP slot admission uses the ranking composite when an explicit regime blend
  is active. Default panel-rank-only behavior is preserved.
- Regression tests added in `tests/test_joint_qp_task.py`: one-slot QP
  admission selects the RS-led candidate when
  `ranking.regime_blend_weights.BULL_CALM=[0,1]`, and still selects the
  raw-rank candidate under default config. Related QP/ranking tests passed.
- Next validation: rerun the same rs-only WF. The ledger must differ from the
  prior diagnostic; if APY/Sharpe still lag SPY, the remaining culprit is the
  QP economic owner (`mu`/expected_return and risk/cost objective), not the
  admission-order bypass.
- Validation result: the ledger changed only in the 2024-01-02 cut; the two
  later cuts stayed byte-identical. Mean WF Sharpe was still only about
  `+0.438` vs SPY `+1.081`, and BULL_CALM score monotonicity still failed.
  Therefore ranking-order bypass was a real bug but not the dominant owner of
  APY/Sharpe. The active bottleneck is QP's economic objective: it still
  optimizes using anti-predictive `mu`/expected-return semantics.
- Follow-up fix: `ForceMuSourceTask` now supports `ranking_composite` and
  `rs_score`, so QP μ can be sourced from the same regime-aware alpha signal
  that RankingJob used. Also fixed the scientific order of operations:
  `AlignQPHorizonUnitsTask` now runs before Grinold-Kahn `alpha_to_mu`, so
  `μ = IC × σ × z(score)` uses sigma on the same rebalance horizon as the QP
  covariance. This follows the Markowitz single-period contract. Defaults
  remain unchanged until a diagnostic config explicitly enables the alternate
  μ source and `alpha_to_mu`.

## 2026-05-24 QP μ Composite / Cash-Drag Root Cause

- Diagnostic config
  `base_featspace_scopefixed_covered_20260523.erfloor_bullcalm040_rsblend100_qpmucomposite.json`
  explicitly set `ranking.qp_mu_source=ranking_composite` and
  `ranking.alpha_to_mu.enabled=true` with `ic=0.08`. It keeps the qpmu change
  diagnostic-only; production defaults are unchanged.
- WF trace
  `horizon60_erfloor_bullcalm040_rsblend100_qpmucomposite_20260524-2124`
  still failed acceptance: mean annual-net Sharpe `+0.465` vs SPY `+1.081`,
  with `0/3` cuts beating SPY. This is not promotion evidence.
- The important forensic change is structural: only `16` closed alpha trades
  appeared, but the alpha sleeve was positive after tax and versus same-capital
  SPY. Gross P/L was about `+$10.1k`, tax about `+$6.0k`, net about `+$4.1k`,
  same-capital SPY about `+$0.9k`, and active net about `+$3.3k`.
  Cut-level active net was positive in all three WF cuts.
- The portfolio-level APY/Sharpe remained weak because average alpha exposure
  was tiny: about `4.8%`, `2.6%`, and `4.1%` across the three cuts, leaving
  roughly `95%+` cash in risk-on periods. The next root is therefore
  benchmark-relative cash drag / portfolio construction, not simply "buy more
  weak alpha names".
- Benchmark-sleeve A/B is now the next mainline test. It uses the existing
  `BenchmarkSleeveTask` core-satellite design and SciPy HiGHS
  `scipy.optimize.linprog`, a mature LP solver, to put residual capital into a
  benchmark core while keeping alpha admission separate. Alpha candidates must
  still pass model/ranking/QP gates; the sleeve may fund up to `15%` alpha
  satellite budget by selling SPY. This tests whether RenQuant should be
  evaluated as benchmark core + active satellite instead of sparse alpha plus
  cash.
- Exit-path false positives remain a separate pending issue. In the same
  qpmu trace, `stop_loss` exits were net negative and barrier-incorrect, while
  some `trailing_stop`/`single_day_loss` exits carried the gains. Do not call
  the system fixed if benchmark sleeve improves Sharpe but exit false-positive
  rates remain high.
- Benchmark-sleeve diagnostic result
  `horizon60_erfloor_bullcalm040_rsblend100_qpmucomposite_benchsleeve_20260524-2300`
  materially improved the portfolio layer: cut Sharpes were `+1.627`,
  `+0.757`, and `+1.018`; mean Sharpe was `+1.134` vs SPY mean `+1.081`;
  `2/3` cuts beat SPY Sharpe and APY. It still failed acceptance because
  `regime_ok=false` with benchmark-lag regime `HIGH_CALM`, and the run was
  explicitly diagnostic-only (`--skip-sanity --skip-config-parity`).
- Clean forensics confirms the sleeve did not hide bad alpha. Alpha-only
  closed trades remained positive: `16` closed alpha trades, gross about
  `+$11.4k`, tax about `+$6.8k`, net after tax about `+$4.6k`, same-capital
  SPY about `+$0.85k`, active net about `+$3.77k`. Average gross exposure
  rose from cash-heavy qpmu to about `99.5%`, `98.2%`, and `89.4%` across
  the three cuts, mostly via SPY benchmark weight.
- Forensic-tool bug fixed after this run: exit-path audit was incorrectly
  including `BenchmarkSleeveJob` / `benchmark_sleeve_rebalance` rows in the
  triple-barrier false-positive report. `alpha_vs_benchmark` already excluded
  sleeve rows, but exit-path audit did not. The tool now filters to alpha
  trades before exit-path labeling; `tests/test_wf_trade_forensics.py`
  covers the sleeve exclusion.
- After that fix, alpha exit-path quality is still a real pending problem:
  `stop_loss` has `5/5` false positives and net `-$2.17k`, with mean
  post-exit 60d SPY-relative excess about `+19.9%`; `trailing_stop` has
  `4/5` false positives despite positive realized P/L, with mean post-exit
  60d excess about `+24.2%`. `single_day_loss` is better but not perfect
  (`2/3` barrier-correct). Next repair should target BULL_CALM path exits,
  not benchmark sleeve.

## 2026-05-24 Stop-Anchor A/B and Sleeve Telemetry Hardening

- Diagnostic config
  `base_featspace_scopefixed_covered_20260523.erfloor_bullcalm040_rsblend100_qpmucomposite_benchsleeve_stopanchor.json`
  tested exactly one exit-policy change: `risk.stop_loss_anchor_policy.mode =
  max_entry_current` for BULL_CALM entries. The hypothesis was that BULL_CALM
  60d momentum entries were being stopped by later current-regime relabels
  using tighter cumulative stops.
- Result: rejected. WF still failed: mean Sharpe `+1.090` vs SPY `+1.081`,
  only `1/3` cuts beat SPY Sharpe and `0/3` beat SPY APY, with benchmark-lag
  regimes `HIGH_CALM` and `LOW_SPIKED`. Compared with the plain benchmark
  sleeve diagnostic, alpha active net fell from about `+$3.77k` to
  `+$2.16k`.
- Trade-level forensics: `stop_loss` count fell from `5` to `3`, but all
  remaining stop-loss exits were still false positives and their net loss
  worsened (`-$2.88k` vs `-$2.17k`). `single_day_loss` became worse
  (`5` exits, `60%` false-positive rate), and `trailing_stop` was unchanged.
  Therefore stop anchoring is not the next promotion path.
- Stronger root cause: score ordering is still wrong inside the actually
  traded BULL_CALM subset. In the stop-anchor trace, entry `rank_score`,
  `panel_score`, and `mu` all had negative Spearman relationships to realized
  P/L; entry `mu` vs 60d forward SPY-relative excess was about `-0.634`.
  This says the next mainline target is score calibration/selection on the
  selected-entry distribution, not another broad stop-loss tweak.
- Engineering fixes completed around the benchmark sleeve:
  `candidate_scores` now excludes the SPY benchmark sleeve holding when the
  sleeve is configured as non-alpha, so alpha candidate audits are not polluted
  by benchmark rows. `ticker_daily_state` still includes the sleeve via
  `decision_trace_tickers`, so portfolio-level decision tracing remains
  complete. LEAN debug logging now tolerates benchmark sleeve BUY orders whose
  alpha fields (`rank_score`, `rs_score`, `confidence`) are intentionally
  missing instead of crashing on float formatting.
- Config validation now treats `risk.stop_loss_anchor_policy.*` as active
  config paths, so future A/B configs cannot carry dead stop-anchor knobs
  silently. This is validation support only; the stop-anchor A/B result above
  is rejected.

## 2026-05-24 Scorer/Calibrator Identity Hardening

- Bug found during code review: scorer/calibrator binding used the full JSON
  file hash as the primary identity. WF gate and acceptance tools append
  mutable metadata such as `wf_gate_metadata` after training, so the same
  predictive model can acquire a new file hash without changing predictions.
  That makes strict calibration pairing unstable and can also hide whether a
  mismatch is a real foreign calibrator or a post-hoc audit stamp.
- Fix direction: split identity into two fields. `artifact_sha256` /
  `artifact_fingerprint` remain full-file tamper/audit hashes. New
  `model_content_fingerprint` hashes only prediction-relevant scorer content
  and explicitly ignores config fingerprints, training metrics, labels, and
  WF metadata. Runtime matching accepts the stable model-content identity and
  legacy file hashes, but never accepts `config_fingerprint` alone.
- Calibrator writers now stamp `scorer_model_content_fingerprint`, and both
  live `LoadGlobalCalibrationTask` and `WalkForwardModelLoader` compare
  identity lists fail-closed. Regression tests cover metadata-only mutation,
  model-content matching, legacy file-hash matching, and rejection of
  config-only calibrators.
- Validation: targeted contract and pipeline tests passed
  (`60 passed, 2 skipped` for scorer/calibrator contracts, `24 passed` for
  sim walk-forward calibration dispatch, `37 passed` for panel scoring jobs),
  and modified files compile. Direct inspection still shows current
  prod/staging calibrators do not match their scorers, and their WF stamps are
  failed. That is expected for old artifacts; do not promote or trade from
  them. The next accepted training run must produce a scorer and calibrator
  with matching model-content identity and passing WF/SPY/regime/sanity gates.
- This fix does not improve APY/Sharpe by itself. It removes a trust failure
  that could make downstream APY/Sharpe diagnostics meaningless. The active
  alpha root remains selected-entry score inversion in BULL_CALM/QP_BUY,
  especially entry `mu` showing negative Spearman to 60d forward excess.

## 2026-05-24 QP Contract Bug-Hunt Follow-Up

- Bug found: `_BuildMuVectorTask` still had an automatic `panel_score`
  fallback when `mu` was missing. Strict production config usually stopped
  this before solve, but any warn/off path could still optimize raw scores as
  expected returns. This violates the no-silent-fallback rule and makes old
  experiments suspect if they relaxed `qp_mu_contract`.
- Fix: QP μ vector now reads only finite `mu`. Explicit raw-score experiments
  must use `ranking.qp_mu_source` plus `alpha_to_mu`; missing forced-source
  fields are recorded as `_qp_forced_mu_missing_tickers` and fail the strict
  contract even after transform. This keeps QP as sizing/rebalance only.
- Bug found: `ComputeFullSigmaTask` repeated the old `0.0 or reverse_lookup`
  correlation bug. A real zero correlation could be replaced by a stale
  reverse-direction value. The task also filled every direction independently,
  so asymmetric stale correlation artifacts could create asymmetric covariance
  matrices.
- Fix: full-Σ construction now uses explicit `None` checks and fills the
  covariance matrix symmetrically from one pair lookup. Regression tests cover
  real `0.0` correlation versus stale reverse `0.95`.
- Validation: QP-focused tests passed (`60 passed`) and modified QP files
  compile. This is another trust fix, not yet proof of better APY/Sharpe.
  Re-run acceptance only after the next scorer/calibrator pair passes strict
  identity and WF gates.

## 2026-05-24 Sim/Live Trace Contract Follow-Up

- Bug found: `SimAdapter` built and cached `asset_embeddings` from the shared
  panel runtime bundle but did not attach them to each per-bar
  `InferenceContext`. Live and LEAN used `attach_panel_runtime_frames()` and
  therefore did attach `_panel_asset_embeddings`. Any scorer with `emb_*`
  columns could validate on a different feature surface in sim than in live.
- Fix: `SimAdapter.make_context()` now carries `_panel_asset_embeddings` into
  the per-bar context; regression coverage pins feature/factor/macro slicing
  plus embedding propagation.
- Bug found: `decision_trace_integrity_report()` existed but was not invoked
  by sim/live/LEAN after DB writes. That meant incomplete `ticker_daily_state`,
  missing blocked reasons, missing trade payloads, missing sell economics, or
  QP attribution gaps could remain a post-hoc test-only finding instead of a
  daily/full failure.
- Fix: added `validate_decision_trace_integrity()` and wired it into
  `SimAdapter.commit()`, `RunnerAdapter.commit()`, and
  `LeanAdapter._record_decision_trace()` after they write candidate scores,
  trades, and ticker daily state. `persistence.strict_decision_trace_integrity`
  is now explicitly `true` in both live config and golden config, with default
  fail-closed behavior.
- Validation: targeted suites passed: adapter/panel scoring (`43 passed`),
  persistence/LEAN/adapter contract (`44 passed`), veto/runner state/DB
  separation (`81 passed`), and config/panel/universe parity (`106 passed`).
  This does not claim APY/Sharpe improvement; it prevents future decision
  quality analysis from being based on incomplete trace rows.
- Follow-up bug found in live sell P/L stamping: the sell branch intended to
  use broker `avg_entry_price` but referenced a stale/nonexistent
  `positions_cache` path behind an unused sentinel. It now reads the actual
  `pos_cache` local and then falls back to `HoldingState.entry_price`.
  Validation: sell-ntfy / runner sell-attribution / runner-state tests passed
  (`62 passed`).
- Follow-up data-quality bug found in `ticker_daily_state`: benchmark sleeve
  rows appended for trace completeness were marked `in_watchlist=1` even when
  the sleeve ticker was not part of the alpha watchlist. This could pollute
  SQL analysis that separates alpha universe from passive sleeve. The row
  builder now writes real watchlist membership while still including the sleeve
  row for trace completeness. Validation: decision-trace / benchmark-sleeve /
  LEAN trace / persistence tests passed (`58 passed`).
- Follow-up trace bug found in `SizeAndEmitTask`: selected candidates skipped
  during second-stage sizing (`bad price`, `Kelly=0`, insufficient cash, or
  cash-invariant guard) only wrote logs and did not populate
  `_blocked_by_ticker`. The DB could then show these tickers as generic
  `candidate_not_selected`, hiding why a model-qualified candidate became no
  order. Sizing now stamps explicit reasons (`size_bad_price`,
  `size_insufficient_cash`, `size_cash_invariant`, or preserved
  `kelly_zero:*`). Validation: buy-quality/Kelly/blocked-by tests passed
  (`64 passed`) and persistence/LEAN trace/buy-emitter/order-attribution tests
  passed (`47 passed`).
- Follow-up QP input-contract bug found: runtime strict QP contract enforced
  finite μ but allowed missing σ to flow through `_BuildSigmaVectorTask`'s
  legacy `0.05` default. That can understate risk and let QP amplify a ticker
  with no real risk evidence. `ValidateQPMuContractTask` now treats missing
  σ as a hard contract failure in strict mode and stamps
  `qp_sigma_contract_block`. Validation: QP Grinold-Kahn/integration/static
  contract tests passed (`34 passed`) and broader QP admission/sector/
  correlation/joint/backend tests passed (`106 passed`).
- Follow-up execution-boundary bug found: `scripts/production_runner.py` was a
  standalone alpha158 scorer with an `--execute` path that submitted Alpaca
  paper orders directly, bypassing `live.runner`, `InferencePipeline`, QP
  admission, risk gates, and decision-trace persistence. Direct execution is
  now fail-closed by default before artifact load; the script remains available
  for dry-run/research scoring. Validation: production-runner guard plus
  sim/live/LEAN adapter parity tests passed (`12 passed`).
- Follow-up calibration drift bug found: `kernel/scoring.py` had already made
  Platt calibration fail closed when scaler metadata is missing, but
  `kernel/models.py::calibrate_score()` still treated missing
  `platt_scale_mean/std` as "use raw score" and returned raw score for unknown
  methods. That is a silent fallback to an uncalibrated score path. The model
  helper now requires finite Platt scaler metadata and falls back to
  `base_rate` for missing scaler or unknown methods, matching the stricter
  scoring helper. Validation: compile passed and kernel/ranking regression
  tests passed (`293 passed`).
- Follow-up QP risk-constraint bug found: when sector/correlation C2 hard caps
  made the QP infeasible, `_retry_with_relaxed_c2_caps()` automatically
  relaxed caps by `1.5x` and could then drop C2 caps entirely. That contradicts
  the "hard diversification constraint" contract and can turn risk limits into
  suggestions. Production/golden configs now set
  `qp_c2_infeasible_policy="strict"`; strict mode keeps the hard constraints
  and blocks QP orders for the bar. Explicit `relax`/`drop` remains available
  only for diagnostic configs. Validation: QP sector/correlation/admission/
  integration/contract tests passed (`70 passed`).
- Follow-up QP backend-boundary bug found: the optional `cvxportfolio`
  backend cannot enforce the project-specific hard sector, correlation, and
  gross-exposure constraints. The previous shim stripped those kwargs before
  solving, so a backend-switch experiment could silently compare a weaker risk
  contract. `SolveMarkowitzQPTask` now fails closed with
  `infeasible:cvxportfolio_unsupported_constraints` whenever those hard
  constraints are present. Validation: backend-switch, cvxportfolio parity,
  sector, and correlation tests passed (`48 passed`).
- Follow-up selection-trace bug found: `RunSelectionTask` recreated
  `_blocked_by_ticker` from an empty dict each time it ran. Any upstream
  terminal reason stamped by model/risk/panel gates could be overwritten by
  the legacy selection path, turning a real reason such as `risk_gate_vol`
  into a generic downstream non-selection reason in the DB. The task now
  preserves and extends the existing map instead of clobbering it. Validation:
  blocked-by, buy-quality, DB persistence, and LEAN trace tests passed
  (`71 passed`).
- Follow-up live sell-state bug found: `RunnerAdapter.commit()` correctly
  cleared state dictionaries after a broker-confirmed full sell, but later
  reused the start-of-bar `ctx.holdings` snapshot to re-persist the sold
  ticker's `sell_streaks` and `position_hwm`, and state GC also treated the
  just-sold ticker as still held. Full exits are now tracked explicitly:
  state persistence skips them and GC computes effective holdings as
  start-of-bar holdings minus confirmed full exits plus accepted buys.
  Validation: runner state/sell attribution/live-state suites passed
  (`97 passed`) plus partial-sell/HWM/sim-live/post-stop tests (`44 passed`).
- Follow-up tax config bug found: an unknown `tax.cash_debit_mode` only logged
  a warning and silently fell back to immediate event-level tax debits. A typo
  in a reporting-only config could therefore corrupt cash/APY while looking
  like a valid run. Unknown modes now fail closed; the legacy
  `event_cash_debit` spelling is accepted only as an explicit alias.
  Validation: vectorbt long-P&L cross-check, sim result tax reporting, and WF
  config parity tests passed (`23 passed`).
- Follow-up data/trace bug found: buy-universe construction correctly excluded
  loaded models whose OHLCV was missing, but the decision trace could later
  label the ticker `no_model_signal`. That is false: the model was present but
  the data was absent. Full inference now stamps `missing_ohlcv` before the
  buy scan, and ticker-daily-state preserves that reason.
- Follow-up data-freshness bug found: an entirely empty `ctx.ohlcv` was allowed
  to pass the freshness gate whenever downstream code was expected to fail.
  That was acceptable for minimal test stubs but unsafe for production configs
  with a watchlist/holdings. The gate now derives expected symbols from the
  watchlist, held tickers, benchmark, and sector ETFs; if any expected symbol
  is absent, buy/full and sell-only paths fail closed before decisions.
  Validation: data freshness, typed adapter, missing-OHLCV trace, buy-universe,
  LEAN trace, and persistence suites passed (`44 + 44 passed` across the two
  targeted runs).
- Follow-up WF/model-acceptance bug found: `run_sanity_battery()` computed
  regime-layered placebo/IC evidence (`sanity_regime_ic`) but the top-level
  sanity verdict, overall WF gate verdict, promotion gate, and full-run
  preflight could still accept old or hand-stamped `passed=true` metadata that
  lacked passing regime sanity. That made BULL_CALM-style placebo-dominated
  evidence visible in metadata but not a hard acceptance boundary. The WF gate
  now requires both global shuffled/placebo sanity and regime-level sanity;
  `promote()` refuses missing/failed `sanity_regime_ic`; P-WF-GATE blocks
  full/buy runs on old `passed=true` metadata without regime sanity while
  still allowing sell-only exits. Validation: WF gate, promotion, preflight,
  preflight-regime-sanity, and runtime regime-admission tests passed
  (`147 passed`).

## Global Pipeline Self-Audit Reset (2026-05-24)

Per operator instruction, previous local/vertical passes do not count. The
next 10 bug-hunting rounds restart from zero and each round must traverse the
entire pipeline end-to-end:

1. Data/fundamentals/news/sector/benchmark ingress and freshness.
2. Feature/label construction, embargo, neutralization, leakage controls.
3. Model training, WF manifests, placebo controls, regime-split evidence.
4. Artifact contracts, fingerprints, promotion, preflight.
5. Runtime inference, decision-tree fields, blocked reasons, DB persistence.
6. Alpha admission versus portfolio sizing/QP/rebalance.
7. Sell/exit/tax/P&L attribution and trade-level accounting.
8. Sim/live/LEAN parity and shared-code enforcement.
9. Operational scripts, cron, broker, ntfy, failure semantics.
10. Repo hygiene, stale artifacts, dead paths, cleanup/backlog/report.

### Global Round 1/10 — Pipeline Contract Pass

- New full/sell-only parity bug found: `InferencePipeline` applied
  `MetaLabelVetoTask` after path-rule sell signals, but `SellOnlyPipeline`
  skipped the same second-stage filter. Current production has meta-label
  disabled, so this did not change today's live behavior, but it was a latent
  contract break: if AFML-style meta-label exit veto is enabled, open/preclose
  sell-only cron paths would bypass the veto while daily full would not. The
  sell-only pipeline now runs `MetaLabelVetoTask` after `TickerSellJob`
  populates `ctx.exits` and before `LimitSellsPerBarTask`, matching the full
  path's exit-filter ordering. Validation: meta-label veto, pipeline,
  runner-meta-label wiring, meta-label preflight, no-trade monitor, and data
  freshness suites passed (`81 passed`, one xgboost warning).

### Global Round 2/10 — Model Evidence / Backtest Parity Pass

- New LEAN data-parity bug found: LEAN subscribes to the full watchlist, but
  `LeanAdapter.make_context()` only loaded OHLCV for `algo._models.keys()`.
  When `LoadUniverseJob` filters models via a universe floor, LEAN could then
  fail `DataFreshnessGateTask` for configured watchlist names that were never
  loaded into `ctx.ohlcv`, while sim/live still had full-watchlist data. LEAN
  now loads history for the full configured watchlist plus loaded models,
  sector ETFs, and SPY. This does not expand the buy universe because
  `_buy_universe()` still requires a loaded model.
- New LEAN execution-parity bug found: QP/selection emit exact whole-share
  orders and sim/live execute those shares, but LEAN buy/top-up used
  `SetHoldings(target_pct)`, letting LEAN recompute a potentially different
  quantity from portfolio value, price, fees, and current holdings. LEAN now
  executes buy/top-up orders with `MarketOrder(sym, shares)`; `target_pct`
  remains audit metadata. The unused `LeanBackend` contract was updated the
  same way so a future execution-backend refactor cannot reintroduce the old
  `SetHoldings` sizing semantics. Validation: LEAN trace/backend,
  data-freshness, sim/live parity, buy-emission, and QP admission tests passed
  (`77 passed`).
- New weekly-promotion feature-cache bug found: the alpha158+fund merge cache
  only compared the merged panel mtime against alpha158 and SEC fundamentals,
  but `build_alpha158_fund_panel.py` also reads `data/earnings_surprise` for
  PEAD/SUE and `data/news_sentiment_alpaca` for sentiment. Weekly promotion
  could therefore stamp a fresh model on a stale merged feature panel after
  those sources changed. `MergeFundFeaturesTask.should_skip()` now includes
  both source directories in cache invalidation. Validation:
  daily-retrain, train/infer feature parity, sentiment panel join, and
  production panel invariant tests passed (`37 passed` + `5 passed`).
- New WF-evidence selection bug found: when `--derive-config-from-prod` saw a
  preferred walk-forward manifest that did not match the candidate recipe,
  `_matching_manifest_for_recipe()` searched the whole sim artifact directory
  and silently replaced it with the highest-coverage same-recipe manifest. That
  could stamp APY/Sharpe evidence from an unintended manifest. The gate now
  treats a configured preferred manifest as the evidence contract: validate it
  and fail closed on mismatch; auto-discovery is allowed only when no preferred
  manifest is supplied. Validation: WF recipe-scope, WF CLI contract, and
  promotion-gate tests passed (`69 passed`).
- New WF acceptance-path bug found: `run_wf_gate.py` invoked each
  `run_sim_104.py` cut with `--skip-preflight`, bypassing the sim side-config
  static-path validator that prevents no-op/unwired configs from spending
  compute and stamping APY/Sharpe. WF cuts now keep that static preflight while
  still using `--no-persist` and `--no-compare` for isolation. Validation:
  WF CLI contract, WF recipe-scope, WF config parity, and promotion-gate tests
  passed (`77 passed`).

### Global Round 3/10 — Data / Label Integrity Pass

- New alpha158 label-alignment bug found: `scripts/build_alpha158_qlib.py`
  aligned SPY benchmark closes to ticker dates with unlimited forward-fill
  before computing `fwd_{5,20,60}d_excess`. If a ticker/calendar row was far
  away from the latest SPY bar, this made stale SPY prices look like valid
  benchmark returns and polluted IC/WF labels without a crash. The label helper
  now bounds SPY forward-fill to 5 calendar days and leaves stale rows as NaN
  for downstream `DropnaLabel` handling. Validation:
  `tests/test_alpha158_label_alignment.py`,
  `tests/test_train_infer_feature_parity.py`, and
  `tests/test_walk_forward_splits.py` passed (`24 passed`).
- New training-preflight bypass found: the production alpha158+fund retrain
  uses `DailyRetrainAlpha158FundPipeline`, not `PanelDataJob`, so it did not
  run the mandated `ScanTrainingDataTask` before rebuilding labels/models.
  The daily retrain pipeline now starts with `ScanDailyTrainingDataTask`, writes
  `artifacts/daily_retrain_training_data_scan.json`, and production/golden
  config sets `panel_ltr.data_scan.strict=true` so missing/stale OHLCV or SPY
  data blocks training before artifacts are stamped. Validation:
  daily-retrain orchestration, smoke-model, and WF-config parity tests passed
  (`60 passed`).
- New sector-neutralization fail-open found: with
  `panel_ltr.neutralize_features=true`, `SectorMomentumTask` and
  `TickerPanelNeutralizeJob` could silently fall back to raw feature frames when
  sector ETF OHLCV, ticker sector metadata, or a sector momentum frame was
  missing. That breaks the training/inference distribution contract and lets
  missing sector metadata disable a key risk-control assumption. Production and
  golden config now set `strict_neutralization=true`; missing sector context
  raises unless an experiment explicitly opts out with
  `strict_neutralization=false`. Validation: panel-training, neutralization,
  train/infer parity, panel-bugfix, and WF-config parity tests passed
  (`55 passed`).
- Follow-up active-source scan gap found by sidecar audit: after adding the
  daily retrain scan, `scan_training_inputs()` still scanned legacy
  per-ticker fundamentals but not the active alpha158+fund sources:
  `data/sec_fundamentals_daily.parquet` and
  `data/news_sentiment_alpaca/`. Strict retrain could therefore pass the
  preflight while the actual model inputs were missing or mostly imputed. The
  scan report now includes both active sources and raises issues for missing,
  unreadable, stale, or materially under-covered active inputs. Current repo
  scan is clean under these active-source thresholds:
  SEC coverage `130/142` (`91.5%`, max date `2026-02-10`, age `103d`) and
  sentiment coverage `142/142` (max date `2026-05-17`, age `7d`).
  Validation: data-scan preflight and daily-retrain tests passed (`32 passed`).

### Global Round 4/10 — Sell / Tax / Trade Accounting Pass

- New tax-lot age bug found by sidecar audit and fixed: sim/LEAN already used
  FIFO/HIFO tax lots for disposed cost basis, but tax rate and reported
  `hold_days` still used aggregate `HoldingState.entry_date`. Under HIFO, an
  old aggregate holding can sell a recently-added high-cost lot; the old path
  could tax a 9-day lot at long-term rates. `apply_sell_lots_detailed()` now
  returns the exact disposed lot slices, sim/LEAN compute event tax from those
  lot acquisition dates, and annual-net tax reporting consumes ST/LT split
  P&L from the trade decision inputs. Validation: partial-sell, tax-lot,
  HIFO-selection, trade-attribution, QP-audit, LEAN backend, and LEAN trace
  suites passed (`69 passed`).
- New live decision-trace selected-state bug found by sidecar audit and fixed:
  live DB persistence built normalized buy trade rows, but selected tickers
  were still derived from raw broker order dicts. Raw live order dicts do not
  necessarily carry `action="buy"`, so a broker-confirmed buy could be written
  as `selected=0` in `candidate_scores` / `ticker_daily_state`. Runner now
  computes `selected_tickers` from normalized `trade_events`, matching sim and
  LEAN. Validation: runner selected-state, trade-event builder, and candidate
  persistence tests passed (`16 passed`).
- New decision-trace migration bug found by sidecar audit and fixed: legacy
  `ticker_daily_state` schemas keyed by `(date, ticker)` were rebuilt into the
  newer `(run_id, ticker)` schema without carrying the post-horizon-contract
  columns `expected_return_horizon_days` and `mu_horizon_days`. Fresh DBs were
  correct, but migrated DBs could silently lose horizon observability. The
  rebuild path now creates and inserts both horizon columns with safe NULL
  defaults for truly old rows, then `record_ticker_daily_state()` can persist
  current 60d horizons normally. Validation: persistence, candidate-score,
  ticker-daily-state, and decision-trace-horizon tests passed (`23 passed`).
- New QP integer-execution bug found by sidecar audit and fixed: the
  `qp_min_share_floor` path could turn a positive solver delta that rounded to
  zero shares into a forced 1-share buy when one share was 5-15% of NAV. This
  helped high-priced stocks enter small accounts, but it violated the QP
  target/cap contract and could manufacture trades larger than the optimizer's
  intended allocation. Production/golden config now sets
  `qp_min_share_floor_pct=0.0`, and the code default is disabled; sub-1-share
  deltas round down to `qp_zero_shares` unless a future experiment uses
  fractional-share execution or a true mixed-integer lot-size optimizer.
  Validation: QP min-share, QP admission, and WF-config parity tests passed
  (`38 passed`).
- New QP held-universe exemption bug found by sidecar audit and fixed:
  holdings must stay in the QP universe so the optimizer can trim/close them,
  but that exit permission was not explicitly separated from buy/top-up
  admission. A held ticker that was no longer a current buy candidate could
  still receive a positive QP delta through the same source map. The source-map
  task now marks held-but-unadmitted names as `qp_universe_exit_only`, a new
  guard caps their QP upper bound at current weight before solve, and the
  emission gate blocks any positive delta with the same reason as a second
  line of defense. Validation: QP admission, min-share, conviction-cap, and
  split-job e2e tests passed (`55 passed`).
- New live universe-held source bug found by sidecar audit and fixed:
  `LoadUniverseJob` used `live_state.position_hwm` to decide which tickers were
  held and therefore exempt from staleness/universe floors. In live mode that
  state can be stale before RunnerAdapter GC reconciles broker positions, so a
  flat ticker could receive a held exemption and become buyable. The live
  runner now connects the broker before universe load and passes authoritative
  broker-held tickers into `UniverseContext`; state-file `position_hwm` remains
  only a fallback for legacy/sim contexts. Validation: universe-held,
  universe-alignment, and daily-e2e loader tests passed (`41 passed`).
- New regime-admission/QP top-up bug found by sidecar audit and fixed:
  `RegimeModelAdmissionTask` correctly cleared new candidates when current
  regime evidence failed, but holdings could still reach QP with refreshed
  rank/μ/σ and receive positive deltas through the held source map. Failed
  regime admission now marks all holdings as QP exit-only with the exact
  `regime_admission:*` reason, the source-map task preserves that marker, and
  QP constraint/emission guards use the specific reason while still allowing
  trims/closes. Validation: regime-admission, QP admission, min-share, panel
  job ordering, and split-job e2e tests passed (`56 passed`).
- New alpha158 train/infer parity bug found by sidecar audit and fixed:
  training used pandas `rolling.std()` sample standard deviation (`ddof=1`),
  while inference single-bar/vectorized paths used population standard
  deviation (`ddof=0`) for `STD*`, `VSTD*`, and `WVMA*`. That silently changed
  model input scales at live/sim/LEAN scoring time. The train builder now
  declares the std contract explicitly, and both inference paths match it.
  Validation: alpha158, feature-cache, and volume-feature tests passed
  (`35 passed`).
- New LEAN/live/sim parity bugs found by sidecar audit and fixed:
  LEAN previously populated `ctx.prices` mainly for holdings/sector ETFs, so
  an unheld ranked candidate could be rejected downstream as `size_bad_price`
  only in LEAN. LEAN now prices every watchlist/model/holding/ETF/SPY ticker
  from Slice, Security price, or latest OHLCV close. LEAN universe loading now
  passes current portfolio-held tickers into `UniverseContext`, matching live's
  broker-held override, and `FilterAutoDropTask` exempts held tickers just like
  staleness/floor filters. Validation: adapter contract, auto-drop, universe
  held-exemption, universe-alignment, and daily-104 loader tests passed
  (`53 passed`).
- Follow-up LEAN held-source hardening fixed: `_current_held_tickers()` now
  admits only finite non-dust quantities (`abs(qty) > 1e-9`), matching live's
  broker-held source semantics. This prevents NaN/dust portfolio quantities
  from granting phantom held exemptions during LEAN universe loading.
- New decision-trace accounting integrity bug found by sidecar audit and
  fixed: `decision_trace_integrity_report()` previously checked only that sell
  economics were non-null. It now requires finite shares/economics,
  `net_pnl_after_tax == gross_pnl - tax`, non-negative tax, no positive tax on
  losing sells, and tax not exceeding positive gross P&L. Validation:
  persistence plus universe/adapter parity tests passed (`88 passed`).
- New raw alpha158 preprocessing replay bug found by sidecar audit and fixed:
  training winsorized raw alpha features at train-only 0.1%/99.9% quantiles
  before z-scoring, but the sidecar/artifact stored only means/stds, so
  runtime raw scoring could not replay the same transform. The alpha158 stats
  sidecar now stores raw clip bounds and preprocessing version; production
  artifacts propagate bounds aligned to `feature_cols`; raw runtime transform
  applies raw clip before z-score, while panel-space scoring remains unchanged.
  Validation: feature-transform, production-artifact, lookahead, train/infer,
  and WF-gate contract tests passed (`86 passed`).
- New trade-evaluation horizon mismatch fixed: 104's current model/rotation
  horizon is 60 trading days, but `ticker_forward_returns` and
  `backfill_trade_evaluations.py` only supported short horizons up to 20d.
  The DB schema/migration, forward-return backfill, and trade-evaluation
  backfill now support `fwd_60d`, and trade evaluation defaults include
  `[1, 5, 10, 20, 60]`. Validation: forward-return, trade-evaluation,
  persistence, reconciliation, and conformal-Gate-B wiring tests passed
  (`92 passed, 1 skipped`).
- New alpha158 extra-feature train/runtime parity bugs fixed: runtime
  fundamentals, PEAD rank, and sentiment imputation were computed over the
  post-filter target set, while training computed per-date fill/rank over a
  stable cross-section. Runtime now computes those extra features over the
  configured/model/watchlist context and reindexes them back to candidates and
  holdings. Sentiment also no longer carries stale prior-day rows into a
  no-news day: it uses exact-date rows plus cross-sectional median/final-zero
  fill, matching the training join. Runtime sentiment zeroing now requires an
  explicit artifact contract that the model was trained for zeroing; otherwise
  the system preserves train-parity values and logs the skipped gate. Validation:
  panel scoring, PEAD, sentiment, daily retrain, train/infer, and alpha158
  inference tests passed (`123 passed`).
- New live execution accounting bug fixed: Alpaca `accepted` / `new` market
  DAY orders were being treated as executed fills. Runner live state, trade
  DB rows, same-bar sell credit, wash-sale clocks, entry dates, P/L stamps, and
  ntfy trade wording now update only from filled/partially-filled quantities.
  Submitted-but-unfilled orders are tracked as pending and surfaced in ntfy
  without mutating execution state. Alpaca order responses now include
  `filled_qty` and `filled_avg_price`; failed exits are urgent ntfy events and
  no longer append a misleading `no trade` suffix. The daily forward-return
  backfill now routes live mode through broker-tagged DB paths such as
  `data/runs.alpaca.db`, matching the actual live trace DB. Validation:
  runner state, trade ntfy, conformal/backfill, portfolio-DB-path, and state
  path tests passed (`107 passed, 1 skipped`; plus broader runner/DB tests
  `105 passed, 1 skipped`).

## Stop Conditions

Stop and fix before reporting performance if any of these happen:

- WF config loses `tax.cash_debit_mode=reporting_only`.
- WF/sanity metadata lacks passing regime-level IC/placebo evidence.
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
- Repo hygiene ledger:
  `doc/research/2026-05-24-repo-hygiene-ledger.md`.
