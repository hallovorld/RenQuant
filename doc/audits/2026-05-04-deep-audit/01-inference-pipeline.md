# InferencePipeline — Deep Audit

Production runtime path. Phases:
RegimeJob → DrawdownJob → BuyGatesJob → SellJob (parallel) → CandidateJob (parallel) → PanelScoringJob → PanelRankVetoJob → RankingJob → RotationJob → SelectionJob → ScoreDistributionJob → (TopUp / Trim / Monitor)

---

## RegimeJob

### HurstTask
File: `kernel/pipeline/task_regime.py:16`
Reads: `ctx.spy_returns` (list[float])
Writes: `ctx.regime_state.{hurst, hurst_regime}`
Edge cases handled: early-exit when `len(spy_returns) < 30` (returns None)
🟡 Concern: early-exit leaves `state.hurst_regime` UNCHANGED from prior bar (could carry stale signal across an actual regime change). Should at minimum fall back to "AMBIGUOUS" on insufficient data.
🟡 No NaN guard on spy_returns — if NaN propagates from upstream `_compute_spy_returns`, `compute_hurst` behavior undocumented.

### CUSUMTask
File: `kernel/pipeline/task_regime.py:44`
Reads: `ctx.spy_returns`
Writes: `ctx.regime_state.cusum_triggered`
🟡 No NaN guard. Empty/short input → `compute_cusum` behavior undocumented.

### GMMTask
File: `kernel/pipeline/task_regime.py:76`
Reads: `ctx.gmm` (loaded artifact), `ctx.spy_returns`, `ctx.ohlcv["SPY"]`
Writes: `ctx.regime_state.gmm_probs`
🔴 **Issue**: NO null guard for `ctx.gmm`. If GMM artifact loading failed upstream, `gmm_predict(None, …)` likely crashes. Production daily104 cron would die hard.
🟡 No null guard for `ctx.ohlcv.get("SPY")` — passes None into `gmm_predict`.

### BEAROverrideTask
File: `kernel/pipeline/task_regime.py:95`
Reads: `ctx.spy_returns`
Writes: `ctx.regime_state.hard_bear`
✅ NaN/inf guard explicit + fail-SAFE to True (line 116-121)
✅ Audit fix RG-1/RG-2 shipped 2026-04-25

### RegimeFinalizeTask
File: `kernel/pipeline/task_regime.py:138`
Reads: `state.gmm_probs`, `state.hurst_regime`, `state.hard_bear`, `state.regime` (prev)
Writes: `ctx.regime`, `ctx.confidence`, `state.{regime,countdown,in_transition,cooldown_start}`
🟡 Empty `gmm_probs` dict → `dominant_gmm` defaults to "BULL_CALM" (line 159) — silent. If GMM failed, the default could push a wrong regime forward into trading.
✅ Cooldown wall-clock + bar-count dual-track logic looks correct.

---

## DrawdownJob

### HWMUpdateTask
File: `kernel/pipeline/task_drawdown.py:12`
✅ NaN/inf guard explicit (DC-1 fix 2026-04-25). Skips update on non-finite portfolio_value, preserves prior HWM. Drawdown gate stays armed.
✅ Order: runs BEFORE DrawdownCircuitTask in `DrawdownJob.tasks`.
🟡 Negative portfolio_value: `max(hwm, -1000) = hwm` — works correctly but undocumented edge case.

### DrawdownCircuitTask
File: `kernel/pipeline/task_drawdown.py:36`
✅ Recomputes `skip_buys` each bar (fixes 133-day no-trade streak bug).
✅ Hysteresis option via `drawdown_resume_pct`.
🔴 **Issue 07**: line 60 `(ctx.hwm - ctx.portfolio_value) / ctx.hwm` — same NaN-propagation pattern as DC-1, but here it's NOT guarded. If `portfolio_value` is NaN despite HWMUpdateTask skipping (e.g., portfolio_value set after HWMUpdateTask returned), drawdown = NaN, `NaN >= halt_pct` is False, halt silently doesn't fire. Should match HWMUpdateTask's defensive `math.isfinite` guard.
🟡 `ctx.hwm <= 0` early-exit — keeps existing `skip_buys` unchanged. If true on first bar before any HWM update, drawdown gate disabled silently.

---

## BuyGatesJob

7 tasks in chain order: DrawdownGate → TransitionWindow → ConfidenceVeto → BullVolOffensiveBlock → BEARBranch → VelocityCrash → EMA50

### DrawdownGateTask
File: `task_gates.py:13`
✅ Simple: reads `ctx.skip_buys` (set by DrawdownCircuitTask), fires `buy_blocked = True`.

### TransitionWindowTask
File: `task_gates.py:23`
✅ wall_time vs bar_count mode handled. Cooldown enforcement in correct task.

### ConfidenceVetoTask
File: `task_gates.py:44`
✅ G-1 fix shipped (NaN/inf confidence → fail-SAFE defensives-only).
✅ Doesn't short-circuit (allows downstream macro gates to still fire).

### BullVolOffensiveBlockTask
File: `task_gates.py:85`
✅ Regime + flag gated. Default OFF.

### BEARBranchTask
File: `task_gates.py:115`
✅ Simple. Doesn't short-circuit chain (correctly).

### VelocityCrashTask
File: `task_gates.py:133`
🟡 **Concern 05**: No NaN guard on `ctx.spy_returns` input — relies on `check_spy_velocity_crash` helper to handle. Helper not audited here. If helper returns False on NaN, gate silently disabled (same fail-OPEN class as Issue 07).

### EMA50GateTask
File: `task_gates.py:150`
🔴 **Issue 06**: `spy_df is None` or empty → `return None` (line 158-160) → BUYS CONTINUE despite missing macro data. **Fail-OPEN, should be fail-SAFE**. The other macro gates (DrawdownGate, VelocityCrash) all default to "block" on missing data; EMA50 is the outlier. With Issue 05 (VelocityCrash silent on NaN), this means a SPY data outage could leave both macro gates disabled and offensive buys flowing in BULL.

---

## BuyGatesJob

7 tasks (DrawdownGate, TransitionWindow, ConfidenceVeto, BullVolOffensiveBlock, BEARBranch, VelocityCrash, EMA50Gate)

🟡 All defined in `kernel/pipeline/task_gates.py`. No NaN handling visible at glance. Each writes `ctx.buy_blocked` or `ctx.bear_only`. No ordering invariants enforced beyond the chain definition.

---

## TickerSellJob (parallel)

### PrepareHoldingTask
File: `task_sell.py:44`
✅ NaN/inf guard on `tc.price` (PH-1/PH-2 fix). NaN prev_close coerced to None.

### ScoreModelTask
File: `task_sell.py:78`
✅ Defaults to "hold" if model/spy/stock missing.
🟡 **Concern 08**: No NaN guard on `tc.features.iloc[-1]` before calling `score_artifact`. If last feature row has NaN cells, score behavior undocumented.

### EvaluateExitsTask
File: `task_sell.py:119`
🟡 Logic delegated to `kernel.exits.compute_exits`. Internal not audited here.

### SellGateBTask
File: `task_sell.py:142`
✅ Documented fall-through cases (5+ defensive branches). Defensive against missing μ/σ.

### PanelConvictionExitTask, EarningsBlackoutSellTask
🟡 Not yet audited. Queued.

---

## TickerCandidateJob (parallel)

7 tasks (EarningsFilter, WashSaleFilter, BuildFeatures, ScoreBuy, ScoreThreshold, RelativeStrength, AssembleCandidate)

### EarningsFilterTask
✅ Defensive: `tc.earnings_calendar or {}`. Returns False if blocked.

### WashSaleFilterTask
✅ Defensive: `tc.last_sell_dates or {}`. Returns False if blocked.

### BuildFeaturesTask
✅ NaN/None handling for missing OHLCV / model / SPY. Cache fallback path.

### ScoreBuyTask
🟡 **Concern 09**: No NaN guard on `tc.features.iloc[-1]` (same as ScoreModelTask Concern 08). Same exposure.

### ScoreThresholdTask
✅ TC-1 fix: NaN rank treated as rejected.

### RelativeStrengthTask
🟡 **Concern 10**: Generic `except Exception` (line 143) masks any error. Could hide specific data integrity issues silently. Should narrow to `(KeyError, ValueError, TypeError, ZeroDivisionError)` or log.warning the exception type.
🟡 No check for negative prices in close column — division by 21-day-old close could break if data has a negative or zero entry.

### AssembleCandidateTask
✅ Simple assembly with defensive defaults.

🟡 Z1 parabolic-exhaustion gate REMOVED 2026-04-28 after A/A falsified hypothesis. Documented in job docstring.

---

## PanelScoringJob

(Already audited — see `02-panel-scoring-job.md`)
✅ VetoWeakBuysTask scale-mismatch bug FIXED 2026-05-04 (P0 #A1)
✅ distribution_floor scale-mismatch FIXED 2026-05-04 (P0 #A5, dormant)
✅ Calibrator NaN-leaf collapse FIXED 2026-05-04 (P0 #B3 + B3a band-aid)
✅ Calibrator rolling window FIXED 2026-05-04 (D11)
✅ row_coverage gate added (training + inference) 2026-05-04 (#13)

---

## PortfolioQP (JointPortfolioQPTask)

File: `kernel/portfolio_qp/task_joint_qp.py:48`
Auditing — see `03-portfolio-qp.md`.

Initial findings (from contract test work tonight):
✅ Bounds enforced in solver (target_w[i] ∈ [w_lower, w_upper])
✅ cash_reserve honored
✅ turnover_max honored
🟡 status enum could include unexpected values from cvxpy — contract test enumerates known set but new cvxpy versions may add more.
🟡 Tax-cost-per-sell parameter exists but plumbing from caller not audited yet.

---

## Findings tally (so far)

| ID | Task | Severity | Status |
|---|---|---|---|
| 01 | GMMTask null-guard for ctx.gmm | 🔴 | OPEN |
| 02 | RegimeFinalizeTask empty gmm_probs default | 🟡 | OPEN |
| 03 | HurstTask early-exit leaves stale state | 🟡 | OPEN |
| 04 | HurstTask / CUSUMTask no NaN guard on spy_returns | 🟡 | OPEN |
| 05 | VelocityCrashTask no NaN guard on spy_returns | 🟡 | OPEN |
| 06 | EMA50GateTask SPY missing → buys CONTINUE (fail-OPEN) | 🔴 | OPEN |
| 07 | DrawdownCircuitTask NaN portfolio_value → halt silently disabled | 🔴 | OPEN |
| 08 | ScoreModelTask no NaN guard on features.iloc[-1] | 🟡 | OPEN |
| 09 | ScoreBuyTask no NaN guard on features.iloc[-1] | 🟡 | OPEN |
| 10 | RelativeStrengthTask generic Exception catch | 🟡 | OPEN |

## Next-up tasks to audit

- DrawdownJob (2 tasks)
- BuyGatesJob (7 tasks)
- TickerSellJob (5+ tasks)
- TickerCandidateJob (7 tasks)
- RankingJob, RotationJob, SelectionJob (chain-of-tasks each)
- TopUpHeldTask, TrimHeldTask, MonitorIdleStreakTask
- PanelTrainingPipeline tasks (29 tasks across the panel-training-pipeline)

---

## PanelTrainingPipeline (training-side, but governs the artifact production runtime reads)

### FactorZScoreTask
File: `training_panel/pp_panel_training.py:1279`
✅ P-16 fix in place (per-ticker raw-feature z-score before cross-sectional z, prevents day-1 NaN explosions).
🟡 **Concern 11**: `raw_cols` list is hand-maintained — adding a new feature in BuildFeaturesTask without updating raw_cols means it silently bypasses the per-ticker normalization. No test asserts the two lists agree. Recommend pulling from a shared registry (or adding a lint test that diffs feature names against `raw_cols`).
🟡 **Concern 12**: `df[col].iloc[-1]` on fundamentals frames — if last row is NaN (common: stale FY data, ticker delisted-then-re-added), z-score evaluates to NaN; downstream NGBoost residual calculation may inherit. No NaN guard.

### LabelsTask
File: `training_panel/pp_panel_training.py:1399`
✅ Hard `raise` on missing benchmark frame (no silent fall-through to zero-label panel).
✅ triple_barrier integration with explicit fail mode (raises if returns can't be sorted by date).
✅ §5.2 sanity hooks present: `panel_ltr.label_shuffle_seed` + `panel_ltr.label_shift_days` config-flag-gated, default off.
🟡 **Concern 13**: `lookahead_days` default 5 — silent if config omits. Production retrains have all set 5 explicitly, but a new caller could ship a model with the wrong horizon and the artifact metadata wouldn't expose the mismatch unless `lookahead_days` is read at inference too.

### CrossValidateTask
File: `training_panel/pp_panel_training.py:2026`
✅ Audit fix #58 (sklearn-CV-index-misalignment): `_SklearnAdapter.fit` raises if X.index has rows not in parent panel. Catches a class of bug where re-indexed CV folds silently produced wrong group_sizes.
✅ Audit fix #14 (CV-vs-FinalFit-epoch-mismatch): transformer CV uses same `num_boost_round` as FinalFit (pre-fix used `// 2`, IC measured a different model than what shipped).
✅ Audit fix #60 (`.to_numpy()` not `.values`): preserves dtype int32 for group_sizes, avoids xgboost ambiguity.
🔴 **Issue 14**: lightgbm and xgboost CV adapters use `num_boost_round=max(num_rounds // 2, 50)` (lines 2137, 2174) — DIFFERENT from FinalFit which uses full `num_boost_round`. Same class of bug as audit fix #14 but for lgbm/xgb backends. CV IC ≠ FinalFit IC because they train for different durations. Should align (either CV uses full rounds, or FinalFit uses //2).
🟡 **Concern 14**: `cv_method == "cpcv"` is the configured production setting. The fallback purged-KFold path (line 2197) is exercised infrequently — possible test gap.

### FinalFitTask
File: `training_panel/pp_panel_training.py:2209`
✅ HIGH-2 fix (purge `lookahead` date-groups between train and eval) in place.
✅ BUG-CV-3 fix (eval size aligned with CPCV fold size) in place.
✅ BUG-CV-2 fix + Task #24 refinement (eval_ic escape clause for strong-univariate-IC features).
✅ External audit fix #7 (`min_eval_ic` second gate) in place, off by default.
🟡 **Concern 15**: `_es_cfg = cfg.get("early_stopping_rounds", 20)` default 20 differs across configs. The min_best_iter gate at 5 protects, but a config with `early_stopping_rounds=5` would let the model save with effectively no learning. Recommend log.warning when `early_stopping_rounds < 10`.
🔴 **Issue 15**: Path `cfg.get('xgb_params', {}).get('eta', cfg.get('xgb_params', {}).get('learning_rate', 0.02))` (line 2373) — checks `eta` first then `learning_rate`. xgboost accepts both; if config sets `learning_rate=0.05` AND `eta` is also present (e.g. 0.02 default leakage from inheritance), `eta` wins. Subtle but real failure mode if a config inherits eta from a base while overriding learning_rate. Should normalize earlier.

### SaveArtifactTask
File: `training_panel/pp_panel_training.py:2401`
✅ Run_id + config_fingerprint stamping (per CLAUDE.md §5.3 invariant — catches the next config/model drift).
✅ Transformer-shim auto-backup before clobbering panel-ltr.json.
🟡 **Concern 16**: `record_training_run` wrapped in bare `except Exception` (line 2568) — failure logged but non-fatal. Acceptable, but if persistence is silently failing for weeks, no alert. Recommend a single ERROR log per run minimum, or a counter that surfaces in retrain summaries.
🟡 **Concern 17**: `meta["panel_shape"]["rows"] = int(ctx.panel_metadata.get("n_rows", len(ctx.panel)))` — if metadata is stale and `len(ctx.panel)` differs (e.g. metadata was set in BuildPanelTask before any later filter), the artifact reports a wrong shape. Check that metadata is always updated AFTER the row_coverage filter.

### NGBoostFitTask
File: `training_panel/pp_panel_training.py:2588`
✅ Audit fix N-2/N-14 (val split + early stopping) in place.
✅ Audit fix N-17 (CPCV for NGBoost, opt-in) in place.
✅ Audit fix HIGH-3 (lookahead purge between train/val for NGBoost) in place.
✅ Audit fix NGB-OVERFLOW-TRAIN: catches numerical blow-up in fit; preserves prior artifact rather than crashing the pipeline.
🟡 **Concern 18**: `cv_params["n_estimators"] = max(50, int(params.get("n_estimators", 400)) // 2)` (line 2653) — same class as Issue 14: CV uses half the production estimators, so CV IC measures a model with less capacity than what ships. Less severe for NGBoost (shallow trees, plateau early), but worth noting.
🟡 **Concern 19**: bare `except Exception` swallowing fit errors (line 2705 + 2725). The CV-side swallow is warning-only and logs the exception type — defensible. The production-side swallow at line 2725 is the existential one (NGBoost dies → silent skip). Existing artifact survives, but no metric/alert tracks "NGBoost has been silently failing for N consecutive runs".

### NGBoostSaveTask
File: `training_panel/pp_panel_training.py:2768`
✅ Inference-side path wins over training-side when both set differently — this is the correct semantic (live runner reads where inference-side points).
🟡 **Concern 20**: not reviewing the save body in this audit pass; recommend a follow-up to verify the NGBoost JSON payload structure stays in sync with NGBoostHead.load() expectations (silent compat break would only show up at next inference run).

### RefreshPanelCalibratorTask
File: `training_panel/pp_panel_training.py:2928`
✅ CAL-7 (auto-refresh after retrain) in place.
✅ CAL-7-PATH (correct config path `ranking.panel_scoring.global_calibration`) in place.
✅ CAL-7-TIMEOUT bumped 600→1800s for CPCV-15.
🔴 **Issue 16**: failure is logged but non-fatal (line 3024-3033). The CAL-7 root incident was *exactly* this — calibrator silently stale because the refresh script was non-fatal. The auto_refresh fix runs the script, but if the subprocess fails (rc != 0), we log a warning and continue with the OLD calibrator. The model artifact has already been overwritten. Result: new model + stale calibrator = same incident class as CAL-7 pre-fix. Recommend: either fail-loud (raise after logging) OR snapshot the previous panel-ltr.json before SaveArtifactTask runs, so a calibrator-refresh failure can roll the model back.
🟡 **Concern 21**: `subprocess.run` with no `env=` argument — inherits the parent process's `OMP_NUM_THREADS` / `MKL_NUM_THREADS`. If the parent didn't export them (per CLAUDE.md §5.10), the calibrator subprocess runs single-threaded. Recommend explicit env propagation.

---

## Findings tally (updated)

| ID | Task | Severity | Status |
|---|---|---|---|
| 11 | FactorZScoreTask raw_cols hand-maintained | 🟡 | OPEN |
| 12 | FactorZScoreTask iloc[-1] NaN unguarded | 🟡 | OPEN |
| 13 | LabelsTask lookahead default silent | 🟡 | OPEN |
| 14 | CV adapters use //2 num_boost_round → CV ≠ FinalFit | 🔴 | OPEN |
| 15 | xgb eta vs learning_rate precedence ambiguity | 🔴 | OPEN |
| 16 | RefreshPanelCalibrator failure non-fatal → CAL-7 redux | 🔴 | OPEN |
| 17 | NGBoost CV n_estimators //2 (lower severity Issue 14) | 🟡 | OPEN |
| 18 | bare except swallowing NGBoost fit errors silently | 🟡 | OPEN |
| 19 | calibrator subprocess inherits no thread env | 🟡 | OPEN |

---

## RankingJob

### BlendScoresTask
File: `kernel/pipeline/task_ranking.py:12`
✅ rs_score channel deprecated — w_rank=1.0, w_rs=0.0 hardcoded; legacy `ranking.blend_weights` config raises a warning if it would have given rs_score weight.
🟡 **Concern 22**: `score_candidates(ctx.candidates, w_rank, w_rs)` — when `w_rs=0.0`, `score_candidates` reduces to a min-max normalization on rank_score. If all candidates have identical rank_score (collapsed calibrator → constant pool), normalization explodes (divide by zero) or maps everything to a single value. Need to confirm `score_candidates` handles zero-spread input. (Saw this exact pathology in B3a band-aid — calibrator constant output.)

### SortCandidatesTask
File: `kernel/pipeline/task_ranking.py:46`
✅ RA-1 fix (NaN treated as -inf so NaN candidates always sink to the bottom deterministically) in place.
🟡 **Concern 23**: stable-sort guarantee depends on Python's `sorted(reverse=True)` (Timsort, stable). If ties exist (e.g. two candidates with identical rank_score), insertion order wins — which is whatever order ctx.candidates came in. No documented invariant on candidate insertion order, so a small upstream re-ordering can flip the top-K. Consider adding ticker as secondary key for determinism.

---

## RotationJob

### BuildPairsTask
File: `kernel/pipeline/task_rotation.py:60`
✅ RG-NaN fix across panel/thesis/Kelly gates — non-finite values fall back to KEEP rather than silent REJECT.
✅ Three rotation modes implemented with explicit dispatch: `"er"` (default), `"thesis_primary"`, `"thesis_symmetric"`.
✅ thesis_symmetric mode warns loudly when `ctx._db is None` (LEAN no-op path documented).
🟡 **Concern 24**: `_drive_score` accepts `"er"`, `"mu_minus_lambda_sigma"`, `"sharpe"` — but a typo (`"mu-lambda-sigma"`, `"Sharpe"`) silently falls through to `expected_return`, masking config errors. Recommend strict whitelist + raise on unknown modes.
🟡 **Concern 25**: `held_diag` decision-tree log only emits for top-K candidates (line 600). If a held was rejected by say `min_hold` and an off-top-K candidate could have rotated in, the audit log never records the comparison. Acceptable but worth surfacing as "non-exhaustive log".

### ValidatePairsTask
File: `kernel/pipeline/task_rotation.py:633`
✅ Re-runs wash/sector/correlation guards on virtual post-swap holdings — the right invariant.
🟡 **Concern 26**: virtual_held construction (line 662) intersects `validated` accumulator + `pair`. Order-dependent: if pair A would block pair B's sector cap, but B comes first, B is accepted and A rejected — different ordering would invert. ctx.rotations comes from BuildPairsTask which sorts by net_advantage descending (deterministic), so the order should be reproducible, but worth documenting.

### EmitRotationsTask
File: `kernel/pipeline/task_rotation.py:690`
✅ ROT-NaN-PRICE: skip ENTIRE pair on bad price (no orphan exit).
✅ PR1-CASH: rolling cash (credit sell, debit buy) inside the loop — fixes the multi-rotation cash-overclaim bug.
✅ Atomic-rotation invariant: buy committed BEFORE exit appended (prevents orphan exits).
✅ ROT-BLOCKED-NTFY: `ctx.rotations_blocked` accumulates so live runner can surface in ntfy.
🟡 **Concern 27**: `held_price = float(ctx.prices.get(pair.sell_ticker, 0.0) or 0.0)` (line 832) — if held's price is missing, sell_proceeds = 0 (entire held value lost in sizing budget). The buy then gets sized as if cash was tighter than reality. After the broker actually ships the sell, real cash exceeds sized cash. Rare edge (held without prices is unusual) but worth a `log.warning` so it's visible.

---

## SelectionJob

### PrepareSelectionTask
File: `kernel/pipeline/task_selection.py:12`
✅ effective_held = (holdings - rotation_sells) | rotation_buys — correctly accounts for already-emitted rotations.
✅ open_slots ≤ 0 returns False (short-circuits the rest of SelectionJob).
✅ BEAR cap honored.
🟡 **Concern 28**: `int(regime_params.get("max_concurrent_positions", config.get("max_concurrent_positions", 8)))` — silent fallback chain. If a config drops `max_concurrent_positions` (typo, missing key, etc.), runs with 8. Recommend log.info on the resolved value at least once per pipeline run.

### RunSelectionTask
File: `kernel/pipeline/task_selection.py:71`
✅ blocked_by_ticker dict populated and stamped to ctx — fed to decision-trace DB.
✅ Counters incremented per-block-reason (wash/sector/corr/defensive_non_bear).
🟡 **Concern 29**: not auditing `kernel.selection.run_selection_loop` body in this pass; recommend a deeper audit of the greedy admission criteria, especially how `tiered_thresholds` interacts with BEAR-only mode.

### SizeAndEmitTask
File: `kernel/pipeline/task_selection.py:100`
✅ SE-1 NaN-price guard in place (treats non-finite price like None).
✅ Rolling cash invariant: remaining_cash decremented after each buy, with explicit assert (`invest > remaining_cash + 1e-6`).
✅ CONF-MULT floored confidence multiplier in place.
✅ Per-session cap honored.
🔴 **Issue 17**: `conv = conviction_multiplier(getattr(c, "panel_score", None), sizing_cfg)` (line 191-193) — uses RAW panel_score (uncalibrated) as conviction input. The VetoWeakBuysTask P0 fix moved to `rank_score` (calibrated). For sizing to be consistent with veto, conviction_multiplier should also use calibrated rank_score (or document why raw panel_score is the correct conviction input). If model scores are bunched (e.g. row_coverage gate kicks in), raw panel_score range can compress, and conv collapses to a uniform multiplier — silent under-sizing. (Same code path is duplicated in EmitRotationsTask line 794-795.)
🟡 **Concern 30**: `conv` and `sig_m` default to 1.0 in pure-Kelly mode, but in legacy mode they multiply max_pct. If conviction_multiplier returns NaN (panel_score = NaN), `max_pct = base × NaN × sig_m = NaN`, then `int(NaN_invest)` raises and silently kills the order. No isfinite check on conv/sig_m output before multiply.

---

## TopUpHeldJob

### TopUpHeldTask
File: `kernel/pipeline/task_topup.py:38`
✅ TU-1..TU-4 NaN guards across kelly_target, price, portfolio, cash.
✅ Bug 26 cash-cap fix in place.
✅ topup_conviction_floor (default 0.20) — fail-CLOSED on missing/low rank.
✅ 2026-05-01 fix: respects EarningsFilter (symmetric ±buffer).
✅ per_session_cap honored.
🟡 **Concern 31**: order_type="TOP_UP" but `conviction=1.0, sigma_mult=1.0` hardcoded (line 195-196). Inconsistent with SizeAndEmitTask which computes dynamic multipliers. May be intentional (TopUp is mechanical Kelly maintenance, not new entry conviction) — but if a TopUp is later analyzed in PR analysis, the 1.0 conviction hides the holding's actual rank. Recommend reading hs.rank_score → conviction for consistency.

---

## TrimHeldJob

### TrimHeldTask
File: `kernel/pipeline/task_trim.py:43`
✅ TR-NaN NaN guards mirror Size + TopUp patterns.
✅ Default OFF (`trim_enabled=False`) per AB-trim 2026-04-24 finding (any trim threshold regresses APY vs no-trim).
✅ μ ≤ 0 guard skips trim when model turned bearish (let full-exit path handle).
✅ "trim shares >= current_shares → skip and let sell paths handle full exit" — preserves trim-vs-exit distinction.
🟡 **Concern 32**: skipped during BEAR/skip_buys (line 65). Asymmetric with TopUp (also skipped). If a position is over-weight and we hit BEAR, the over-weight stays. Documented as intentional ("rebalancing in risk-off is counter-productive") but worth surfacing in a regime-transition test.

---

## Findings tally (final updated)

| ID | Task | Severity | Status |
|---|---|---|---|
| 17 | SizeAndEmitTask uses raw panel_score for conviction (vs calibrated) | 🔴 | OPEN |
| 22 | BlendScoresTask zero-spread normalization undefined | 🟡 | OPEN |
| 23 | SortCandidatesTask ties non-deterministic | 🟡 | OPEN |
| 24-28 | Various rotation / selection silent-fallback concerns | 🟡 | OPEN |
| 30 | conv/sig_m NaN propagation in SizeAndEmit | 🟡 | OPEN |
| 31 | TopUp hardcoded 1.0 conv vs ranked conviction | 🟡 | OPEN |
