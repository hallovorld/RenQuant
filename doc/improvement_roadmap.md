# RenQuant 104 — Planned Work

Single source of truth for what's next. Ordered by expected value.
When starting an item, flip status to **🟡**. When done, flip to **✅** and drop a
1-paragraph result + commit sha. Archive anything closed to the History
section at the bottom.

**Current golden state** — `strategy_config.golden.json` @ commit `ee4faab`:
after-tax +40.1% APY / +111.6% total / 77% win rate / 37d avg hold / 22d
longest-no-trade-streak on the 27-month 2024-01-01 → 2026-03-26 OOS sim.
Panel-LTR OOS IC (CPCV 15-split) = +0.0411.

Full golden-config details: `doc/golden_config_2026-04-23.md`.
Per-run training history: `doc/panel_training_runs.md`.
Panel-LTR methodology primer: `doc/panel_ltr_primer.md`.

---

## Active queue (next pickup → drop to the bottom)

Ordered top-to-bottom by pickup order. Block letter is just a handle —
pick the topmost unblocked item when starting the next session.

| # | Item | Priority | Est. | Expected value |
|---|---|---|---|---|
| ~~A~~ | ~~Clear stale live_state `high_water_mark`~~ | ✅ done — `ab1006d` | 10 min | `resolve_hwm()` helper snaps HWM down when stored > 1.5× equity. 10 regression tests. Smoke-tested. Next scheduled daily_104.sh will pick it up. |
| ~~B~~ | ~~LightGBM backend A/B vs XGBoost on current golden~~ | ❌ shelved — `8d6b08a` | 4 h | LGBM regressed APY 34.4% → 21.7% (−12.7 pts) and win rate 84% → 74%. Per-row-weight bug fixed on the way in (regression test shipped). LGBM infra stays for future rerun (post-G). |
| C | σ-penalty λ sweep: λ ∈ {0.1, 0.25, 0.5} | MED — fast config test | 1-2 h sim + analysis | Task-#2 A/B showed λ=1.0 crashes APY (32% → 5%); there may be a non-zero optimum. Low-risk because code refactor already shipped; this is config-only. |
| ~~D~~ | ~~Sustainability watch on T4 golden — 30-day live APY tracker~~ | ✅ done — (this commit) | 2 h | daily_104.sh appends audit JSONL; `scripts/weekly_apy_check.py` computes rolling 30d APY + drawdown-streak, ntfy alert below 25% / 20% / 5d. Launchd plist loaded Sun 12 PT. Smoke-tested 4 cases. |
| E | Re-fit global calibrator on μ−λσ distribution | MED — only if C wins | 1 day | Conditional on C finding a winning λ. Metric-calibrates the μ−λσ mode instead of relying on the directionally-monotone raw-panel calibrator. |
| F | Item 6 — regime-conditional calibration | MED — waits on DB data | 2 days | Per-regime calibrators (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR). Wants ≥ 2 months of decision traces in `data/runs.db` first. |
| G | Item 8 — hourly-bar panel features (keep as eventual, not deferred) | MED — larger build | 2-3 days | +0.01 IC target per Item 8 spec. User wants panel trained on hourly data as a real objective, independent of the transformer track. Does not depend on A-F finishing; can run whenever there's a 2-3 day block. Uses Alpaca intraday fetcher from Item 5. |
| H | Revisit transformer on the hourly-enhanced panel | downstream of G | 1 day | After G lands, the panel grows in both features AND rows (if we also fetch hourly bars across the full 10yr sample). Rerun `scripts/compare_panel_backends.py`. Current transformer baseline shelved at ratio 0.49× XGBoost on the 47k daily-only panel. |

---

## Detailed specs

### A. Clear stale live_state `high_water_mark`  🟢 HIGH

**Problem:** `backtesting/renquant_104/live_state.json` has `high_water_mark: 100_000` from the original adapter seed. Real Alpaca account equity is lower → `DrawdownCircuitTask` computes drawdown = 89.9% ≥ 35% halt → `buy_blocked=True` every bar. Today's e2e `daily_104.sh` run fired the halt and placed zero orders.

**Fix options:**
1. One-shot: read current Alpaca equity via the broker API at the start of the next live run and overwrite `high_water_mark` if it's wildly above current equity.
2. Passive: let `HWMUpdateTask` ratchet naturally once live portfolio exits BEAR/halt and hits a new peak.

**Recommended:** option 1, with a safety guard — if stored HWM > 1.5× current equity, snap to current equity. This is a one-line change in `adapters/runner.py::make_context` or `live/runner.py`. Add a regression test: synthetic context with HWM=150k + equity=50k should snap HWM to 50k at next bar.

**Acceptance:** next `daily_104.sh` invocation shows `DrawdownCircuitTask` log message `drawdown=X.X%` with X well below 35%, and at least one buy order is attempted (market open, no other gates blocking).

### B. LightGBM backend A/B  🟢 MED

**Problem:** Item 7 shipped `PanelLGBMModel` + `PanelLGBMScorer` + dispatcher but never measured live panel performance. NDCG@10 objective (LightGBM's `lambdarank`) should match the top-8 selection better than XGBoost's `rank:pairwise`.

**Plan:** use `scripts/compare_panel_backends.py --strategy renquant_104` but extend it to accept `--backends xgboost,lightgbm`. Need to pin LightGBM hparams first (current config has `lightgbm_params` populated but untested). Do one warm-up LightGBM retrain, tune if training fails, then A/B.

**Acceptance:** LightGBM reaches OOS IC ≥ 1.10× XGBoost (per ship gate). If yes, flip backend in golden. If 1.00-1.10×, ensemble (infra exists — `kernel/panel_pipeline/ensemble_scorer.py`). If < 1.00×, shelve.

### C. σ-penalty λ sweep  🟢 MED

**Problem:** Task-#2 A/B tested only λ_σ=1.0 and it regressed −27 APY pts. There could be a small-but-positive λ where μ−λσ ranking beats additive. The code refactor (commit `339944b`) already allows flipping `score_mode=mu_minus_lambda_sigma` via config.

**Plan:** same harness as `/tmp/test_pending_reverts.py` but iterate over `lambda_sigma` ∈ {0.0, 0.1, 0.25, 0.5, 1.0}. The 0.0 case ≡ additive (baseline). Keep `score_mode=mu_minus_lambda_sigma` across all runs so calibration-after-NGBoost is consistent.

**Acceptance:** at least one λ beats additive by ≥ 3 APY points on the 27-month OOS window AND doesn't drop win-rate below 80%. If yes → commit + promote golden. If no → lock additive permanently, delete this experiment branch.

### D. Sustainability watch on T4 golden  🟢 MED

**Problem:** T4's panel hyperparameters (depth=3, mcw=60, λ=5, α=2, rounds=300) are historically validated but the current 27-month backtest is largely in-sample for the per-ticker tournament models. The 40.1% APY could be backtest-specific. The golden doc logged a 30-day live-trading guardrail.

**Plan:**
- Add a daily audit step in `scripts/daily_104.sh` that appends a JSONL row to `logs/live_104/audit.jsonl`: `{date, equity, hwm, n_positions, n_orders_today, regime, confidence}`.
- Weekly: compute rolling 30-day APY from the audit stream, emit a ntfy alert if below 25% (halfway between T4's 40% and the v1 golden's 33%).
- If 30-day live APY < 30% sustained, revert to v1 golden (`doc/golden_config_2026-04-23.v1.md`).

**Acceptance:** audit file populated for ≥ 5 consecutive trading days, alert path manually tested.

**Can run in parallel with A/B/C** — no sim dependency; it's a logger + cron.

### E. Re-fit global calibrator on μ−λσ distribution  🟢 MED — conditional on C

**Problem:** Current `scripts/fit_panel_calibrator.py` fits the isotonic head on raw panel scores. In `mu_minus_lambda_sigma` mode, the calibrator's input distribution shifts — μ−λσ has similar scale but different shape vs raw panel scores. Today's A/B reuses the raw-panel calibrator (directionally monotone but not metric-calibrated).

**Plan:** extend `scripts/fit_panel_calibrator.py` with a `--input-distribution` flag: `raw` (current) or `mu_minus_lambda_sigma`. The latter path needs to compute μ, σ for each (ticker, date) via `ngboost_head.predict_distribution`, form μ − λ·σ with the configured λ, pool, and fit isotonic against future excess returns. Artifact saved as `panel-rank-calibration-musigma-L{lambda}.json`.

**Acceptance:** two calibrator artifacts (raw + musigma) coexist. `ApplyGlobalCalibrationTask` loads the right one based on `score_mode`. Sim with μ−λσ mode + fitted-for-it calibrator beats the directionally-monotone baseline by ≥ 1 APY point.

**Depends on:** C (don't fit a calibrator for a mode we've decided to shelve).

### F. Item 6 — regime-conditional calibration  🟡 partial

**Status:** CPCV 15-split shipped (commit `c5fd154`). Regime-conditional calibrator part is still pending.

**Plan:**
- `training_panel/global_calibrator.py` gains `fit_regime_conditional(panel_scores, returns, regime_series) → dict[regime, GlobalPanelCalibration]`.
- `ApplyGlobalCalibrationTask` picks the calibrator by `ctx.regime`, falls back to pooled calibrator when regime absent.
- Artifacts: `panel-calibration-BULL_CALM.json`, `..._BEAR.json`, etc.

**Acceptance:** 4 regime calibrator JSONs produced. Sim with regime-conditional beats pooled by ≥ 2 APY points (target justifies the 2-day build).

**Depends on:** DB (`data/runs.db` — Item 3 shipped) should have at least 2 months of real-world decision traces before per-regime fitting is data-driven enough.

### G. Panel trained on hourly data (was Item 8)  🟢 MED — keep in queue, not deferred

**User objective:** train the panel model — whether XGBoost or transformer — on
hourly-bar data, not just daily. This is a real planned item, not a fallback.
Timing is flexible (no deadline) but it stays on the queue; pick it up after
A-F when there's a 2-3 day block.

**Plan:**
1. Extend `kernel/data.py::fetch_intraday_bars` (Item 5 infra) to cache hourly bars
   across the full 10yr sample at `data/intraday/{SYMBOL}/1h.parquet`.
2. Add `training_panel/hourly_features.py` with 6 per-(ticker, date) aggregates:
   - `morning_drift`   = (hr1_close − open) / open
   - `afternoon_drift` = (close − hr1_close) / hr1_close
   - `vwap_premium`    = (close − intraday_vwap) / intraday_vwap
   - `vol_ratio`       = last-hour volume / first-hour volume
   - `intraday_realized_vol` = std of 7 hourly returns
   - `overnight_gap`   = (open − prev_close) / prev_close
3. New `HourlyFeatureTask` in `PanelDataJob` (runs once per day after close).
4. Panel feature set grows 25 → ~31 columns.
5. Retrain + A/B vs daily-only baseline.

**Acceptance:** OOS mean-IC improvement ≥ +0.01 vs current golden (0.0411 → ≥ 0.051).
Training time stays under 20 min end-to-end. Hourly features together account for
> 10% of feature importance.

**Sequencing note:** H's transformer rerun is a natural follow-on once G lands —
the enlarged panel is exactly the data-growth condition the transformer was
waiting on.

### H. Revisit transformer on the hourly-enhanced panel  🟢 downstream of G

**Trigger:** G lands (panel trained on hourly bars → more features + likely more
rows once the 10yr hourly cache is built).

**Plan:** flip `panel_ltr.backend: "transformer"` in config, run
`python scripts/compare_panel_backends.py --strategy renquant_104 --device mps`.
Current infra + tests (27 green) are preserved. The transformer's attention
across a date-group should benefit most from the intraday structure that
XGBoost's axis-aligned splits can't fully exploit.

**Ship gate:** transformer OOS IC ≥ 1.30× XGBoost on the hourly-enhanced panel
(design doc §5 — unchanged). If 1.10-1.30×, ensemble via rank-averaging
(`kernel/panel_pipeline/ensemble_scorer.py`).

---

## Watch items (not planned work, but monitor)

- **Training audit table schema drift.** `SaveArtifactTask` logs warning `table training_runs has no column named elapsed_sec`. `kernel/persistence.py` schema doesn't have `elapsed_sec` but the code tries to write it. Low-impact (try/except swallows) but noisy. Fix: one-line schema extension + `ALTER TABLE` migration.
- **Panel IC variance across daily retrains.** Today's A/B sim baseline produced 32% APY on a panel with IC=0.0363, vs yesterday's explicit retrain at IC=0.0411 / 40.1% APY. Same T4 xgb_params. Root cause: today's daily_104 retrained per-ticker tournament models first, which changes the feature frames feeding the panel. Variance is expected but worth tracking — if IC swings >0.005 day-to-day, the signal is noisier than we'd like.
- **Recalibrate write-after-read race.** Fixed in commit `bc81360` (merge-on-write shield). Sanity-check weekly via the JSONL audit log.
- **`NoCandidateAlert` streaks ≥ 20 days in BEAR regime.** Normal behavior (BEAR blocks offensive buys) but occasionally fires outside BEAR due to ScoreBuyTask per-ticker models being conservative. Monitor via `monitoring.max_no_candidate_days` alert; 15d limit is current.

---

## Working rhythm

1. Pick topmost item from Active queue.
2. Flip to 🟡, announce in chat.
3. Ship smallest reversible step + commit before next step.
4. Test vs golden (sim + sustainability check for any APY-affecting change).
5. If item lifts APY ≥ 2 pts: promote golden (`strategy_config.golden.json` + `doc/golden_config_*.md`) in same commit.
6. Flip to ✅ with 1-paragraph result + commit sha.
7. Move item to History.

**Stop rule:** if an item's 2× cost budget blows and no path to completion, mark as blocked with diagnosis, move to Deferred.

---

## Completed (archive)

Condensed summaries. Full spec for each is in the commit body + per-doc trails.

| # | Item | Commit(s) | Result |
|---|---|---|---|
| 1 | Run 3 — lookahead=10d + regularization sweep | `5fdba09` prep → superseded by T4 | OOS IC 0.025 → 0.040; all 5 folds positive. Train/OOS ratio 16× → 8×. |
| 2 | Global calibrator on panel | (pre-session baseline) | Pool IC 0.071 on 89k rows. `ApplyGlobalCalibrationTask` + `fit_panel_calibrator.py` shipped. Enabled live. |
| 3 | SQLite decision-trace database | (pre-session baseline) | 5 tables + 6 canned queries + sim/live hooks. `data/runs.db` enabled. |
| 4 | BaselineTournament winner by IC | (pre-session) | `oos_single_ticker_ic` + `winner_metric` flag. Default `sharpe` preserved. |
| 5 | Alpaca minute bars + intraday sell | (pre-session) | IEX feed, 20 slots 07:00-12:30 PT Mon-Fri via plist. Smoke-tested 44/44 overlay. |
| 7 | LightGBM LambdaRank backend | (pre-session) | `PanelLGBMModel` + dispatcher. Default still XGBoost. **Not A/B'd yet — re-queued as F above.** |
| 9 | Cross-sectional transformer panel backend | `8f38f80`, `7eb05db`, `58ff882`, `99dd08d`, `2c06d19`, `fff97af`, `908019a` | Shelved at ratio 0.49 vs XGBoost on 47k-row panel. Full infra kept (model / scorer / ensemble / A/B driver / 27 tests) for future rerun when panel grows. |
| — | **Universe floor regression fix** | `2df4e21` | `_eval_sharpe` prefers tournament `sharpe` over noisy `live_holdout_sharpe`. Floor raised 0.5 → 1.0. Admits 36/52 vs 21/52 previously. APY 2.4% → 10.1%. |
| — | **ConfidenceVeto disabled** | `33c0e9b` | Threshold 0.30 → 0.0. GMM posterior structurally capped at ~0.25, so any threshold ≥ 0.26 was an always-on veto blocking all offensive buys. |
| — | **DrawdownCircuitTask resets on recovery** | `e586018` | THE bug behind the 153-day no-trade streak. `skip_buys` used to latch on and never clear. Now recomputed each bar with optional hysteresis via `drawdown_resume_pct`. APY 10.9% → 33.1%. |
| — | Golden v1 snapshot (33.1% APY) | `d3ef68f` | First post-fix golden baseline. Archived as `doc/golden_config_2026-04-23.v1.md`. |
| — | **T4 — xgb_params revert** | `ee4faab` | Reverted panel `xgb_params` to pre-regression values (depth 2→3, mcw 100→60, λ 10→5, α 5→2, rounds 150→300). APY 33.1% → **40.1%**. OOS IC 0.0397 → 0.0411. Win rate 72% → 77%. **Current golden.** |
| — | recalibrate_scores.py race-condition shield | `bc81360` | Merge-on-write instead of dump-whole-config. Protects golden from scheduled Tue/Thu/Sun recalibrate overwriting concurrent edits. |
| — | daily_104.sh notification surfaces panel info | `d71ee5d` | "Models retrained: N" ntfy alert now includes `panel@DATE IC=±std | ngb@DATE n=ROWS` so phone can see the panel retrain metadata. |
| — | PanelScoringJob reorder (NGBoost before calibration) | `339944b` | Architectural cleanup. Removes the mu_minus_lambda_sigma short-circuit footgun. No-op in additive mode (provably). Enables future σ sweeps as config-only experiments. |
| — | Task #2 A/B (mu_minus_lambda_sigma + λ=1.0) | `913c929` (doc only) | Shelved at −27 APY pts. λ=1.0 too aggressive. See Active queue C for a narrower sweep. |

---

## Deleted / no-action

- **Revert `lookahead_days` 10 → 5** — prior evidence (Run 3 writeup in `doc/panel_training_runs.md`) showed 5d regresses. No test needed.
- **Revert `ngboost.score_mode` to `mu_minus_lambda_sigma` at λ=1.0** — A/B ran, regressed. See Active queue C for a survivable reformulation.
