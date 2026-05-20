# RenQuant 104 Deep Audit - 2026-05-15


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

Scope: actual `renquant_104` code, config, artifacts, local data, live telemetry, and targeted reproductions. Documentation was used only as a map; findings below are from executable code, artifact contents, and local databases.

## Executive Judgment

RenQuant 104 has evolved from a model-driven strategy into a layered production system: per-ticker models admit candidates, panel-LTR scores the cross-section, a global calibrator maps raw scores, Kelly/NGBoost fields are partly present, and a Markowitz QP now emits portfolio actions.

The largest risk is not one isolated bug. It is that several layers no longer agree on what a score means:

- Live candidate probabilities are saturated near `1.0`.
- Kelly is enabled but NGBoost is disabled, so Kelly targets are mostly zero.
- The QP still buys using raw panel scores as if they were expected returns.
- The QP silently falls back to diagonal covariance because it reads the wrong correlation path.

That means the current live behavior is dominated by raw XGBoost margin plus QP constraints, while much of the visible probability/Kelly telemetry is no longer decision-useful.

## Severity Legend

- P0: Can directly send wrong live orders or defeat a core safety invariant.
- P1: Material systematic trading/portfolio risk or regime behavior mismatch.
- P2: Broken validation, experiment integrity, or important operational reliability issue.
- P3: Hygiene, confusing config, or non-critical drift.

## Findings

### P0 - Data freshness gate can approve yesterday's data after today's close

Evidence:

- `backtesting/renquant_104/kernel/pipeline/task_data_freshness.py:119-133`
- `backtesting/renquant_104/kernel/data.py:154-167`
- Reproduction:
  - `2026-05-15 => 2026-05-14`
  - `2026-05-16 => 2026-05-15`

The gate computes the "last completed NYSE close" as the last session strictly before `ctx.today`. On Friday 2026-05-15, it expects Thursday 2026-05-14 even after the Friday close. `daily_104.sh` runs live trading after close, so the system can trade the after-close daily pass using yesterday's cache and still pass freshness.

Impact:

- The exact incident the gate was designed to prevent can recur on any trading day after close.
- It also exists in `LocalStore.has_range`, so cache fetches can decide not to refresh.

Fix:

- Make freshness time-aware, not date-only. If the run timestamp is after the NYSE close for the current session, require today's bar.
- For after-close daily live trading, require `max(ohlcv.index.date) >= current_session_date`.
- For intraday sell-only runs, either require today's intraday overlay for held tickers or explicitly mark daily bars as pre-close data.

### P1 - QP covariance is silently disabled by a wrong artifact path

Evidence:

- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py:89-98`
- Config/adapters use `artifacts/prod/watchlist-correlation.json`.
- `ComputeFullSigmaTask` hardcodes `artifacts/watchlist-correlation.json`.
- Reproduction:
  - flat path exists: `False`
  - prod path exists: `True`
  - `ctx._qp_Sigma_full is None`: `True`

Impact:

- `rotation.joint_actions.qp_use_full_sigma` defaults true, but production QP falls back to diagonal covariance.
- Correlation risk is not modeled in the Markowitz objective; only later hard pair/group constraints can still operate.
- Ledoit-Wolf shrinkage is a no-op when `_qp_Sigma_full` is `None`.

Fix:

- Use `ctx.corr_matrix` already loaded by the adapters, or resolve the configured `regime.correlation_artifact`.
- If `qp_use_full_sigma=true` and the configured artifact is absent, fail loudly or at least hard-warn in preflight.

### P1 - Joint QP ignores `rotation.enabled_regimes`

Evidence:

- Legacy `RotationJob` respects the allow-list at `backtesting/renquant_104/kernel/pipeline/job_rotation.py:32-44`.
- `JointPortfolioQPJob.should_skip()` does not check it at `backtesting/renquant_104/kernel/portfolio_qp/job_qp.py:136-147`.
- Reproduction with `enabled_regimes=["BULL_CALM"]`:
  - `BULL_CALM should_skip=False`
  - `BULL_VOLATILE should_skip=False`
  - `CHOPPY should_skip=False`
  - `BEAR should_skip=True`

Impact:

- The system comment says BULL_VOLATILE rotation caused whipsaw and was disabled.
- With joint QP enabled, that protection does not apply. The QP can rebalance/buy/sell in BULL_VOLATILE and CHOPPY even though the config says rotation is allowed only in BULL_CALM.

Fix:

- Add the same `rotation.enabled_regimes` check to `JointPortfolioQPJob.should_skip()`.
- Add a regression test for joint QP specifically; existing tests cover only legacy `RotationJob`.

### P1 - Current live calibrated scores are saturated and no longer discriminate candidates

Evidence from `data/runs.alpaca.db`:

- 2026-05-15 score distribution: 40 scored rows, 38 with `rank_score >= 0.999`.
- 2026-05-12 through 2026-05-15: most live candidates map to `rank_score=1.0`.
- Current calibrator support:
  - probability `x` max is `0.5917923450`
  - current raw panel scores reach `0.8950151205`
  - values above about `0.4918` map to `1.0`
- Latest selected live buy:
  - `META panel_score=0.7152 rank_score=1.0 mu=NULL sigma=NULL kelly_target_pct=0.0 blocked_by=kelly_zero:mu_none`

Impact:

- `rank_score`, tier gates, score distribution telemetry, and "probability" language are not currently telling the truth at the high end.
- The QP is still making buys, but those buys are driven by raw panel score/QP mechanics, not by a calibrated probability distribution.
- Buy-floor/adaptive threshold logic becomes weak when nearly every candidate is at the ceiling.

Likely cause:

- The active scorer's runtime raw-score distribution shifted above the calibrator's fitted support, while the calibrator metadata still references an older flat artifact path and fingerprint.

Fix:

- Add preflight that scores the current cross-section and fails/warns if more than a threshold, e.g. 25%, of candidates saturate at the calibrator ceiling.
- Validate calibrator `scorer_artifact_fingerprint` against the active scorer artifact fingerprint, not only `n_unique_prob_y`.
- Refit calibrator against the exact active artifact path and runtime feature path.

### P1 - Kelly is enabled while NGBoost is disabled; BEAR defensive buys can be dead

Evidence:

- Config:
  - `ranking.kelly_sizing.enabled = true`
  - `ranking.panel_scoring.ngboost.enabled = false`
  - `ranking.panel_scoring.sigma_sizing.enabled = true`
- `ApplyKellySizingTask` sets `kelly_target_pct=0` when `mu` or `sigma` is missing.
- `SizeAndEmitTask` skips a selected ticker when Kelly is on and `kelly_target_pct <= 0`.
- Reproduction with a BEAR defensive candidate `TLT`:
  - `kelly_target_pct 0.0`
  - `blocked {'TLT': 'kelly_zero:mu_none'}`
  - `orders []`

Impact:

- In BEAR, joint QP is skipped and the legacy selection path is used.
- Defensive buys can be selected but then size to zero because Kelly has no `mu/sigma`.
- This contradicts the defensive branch's purpose.

Fix:

- In BEAR defensive mode, bypass Kelly sizing or provide a defensive fixed target.
- Alternatively disable Kelly automatically when NGBoost is off.
- Add a test: BEAR regime + defensive candidate + Kelly enabled + NGBoost disabled must emit a defensive order when cash allows.

### P1 - QP objective uses raw panel margins as expected returns with default sigma

Evidence:

- `_BuildMuVectorTask` uses `attr="mu"` with fallback to `panel_score` in `backtesting/renquant_104/kernel/portfolio_qp/job_qp.py:59-69`.
- `_BuildSigmaVectorTask` defaults missing `sigma` to `0.05` in `job_qp.py:76-85`.
- `ranking.alpha_to_mu` is missing from config.
- `ranking.qp_mu_source` is missing from config.
- NGBoost is disabled, so `mu/sigma` are absent for current live candidates.

Impact:

- The QP treats arbitrary XGBoost ranking margins as return forecasts.
- Risk aversion, turnover cost, and min-invested constraints are calibrated against a made-up scale.
- Because full covariance is also disabled by the path bug, the optimizer is closer to "ranked score allocator with constraints" than a real risk/return optimizer.

Fix:

- Pick one coherent contract:
  - raw score -> z-score -> `mu = IC * sigma * z(score)` via `alpha_to_mu`, or
  - enable NGBoost and use its `mu/sigma`, after validating the scale, or
  - do not call the solver Markowitz/QP; call it constrained rank allocation.
- Make `qp_mu_source` explicit in config and preflight-print the active source.

### P2 - Active artifact and calibrator metadata are not acceptance-grade

Evidence:

- Active scorer `artifacts/prod/panel-ltr.alpha158_fund.json`:
  - `oos_mean_ic = None`
  - `train_ic = None`
  - metadata only contains `_fingerprint_restamped_2026_05_11`
- Active calibrator metadata:
  - `scorer_artifact = .../artifacts/panel-ltr.alpha158_fund.json`
  - active scorer path is `.../artifacts/prod/panel-ltr.alpha158_fund.json`
  - `scorer_artifact_fingerprint = sha256:4f1e25989d475225`
  - active scorer fingerprint in config check is `sha256:e885d0d305e3fd2a`
- `daily_104.sh` reports model IC as `—` because the active artifact lacks top-level IC.

Impact:

- Operators cannot see active model quality from the production artifact.
- Preflight checks `best_iter`, config fingerprint, and calibrator uniqueness, but does not prove scorer/calibrator pairing or current score support.

Fix:

- Stamp active scorer with `oos_mean_ic`, `training_train_ic`, `run_id`, feature schema hash, and calibrator fingerprint.
- Add preflight checks for scorer-calibrator pair identity and current score saturation.

### P2 - Parallel timeout settings are ineffective

Evidence:

- Inference: `backtesting/renquant_104/kernel/pipeline/pipeline.py:109-115`
- Training: `backtesting/renquant_104/training_panel/pp_panel_training.py:177-184`
- Both use `as_completed(futures, timeout=None)` and then call `fut.result(timeout=timeout_seconds)` after the future has already completed.

Impact:

- `parallel_ticker_timeout_seconds` does not protect against a hung ticker task.
- A hung worker can stall inference/training indefinitely.

Fix:

- Use `concurrent.futures.wait(..., timeout=deadline)` or drive futures with explicit deadlines.
- For tasks that can hang in native/network code, prefer process isolation or enforce timeouts inside the task itself.

### P2 - Production `sample_end` is in the future and changes OOS semantics

Evidence:

- `sample_end = 2026-06-30`
- Current audit date is 2026-05-15.
- `resolve_oos_cutoff()` anchors to `sample_end` when present.
- Reproduction:
  - `sample_end 2026-06-30`
  - `cutoff 2026-04-01`
  - OHLCV max mostly `2026-05-14`, with three symbols at `2026-05-15`.

Impact:

- The per-ticker tournament thinks the OOS window is anchored to 2026-06-30, but actual data ends in mid-May.
- The intended 90-calendar-day OOS window is effectively much shorter until data catches up to the future sample end.

Fix:

- For live/prod retraining, set `sample_end` to absent/today or clamp to the minimum data max date.
- Fail training if `sample_end > available_data_max + tolerance`.

### P2 - Panel fetch ignores configured `sample_start`/`sample_end`

Evidence:

- `FetchPanelDataTask` calls `fetch_ohlcv(sym, provider=provider)` without `start` or `end` at `backtesting/renquant_104/kernel/pipeline/pp_training_full.py:159-162`.
- Config contains `sample_start=2016-01-01`, `sample_end=2026-06-30`.

Impact:

- Training consumes whatever local cache currently contains.
- Reproducibility depends on cache history rather than config.

Fix:

- Pass `start=config.get("sample_start")` and `end=min(sample_end, today/data_max)` explicitly.
- Log per-symbol actual date range and fail on materially inconsistent ranges.

### P2 - Universe staleness uses wall-clock date, not sim/backtest date

Evidence:

- `FilterStalenessTask` uses `date.today()` at `backtesting/renquant_104/kernel/pipeline/job_universe.py:75-96`.

Impact:

- Historical sim/LEAN runs can admit or reject models based on the machine's current date, not the simulated date.
- This is a reproducibility and parity risk.

Fix:

- Add `as_of_date` to `UniverseContext`.
- Live uses wall-clock; sim/LEAN pass their bar date.

### P2 - Fundamentals are stale/incomplete relative to the current watchlist

Evidence:

- `data/sec_fundamentals_daily.parquet` max date: `2026-02-10`.
- Watchlist coverage: `94 / 103`.
- Missing fundamentals for: `ASML`, `GLD`, `GOOG`, `NVO`, `NVTS`, `NXPI`, `SHOP`, `SPOT`, `TSM`.
- Latest watchlist missing rates:
  - `earnings_yield`: 23.4%
  - `book_to_price`: 24.5%
  - `gross_profitability`: 37.2%
  - `roe`: 2.1%
  - `asset_growth`: 0.0%

Impact:

- The scorer has 5 fundamental features, but runtime often cross-sectionally imputes them.
- Some misses are expected for ETFs/foreign listings, but the system should surface this as model-input degradation rather than quiet median fill.

Fix:

- Add a daily data coverage check to preflight for active feature groups.
- Distinguish "expected missing" tickers like GLD from broken equity fundamentals.

## What Looked Healthy

- OHLCV cache coverage is complete for the current watchlist plus benchmark/sector ETFs:
  - 112 symbols checked.
  - 0 missing.
  - 0 duplicate index rows.
  - 0 OHLC relationship violations.
  - 0 negative volume rows.
  - 109 symbols max at `2026-05-14`; 3 symbols max at `2026-05-15`.
- Earnings surprise cache covers 102 / 103 watchlist names; only `GLD` is missing.
- Universe load currently admits 82 / 103 watchlist names; 21 are rejected by the configured Sharpe floor.
- Focused tests passed:
  - `102 passed, 2 skipped`
  - Command: `python -m pytest tests/test_buy_sell_audit_fixes.py tests/test_qp_grinold_kahn_transform.py tests/test_qp_force_mu_source.py tests/test_kelly_sizing.py tests/test_correlation_guard.py tests/test_preflight.py -q`

## Highest-Value Fix Order

1. Fix date freshness for after-close daily live runs and `LocalStore.has_range`.
2. Fix QP correlation artifact resolution and add preflight for `_qp_Sigma_full`.
3. Make joint QP respect `rotation.enabled_regimes`.
4. Add scorer/calibrator pairing and saturation preflight; refit calibrator against the active prod artifact.
5. Resolve the Kelly/NGBoost/QP contract. Do not leave Kelly enabled with missing `mu/sigma` while QP buys from raw panel margins.
6. Add BEAR defensive regression test.
7. Replace ineffective per-ticker timeout patterns.
8. Clamp/remove future `sample_end` in live training and pass sample bounds into OHLCV fetch.

## Commands And Reproductions

Key commands run:

```bash
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest \
  tests/test_buy_sell_audit_fixes.py \
  tests/test_qp_grinold_kahn_transform.py \
  tests/test_qp_force_mu_source.py \
  tests/test_kelly_sizing.py \
  tests/test_correlation_guard.py \
  tests/test_preflight.py -q
```

Targeted reproductions:

- Freshness date mapping via `DataFreshnessGateTask._last_completed_nyse_close`.
- QP `enabled_regimes` behavior via `JointPortfolioQPJob().should_skip(ctx)`.
- QP covariance path via `ComputeFullSigmaTask`.
- BEAR defensive Kelly-zero path via `ApplyKellySizingTask` + `SizeAndEmitTask`.
- Live score saturation via `data/runs.alpaca.db`.
- Data coverage via local parquet scans.

