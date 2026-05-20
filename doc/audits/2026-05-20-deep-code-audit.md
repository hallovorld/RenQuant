# Deep Code Audit — 2026-05-20

> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Scope**: 6 categories per user mandate — §5.12 mature-lib substitution, dead code, logic bugs, logging gaps, **bugs** (highest priority), missed parallelism. 5 parallel audit agents covered:

1. `kernel/` + `backtesting/renquant_104/kernel/` (~130 files)
2. `training_panel/` + `scripts/train_*` + `scripts/patchtst_*` + cron shell wrappers
3. `live/` + `adapters/` + sim infrastructure
4. `scripts/*` ops + plists + dashboard
5. `tests/` coverage + integration

**Total findings**: ~180 issues across the 5 reports. This doc lists P0 (production-impacting) + P1 (significant) only. Full reports in agent transcripts (2026-05-20 session).

---

## 🔴 P0 CRITICAL — Production-impacting; fix before more sim/eval

### P0-1 Walk-forward splits have ZERO embargo → all current BG eval leaked
- **File**: `kernel/walk_forward_splits.py:79-81` (`assign_split_column`)
- **Bug**: `train = dates < val_start`, `val = val_start <= dates < val_end`. No gap. With default label `fwd_60d_excess`, the last 60 trading days of every train fold have label windows that reach INTO val. Direct train→val label leakage.
- **Impact**: All 5-cut × 5-seed eval running BG since 2026-05-19 evening (HF Trainer baseline + FiLM A/B + DLinear = 75 trainings) is contaminated. Reported per-regime IC numbers (e.g., pt_07 bull_IC +0.098, DLinear cut1 +0.42) are inflated.
- **Comparable arches** (PatchTST vs FiLM vs DLinear) still A/B-meaningful because all share same leakage; **absolute IC numbers** are not.
- **§5.13.5 violation**: `kernel/purged_cv.py::PurgedKFold` (same codebase) has correct embargo+purge implementation. Two parallel splitters with divergent semantics.
- **Fix**: add `embargo_days` parameter (default = `lookahead_days`); `train: dates < val_start - embargo_days`.

### P0-2 FactorZScoreTask look-ahead bias in fund features
- **File**: `backtesting/renquant_104/training_panel/pp_panel_training.py:1542-1551, 1587`
- **Bug**: Loop reads `df[col].iloc[-1]` (last row's value) then writes that scalar back into every historical date via `pd.Series(v, index=idx)`. Affects `earnings_yield`, `roe`, `gross_profitability`, `book_to_price`. Every historical training row sees the MOST RECENT fundamental, not as-of-bar fundamental.
- **Impact**: Production alpha158+fund (172-feature) panel has 4 leaked features baked in. Affects all panel-LTR + NGBoost + calibrator artifacts since this code path shipped.
- **Code comment self-incriminates**: "broadcast scalar — any row works" is the bug talking.

### P0-3 `weekly-wf-promote.plist` Weekday=7 = Sunday not Saturday
- **File**: `scripts/launchd/com.renquant.weekly-wf-promote.plist:38`
- **Bug**: macOS `launchd.plist(5)` spec: `Weekday=0` AND `Weekday=7` both = Sunday; Saturday = `6`. Plist comment claims "Sat=7 in modern macOS launchd convention" — WRONG.
- **Impact**: Weekly walk-forward promote running Sunday 04:00 PT, 24h after intended Saturday. Sibling `com.renquant.weekly-fundamental-refresh.plist` correctly uses `Weekday=6`.
- **Fix**: `Weekday=7` → `Weekday=6` (2-char fix).

### P0-4 `conditional-retrain104.plist` Weekday=2-6 = Tue-Sat not Mon-Fri
- **File**: `scripts/launchd/com.renquant.conditional-retrain104.plist:32-55`
- **Bug**: macOS launchd Mon=1..Fri=5. Plist uses 2-6 = Tue-Sat. Comment says "Mon=2..Fri=6" — WRONG.
- **Impact**: SPY/VIX-anomaly triggered retrain misses Monday shocks, fires on Saturday with stale data. Sibling `daily-news-sentiment.plist` correctly uses 1-5.
- **Fix**: 2-6 → 1-5.

### P0-5 `build_dashboard.py` reads legacy data/ path not artifacts/prod/
- **File**: `scripts/build_dashboard.py:171`
- **Bug**: hardcoded `REPO_ROOT / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"`. Canonical path post-2026-05-11 sim/prod isolation is `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`.
- **Impact today**: both files have identical md5 (`4e1277...`) — coincidence, likely a `_train_*.py` leftover. Next weekly promote only updates `artifacts/prod/`. Dashboard then silently reports stale fingerprint.
- **§5.13.14 violation**: hardcoded artifact filename.
- **Fix**: resolve via `cfg["ranking"]["panel_scoring"]["artifact_path"]`.

### P0-6 Calibrator default = isotonic (docs claim Platt since 2026-05-18)
- **Files**: `scripts/fit_panel_calibrator.py:348-350`, `backtesting/renquant_104/strategy_config.golden.json:967` (`calibration_method="isotonic"`)
- **Bug**: CLAUDE.md + `doc/components/calibration.md` claim "Platt scaling default since 2026-05-18". Code still defaults to isotonic. Either rollout never landed, OR config knob never reaches runtime fitter.
- **Impact**: documented current state ≠ actual code behavior. Need to verify which prod artifact (`panel-rank-calibration.json` metadata `calibration_method`) is in production today.

### P0-7 Test collection errors — `kernel/__init__.py` missing
- **Files**: `tests/test_hmm_regime_labels.py:17`, `tests/test_walk_forward_splits.py:17`
- **Bug**: `ModuleNotFoundError: No module named 'kernel.hmm_regime_labels'`. Root cause: `/kernel/` (top-level) has no `__init__.py`; `backtesting/renquant_104/kernel/__init__.py` exists and shadows the namespace when sys.path includes 104 dir first.
- **Impact**: 2 test files don't collect — confirmed as 2 of the 12 pre-existing failures in CLAUDE.md status.
- **Fix**: `touch kernel/__init__.py` (empty file). Test re-collection automatic.

### P0-8 Six newly-shipped scripts have ZERO test coverage
- **Files**:
  - `scripts/eval_dlinear_5cut_5seed.py` (2026-05-19)
  - `scripts/eval_hf_film_5cut_5seed.py` (2026-05-20)
  - `scripts/eval_hf_trainer_5cut_5seed.py` (2026-05-19)
  - `scripts/compare_arch_5cut_5seed.py` (2026-05-19) — **the aggregator that drives next promote decision**
  - `scripts/dlinear_baseline.py` (2026-05-19)
  - `scripts/verify_sigma_calibration.py` (2026-05-19)
- **Bug**: 0% test coverage. All <72h old. All running BG / about to drive verdicts. A typo in argv handling or path bug is silent.
- **Fix**: 5-line `pytest` smoke per script — import + `--help` works, plus argv parse round-trip.

### P0-9 BUG D still open — settled-only cash on Alpaca
- **File**: `live/alpaca_broker.py:110-113` returns `account.cash` (settled only). `adapters/runner.py:278` consumes as dispatchable budget.
- **Bug**: Alpaca margin allows T+2 unsettled = `account.non_marginable_buying_power`. Sim path includes T+2 pending (sim.py:1482-1486); live path does not.
- **Impact**: Live over-trades when fresh sell proceeds are unsettled (sim under-states cash for 2 bars post-sell vs live).
- **Status**: Known open per CLAUDE.md 2026-05-09 status — still unresolved.

### P0-10 `--broker alpaca` has no LIVE-account positive assertion
- **File**: `live/runner.py:67-68`
- **Bug**: returns `AlpacaBroker(paper=False)` without asserting `account.account_number` matches expected LIVE id. If `.env` has paper-keys accidentally, gets 401 at connect (loud failure — OK), but no positive verification.
- **Impact**: Per 2026-05-17 mandate "我他妈说了一万遍了 LIVE account" — defense in depth missing.
- **Fix**: at connect, log `account.account_number + account.equity`, assert against env-pinned LIVE id.

### P0-11 `strategy_config.shadow.json` violates §5.13.13 — 4 prod artifact paths
- **Tested in**: `tests/test_side_config_artifact_paths.py:114`
- **Bug**: Shadow config has prod artifact paths for ngboost / calibration / panel_ltr / ngboost-head. Running shadow would silently overwrite prod artifacts.
- **Fix**: Rewrite shadow paths to include `.shadow.` segment.

### P0-12 Feature count test drift — prod has 172, test expects 169
- **File**: `tests/test_train_infer_feature_parity.py:161`
- **Bug**: Prod artifact has 172 features (alpha158 + 5 fund + 3 PEAD + 3 SUE + 3 sentiment), test still expects 169 (pre-sentiment).
- **Fix**: derive expected count from registered feature lists, not hardcoded.

### P0-13 DDV attribute name bug (sue_score / pead_score never populated)
- **File**: `backtesting/renquant_104/kernel/panel_pipeline/task_buy_quality_gates.py:195-198`
- **Bug**: `DeepDrawdownVetoTask` reads `_feature(cand, "sue_score")` / `"pead_score"`. Job writes `sue_signal` and `pead_signal`. Name mismatch → `_feature()` returns `None` 100% of the time → `confirmed = False` → DDV vetos every deep-DD candidate.
- **Impact**: Currently masked because DDV globally disabled 2026-05-17 per HXZ 2020. The DAY DDV is re-enabled (planned regime-conditional), the entire DDV-bypass path is dead.

### P0-14 Sunday-sweep cross-contamination — baseline read after train#1 overwrites
- **File**: `scripts/sunday_panel_sweep.py:324-377`
- **Bug**: `_backup_artifacts` runs AFTER each backend train regardless of acceptance gate result. If train#1 trashes `panel-ltr.json` (writes 21-feat stub), train#2's gate-check reads contaminated state.
- **Impact**: Best-by-OOS-IC selection only valid when each backend produces isolated output. Currently not isolated.

### P0-15 `train_ngboost_proper.py` validates against stale XGB baseline + single-seed default
- **File**: `scripts/train_ngboost_proper.py:108, 128`
- **Bugs**:
  1. `XGB_BASELINE_MEAN=0.0294` hardcoded from 2026-05-15 measurement on PRE-wl200 panel. Stale.
  2. Default `NGB_SEEDS=42` (single seed) but reports t-stat against 5-seed XGB baseline → §5.13.4 single-measurement violation.

### P0-16 `monthly_calibrator_refresh.sh` rollback `cp` not atomic
- **File**: `scripts/monthly_calibrator_refresh.sh:186, 192`
- **Bug**: POSIX `cp "$ROLLBACK_CAL" "$PROD_CAL"` is two syscalls. SIGKILL mid-copy → half-written JSON in prod.
- **Fix**: `cp ... .tmp && mv .tmp ...`.

### P0-17 `backup_to_github.sh` no 100MB file-size guard
- **File**: `scripts/backup_to_github.sh:122`
- **Bug**: `git add -A` includes `runs.alpaca.db`. GitHub blocks pushes >100MB. Currently ~45-50MB, growing ~50K/day. Crosses 100MB sometime in 2026 — silent rejection (ntfy doesn't fire because `git push` exits 1 silently).

---

## 🟠 P1 SIGNIFICANT — code quality / efficiency / dormant landmines

### §5.12 mature-lib violations (selected, full list ~15 items)
- `regime.py:97-135` — hand-rolled R/S Hurst; `nolds.hurst_rs` exists
- `regime.py:149-171` — hand-rolled CUSUM; `ruptures` exists
- `regime_hmm.py:163-181` — manual forward algorithm; `hmmlearn.GaussianHMM.predict_proba` exists, `hmmlearn` already in requirements
- `portfolio_qp/qp_solver.py:327-329` — manual matrix sqrt via eigh; `scipy.linalg.cholesky` standard
- `portfolio_qp/tasks.py:184-191` — manual Ledoit-Wolf with hardcoded λ=0.2; `sklearn.covariance.LedoitWolf` derives optimal λ
- `portfolio_qp/qp_solver.py:194-200` — manual Garleanu-Pedersen signal decay; cvxportfolio backend has the canonical class
- `regime_labels.py:47-55` — `_pct_rank` apply-lambda is 50× slower than `pd.Series.rolling().rank(pct=True)`
- `panel_pipeline/hf_patchtst_scorer.py:31-37` — manual CSRankNorm; `qlib.processor.CSRankNorm` is canonical
- `panel_pipeline/job_panel_scoring.py:300-303` — manual XS median imputation; `sklearn.impute.SimpleImputer(strategy='median')` stores training-time medians
- `live/alpaca_broker.py:152,211` — no `tenacity` retry on Alpaca 429/5xx
- `live/runner.py:599-603` — no retry on ntfy POST (silent trade-notification loss)
- `scripts/fetch_news_alpaca.py:52-81` — custom TokenBucket; `aiolimiter` exists
- `scripts/fetch_sec_fundamentals.py:55,85-127` — custom retry; doesn't honor `Retry-After` header
- `scripts/check_config_drift.py:35-74` — hand-rolled dict diff; `dictdiffer` / `jsondiff` exist
- multiple test files use 432 `pytest.approx` + 9 fragile `==` float compares; convert remaining

### Dead code (selected, full list ~12 items)
- `kernel/panel_pipeline/regime_router.py` — orphan Phase A skeleton
- `kernel/panel_pipeline/ensemble_scorer.py` — only test imports
- `kernel/panel_pipeline/transformer_scorer.py` — dispatcher branch dead (no `panel_transformer` artifact in prod)
- `kernel/portfolio_qp/signal_combiner.py` — only test imports
- `kernel/portfolio_qp/qp_solver.py:438-498` — DEPRECATED helpers retained 5+ months
- `kernel/portfolio_qp/qp_solver.py:114-116` — 3 ignored kwargs accepted silently
- `kernel/pipeline/task_sell.py:286-402` — `PanelConvictionExitTask` legacy (replaced by cross-sectional)
- `scripts/transformer_v4.py` (784 LOC) still alive via legacy `patchtst` kind in model_registry
- `scripts/qlib_transformer_v5.py` (371 LOC) — 0 prod imports
- `scripts/patchtst_doe_sweep.py` (417 LOC) — superseded by doe_hf
- `scripts/train_horizon_blender{,_v2,_v3}.py` (1059 LOC) — 0 prod imports
- `scripts/_train_BB_{00..12}.py` (16 files) — DOE expansion temp files

### Logic bugs (P1)
- `exits.py:482, 556` — short-detection uses `total_shares()` method AND `state.shares` field inconsistently
- `regime.py:530` — `dominant_gmm != BEAR` short-circuits BULL_VOLATILE arbitrary demotion
- `portfolio_qp/qp_solver.py:259-261` — gross_max constraint vs cash_reserve interaction undocumented
- `portfolio_qp/tasks.py:329` — `min_reentry_days` uses CALENDAR days but `feedback_anti_churn_principle` mandates BUSINESS days
- `task_post_stop_cooldown.py:67-91` — same calendar-vs-trading day issue
- `live/runner.py:957-959` — `_positions_cache_for_pl` attribute never assigned → cost-basis hack returns 0
- `live/runner.py:1224-1228` — STATE-EXT-SELL only protects pending-BUY direction, not pending-SELL
- `portfolio_qp/tasks.py:1294-1297` — `min_share_floor` checks NAV not cash
- `live/paper_broker.py:143-148` — accepts impossible negative cash with warning
- `live/runner.py::main()` — no lock file in runner itself (cron-lock not enough)

### Logging gaps (P1)
- 3 silent crons: `daily_news_sentiment_refresh.sh`, `daily_iv_snapshot.sh`, `preopen_cancel_gate.sh` — no ntfy on failure. `tee` swallows exit code. **`preopen_cancel_gate.sh` is the macro-shock cancel — silent failure worst possible mode**
- `live/alpaca_broker.py` — 429 rate-limit hits never logged
- `live/runner.py:1241` — partial fills tagged but never specifically logged; downstream treats partial-buy as full → next-bar STATE-EXT-SELL false-positive
- `live/runner.py:605-607` — ntfy timeout WARN doesn't escalate; no health-check at startup
- `regime.py:540` — Plan B cooldown arming has no log
- Multiple `Task.run()` return-False short-circuits without log

### Missed parallelism (P1)
- `panel_pipeline/job_panel_scoring.py:241-251` — 172-feat alpha158 compute for 142 tickers serial → ~10× speedup via `joblib.Parallel`
- `adapters/runner.py:723`, `adapters/sim.py:134`, `adapters/lean.py:215` — `prepare_inference_panel_frames` single-thread per ticker; 142 tickers parallel
- 3 eval drivers `eval_*_5cut_5seed.py` run 25 subprocesses sequentially; CPU (DLinear) trivially parallel via `xargs -P 5` or `concurrent.futures`
- `live/runner.py:298,316,342` — 3 serial Alpaca API calls (positions/orders/fills) — `asyncio.gather` candidate
- `scripts/fetch_news_alpaca.py`, `fetch_sec_fundamentals.py` — sequential 142 tickers; ThreadPoolExecutor + shared semaphore
- `scripts/export_lean_watchlist.py:120` — 142 tickers parquet→zip sequential
- `live/runner.py:1066` — `submit_order` not batched; alpaca-py supports list
- HF Trainer per-day batch_size=1 — 4-8× MPS speedup left on table

---

## Test coverage gap matrix (selected critical gaps)

| Module | Tests? | Critical gap? |
|---|---|---|
| `scripts/compare_arch_5cut_5seed.py` | NONE | 🔴 aggregator drives next verdict |
| `scripts/eval_*_5cut_5seed.py` (3 files) | NONE | 🔴 running BG now |
| `scripts/dlinear_baseline.py` | NONE | 🔴 §5.12 baseline |
| `scripts/verify_sigma_calibration.py` | NONE | 🔴 σ-calibration gate tool |
| `scripts/patchtst_hf.py --save-model` | indirect | 🟡 save-mismatch bug class |
| `kernel/portfolio_qp/tasks.py::min_share_floor` | source-string only | 🔴 string-presence ≠ code-path test |
| `scripts/train_104.py::RQ_ALLOW_NO_WF` removal | source-string only | 🔴 §5.13.1 test fixtures lie |
| `scripts/sunday_panel_sweep.py` H1-H4 gates | source-string only | 🔴 §5.13.1 |
| `kernel/exits.py` HIFO `apply_sell_lots` | real test ✓ | (template for others) |

---

## Triage proposal

**Immediate (this session)**:
1. P0-1 walk-forward embargo — fix splitter + decide on running BG eval (kill or let finish)
2. P0-3, P0-4 plist Weekday fixes — 4-char edits, 2 plists
3. P0-5 dashboard artifact path — 1-line fix
4. P0-7 `kernel/__init__.py` — empty file
5. P0-11 shadow config artifact paths — 4 rename edits

**Next session (high-priority but bigger scope)**:
6. P0-2 FactorZScoreTask look-ahead — needs verify + retrain
7. P0-6 calibrator method reconciliation
8. P0-8 6 zero-coverage smoke tests
9. P0-9 BUG D settled-cash semantics
10. P0-10 LIVE-account assertion

**Tier 2 / 3 deferred** (P1 items): mature-lib substitutions, dead code removal, parallelism, full coverage matrix.

---

## Source agent reports

5 parallel audit agents produced detailed punch lists (~180 issues total) on 2026-05-20. Reports in session transcripts. File:line citations in each report.
