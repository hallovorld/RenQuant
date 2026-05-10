# CLAUDE.md

Guidance for Claude Code working in this repository. **Concise on purpose** — pointers to detailed docs, not duplicated content.

---

## 🗂 项目状态速览（2026-05-09 EOD — AUDIT OPEN; 所有性能数字未经验证不可信任）

> 每次进入此项目时先读这段，5 分钟上下文。详细历史见 [`doc/archives/sessions/`](doc/archives/sessions/)；最新 roadmap 在 [`doc/roadmap.md`](doc/roadmap.md)；**当前审计在 [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md)**。

**🔴 2026-05-09 EOD 审计触发的关键事实：**

1. **所有"基线"数字目前都不可信任**。早上跑出 27-mo APY +6.77% / Sharpe +0.40 单次测量；同一天晚上同样 config + artifact 重跑得出 +1.97% / +0.20。这是单次测量复现失败，而 CLAUDE.md §5.2 早就要求每个新数字必须配 sanity check（A/A、shuffled-label、time-shift placebo），从未对 27-mo APY 做过 A/A multi-seed → σ_APY 未知 → 任何 APY 差异都可能纯噪声。

2. **审计找到 17 个问题**，已修 6 个 RED（commits 9ff4984 / 0fa2557 / 7091e57 / de8d7ea / 0a9c39c / b11dadd / 501ede0）：
   - dashboard 读错 broker DB → 修复 + 7 测试
   - panel-ltr.json 21-feat 死文件 → 同步到 169-feat 生产 artifact
   - sim 没传 last_sell_pls → cost-aware wash-sale 在 sim 里降级二进制 → 修 + 8 测试
   - QP wash-sale 用二进制 + ISO 字符串日期崩溃 → 修 + 10 测试
   - selection greedy 路径用二进制 wash-sale → 修 + 7 测试

3. **未修的 YELLOW 问题**：
   - BUG #5（asset_growth periods 修了 fetch 脚本但 `data/sec_fundamentals_daily.parquet` 是 5/8 旧版）→ 生产模型还在用 bug 的特征值。需重跑 fetch + 重训 panel
   - WF 闸门 daily cron 用 `RQ_ALLOW_NO_WF=1` 全程绕过 → 所有 promote 实际未经 walk-forward 验证
   - +6.77% 不可复现的根本原因未隔离（可能 sim 非确定性、可能 cron 干扰、可能我撤掉的 exec_tactics 改动）

4. **执行战术（5 fix）整批撤销**（commit 501ede0 之后清理）：
   - σ-自适应止损 (#0a)、利润分级 (#0c) 字面死代码 — 依赖 `state.sigma`，但 NGB 关了后 prod 永远是 None
   - 时间衰减止损 (#0b) 实测 −4.33 pp APY 有害
   - 赢家豁免 (#0e) +0.05 pp 在噪声内
   - VWAP smart_orders 模块从未接进生产代码（grep 零引用）

**生产模型：** XGBoost rank:pairwise + alpha158 + 5 fund + 3 PEAD + 3 SUE = 169 features，artifact `panel-ltr.alpha158_fund.json` 训练于 2026-05-09 03:44。NGBoost OFF。Calibrator pool_ic=+0.094。**性能：TBD pending bug-fix + A/A multi-seed protocol。**
**实盘：** Alpaca live ~$10.5k。
**Watchlist：** 103 ticker（runtime）/ 292 ticker（training panel）/ wl162 quality-first selected pending evaluation。

**已关闭（不要再讨论）：**
- **Macro 路线**：v1 broadcast 零梯度，v2 per-ticker β 修复 3 bug 后仍 −23% IC，v3 扩展 IC 单调递减，v4 macro-as-panel-row OOS IC −28.8%。所有 macro 形式都被否决。等 watchlist 扩到 200+ 再重评。
- **Asset Embeddings (T2-2)** — 16D embeddings 全 watchlist 重训覆盖 104 ticker，paired CPCV OOS IC = +0.0341 vs baseline +0.034 ≈ 0 提升。`asset_embeddings.enabled` 已设回 `false`。
- **LightGBM 替换**：在当前面板 −60% IC，已拒绝（2026-04-27）。
- **T2-4 Boyd Rotation**：rotation 每次 −2.5 APY pts，基础设施保留但默认关闭。
- **Track A — Insider 信号 (E22, 2026-05-02)**：44% 覆盖下贡献 −0.0008 在噪声内。Resume 条件：SEC IP 节流恢复 + Sunday retrain 补全数据后重测。
- **Track B — PEAD enrichment (E23, 2026-05-02)**：days_since/decay/signal 三列 A/A delta = −0.0010 ~ −0.0013 = **17-22σ 显著负向**。fwd_5d horizon 太短捕捉不到 30-60d drift。Resume 条件：fwd_20d / surprise quintile-rank。
- **Track D — wl103→wl183 expansion (E26, 2026-05-05)**：Stage 3 IC-additive admission 出 +9bp Stage IC，但 27-mo B2 holdout post-fix Sharpe **−0.07** / APY **−1.60%** vs wl103 baseline +1.10 / +13.27%。78% win rate 但 average loser ≫ average winner。Resume 条件：见 E26。**wl183 with all 14 bug fixes**: Sharpe +0.55 vs wl103 +0.68 — wl183 仍然输给 wl103 0.13 Sharpe，confirms TC collapse (Fundamental Law violation when breadth doubles but transfer coefficient halves)。
- **🔴 整个 active-alpha 路线 (E27, 2026-05-05)**：walk-forward 3-cut 显示模型 mean alpha vs SPY 一致负向 (−15.62% ± 10.21%)，所有 cut 都输给 SPY。单切 27-mo Sharpe 0.68 是 regime-smoothing 假象。**当前模型架构 + label (fwd_5d) + 训练数据 (2.5y) 不能产出真 alpha**。Resume 条件：换 label (fwd_20d/60d)、扩训练数据 (5y+)、换 architecture (Transformer)、或做真正的 walk-forward retraining。在那之前，建议 (a) cap allocation 至 30%，剩 70% 直接持 SPY；或 (b) 完全切 passive。
- **Track F — Triple-barrier label (E25, 2026-05-02)**：v3 hit-time-matched 出 mean_ic +0.0438（vs baseline +0.034 = +98bp 假象）。但 **time-shift +60d placebo 也 +0.0458 ≈ real**——disambiguation vs fwd_5d 显示 triple-barrier 是 regime persistence 拟合，无 10 天 alpha。Production 保留 fwd_5d。

**当前优先顺序：**
1. 🟡 **Calibrator retrain** (P0, blocks Track D resume)：production `panel-rank-calibration.json` `n_unique_prob_y=7` < runtime floor 10。Underlying cause: panel-LTR `best_iter=4` (XGB plateaued at 4 rounds → ~7 distinct probabilities)。Fix: bump `min_best_iter` floor + retrain panel-LTR + refit calibrator。
2. 🔴 **Microstructure / hourly bar 信号 (Track C)**：Alpaca hourly 数据空（0/178 缓存），数据获取 + 特征构建 ~3-5 天。
3. 🔴 **Regime Ensemble (T2-3)**：等面板 > 150k rows。Track D 已 shelved，等下一次 wl 扩或更宽 horizon 标签提供更多 rows。
4. 🟡 **B2 holdout sim 基础设施完善**：Sharpe/Sortino/Calmar 已 instrumented；wl103 baseline Sharpe 1.10 / APY 13.27%。Future runs anchor on this benchmark。

**评估基准共识：** CPCV mean_ic 可信（**run-to-run σ = 0.6bp** per 3-run 2026-05-02 estimation）。**train_ic 和 best_iter 跨 seed 高方差**——只用 mean_ic 比较实验。

**§5.2 sanity 三件套强制要求**（任何架构试验声明 +IC 之前必跑）：
1. A/A test（同 config 多次 → σ）
2. Shuffled-label（panel_ltr.label_shuffle_seed=N → mean_ic 应 ≈ 0）
3. Time-shift placebo（panel_ltr.label_shift_days=N → mean_ic 应 ≈ 0）

infra 已接入 standard pipeline，一行 config 就跑。Track F 的 +98bp 假象就是漏跑 placebo 才被骗。

**IC 评估方法（强制读 [`doc/research/ic-evaluation-methodology.md`](doc/research/ic-evaluation-methodology.md)）：**
- 任何 IC 数字必须来自 walk-forward（≥5 cuts），不能用单次 train/val/test split
- 报告必须包含 mean ± std，per-cut 明细
- Linear/Ridge baseline 必须在同一组 cuts 上跑过
- Param/sample ratio > 1/100 的模型（transformer 等）在我们当前数据规模上**禁止训练**——必然过拟合
- 当前 baseline (2026-05-08, 7-cut WF, fwd_20d, 291-ticker + fundamental):
  OLS mean=+0.029 std=0.038 / Ridge +0.030 / XGB +0.039 std=0.046
- 任何新模型/特征声明优于 baseline，必须在同一 7 cuts 上 paired comparison，差距 > 0.01 且 ≥ 5/7 cuts 胜出

---

## ✅ P0 CV bugs (discovered 2026-04-28, all fixed; refined 2026-05-02)

Three bugs that corrupted CPCV IC measurements. **All fixed with regression tests in `tests/test_p0_cv_bug_fixes.py` (14 tests green).** Kept here for historical reference — do not re-investigate.

- **BUG-CV-1** (linspace fold drift) — fixed in `training_panel/purged_cv.py` (integer division `fold_size = n_dates // n_splits`, last edge clamped). Test: `TestBugCV1FoldStability::test_fold_assignment_stable_across_n_dates_roll`.
- **BUG-CV-2** (best_iter guard) — fixed in `training_panel/pp_panel_training.py` (`min_best_iter=5` default raises RuntimeError if early-stop fires below threshold). **Refined 2026-05-02 (Task #24)**: iter-count check is FALSE POSITIVE on strong-univariate-IC features (e.g. days_since_earnings IC=+0.02 → XGB plateaus at round 4-9). Added eval_ic escape clause: when `best_iter < min_best_iter`, accept if `eval_ic >= min_best_iter_eval_ic_floor` (default 0.02). Pathological case (eval_ic ≈ 0) still raises. Test: `test_guard_has_eval_ic_escape_clause`.
- **BUG-CV-3** (eval set misaligned with CPCV) — fixed in `training_panel/pp_panel_training.py` (`n_eval = max(2, n_total // cv_n_splits)` instead of hardcoded 20%). Test: `test_eval_size_matches_cpcv_fold_size`.

**XGB run-to-run variance characterized (2026-05-02 σ estimation)**: same config (pead_off) produces highly variable `best_iter` (4 vs 25) and `train_ic` (±0.025) across seeds, but **CPCV `mean_ic` is robust (Δ=0.0001 across 2 paired runs)**. Lesson: compare experiments on `mean_ic`, not `train_ic` or `best_iter`.

---

## Project

RenQuant — personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live trading (Alpaca/IBKR). Statistically interpretable, strictly decoupled.

**Active strategy**: `renquant_104` (panel-LTR cross-sectional ranking). 103 retained for rollback.

## Environment

```bash
conda create -n renquant python=3.10
conda activate renquant
pip install pandas numpy matplotlib seaborn yfinance scikit-learn xgboost jupyterlab pyarrow
pip install "openbb[all]" openbb-cli backtesting scipy lean alpaca-py
lean login
```

Docker ≥ 16GB for LEAN. Alpaca creds in `.env` (gitignored).

Detailed reproducibility: [`doc/ops/environment.md`](doc/ops/environment.md).

## Workflow modes (5)

| Mode | Command | Use when |
|---|---|---|
| Research | Open notebook | Train + iterate, no Docker |
| Validation | `lean backtest .` (after `export_lean_watchlist.py --strategy X`) | Final OOS check |
| Analysis | `python scripts/analyze_backtest.py --strategy X` | Visualize a finished backtest |
| Live | `python -m live.runner --strategy X --broker {paper,alpaca-paper,alpaca,ibkr} --once` | One-shot trade |
| Scheduled | macOS launchd via 5 plists for renquant_104 | Daily cron-style |

**LEAN data isolation**: LEAN reads `backtesting/data/equity/usa/daily/{sym}.zip`, NOT `data/ohlcv/`. Run `export_lean_watchlist.py` before backtesting.

Detailed CLI + scheduled runs + plist files: [`doc/ops/usage.md`](doc/ops/usage.md).

## Architecture (one paragraph + cross-refs)

Three pipelines are the source of truth for ALL decision logic:
- **InferencePipeline** / **SellOnlyPipeline** — used by LEAN, live runner, sim. Phases: regime → drawdown → buy gates → sell (parallel) → buy candidates (parallel) → ranking → rotation → selection. Code in `kernel/pipeline/`.
- **FullTrainingPipeline** — orchestrates `BaselineTournamentJob → PanelTrainingJob → RecalibrationJob`. Code in `kernel/pipeline/pp_training_full.py`.
- **PanelTrainingPipeline** — `PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob`. Code in `training_panel/pp_panel_training.py`.

Strategy specs:
- 101 / 102 / 103: archived (rolled-back / superseded). Code lives at `backtesting/renquant_10{1,2,3}/` for forensic / rollback only.
- **104 (ACTIVE, panel-LTR)**: [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md) — cross-sectional XGBoost rank + NGBoost μ,σ head (currently OFF) + global calibration + portfolio QP + acceptance gates

Component deep dives (verified to exist 2026-05-09):
- [`doc/components/panel-ltr.md`](doc/components/panel-ltr.md) — primer + glossary
- [`doc/components/buy-logic.md`](doc/components/buy-logic.md) — 3 quality gates + portfolio QP
- [`doc/components/sell-logic.md`](doc/components/sell-logic.md) — SellGateB + LimitSellsPerBar
- [`doc/components/calibration.md`](doc/components/calibration.md) — score-DB + isotonic
- [`doc/components/rotation.md`](doc/components/rotation.md)
- [`doc/components/portfolio-qp.md`](doc/components/portfolio-qp.md) — cvxpy CLARABEL backend
- [`doc/components/databases.md`](doc/components/databases.md) — runs.db schema, broker isolation
- [`doc/components/training-pipeline.md`](doc/components/training-pipeline.md)

**Deleted 2026-05-09 (audit code-vs-doc reconciliation, code is source of truth):**
- `doc/arch/decision-graph-103.md` — removed (renquant_103 archived; canonical flowchart is now `kernel/pipeline/pp_inference.py`)
- `doc/arch/strategy-103.md` — removed (renquant_103 archived)
- `doc/components/transformer.md` — removed (transformer DISABLED, see closed-track list above)
- `doc/components/macro-factor-frame-design.md` — removed (all macro variants NO-GO per closed-track list)
- `doc/components/trade-evaluation.md` — removed (RL-OPE deferred)
- `doc/ops/maintenance-103.md` — removed (renquant_103 archived)
- `doc/ops/transformer-promotion.md` — removed (transformer DISABLED)
- `doc/experiments/post-tier1-followups.md` — removed (Tier 1 era closed)
- `common/` directory — empty, never re-populated post-pipeline-refactor

## Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
cd backtesting/foo && lean backtest .
python -m live.runner --strategy foo --broker paper --once
```

---

## Development Rules (mandatory, always follow)

### 1. Code is the Source of Truth (per user mandate 2026-05-09)
- The flowchart for renquant_103 (`doc/arch/decision-graph-103.md`) was deleted along with renquant_103's active status. **Source of truth for decision logic is `kernel/pipeline/pp_inference.py`** — read the Pipeline / Job / Task chain there before consulting any doc.
- For renquant_104 specifically: `pp_inference.py::InferencePipeline.run` enumerates all phases. Never trust a doc that contradicts a Job/Task body in `kernel/pipeline/` or `kernel/panel_pipeline/`.
- Doc-vs-code reconciliation rule: when in conflict, **code wins, doc gets updated**. Stale docs are bug-class — they mislead future-Claude into writing wrong code.

### 1b. Every Logical Unit Is a Task, Job, or Pipeline
**All new decision logic belongs in `kernel/pipeline/` or `kernel/panel_pipeline/` as a Task, Job, or Pipeline.** No new hand-written loops that bypass the orchestration layer.
- **Task**: an atomic step that reads/writes `InferenceContext` (or a ticker slice). Return `False` to short-circuit the enclosing Job's chain.
- **Job**: a sequential Task chain with a `should_skip(ctx)` gate. Jobs may run serially or via `run_parallel()` for per-ticker work.
- **Pipeline**: orders Jobs into phases and owns the full run.
- LEAN, live, and sim all go through `InferencePipeline` via `LeanAdapter` / `RunnerAdapter` / `SimAdapter`. Universe admission is consolidated in `kernel/pipeline/job_universe.py::LoadUniverseJob`.
- When adding a new decision: write as Task → wire into Pipeline phase → add paired alignment tests in `tests/test_panel_alignment.py` or `tests/test_policy_alignment.py`.

### 1c. Split Every Complex Structure (Principle)
**Every complex structure — not just decision logic — must be split into small Tasks. The Task chain is universal: sequence or concurrent.** A monolith of 200+ lines doing 5 logical steps is a bug factory. The user mandate (2026-05-04, formalized): "每个复杂结构都应该split成小的，或者可以sequence，或者可以concurrent的task！"

**Limits:**
- Each Task body **soft target ≤ 50 lines**. Going over is a smell, not a build break — but if you're at 100+ you almost certainly have ≥ 2 responsibilities and should split. Use judgment: a single complicated math operation can legitimately exceed the target; multiple sequential operations cannot.
- Each Task is **single-responsibility** — one of {extract, validate, compute, transform, persist, emit}. If you can't name it in one verb, it's two Tasks.
- Each Task has its **own unit test** with a stub ctx. Tests assert the ctx mutations the Task is responsible for, nothing more.
- Tasks communicate via **documented ctx fields** (public `ctx.X` for cross-Job state, private `ctx._job_*` for intra-Job state). No hidden coupling via globals.

**Why monoliths kill us:**
- A 459-line `JointPortfolioQPTask` (pre-split 2026-05-04) bundled vector extraction + tax cost + constraints + solve + emit into one function. Bugs in tax math hid because tests had to mock the entire context to exercise that branch. Refactor: 5 Tasks of 23-45 lines each, each independently testable.
- The 2026-04-29 calibrator NaN-leaf collapse incident lived for days inside `BuildFeatureMatrixTask` — a single Task doing matrix build + drift guard + row-coverage filter. The bug was in the row-coverage interaction with the drift guard, but isolating it required reading the full Task. Splitting would have made the failure mode obvious.
- Future-Claude reading a 200-line Task can't tell which lines are load-bearing for which downstream assertion.

**How to apply:**
- **New code**: design as a Job (or Task chain in an existing Job). Each step gets its own Task class with explicit reads/writes documented in the docstring. Never start with a monolith intending to "split later".
- **Splitting existing monoliths**: identify the logical phases (extract / validate / compute / emit), promote each to a Task, store cross-step state on `ctx._job_*` private fields, keep a back-compat shim at the original entry point. Reference: `kernel/portfolio_qp/{tasks.py, job_qp.py, task_joint_qp.py}` (the 2026-05-04 QP split pattern).
- **Sequence vs concurrent**: default sequence. Promote to `run_parallel()` only when (a) Tasks operate on independent rows (per-ticker chains) AND (b) measured speedup ≥ 1.5x on M2 Pro 10-core (per §5.10).
- **No partial splits**: if you split 3 of 5 phases and leave 2 inside a giant Task, you've doubled the maintenance surface. Either split all or none.

**Pending-splits backlog (status as of 2026-05-04 evening):**

| Monolith | Status | Notes |
|---|---|---|
| `JointPortfolioQPTask` (459 lines) | ✅ split → `JointPortfolioQPJob` (5 domain Tasks + 8 atoms) | reference pattern in `kernel/portfolio_qp/{tasks.py,job_qp.py}` |
| `BuildFeatureMatrixTask` (~165 lines) | ✅ split → `BuildFeatureMatrixJob` (4 Tasks) | `kernel/panel_pipeline/tasks_feature_matrix.py` |
| `BuildPanelTask` (~155 lines) | ✅ split → `BuildPanelJob` (6 Tasks) | `training_panel/tasks_build_panel.py` |
| `BuildHourlyResolutionPanelTask` (~160 lines) | ✅ split → `BuildHourlyResolutionPanelJob` (5 Tasks) | `training_panel/tasks_build_hourly_panel.py` |
| `JointActionTask` legacy greedy (~700 lines) | ⏸ **DEFERRED** | 6-phase state machine with 10+ load-bearing audit fixes (BUG-MM defer, BUG-L two-pass, BUG-Q net-alpha, etc.). NOT in production path (`solver=qp` is the live config). Splitting carries high regression risk for low ROI; revisit when production switches back to greedy or when full integration tests are in place to validate parity. |
| `record_trades` / `record_pipeline_run` | ✅ NOT-A-MONOLITH | both functions ≤ 50 lines already; do not split. |

### 2. Tests for Every Feature — and Every Bug
Every policy in notebook and LEAN must have a corresponding test. **Every bug gets fixed as soon as it's found, and the fix ships with a regression test that would fail before the fix.** A bug without a test is a bug you'll see again.

Bug-fix workflow:
1. Reproduce the bug with a failing test (unit or sim-level).
2. Land the fix. The test now passes.
3. Keep both together in the same branch — never merge a fix without its regression test.

This applies to:
- Production incidents (e.g. empty policy-metadata.json crashing daily_104.sh → `tests/test_universe_alignment.py::TestResilience`)
- Silent failure modes (e.g. systemic no-trade periods → `tests/test_no_trade_monitor.py` + `tests/test_no_trade_invariant.py`)
- Upstream-task ordering bugs (e.g. ApplyGlobalCalibrationTask skipped in additive mode → `tests/test_panel_bugfixes.py`)

Run tests via `python -m pytest tests/ -v`. Total **2929 passed, 10 skipped** as of 2026-05-02 evening (after Track D + Track F sanity infrastructure, +599 since 2026-04-26 round-7's 2330).

Notable test groupings (just summary — full file listing in [`doc/arch/overview.md`](doc/arch/overview.md)):
- `tests/test_policy_alignment.py` — 235 paired NB/LEAN alignment
- `tests/test_panel_alignment.py` — 34 panel adapter parity
- `tests/test_kernel_units.py` — 140 kernel module units
- `tests/test_universe_alignment.py` — 18 universe admission
- `tests/test_model_acceptance.py` — 29 acceptance gate tests (round-7)
- Plus dozens more — see `tests/` directory.

### 2a. Promotion Thresholds Are Not Floors for Theoretically-Sound Wins

**Default rule:** APY win ≥ +2 pts on 27-mo OOS = promote to golden.

**Exception (user spec 2026-04-24):** when variables are rigorously controlled, **any positive margin is meaningful**. Specifically:
- **Live/sim parity fixes** (e.g. CUSUM-v2 wall_time mode closing known bar-count-vs-calendar-day drift) ship even at < +2 pt — the fix correctness is the point; the APY lift is incidental confirmation.
- **Theory-aligned wins where the predicted magnitude matches** (e.g. CUSUM-v2 predicted "~2 pt drift closure" in the roadmap, result +1.97) are signal, not noise.
- **Mechanism-clean changes** (no hyperparameter drift, same panel) with positive margin are shipped.

**Not exceptions:** new strategies, hyperparameter sweeps, panel retrains — those need the full +2 pt to clear the promotion floor.

### 2b. Unexpected A/B Results = Audit Before Accepting

**When a theory we believed would improve the model produces the OPPOSITE result, the first hypothesis must be: "my implementation has a bug or my assumptions were wrong" — not "the theory is wrong".** Accept the negative result only AFTER the implementation audit.

Typical bugs that masquerade as "theory failed": ordering bug; wrong-sign delta; unit mismatch; silent guard fires; stale data; upstream flag default wrong.

**Audit checklist before shipping a negative finding:**
1. Print a per-bar log of the new Task's inputs on ≥3 sample bars — do they look sane?
2. Reason through every input/output independently: if everything were correct, what would we *expect*?
3. Re-read the commit's defaults — does the "GOLDEN" variant in the A/B actually preserve baseline behaviour?
4. Check whether any other task/config reads the same data — possible interaction.
5. Only after all four → document the audit and shelve the theory with evidence.

### 3. Git Commits — Sync Everything, Guard Secrets
Commit and push all changed files so the remote is always up to date.

**Before every commit:**
- Check `git status` for untracked or modified files.
- If a file contains sensitive data, add to `.gitignore` first, commit `.gitignore`, then handle the file.
- Currently gitignored sensitive/large paths: `.env`, `live/logs/`, `data/`, `backtesting/data/`, `backtesting/*/backtests/`.

If it's not gitignored, it should be in the remote.

### 4. Always Keep Docs Up to Date (per user mandate "code is source of truth")
After any non-trivial change, sync these. Code wins on conflict, doc gets corrected.
- [`doc/arch/overview.md`](doc/arch/overview.md) — pipeline + data flow
- [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md) — current active strategy (only renquant_104; 101/102/103 archived)
- [`doc/ops/schedule.md`](doc/ops/schedule.md) — cadence (daily / weekly / monthly / event-triggered)
- [`doc/roadmap.md`](doc/roadmap.md) — living roadmap
- This file — keep test counts and rule set current

### 5. Engineering Principles (mandatory, adopted 2026-04-28 after deep-audit + NVTS post-mortem)

These are not "nice to haves" — they're the response to a single 24h period where we shipped a stub as if it worked, ran a 10h experiment whose chain script was edited mid-run, reported a +54% IC win that was selection bias, and bought NVTS at +91%/20d on a model that had no parabolic-regime samples. Source: [`doc/archives/audits/2026-04-28-deep-audit.md`](doc/archives/audits/2026-04-28-deep-audit.md), [`doc/archives/audits/2026-04-28-nvts-buy-postmortem.md`](doc/archives/audits/2026-04-28-nvts-buy-postmortem.md).

**5.1 Run relevant tests before every commit/push.**
- `git push` is not a probe. Before any push, run `pytest` on at least the file(s) you touched + their tests.
- Even "obviously correct" scripts get a dry-run import check (`python -c "import the_module"`) before they run overnight. M2 blender shipped with a wrong function name and silently failed the whole 10h chain because nobody did the 1-second import check.
- If a test suite for the touched area doesn't exist, write a minimal one *before* shipping the change.

**5.2 Every new number ships with at least one sanity check.**
- "+54% IC" or "edge_sharpe = +0.139" with no falsification check is not a result, it's a guess.
- Mandatory minimum sanity tests for any new metric:
  - **A/A test**: same data, randomly resplit — does the lift persist? (catches selection bias)
  - **Shuffled-label test**: shuffle y, retrain — IC should be ≈ 0
  - **Placebo / time-shift**: shift labels by 1y, retrain — IC should not match
- Pick the one most relevant to the failure mode you're worried about. Without it, the number is not a finding.

**5.3 Every "fix" must name the invariant that prevents the entire class of bug.**
- Single-bug patches breed: same class returns under a new name. Always ask: *what invariant would have made this impossible?*
- NGBoost feature drift wasn't fixed by retraining the head — that's a patch. The fix is the `max_feature_drift_pct` hard guard + config-fingerprint stamping that makes silent train/inference column drift architecturally impossible.
- NVTS wasn't fixed by selling NVTS — that's a patch. The fix is `ParabolicExhaustionGateTask` that rejects *all* parabolic-top candidates regardless of edge_sharpe.
- If you can't name the invariant in one sentence, you wrote a patch, not a fix.

**5.4 Don't edit on-disk files of running scripts.**
- A running cron, a 10h overnight chain, a launchd-spawned job — its files are read-only until it finishes.
- Mid-run edits to chain scripts are silent failures. The 2026-04-28 chain skipped its M2 phase because phase-4 was appended to the script *after* it started executing.
- If you need to change behavior of a running job: schedule a wakeup with `ScheduleWakeup` (which restarts cleanly), or stop+restart the job. Never `Edit`/`Write` a file that has a live reader.

**5.5 Any production-touching change must rehearse its rollback.**
- "We have an auto-revert script" is not the same as "rollback works". The 2026-04-28 auto-revert restored the model checkpoint but *not* `strategy_config.json` — the watchlist stayed at 227 while the model knew about 103. Result: another fingerprint mismatch at the next cron tick.
- Rollback rehearsal = manually trigger the rollback path on a non-prod copy and verify the post-rollback state is fully self-consistent (config + model + state files all aligned).
- Production-touching includes: `strategy_config.json`, `strategy_config.golden.json`, models in `artifacts/`, anything launchd reads, anything live runner reads.

**5.6 Definition of "fixed" = full 24h audit clean.**
- A bug is not "fixed" until: (a) it's patched, (b) the regression test is green, (c) every file you touched in the last 24h has been re-audited end-to-end, (d) every upstream/downstream pipeline that consumes those files has been re-audited end-to-end, and (e) no remaining issues are open in any of those files.
- "I think it's done" is not done. Show the audit log.
- This is the gate before re-running any experiment after a fix-up cycle. Experiments run on broken infra produce broken results — fix infra first, *then* re-run.

**5.7 Every failed experiment must be documented in `doc/research/failed-experiments-log.md`.**
- Hypothesis, implementation, exact numbers, sanity check, conclusion, reproduction recipe — all required. Without the recipe an independent agent (Codex / a future Claude session) cannot verify the result and the team will re-run it 6 months later.
- Write the entry the same day as the result lands, before moving on. Memory of "why we didn't do X" decays fast; the entry is the durable record.
- Why this is load-bearing: 6 months from now, "should we try blending horizons?" will come up again. If `failed-experiments-log.md` doesn't have an answer, we will burn 3-6 hours of compute re-discovering. The log is cheaper than the rerun.
- A failed experiment is not a failure of the team — failing fast and recording why is the engine of progress. The failure mode is *not recording* it.

**5.8 Status reports use concepts, not code.**
- When briefing the user on progress, avoid commit hashes, file paths, variable names, task IDs, and bug labels (`BUG-CV-2`, `min_best_iter=0`, `panel-ltr.json`, etc.). Use the conceptual name of the thing instead — "未训练守卫", "诊断侧配置", "面板排序模型".
- Why: code references force the user to context-switch into the codebase to interpret the message. Concept names communicate the *what* and *why* directly. Status briefings exist to give the user a fast read on where things are, not to repeat the implementation.
- Code-level identifiers belong in commit messages, doc files, and direct technical questions — *not* in recurring status reports.
- When the user asks a specific technical question ("what does file X do?"), code references are appropriate. Distinguish status briefings (concepts) from technical answers (code OK).

**5.9 Multi-track plans ship as a timeline diagram.**
- When proposing a plan with parallel work streams or sequenced phases, draw it as an ASCII timeline before starting execution. Tracks on rows, time on the X-axis, dependencies as arrows.
- Why: a paragraph-style plan hides ordering and parallelism. A diagram makes both visible in one glance — the user can see at-a-glance which steps gate the next, which run in parallel, and where the merge points are. They can redirect ("flip Track A and B") with a single line instead of re-reading prose.
- Required elements: phase markers on the X-axis (e.g. `[现在]`, `[+2h]`, `[+完成]`), each track on its own row, arrows for sequence (`──→`), vertical bars for parallel streams (`↓`).
- Apply for: any plan with ≥2 sequential phases, OR ≥2 parallel work streams, OR a stop-gate (e.g. acceptance check) followed by branching paths. Single-step actions don't need a diagram.
- Example pattern (use when a fix queue must complete before a retrain, with secondary work in parallel):
  ```
  [现在]              [+2h]               [+3h]                [+并行]
  P0 fixes ─────────→ retrain ──────────→ sim ──────────────→ promote/reject
                       ↓
                      P1 fixes 同时跑
  ```

**5.10 Saturate the local hardware. Memory is for spending.**
- M2 Pro has 10 cores (8 performance + 2 efficiency) and 32 GB RAM. Every long-running compute job MUST be configured to use them. Default XGBoost / NumPy / PyTorch settings often leave 60-75% of the machine idle — that's wallclock you're throwing away.
- Mandatory env vars or config flags before any training / sim:
  ```
  export OMP_NUM_THREADS=10
  export MKL_NUM_THREADS=10
  export OPENBLAS_NUM_THREADS=10
  ```
  XGBoost: set `xgb_params.nthread=10` (or `n_jobs=-1`); NGBoost: set `n_jobs=-1`; sklearn helpers: `n_jobs=-1`.
- Memory-side: don't conservatively load chunks. If the dataset is < 5 GB, load fully into RAM. `pd.read_parquet` defaults are already fast; the lazy-loading is for 100 GB+, not us.
- **Verification step** before declaring a long job "running": after dispatch, check `top -l 1 | grep CPU` — should see ≥80% user CPU. If 30% user / 60% idle → the job is bottlenecked, fix BEFORE letting it eat hours.
- Why this is load-bearing: in 2026-05-02 Stage 3 wl-expansion experiment I ran 18 sequential XGB retrains over ~10 hours wallclock. Mid-run sampling showed 25% CPU usage / 62% idle — the job was running on ~2.5 cores out of 10. **Same compute completes in ~3-4 hours with proper saturation.** The user is paying for an M2 Pro, not an M1 Air.

**5.11 Experiment design optimizes for time-to-answer, not "thoroughness".**
- Before running a long experiment, ask: **"What's the cheapest experiment that would change my decision?"** Run that FIRST. The greedy / iterative / "leave-no-stone-unturned" approach is for OPTIMIZATION, not range-finding.
- Decision tree before any multi-hour experiment:
  1. **Range-finding question?** ("Does X work at all?") → single endpoint test (top-down, max-everything), 30 min wallclock.
  2. **Optimization question?** ("Find best subset / hyperparam") → ONLY after range-finding shows X works at all. Greedy / batch / sweep is fine here.
  3. **Diagnostic question?** ("Why did Y fail?") → §5.2 sanity sequence FIRST (A/A + shuffled-label + time-shift placebo), THEN dig into mechanism.
- Why this matters: in 2026-05-02 Track D Stage 3 I ran a 9-hour greedy admission experiment to answer "does watchlist expansion lift IC?" — a 30-minute top-down single training would have answered that range-finding question and saved 8.5 hours. Bottom-up greedy is for FINDING the best subset given that breadth helps, not for testing whether it does.
- Apply also to: A/A vs A/B (run A/A noise FIRST), full panel vs ablation (run smallest-meaningful-ablation first), training cost (single seed first to verify pipeline before launching 5-seed σ estimation).
- Sunk-cost guard: if mid-experiment evidence already answers the question, **kill the experiment**. The 12 batches of Stage 3 already proved "wl expansion lifts IC by ~9 bp"; the remaining 6 batches were optimization, not new information. Don't keep running just because the script is configured to keep running.

**5.12a Default to widely-accepted open-source solutions and the methods of highly-cited references — refuse to reinvent the wheel.**
- The first question for every algorithm / data-pipeline / model-choice decision is: *what does Qlib / cvxportfolio / scikit-learn / PyPortfolioOpt / the canonical paper do?* The second question is *can we use that directly?* Inventing your own variant is the last resort, only after the canonical method is shown insufficient on RenQuant data.
- Specific reference set this codebase pegs to:
  - **Cross-sectional ranking models / features**: `microsoft/qlib` (Alpha158, LinearModel, LGBModel, TransformerModel)
  - **Convex portfolio optimization**: `cvxportfolio` (Boyd group, Stanford) and CVXPY for the QP layer
  - **Asset-pricing factor design**: Bryan Kelly + Gu + Xiu RFS 2020 firm-characteristics list
  - **Risk + transaction cost**: Almgren-Chriss 2000, Ledoit-Wolf 2004 (already wired in)
  - **Time-series ML reference**: Patch-TST (Nie 2023), TFT (Lim 2021), Qlib's pytorch_*_ts.py
- Concrete failure mode this prevents: 2026-05-06 — I built `alpha158_lite` from documentation impressions instead of cloning Qlib and reading `qlib/contrib/data/loader.py` line-by-line. Result: 8 substantive deviations (sign-flipped ROC, wrong CORR formula, missing CNTP/SUMP/RSQR/RESI families, wrong normalization). When I went back and faithfully replicated Qlib's 158 features + LinearRegression(MSE), test IC jumped from +0.010 → +0.0316 (3× improvement).
- Workflow: before writing a new algo or design choice (a) name the canonical reference, (b) confirm a clean license / pip-installable / cloneable version exists, (c) port the source with adaptation comments rather than rewriting, (d) only then test variations.

**5.12 Every architectural / hyperparameter / data-design decision must be backed by literature OR a mature open-source reference — and "backed by" means the cited source was actually READ, not name-dropped as decoration.**
- Citing "Qlib alpha158" without cloning `microsoft/qlib` and reading `qlib/contrib/data/handler.py` is decoration, not support. Citing "PatchTST" without reading Nie et al. 2023 to confirm `patch_len`/`stride`/`d_model` defaults is decoration.
- Why this is load-bearing: in 2026-05-06's Transformer rebuild, I cited "Qlib alpha158" but built my own 40-feature subset by reading docs — features were OHLCV-derived and redundant with existing TA indicators. Result: 51-feature linear gave **+0.006 test_ic vs 11-feature +0.010** — strictly worse. Wasted ~2 hours of compute. Had I cloned Qlib first, I would have seen alpha158 needs to combine with cross-sectional and fundamental features (which production XGB already has).
- Mandatory checklist for any non-trivial design decision:
  1. **Cite a specific source** (paper title + year, or repo path `org/repo:file.py`).
  2. **Read it** — open the paper section / clone the repo and read the relevant file. Add the SHA / commit hash if relying on a specific version.
  3. **Confirm the decision matches the source** — same hyperparameter values, same data preprocessing, same evaluation metric. List discrepancies if you intentionally diverge, with reason.
  4. **If no source can be found** — the decision is exploratory; mark it as "unsupported, will tune via A/B".
- Apply for: model architecture (layers / d_model / heads), loss function, optimizer hyperparameters, data preprocessing, label construction, train/val/test split methodology, evaluation metrics, sanity-test design, feature engineering. Apply NOT for: trivial coding choices (variable names, file layout), task-specific glue (file paths, log formats).
- Failure mode this prevents: re-deriving things from scratch when a 14k-star repo or 2k-citation paper has the answer. The user repeatedly asked "have you read [Qlib / Kelly RFS 2020 / etc]?" because my decisions kept being ad-hoc.
- The 2026-05-06 self-audit listing **9 decisions where citation was decoration not real reference** (per-day batched DataLoader, listwise pairwise BCE, 60-day seq_len, alpha158-lite 40 features, label clip ±30%, per-horizon standardize, AdamW lr=1e-3, per-day demean labels, train/val/test split methodology) is the durable evidence of this principle's necessity.

### 5.13 — 2026-05-09 audit lessons (mandatory anti-patterns to forbid)

After a single-day audit found **17 RED bugs** (silent corruption, dead code, calibrator output up to +401%, sim/live divergence, theatrical WF gate), these patterns are now banned. Each rule is paired with the bug-class it prevents.

**5.13.1 — Test fixtures lie. Tests must walk real prod data flow.**
- I wrote 124 unit tests for σ-aware stop loss + profit ladder, all green. All used `HoldingState(sigma=0.30)` hand-construction. Production has `state.sigma = None` (NGB OFF). The 124 tests **never executed the prod path code being tested**.
- **New rule:** every fix has at least one test that calls through the actual SimAdapter / RunnerAdapter / Pipeline entry point — not a hand-constructed fixture. Reference: `tests/test_pipeline_invariants.py::test_missing_ticker_gets_xs_median_not_zero` (synthetic SEC parquet → real ApplyScoresTask → assert prod behavior).

**5.13.2 — Any new module is dead until grep proves prod imports it.**
- `kernel/execution/smart_orders.py` (156 lines) + 42 tests. Production code: zero imports. Module orphaned for hours before anyone noticed.
- **New rule:** before declaring a module "shipped", run `grep -rn '<module_name>' backtesting/<strategy>/{adapters,kernel,live,scripts}/` and verify ≥1 production import. If only tests reference it, it's not shipped.

**5.13.3 — Every fix names its class-of-bug invariant + AUDIT REGRESSION GUARD test.**
- BUG #6 (μ̂ collapse) was fixed by stamping `ctx._panel_matrix` AND adding `soft_check_score_series` (output diversity). Future-Claude can't reintroduce μ̂ collapse via a different mechanism — the diversity guard catches it.
- **New rule:** every fix commit includes a test class named `class Test<Bug>RegressionGuard` (or AUDIT REGRESSION GUARD as docstring) that pins the invariant which prevents the entire bug class. Examples in repo: `test_qp_wash_sale_cost_aware.py`, `test_smoke_test_model.py::TestNoRetrainInDailyShell`.

**5.13.4 — Single performance number = unverified claim. Multi-seed mean ± std required.**
- "27-mo APY +6.77% / Sharpe +0.40 honest baseline" was cited in 3 doc files. Re-run 8 hours later (same config + artifact) produced +1.97% / +0.20. **Single-measurement claims are unfalsifiable.**
- **New rule:** any APY / Sharpe / IC number quoted in commit / doc / roadmap MUST be `mean ± std` from ≥5 runs (different seeds OR different bar-orderings). Claims without σ are forbidden.

**5.13.5 — Single source of truth: same business decision = same function.**
- Wash-sale logic had 5 call sites: 4 used cost-aware `is_wash_sale_blocked_with_cost`, 1 (greedy selection) used binary `is_wash_sale_blocked`. Tickers got contradictory rulings at different stages.
- **New rule:** any business-rule decision (wash-sale, position cap, drawdown halt, post-stop cooldown, earnings blackout) has exactly **one** function. All callers route through it. Adding a parallel implementation requires deleting the original first.

**5.13.6 — Cron cadence must be info-theoretically justified.**
- Daily retrain on a fwd_60d-label model adds 0.014% new info per day. Daily was **cargo-cult** — the trust boundary belongs at the cadence where new label info materializes (weekly+).
- **New rule:** any new cron must answer in its docstring: "this frequency adds N% new training-relevant information per tick, vs M% from the next-coarsest alternative". If the answer is < 5% / tick, the cadence is wrong.

**5.13.7 — Code change ≠ data update. Mark "requires data regen" explicitly.**
- BUG #5 fixed `pct_change(periods=4) → 252` in `fetch_sec_fundamentals.py`. Commit said "FIXED". But `sec_fundamentals_daily.parquet` on disk was still the buggy version → production model still trained on bug values.
- **New rule:** any commit modifying a data-pipeline script must include in the commit message: `⚠️ requires data regen: <command>`. Until the regen runs and the artifact mtime updates, the fix is **not** in production.

**5.13.8 — Full pytest pass is a CI-gate invariant, not a vibe.**
- Today's audit started from "the test suite has 67 failures" (down to 4 after fixes). Nobody had run the full suite in days. Some failures were missing optional deps; some were real bugs (8 side-config artifact paths, calibrator y > +1, panel_shape list/dict).
- **New rule:** every push of `kernel/`, `adapters/`, `training_panel/`, `live/`, or `scripts/{train,daily,weekly,monthly}_*` must include a fresh `pytest tests/ --tb=no -q` snapshot in the commit message ("N passed / M failed"). Failure ≥ 5 → the commit is broken; investigate before merge.

**5.13.9 — "Audit subsystem X" is a 4-step protocol, not a grep.**
1. List X's data-flow inputs (where they come from, who writes them)
2. List X's data-flow outputs (where they go, who reads them)
3. For each input → output edge, write or find an integration test
4. For each "if X is not None" branch, prove via grep that X has a non-None path in production

If only step 1+2 done = audit at risk. Phase 2 audit on 2026-05-09 was 6 sampled / 2 deep — that's how σ-unwiring slipped through.

**5.13.10 — `if optional_field is not None` defaults to dead code unless verified.**
- σ-aware stop_loss / profit ladder both had `if state.sigma is None: return` short-circuits. NGBoost is OFF in production → `state.sigma` is never populated → the entire feature is **architecturally dead code**. The fix had passed 124 tests because all fixtures set sigma manually.
- **New rule:** any new code path with `if optional_field is not None and optional_field > 0` must `grep -r 'optional_field = '` in the production codebase + show one prod write site that fires under current config. Otherwise the path is dead and the whole "fix" must be reverted or guarded by a config flag that's documented as "requires <feature> ON".

**5.13.11 — NaN / inf must be guarded explicitly. `>` and `<` evaluate False on NaN.**
- Found 5 separate places where `if x > 0` let NaN pass: `qty * price` cash math, `mkt / qty` price extraction, `panel_shape` schema, `tax` arithmetic, `position_value` updates. Pattern: `NaN > 0` is `False`, so the "nonpositive-reject" guard silently passes the NaN through.
- **New rule:** any `>` or `<` comparison on a value that *could* be NaN (broker returns, broker fills, config-driven floats, OHLCV closes, fund features, calibrator output) must be paired with `math.isfinite(x)`. Pattern: `if math.isfinite(x) and x > threshold:`.

**5.13.12 — Calibrator / regression output must be range-bounded.**
- Production calibrator's `expected_return.y` ranged from -0.30 to +**4.01** (i.e. predicted +401% over fwd_60d). 32% of knots > ±100%. The QP solver consumed this as μ_i → top-scored tickers' position weights inflated 5-10×.
- **New rule:** any regression / calibration output that feeds position sizing / Kelly / cash-budget math must have explicit `np.clip(out, lo, hi)` applied AT THE TRAIN SITE (so the artifact stores sane bounds). Defense in depth: also clip at the consumer site. Reference: `training_panel/global_calibrator.py:362-388` (commit `e715455`).

**5.13.13 — Side configs are loaded weapons. Aliased artifact paths or fail-fast.**
- 8 historical "side" configs (`strategy_config.golden.previous_*`, `live.previous_*`, `ngb_off_ab`, `alpha158_fund_paper`) all referenced production artifact paths. `--strategy-config-name <side>.json` would have **silently overwritten production** during retrain.
- **New rule:** any `strategy_config.<label>.json` (where label != empty string) must alias all `artifact_path` keys to side-paths containing the label. Pinned by `tests/test_side_config_artifact_paths.py`.

**5.13.14 — `panel-ltr.json` is a stub, not the live model. Tools must read `strategy_config.json::artifact_path`.**
- 5 tooling scripts (finalize_challenger, audit_oos_ic_drift, model_dashboard, fit_panel_calibrator, train_panel_model) defaulted to loading `panel-ltr.json`. After alpha158 promotion that file became a 21-feat stub, while inference loaded `panel-ltr.alpha158_fund.json` (169 feats). Cross-tool comparisons were nonsense for days.
- **New rule:** no tool defaults to a hardcoded artifact filename. Every load resolves through `cfg["ranking"]["panel_scoring"]["artifact_path"]`. Hardcoded filenames are a regression flag.

**5.13.15 — WF gate exists in code ≠ WF gate enforced in production.**
- `_check_wf_gate` was committed with regression tests. But `train_104.py:171` set `RQ_ALLOW_NO_WF=1` unconditionally on every daily promote. **Every prod promote bypassed the gate.** No weekly cron actually ran `run_wf_gate.py`. The gate was theatrical.
- **New rule:** any safety gate added to the codebase has TWO required artifacts: (a) the gate function + tests, AND (b) a scheduled cron (plist + .sh) that invokes it WITHOUT override. If only (a) ships, the gate is decoration. Reference cadence: `doc/ops/schedule.md`.

---

### Documentation Index (canonical pointers; reconciled 2026-05-09 — all links verified to resolve)

**Foundation**: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/arch/indicators.md`](doc/arch/indicators.md), [`doc/arch/models.md`](doc/arch/models.md)

**Components**: [`doc/components/panel-ltr.md`](doc/components/panel-ltr.md), [`doc/components/buy-logic.md`](doc/components/buy-logic.md), [`doc/components/sell-logic.md`](doc/components/sell-logic.md), [`doc/components/calibration.md`](doc/components/calibration.md), [`doc/components/rotation.md`](doc/components/rotation.md), [`doc/components/portfolio-qp.md`](doc/components/portfolio-qp.md), [`doc/components/databases.md`](doc/components/databases.md), [`doc/components/training-pipeline.md`](doc/components/training-pipeline.md)

**Operations**: [`doc/ops/golden-config.md`](doc/ops/golden-config.md), [`doc/ops/usage.md`](doc/ops/usage.md), [`doc/ops/setup.md`](doc/ops/setup.md), [`doc/ops/environment.md`](doc/ops/environment.md), [`doc/ops/schedule.md`](doc/ops/schedule.md) — **single source of truth for cron / weekly / monthly / event-triggered cadence**

**Research**: [`doc/research/papers-implemented.md`](doc/research/papers-implemented.md), [`doc/research/scoring-research.md`](doc/research/scoring-research.md), [`doc/research/rotation-research.md`](doc/research/rotation-research.md), [`doc/research/alpaca-crypto-btc.md`](doc/research/alpaca-crypto-btc.md), [`doc/research/watchlist-100.md`](doc/research/watchlist-100.md), [`doc/research/panel-sunday-sweep.md`](doc/research/panel-sunday-sweep.md), [`doc/research/ic-evaluation-methodology.md`](doc/research/ic-evaluation-methodology.md), [`doc/research/failed-experiments-log.md`](doc/research/failed-experiments-log.md)

**Experiments / measured A/B**: [`doc/experiments/ab-journal.md`](doc/experiments/ab-journal.md), [`doc/experiments/panel-training-runs.md`](doc/experiments/panel-training-runs.md), [`doc/experiments/sim-ab-results.md`](doc/experiments/sim-ab-results.md)

**Roadmap + audit**: [`doc/roadmap.md`](doc/roadmap.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md), [`doc/AUDIT_2026-05-09_SUMMARY.md`](doc/AUDIT_2026-05-09_SUMMARY.md)

**History**: `doc/archives/sessions/`, `doc/archives/audits/` (browseable directories — `git log --follow doc/archives/...` for provenance)

---

## General Coding Guidelines

**Bias toward caution over speed. For trivial tasks, use judgment.**

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Transform tasks into verifiable goals before implementing:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan with verify steps before starting.
