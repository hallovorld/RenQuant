# CLAUDE.md

Guidance for Claude Code working in this repository. **Concise on purpose** — pointers to detailed docs, not duplicated content.

---

## 🗂 项目状态速览（2026-04-27 晚间更新 — embeddings/macro A/B 决定性结果 + 今日交易事故根因）

> 每次进入此项目时先读这段，5 分钟上下文。详细记录见 [`doc/archives/sessions/2026-04-27-decisions.md`](doc/archives/sessions/2026-04-27-decisions.md)。

**生产模型：** XGBoost rank:pairwise，27 特征，无 macro，无 embedding，**CPCV OOS IC = +0.0418**（15-fold，2026-04-27 重训复现）
**实盘：** Alpaca ~$10k，持仓 PLTR / TSM / CAT / AMZN / GOOG / XLU，真实回撤 ~0.17%
**Watchlist：** 103 ticker，面板 ~77k rows

**已关闭（不要再讨论）：**
- **Macro 路线**：v1 broadcast 零梯度，v2 per-ticker β 修复 3 bug 后仍 −23% IC，v3 扩展 IC 单调递减。**v4 macro-as-panel-row** 实测 OOS IC −28.8%（2026-04-27 paired t=−1.98）。所有 macro 形式都被否决。等 watchlist 扩到 200+ 再重评。
- **Asset Embeddings (T2-2)** — **2026-04-27 NO-GO**：16D embeddings 全 watchlist 重训覆盖 104 ticker，paired CPCV 实测 OOS IC = +0.0341 vs baseline +0.0418（−18.5%, t=−1.45）。早先 dispatch agent 的 "GO" 判断（基于 OLS A/B + per-feature IC 单变量正交）在 XGB rank 树模型下不成立。`asset_embeddings.enabled` 已设回 `false`。
- **LightGBM 替换**：在当前面板 −60% IC，已拒绝（2026-04-27）。
- **T2-4 Boyd Rotation / rotation 作为 APY 杠杆**：rotation 每次 −2.5 APY pts，基础设施保留但默认关闭。

**当前优先顺序：**
1. 🔴 **Watchlist 99→200**（breadth 扩展，+42% IR ceiling — 现在是最有希望的 lever）
2. 🔴 **T2-3 Regime Ensemble**（等面板 > 150k rows）
3. 🔴 **OOS Backtest 基础设施 B1-B3**（等 live 数据积累）
4. 🟡 **`bypass_ticker_gate=true` 实验**：今天 41 个 in-universe 票里只有 10 个进 candidate（NVDA/AMD 都被 per-ticker model 预筛掉），让 Panel-LTR 真正成为主控可能能多放出 candidates；先在 sim 验证。

**评估基准共识：** CPCV OOS IC 可信（轻微虚高不影响相对比较）。Sim backtest 泄漏是独立的 Roadmap P0 问题，当前阶段不处理——不需要每次解释。

**🚨 2026-04-27 已修复事故 — NGBoost feature drift：** 今日盘中 0 buy。根因是部署的 `ngboost-head.json` 是某次 macro v3 实验时训出，含 140+ macro feature cols（vxx/hyg/dgs10/cpiaucsl/...）；而当前 inference panel 不再产生这些列（macro 已关），ApplyNGBoostTask 把缺失列零填充 → σ 失真 → 所有 candidate 的 `edge_sharpe` 被压到 < 0.10 阈值 → Gate B 全否。修复：(a) 已重训 panel + NGBoost head 与当前 panel 对齐；(b) 已加 `max_feature_drift_pct` 硬阈值（默认 5%），缺失超过就 SKIP NGBoost 而不是静默零填充。

---

## ✅ P0 CV bugs (discovered 2026-04-28, all fixed by 2026-04-29)

Three bugs that corrupted CPCV IC measurements. **All three are fixed and have regression tests in `tests/test_p0_cv_bug_fixes.py` (13 tests, all green).** Kept in this file for historical reference — do not re-investigate.

- **BUG-CV-1** (linspace fold drift) — fixed in `training_panel/purged_cv.py` lines 70-72 + 261-263 (integer division `fold_size = n_dates // n_splits`, last edge clamped to `n_dates`). Test: `TestBugCV1FoldStability::test_fold_assignment_stable_across_n_dates_roll`.
- **BUG-CV-2** (best_iter guard) — fixed in `training_panel/pp_panel_training.py` line 2438 (`min_best_iter=5` default; raises RuntimeError if XGBoost early-stopped below threshold). Test: `test_guard_raises_when_best_iter_below_threshold`.
- **BUG-CV-3** (eval set misaligned with CPCV) — fixed in `training_panel/pp_panel_training.py` line 2352 (`n_eval = max(2, n_total // cv_n_splits)` instead of hardcoded 20%). Test: `test_eval_size_matches_cpcv_fold_size`.

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

## Shared library `common/`

Import as `import common`. **Do not import `common/` from inside `backtesting/`** — LEAN Docker can't access it.

Modules: `common.{config, data, indicators, models, strategy, portfolio, tax, plotting}`. Detailed module exports: [`doc/arch/overview.md`](doc/arch/overview.md).

## Architecture (one paragraph + cross-refs)

Three pipelines are the source of truth for ALL decision logic:
- **InferencePipeline** / **SellOnlyPipeline** — used by LEAN, live runner, sim. Phases: regime → drawdown → buy gates → sell (parallel) → buy candidates (parallel) → ranking → rotation → selection. Code in `kernel/pipeline/`.
- **FullTrainingPipeline** — orchestrates `BaselineTournamentJob → PanelTrainingJob → RecalibrationJob`. Code in `kernel/pipeline/pp_training_full.py`.
- **PanelTrainingPipeline** — `PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob`. Code in `training_panel/pp_panel_training.py`.

Strategy specs are detailed elsewhere:
- 101 (single-stock): minimal, kept for reference
- 102 (multi-stock scanner): [`doc/arch/strategy-103.md`](doc/arch/strategy-103.md) (102 is parent of 103)
- 103 (regime-adaptive): [`doc/arch/strategy-103.md`](doc/arch/strategy-103.md) — 3-layer regime + relative strength + rotation
- **104 (active, panel-LTR)**: [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md) — cross-sectional XGBoost rank + NGBoost μ,σ head + global calibration + portfolio QP + acceptance gates

Component deep dives:
- [`doc/components/panel-ltr.md`](doc/components/panel-ltr.md) — primer + glossary
- [`doc/components/buy-logic.md`](doc/components/buy-logic.md) — 3 quality gates + portfolio QP (operator runbook merged)
- [`doc/components/sell-logic.md`](doc/components/sell-logic.md) — SellGateB + LimitSellsPerBar (round-7)
- [`doc/components/calibration.md`](doc/components/calibration.md) — saturation finding + score-DB design
- [`doc/components/rotation.md`](doc/components/rotation.md)
- [`doc/components/transformer.md`](doc/components/transformer.md) — daily + hourly + Bug #21/23/24 + acceptance protections
- [`doc/components/macro-factor-frame-design.md`](doc/components/macro-factor-frame-design.md) — VIX/HYG/UUP cross-asset broadcast
- [`doc/components/trade-evaluation.md`](doc/components/trade-evaluation.md) — RL-OPE design

## Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
cd backtesting/foo && lean backtest .
python -m live.runner --strategy foo --broker paper --once
```

---

## Development Rules (mandatory, always follow)

### 1. Logic Graph is the Source of Truth
[`doc/arch/decision-graph-103.md`](doc/arch/decision-graph-103.md) is the canonical decision flowchart for renquant_103.
- **Whenever the notebook simulation cell (657a4a6c) changes**, update the logic graph first, then verify LEAN matches every node.
- **Whenever LEAN main.py changes**, check it against the logic graph and update if LEAN is intentionally extended.

### 1b. Every Logical Unit Is a Task, Job, or Pipeline
**All new decision logic belongs in `kernel/pipeline/` or `kernel/panel_pipeline/` as a Task, Job, or Pipeline.** No new hand-written loops that bypass the orchestration layer.
- **Task**: an atomic step that reads/writes `InferenceContext` (or a ticker slice). Return `False` to short-circuit the enclosing Job's chain.
- **Job**: a sequential Task chain with a `should_skip(ctx)` gate. Jobs may run serially or via `run_parallel()` for per-ticker work.
- **Pipeline**: orders Jobs into phases and owns the full run.
- LEAN, live, and sim all go through `InferencePipeline` via `LeanAdapter` / `RunnerAdapter` / `SimAdapter`. Universe admission is consolidated in `kernel/pipeline/job_universe.py::LoadUniverseJob`.
- When adding a new decision: write as Task → wire into Pipeline phase → add paired alignment tests in `tests/test_panel_alignment.py` or `tests/test_policy_alignment.py`.

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

Run tests via `python -m pytest tests/ -v`. Total ≈ 2330+ tests as of 2026-04-26 round-7.

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

### 4. Always Keep Docs Up to Date
After any non-trivial change, sync these:
- [`doc/arch/decision-graph-103.md`](doc/arch/decision-graph-103.md) — decision flowchart
- [`doc/arch/overview.md`](doc/arch/overview.md) — pipeline + data flow
- [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md) — current active strategy
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

### Documentation Index (canonical pointers)

**Foundation**: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/arch/decision-graph-103.md`](doc/arch/decision-graph-103.md), [`doc/arch/indicators.md`](doc/arch/indicators.md), [`doc/arch/models.md`](doc/arch/models.md)

**Components**: [`doc/components/panel-ltr.md`](doc/components/panel-ltr.md), [`doc/components/buy-logic.md`](doc/components/buy-logic.md), [`doc/components/sell-logic.md`](doc/components/sell-logic.md), [`doc/components/calibration.md`](doc/components/calibration.md), [`doc/components/rotation.md`](doc/components/rotation.md), [`doc/components/transformer.md`](doc/components/transformer.md), [`doc/components/portfolio-qp.md`](doc/components/portfolio-qp.md), [`doc/components/databases.md`](doc/components/databases.md), [`doc/components/training-pipeline.md`](doc/components/training-pipeline.md), [`doc/components/trade-evaluation.md`](doc/components/trade-evaluation.md), [`doc/components/macro-factor-frame-design.md`](doc/components/macro-factor-frame-design.md)

**Operations**: [`doc/ops/golden-config.md`](doc/ops/golden-config.md), [`doc/ops/usage.md`](doc/ops/usage.md), [`doc/ops/setup.md`](doc/ops/setup.md), [`doc/ops/environment.md`](doc/ops/environment.md), [`doc/ops/transformer-promotion.md`](doc/ops/transformer-promotion.md), [`doc/ops/maintenance-103.md`](doc/ops/maintenance-103.md)

**Research**: [`doc/research/papers-implemented.md`](doc/research/papers-implemented.md), [`doc/research/scoring-research.md`](doc/research/scoring-research.md), [`doc/research/rotation-research.md`](doc/research/rotation-research.md), [`doc/research/alpaca-crypto-btc.md`](doc/research/alpaca-crypto-btc.md), [`doc/research/watchlist-100.md`](doc/research/watchlist-100.md), [`doc/research/panel-sunday-sweep.md`](doc/research/panel-sunday-sweep.md)

**Experiments / measured A/B**: [`doc/experiments/ab-journal.md`](doc/experiments/ab-journal.md), [`doc/experiments/panel-training-runs.md`](doc/experiments/panel-training-runs.md), [`doc/experiments/sim-ab-results.md`](doc/experiments/sim-ab-results.md), [`doc/experiments/post-tier1-followups.md`](doc/experiments/post-tier1-followups.md)

**Roadmap**: [`doc/roadmap.md`](doc/roadmap.md)

**History**: [`doc/archives/sessions/`](doc/archives/sessions/), [`doc/archives/audits/`](doc/archives/audits/)

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
