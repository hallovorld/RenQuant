# CLAUDE.md

Guidance for Claude Code working in this repository. **Concise on purpose** — pointers to detailed docs, not duplicated content.

---

## 🔴 PRIME DIRECTIVE — RenQuant is a REGIME-CONDITIONAL strategy

**This is non-negotiable architecture, set by user 2026-05-14.** Every feature, every knob, every experiment must be designed and evaluated through a regime-conditional lens. Pooled-mean metrics across regimes are MISLEADING and produce false NEITHER verdicts.

### Implications

1. **Every numeric knob should live in `regime_params.<REGIME>.<knob>`**, not as a global scalar. Examples that should be per-regime (most are not yet — wire them progressively):
   - `long_short.enabled`, `long_short.max_short_pct`, `long_short.max_shorts`
   - `stop_loss_pct`, `trailing_stop_trigger_pct`, `trailing_stop_trail_pct`
   - `max_holding_days`, `take_profit_pct`, `drawdown_halt_pct`, `drawdown_resume_pct`
   - `kappa` (risk aversion), `qp_dw_max` (rebalance throttle), `qp_turnover_penalty`
   - `vol_target`, `kelly_scale`, `cash_reserve_pct`, `min_model_score`
   - `bear_defensive_slots`, `bear_defensive_pct`, `defensive_tickers`
   - `entry_mode`, `min_price_move_pct`

2. **Every experiment design starts with: "which regime does this thesis apply to?"** If the thesis is regime-neutral, that's a red flag — most signals are not. The experiment config should differ per regime, not flip one global switch.

3. **Every evaluation reports per-regime numbers first, pooled-mean second.** The 5-test framework already has a regime-stratified test — that test is the PRIMARY signal, not the pooled mean. A "WIN in 3 regimes / LOSE in 2 regimes" is actionable (deploy conditionally); a "+6.23pt pooled NEITHER" hides the signal.

4. **Regime detector quality is P0.** If the detector mis-labels regimes (e.g., 2026-05-14 bug where 2022-Q2 bear labeled BULL_CALM 100% of bars), all regime-conditional logic is theatrical. Detector accuracy on known objective regimes must be tested whenever the detector code changes.

5. **Promotion gating uses regime-stratified Tier 2/3 criteria.** A change that wins in regimes A+B and loses in regime C is promotable ONLY as a regime-conditional config edit (enable in A+B, leave default in C), NEVER as a global flip. No exceptions.

### Why this exists

2026-05-14 session: shorts ON pooled across 16 windows scored NEITHER (+6.23pt, p=0.23). But regime-stratified: BEAR +22pt / CHOPPY +14pt / BULL_VOL +13pt (wins) vs BULL_CALM −8pt / BULL_STRONG −2pt (losses with 2 catastrophes). The right answer was "deploy shorts conditional on regime ∈ {BEAR, CHOPPY, BULL_VOL}", not "reject shorts globally". Pooled-mean evaluation buried the actionable signal.

Same week: regime detector was found to label 2022 Q2 bear market as BULL_CALM 100% of bars because Hurst > 0.65 (trending) routes to BULL_CALM regardless of direction. Fixed in commit `3925c0d` via SPY < MA50 direction signal. Without this fix, every `regime_params.BEAR.*` setting in the config was decorative.

### Where to start reading

- Current regime detector: `kernel/regime.py` (standalone) + `kernel/pipeline/task_regime.py` (production task path — both must stay in sync)
- Current regime params: `strategy_config.golden.json::regime_params.{BULL_CALM,BULL_VOLATILE,BEAR,CHOPPY}`
- Evaluation methodology: `feedback_eval_robust_methodology.md` memory + `doc/research/2026-05-14-longshort-clean-FINAL.md`
- Promotion gating: `doc/research/promotion-methodology.md` (3-tier)

---

## 🗂 Current state

**Start every RenQuant 104 continuation with**
[`doc/research/2026-05-23-mainline-memory.md`](doc/research/2026-05-23-mainline-memory.md).
It is the live mainline memory for the current repair campaign: what is fixed,
what is still broken, which numbers are authoritative, and what to do next.

For day-to-day status, read `git log --oneline -30` and the docs below. Recent session snapshots are archived in `doc/archives/sessions/` — most recent:
- [`2026-05-17-night.md`](doc/archives/sessions/2026-05-17-night.md) — detector + per-regime σ-wire + 3 prod-corruption gates + HIFO
- [`2026-05-15-evening.md`](doc/archives/sessions/2026-05-15-evening.md) — calibrator P0 + NGBoost confirmed + regime-aware re-eval
- [`2026-05-12-evening.md`](doc/archives/sessions/2026-05-12-evening.md) — Bug C + industry-grade evaluation rebuild

**Active strategy:** `renquant_104` (panel-LTR cross-sectional ranking, XGBoost rank:pairwise on alpha158 + fund + PEAD + SUE = 169 features). `renquant_103` archived for rollback only.

**Live mode:** Alpaca PAPER for cron schedules (2026-05-11 safety mandate). Explicit `"e2e"` requests run against the LIVE Alpaca account (see Environment §"e2e" below).

**NAV invariant (Bug C regression guard):** `NAV ≡ free_cash + pending_settle + Σ(shares × price)`. Pinned by `tests/test_sim_nav_t2_settlement.py`.

**Roadmap + audits:** [`doc/roadmap.md`](doc/roadmap.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md), [`doc/research/failed-experiments-log.md`](doc/research/failed-experiments-log.md).

---

## Project

RenQuant — personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live trading (Alpaca/IBKR). Statistically interpretable, strictly decoupled.

## Environment

Activate the project venv: `source .venv/bin/activate` (NOT conda). Docker ≥ 16GB for LEAN. Alpaca creds in `.env` (gitignored). Full setup: [`doc/ops/environment.md`](doc/ops/environment.md).

### "e2e" = LIVE Alpaca account (real money) — user mandate

When the user says **"e2e"**, **"daily e2e"**, **"run with alpaca account"**, or any variant, run `python -m live.runner --strategy <name> --broker alpaca --once` against the **LIVE Alpaca account with real money**. Do NOT propose paper variants. Do NOT switch to `--broker alpaca-paper` or `--broker paper` silently or via a clarifying question. The `.env` only has LIVE credentials — paper-API calls 401.

This overrides the 2026-05-11 PAPER safety mandate **for explicit e2e invocations**. The PAPER mandate still applies to launchd cron schedules and any non-explicit invocation. Locked in 2026-05-17 after user verbatim said: `"我他妈的说了一万遍了！live account！写进claude.md！"`.

Standard invocation (venv + .env exported so Alpaca SDK reads keys):
```bash
nohup bash -c 'set -a; source .env; set +a; .venv/bin/python -m live.runner --strategy renquant_104 --broker alpaca --once' \
  > logs/live_e2e/e2e_alpaca_live_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

### "daily full" / "full daily" = force the full path, not wrapper-only

When the user says **"daily full"**, **"full daily"**, **"run daily full"**,
**"跑 daily full"**, or asks for daily shadow full, do not treat a wrapper
holiday/calendar/cadence skip as completion. Try the wrapper if useful, but if
`scripts/daily_104.sh` skips due NYSE calendar, lock, cadence, or schedule
guard, immediately run the direct full paths and report the complete
decision-tree / blocker output:

```bash
set -a; source .env; set +a; .venv/bin/python -m live.runner --strategy renquant_104 --broker alpaca --once
set -a; source .env; set +a; .venv/bin/python -m live.runner --strategy renquant_104 --broker readonly-alpaca --once --strategy-config-name strategy_config.shadow.json
```

This means execute the flow and surface hard-gate outcomes. It does not
authorize disabling WF/model/preflight hard gates, ignoring failed artifact
contracts, or inventing discretionary live orders. If the user gives an
explicit manual order, require exact ticker, quantity/notional, and order type.

## Workflow modes

| Mode | Command | Use when |
|---|---|---|
| Research | Open notebook | Train + iterate, no Docker |
| Validation | `lean backtest .` (after `export_lean_watchlist.py --strategy X`) | Final OOS check |
| Analysis | `python scripts/analyze_backtest.py --strategy X` | Visualize a finished backtest |
| Live | `python -m live.runner --strategy X --broker {paper,alpaca-paper,alpaca,ibkr} --once` | One-shot trade |
| Scheduled | macOS launchd via 5 plists for renquant_104 | Daily cron-style |

**LEAN data isolation**: LEAN reads `backtesting/data/equity/usa/daily/{sym}.zip`, NOT `data/ohlcv/`. Run `export_lean_watchlist.py` before backtesting. CLI + plist details: [`doc/ops/usage.md`](doc/ops/usage.md).

## Architecture

Three pipelines own ALL decision logic (code is source of truth — never trust a doc that contradicts a Job/Task body):
- **InferencePipeline / SellOnlyPipeline** (`kernel/pipeline/`) — used by LEAN, live, sim. Phases: regime → drawdown → buy gates → sell (parallel) → buy candidates (parallel) → ranking → rotation → selection.
- **FullTrainingPipeline** (`kernel/pipeline/pp_training_full.py`) — `BaselineTournamentJob → PanelTrainingJob → RecalibrationJob`.
- **PanelTrainingPipeline** (`training_panel/pp_panel_training.py`) — `PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob`.

LEAN / live / sim all enter through `InferencePipeline` via `LeanAdapter` / `RunnerAdapter` / `SimAdapter`. Universe admission: `kernel/pipeline/job_universe.py::LoadUniverseJob`.

## Adding a New Strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
cd backtesting/foo && lean backtest .
python -m live.runner --strategy foo --broker paper --once
```

---

## Development Rules

### 0. Execute immediately, never wait for next session (2026-05-18 mandate, re-emphasized 2026-05-20)

**User verbatim** (2026-05-18): "任何 planned job 马上开工, 不要等下个 session". **Re-emphasis 2026-05-20**: "我说过很多次了！不要推迟到下个session！" (after I violated by proposing "明天重跑 ..." for a planned eval).

When a job is planned (in this session's task list, in roadmap.md, or just verbally agreed), START IT NOW. Don't write "let me schedule for next session" or "I'll do this tomorrow". The user is here, the context is loaded, the env is warm — execute.

**Sessions are MY implementation detail, not the user's project unit.** A task too big for one session doesn't get split into "today" + "tomorrow" — it just runs as far as it can. The user resumes themselves when ready.

Examples of what NOT to say (ALL §0 violations):
- "Will do in next session"
- "Let me schedule wakeup for tomorrow"
- "Planned for later sprint"
- "Will start when prerequisites ready" (instead: start what CAN start, in parallel)
- "Next step: rerun X tomorrow" / "明天重跑 X" / "下次 session 继续"
- "Wait for verdict, then ship Y" (when Y can prep in parallel)
- ANY sentence with "next session", "tomorrow", "later this week", "明天", "下次" about agreed work

What TO do:
- Concurrent BG jobs when ROI permits (multiple training runs, multiple sims, multiple data fetches)
- Foreground tasks that don't compete for the BG resource (CPU vs MPS GPU)
- Maximize the user's session time. Their attention is the rate-limiting resource.
- If a long compute is needed, START it BG and report estimated time + monitor.
- When asking for triage, frame as "which order should I execute" NOT "should we save X for later"

**Self-check before any "later" / "next" word**: would the user be happier if this work were running BG right now, vs deferred? If yes — start it.

Caveat: still respect risk gates (don't ship to live without §5.2 sanity, don't promote without §5.13.4a Tier 3, don't skip preflight). But "respect gates" ≠ "defer to next session" — gates fail fast, then execute the next plan.

**Source incidents** (chronological violations I committed):
- 2026-05-19 14:30 PT: shipped HF Trainer + eval drivers without §5.2 sanity (P0-1 leakage rooted here)
- 2026-05-20 11:18 PT: proposed "下一步建议: 明天重跑 PatchTST + iTransformer + FiLM 三方对比" after partial P0 fixes — user response: "我说过很多次了！不要推迟到下个session！"

### 1. Code is the source of truth
When code and doc conflict, **code wins, doc gets corrected**. Stale docs mislead future-Claude into writing wrong code. For renquant_104 specifically, start at `kernel/pipeline/pp_inference.py::InferencePipeline.run`.

### 1b. Every logical unit is a Task, Job, or Pipeline
- **Task**: atomic step; reads/writes `InferenceContext`; returns `False` to short-circuit the Job.
- **Job**: sequential Task chain with a `should_skip(ctx)` gate. May run serially or via `run_parallel()` for per-ticker work.
- **Pipeline**: orders Jobs into phases.

New decision logic always wires in as Task → Job → Pipeline. No hand-written loops bypassing the orchestration. Add paired alignment tests in `tests/test_panel_alignment.py` or `tests/test_policy_alignment.py`.

### 1c. Split every complex structure
Each Task ≤ 50 lines (soft target), single-responsibility (one of `{extract, validate, compute, transform, persist, emit}`), with its own unit test asserting only its ctx mutations. Tasks communicate via documented ctx fields (`ctx.X` public; `ctx._job_*` private). Reference split pattern: `kernel/portfolio_qp/{tasks.py, job_qp.py, task_joint_qp.py}` (the 5-Task QP refactor). Never start with a monolith intending to "split later". Promote to `run_parallel()` only when independent rows AND measured speedup ≥ 1.5×.

### 2. Tests for every feature — and every bug
Every policy in notebook and LEAN has a paired test. **Every bug-fix ships with a regression test that would fail before the fix.** Run `python -m pytest tests/ -v` before commit/push. Bug-fix workflow: (1) reproduce with a failing test, (2) land fix, (3) keep both in the same commit.

### 2a. Promotion thresholds aren't floors for theoretically-sound wins
**Default:** APY win ≥ +2 pts on 27-mo OOS = promote to golden. **Exception** (variables rigorously controlled): live/sim parity fixes, theory-aligned wins where predicted magnitude matches, and mechanism-clean changes with positive margin ship even at < +2 pt. **Not exceptions:** new strategies, hyperparameter sweeps, panel retrains.

### 2b. Unexpected A/B results = audit before accepting
When a theory predicts X and the result is ¬X, the first hypothesis is "my implementation has a bug or my assumptions were wrong" — not "the theory is wrong". Checklist before shipping a negative finding:
1. Per-bar log of new Task's inputs on ≥3 sample bars — sane?
2. Reason through every input/output independently: with all inputs correct, what would we *expect*?
3. Re-read commit defaults — does the GOLDEN variant in the A/B preserve baseline?
4. Check if any other task/config reads the same data — interaction possible?

### 3. Git commits — sync everything, guard secrets
Commit and push all changed files. Before every commit: `git status` for untracked/modified, then if any file contains sensitive data, gitignore-first then commit `.gitignore` then handle the file. Currently gitignored: `.env`, `live/logs/`, `data/`, `backtesting/data/`, `backtesting/*/backtests/`. If it's not gitignored, it should be in the remote.

### 3a. PR-based workflow — STRICT, all 10 repos (2026-05-30 mandate)

**User verbatim** (2026-05-30): *"future code changes should be PR based! this is a strict rule for all repos! enforce it! main branch of all repos should be protected from now!"*

**Hard rule, no exceptions, applies to ALL 13 renquant repos** (umbrella `RenQuant` + 12 subrepos under `hallovorld/`):
- **11 public** (server-side protected 2026-05-30): `RenQuant`, `renquant-artifacts`, `renquant-backtesting`, `renquant-base-data`, `renquant-common`, `renquant-execution`, `renquant-model-gbdt`, `renquant-model-patchtst`, `renquant-orchestrator`, `renquant-pipeline`, `renquant-strategy-104`.
- **2 private** (no server-side; agent-rule + pre-push hook enforcement): `renquant-model`, `renquant-state-backup`.
- **Future renquant-\* repos** automatically covered by the agent rule. Apply server-side protection at creation time (`gh api ... /branches/main/protection`); if private, install hook via `bash scripts/install_pr_hook.sh <repo-path>`.
1. **NEVER commit directly to `main`.** Always create a feature branch first (`feat/`, `fix/`, `chore/`, `docs/`, `bug/` prefix).
2. **Open a GitHub PR** via `gh pr create --base main --head <branch>` after pushing the branch. Body must include: change summary, test evidence, and rollback notes if production-touching.
3. **Self-merge allowed** (solo dev) but the PR is required as the audit surface. Use `gh pr merge --merge|--squash <PR#>` (default `--merge` so merge commit retains branch lineage).
4. **NEVER run `git push origin main` from a branch checkout.** Period. Server-side branch protection on 11 public repos will reject; for the 2 private repos (`renquant-model`, `renquant-state-backup`) the local pre-push hook + this rule are the enforcement.
5. **Server-side protection** (already applied 2026-05-30 to 11 public repos): `enforce_admins=true`, `required_pull_request_reviews.required_approving_review_count=0`, `allow_force_pushes=false`, `allow_deletions=false`. The 2 private repos (`renquant-model`, `renquant-state-backup`) not server-protected on free plan — agent rule + local pre-push hook are the gate.
6. **Pre-push hook installed on every local clone** via `bash scripts/install_pr_hook.sh --all`. Blocks `git push origin main` even when server-side is unavailable. Re-run after `git clone` of any new renquant repo.
7. **Reverses the deleted 2026-05-27 verbal-merge convention** (`feedback_no_pr_verbal_merge.md`). That convention's rationale (solo dev = no second pair of eyes) is overridden by the explicit 2026-05-30 mandate. Verbal approval is still gating ("ok merge?"), but the *mechanism* is now `gh pr merge`, not `git merge --no-ff` on local `main`.

**Workflow template:**
```bash
# 1. Branch first
git checkout -b feat/foo-bar
# 2. Make changes, commit
git add -A && git commit -m "..." 
# 3. Push branch
git push -u origin feat/foo-bar
# 4. Open PR
gh pr create --base main --head feat/foo-bar --title "..." --body "..."
# 5. After verbal approval, merge
gh pr merge --merge --delete-branch
```

**Source incident:** 2026-05-30 session — user halted free-flow chat to lay down the rule. Captured before next commit to ensure agent compliance.

### 3b. Sync-from-remote-before-work — STRICT, all 13 renquant repos (2026-05-30 multi-agent mandate)

**User verbatim** (2026-05-30): *"告诉所有repo，记得经常sync from remote！现在是多agent互相协作！"*

Multi-agent collaboration (Claude main session + codex + user + future agents) means each agent starts cold and assumes nothing about its local clone's freshness. **Without an explicit pull-from-remote step at every task boundary**, agents work on stale state, create duplicate work, miss in-flight fixes from other agents, and produce silent merge conflicts.

**Hard rule** for every renquant repo (`RenQuant` umbrella + all 12 subrepos):

```bash
# At the start of EVERY task, before any edit/commit/PR:
git fetch origin
git checkout main && git pull --ff-only origin main

# Before opening a PR or merging:
git fetch origin && git rebase origin/main   # if working on a feature branch
```

**Mandatory sync points:**
1. Starting any new task (even resuming after a 10-minute break)
2. Before editing any file you haven't touched in the current task
3. Before opening any PR (rebase feature branch on latest `origin/main`)
4. After receiving a task-notification or codex-merged event
5. Before declaring a PR ready for verbal-approve + merge

**Sync ALL repos when work touches multiple:**
```bash
for d in /Users/renhao/git/github/RenQuant /Users/renhao/git/github/renquant-*; do
    [ -d "$d/.git" ] && (cd "$d" && git fetch origin -q && git pull --ff-only origin main 2>&1 | grep -v "^Already up to date")
done
```

**Source incidents** that triggered this rule (all 2026-05-30):
- Codex pushed `9982de8` (umbrella) and `7f3cd14` (subrepo pipeline) to in-flight PRs while my local main was unaware. I had to manually `git fetch + git show` to discover the hardening.
- Codex opened renquant-backtesting PR #6 + #7 (refining PR #5 + adding review protocol) while I was working on unrelated tasks. Without periodic sync, I'd have missed both.
- Cross-repo recorder lift PRs (7 PRs across common / pipeline / model / base-data / artifacts / strategy-104 / backtesting) all landed in ~1 hour. Any agent without sync would have worked against stale dep pins.

**Cross-repo aware:** When you `pull` one repo, also check sibling renquant repos for recent merges. Even if your current task only touches one repo, downstream Phase 1 byte-equivalence tests assert state across multiple — drift on one repo can break tests on another.

**Inherits to subrepos:** This rule, like §3a, is renquant-wide canon. Subrepo CLAUDE.md files do NOT need to repeat it — they inherit via this umbrella declaration.

### 4. Keep docs current
After any non-trivial change, sync: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/ops/schedule.md`](doc/ops/schedule.md), [`doc/roadmap.md`](doc/roadmap.md), and this file. Code wins on conflict.

### 5. Engineering Principles

Each rule names the bug-class it prevents. Source incidents: [`doc/archives/audits/2026-04-28-deep-audit.md`](doc/archives/audits/2026-04-28-deep-audit.md), [`2026-04-28-nvts-buy-postmortem.md`](doc/archives/audits/2026-04-28-nvts-buy-postmortem.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md).

**5.1 Run relevant tests before every commit/push.** Even "obviously correct" scripts get a 1-second `python -c "import the_module"` check before overnight runs.

**5.2 Every new number ships with at least one sanity check.** Mandatory triad — A/A (resplit → does lift persist?), shuffled-label (IC ≈ 0), time-shift placebo (IC ≈ 0). Without one, the number is a guess. Track F's +98bp "win" was regime-persistence fitting that the placebo would have caught.

**5.2a Audit the data-pipeline foundation BEFORE reporting numbers from new training code.** (2026-05-20 walk-forward leakage incident, see `doc/research/2026-05-20-jiantaoshu.md`.) Any new training / eval script that depends on a splitter, label, or calibrator MUST audit those dependencies at WRITE time:
1. **Splitter invariant**: `max(train_date) + label_lookahead_days < min(val_date)`. Grep the actual `assign_split_column` / equivalent for an `embargo` or `purge` parameter. **Default to absent = bug.** Use `kernel.purged_cv.PurgedKFold` or pin embargo explicitly.
2. **Label causality**: every feature input to the model has timestamp `t`; label uses only data from `t+1` to `t+lookahead_days`. Grep feature builder for `iloc[-1]` / `tail(1)` / `.last(...)` broadcast-to-history patterns — those are look-ahead red flags.
3. **Calibrator scope**: calibrator fit on train rows ONLY (NOT val OR test). Verify the script reads `split_label == "train"` (not `!= "val"` which includes test).
4. **§5.2 sanity triad executed** on the new code path before the FIRST log line reporting an IC/Sharpe number. If `--label-shift-days 10` doesn't drop IC to ≈ 0, there is leakage — STOP, audit, do not report numbers.

**Pinned invariant**: `tests/test_walk_forward_splits.py::TestEmbargo::test_no_train_row_has_label_window_overlapping_val` (added 2026-05-20).

**5.3 Every "fix" names the invariant that prevents the entire bug class.** A patch fixes the symptom; a fix names "what invariant would have made this impossible". Example: NGBoost feature drift → `max_feature_drift_pct` hard guard + config-fingerprint stamping (architectural impossibility), not "retrain the head".

**5.4 Don't edit on-disk files of running scripts.** Mid-run edits to chain scripts are silent failures. Stop+restart, or use `ScheduleWakeup`.

**5.5 Production-touching changes rehearse rollback.** "We have an auto-revert script" ≠ "rollback works". Manually trigger the rollback path on a non-prod copy and verify post-rollback state is fully self-consistent (config + model + state files all aligned). Production-touching = anything live runner / launchd reads, plus models in `artifacts/`.

**5.6 "Fixed" = full 24h audit clean.** A bug is fixed only when (a) patched, (b) regression test green, (c) every file touched in last 24h re-audited end-to-end, (d) every up/downstream consumer re-audited, (e) zero open issues across all of them. Show the audit log. This gates re-running any post-fix experiment.

**5.7 Document every failed experiment in [`doc/research/failed-experiments-log.md`](doc/research/failed-experiments-log.md).** Hypothesis, implementation, exact numbers, sanity check, conclusion, reproduction recipe — all required, written same day. Without the recipe an independent agent cannot verify; we will burn 3-6 hours of compute re-discovering 6 months later.

**5.8 Status reports use concepts, not code.** Avoid commit hashes, file paths, variable names, task IDs, bug labels in status briefings. "面板排序模型" not `panel-ltr.json`. Code-level identifiers belong in commit messages and direct technical answers only.

**5.9 Multi-track plans ship as ASCII timeline.** Tracks on rows, time on X-axis, dependencies as arrows. Apply when ≥2 sequential phases OR ≥2 parallel work streams OR a stop-gate followed by branching. Example:
```
[现在]              [+2h]               [+3h]                [+并行]
P0 fixes ─────────→ retrain ──────────→ sim ──────────────→ promote/reject
                     ↓
                    P1 fixes 同时跑
```

**5.10 Saturate hardware.** M4 Pro = 14 cores + 48 GB RAM. Before any long compute job: `OMP_NUM_THREADS=14`, `MKL_NUM_THREADS=14`, `OPENBLAS_NUM_THREADS=14`; XGBoost `nthread=14` / `n_jobs=-1`; sklearn `n_jobs=-1`. Datasets < 5 GB load fully into RAM. **Verify** ≥80% user CPU via `top -l 1 | grep CPU` after dispatch — fix bottleneck BEFORE letting it eat hours.

**5.11 Experiment design optimizes for time-to-answer.** Decision tree before any multi-hour run:
0. **No-run path first**: before launching a new experiment, explicitly ask
   "can code review, data audit, procedure audit, an existing trace, or a
   focused unit/integration test answer this?" If yes, do that first. New
   experiments are for unresolved empirical questions, not for debugging
   invariants that can be proven from code/data already on disk.
1. **Range-finding** ("does X work at all?") → top-down single endpoint, 30 min wallclock.
2. **Optimization** ("find best subset/hyperparam") → ONLY after range-finding shows X works.
3. **Diagnostic** ("why did Y fail?") → §5.2 sanity sequence FIRST, THEN mechanism.

Sunk-cost guard: kill the experiment when mid-run evidence already answers the question.

**5.11a Never idle behind a running experiment.** Once a long experiment is
running, immediately switch to non-competing work: code review, data-flow
audit, decision-trace audit, cleanup, regression tests, docs, or the next
bounded feature/fix. Waiting is allowed only when the next action truly depends
on the result or additional work would contend for the same scarce resource
(for example MPS/GPU memory). Record what was done while the run was active.

**5.11b Experiments must reuse existing evidence before spending compute.**
Before re-running a sim/WF/training job, inspect prior artifacts/logs/traces and
write down why they are insufficient (stale code, missing fields, failed
contract, wrong regime split, etc.). If the only gap is missing analysis, write
the analyzer and use existing data. Re-running without first checking available
evidence is a process bug.

**5.12 Default to canonical references; never reinvent.** First question for any algorithmic / data / model decision: *what does Qlib / cvxportfolio / scikit-learn / PyPortfolioOpt / the canonical paper do?* Reference set:
- Cross-sectional ranking & features: `microsoft/qlib` (Alpha158, LinearModel, LGBModel, TransformerModel)
- Convex portfolio optimization: `cvxportfolio` (Boyd) + CVXPY
- Asset-pricing factors: Kelly-Gu-Xiu RFS 2020 firm-characteristics list
- Risk + transaction cost: Almgren-Chriss 2000, Ledoit-Wolf 2004
- Time-series ML: PatchTST (Nie 2023), TFT (Lim 2021), Qlib `pytorch_*_ts.py`

"Backed by literature" means **READ**, not name-dropped. Mandatory checklist for non-trivial design decisions:
1. Cite specific source (paper title + year, or `org/repo:file.py` + commit SHA).
2. Read it.
3. Confirm the decision matches the source — same hyperparameters, preprocessing, evaluation. List intentional divergences with reason.
4. If no source — mark "exploratory, will tune via A/B".

Apply to: model architecture, loss, optimizer, preprocessing, label construction, train/val/test split, evaluation metrics, sanity-test design, feature engineering. Decoration-citing burned ~2h on `alpha158_lite` (8 substantive deviations from canonical Qlib alpha158); faithful replication tripled test IC.

### 5.13 — 2026-05-09 audit anti-patterns

**5.13.1 Test fixtures lie.** Every fix has at least one test that calls through the actual `SimAdapter` / `RunnerAdapter` / `Pipeline` entry point — not a hand-constructed fixture. 124 σ-aware-stop-loss tests passed using `HoldingState(sigma=0.30)` while production has `state.sigma = None` (NGB OFF) — the prod path was never executed.

**5.13.2 Any new module is dead until grep proves prod imports it.** Before declaring shipped: `grep -rn '<module>' backtesting/<strategy>/{adapters,kernel,live,scripts}/`. Tests-only references = orphaned. (`smart_orders.py` was 156 lines + 42 tests with zero prod imports.)

**5.13.3 Every fix = regression-guard test class.** Name it `class Test<Bug>RegressionGuard` (or include "AUDIT REGRESSION GUARD" in the docstring) and pin the invariant. Reference: `tests/test_qp_wash_sale_cost_aware.py`.

**5.13.4 Single performance number = unverified claim.** Any APY / Sharpe / IC quoted in commit / doc / roadmap MUST be `mean ± std` from ≥5 runs (different seeds OR bar-orderings). Single-measurement claims are forbidden.

**5.13.4a Promotion gating uses the 3-tier methodology** (see [`doc/research/promotion-methodology.md`](doc/research/promotion-methodology.md)): Tier 1 (REJECT: mean ΔAPY < 0 ∧ mean ΔSharpe < 0); Tier 2 (SCREEN, NOT live-promotable: mean ΔAPY > 0, mean ΔSharpe ≥ 0, ≥ 4/N consistent, ΔSPY-α ≥ 0); Tier 3 (LIVE-PROMOTABLE: Tier 2 + DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 with t > 3.0). No live config flip without Tier 3. Run `python scripts/analyze_experiments.py` at the end of every multi-config batch.

**5.13.5 One business decision = one function.** Wash-sale, position cap, drawdown halt, post-stop cooldown, earnings blackout — exactly **one** implementation; all callers route through it. Adding a parallel impl requires deleting the original first.

**5.13.6 Cron cadence must be info-theoretically justified.** Any new cron's docstring answers: "this frequency adds N% new training-relevant info per tick vs the next-coarsest alternative". < 5%/tick = wrong cadence. Daily retrain on a fwd_60d-label model adds ~0.014%/day — cargo cult.

**5.13.7 Code change ≠ data update.** Any commit modifying a data-pipeline script includes in its message: `⚠️ requires data regen: <command>`. Until the regen runs and artifact mtime updates, the fix is **not** in production. (BUG #5: `pct_change(periods=4) → 252` was committed; on-disk parquet was still buggy.)

**5.13.8 Full pytest pass is a CI-gate invariant.** Every push touching `kernel/`, `adapters/`, `training_panel/`, `live/`, or `scripts/{train,daily,weekly,monthly}_*` includes a fresh `pytest tests/ --tb=no -q` snapshot in the commit message. ≥5 failures = broken commit, investigate before merge.

**5.13.9 "Audit subsystem X" = 4-step protocol.** (1) List X's data-flow inputs. (2) List X's outputs. (3) For each input→output edge, find or write an integration test. (4) For each `if X is not None` branch, prove via grep that X has a non-None path in production. Steps 1+2 only = audit at risk.

**5.13.10 `if optional_field is not None` defaults to dead code unless verified.** Any new code path with this pattern must `grep -r 'optional_field = '` and show ≥1 prod write site that fires under current config. Otherwise the path is dead and the "fix" must be reverted or guarded by a documented config flag ("requires <feature> ON").

**5.13.11 NaN / inf must be guarded explicitly.** `NaN > 0` is `False` — silently passes through "nonpositive-reject" guards. Any `>` / `<` on a value that *could* be NaN (broker returns/fills, config floats, OHLCV closes, fund features, calibrator output) must pair with `math.isfinite(x)`: `if math.isfinite(x) and x > threshold:`.

**5.13.12 Calibrator / regression output must be range-bounded.** Any output that feeds position sizing / Kelly / cash-budget math: `np.clip(out, lo, hi)` AT THE TRAIN SITE so the artifact stores sane bounds. Defense in depth: also clip at consumer site. (Production calibrator's `expected_return.y` ranged -0.30 to +**4.01** until `training_panel/global_calibrator.py:362-388`.)

**5.13.13 Side configs are loaded weapons.** Any `strategy_config.<label>.json` (label != "") must alias all `artifact_path` keys to side-paths containing the label. Pinned by `tests/test_side_config_artifact_paths.py`. Otherwise `--strategy-config-name <side>.json` silently overwrites production during retrain.

**5.13.14 No tool defaults to a hardcoded artifact filename.** Every load resolves through `cfg["ranking"]["panel_scoring"]["artifact_path"]`. After alpha158 promotion, `panel-ltr.json` became a 21-feat stub while inference loaded the 169-feat artifact — 5 tools were comparing nonsense for days because they hardcoded the filename.

**5.13.15 Safety gate in code ≠ safety gate enforced in production.** Every safety gate ships TWO artifacts: (a) gate function + tests, AND (b) a scheduled cron (plist + `.sh`) that invokes it WITHOUT override. If only (a), the gate is decoration. (`_check_wf_gate` was committed with tests, but every daily promote set `RQ_ALLOW_NO_WF=1` — theatrical gate.)

**5.13.16 Splitter embargo invariant.** (2026-05-20 walk-forward leakage, see `doc/research/2026-05-20-jiantaoshu.md`.) Every train/val splitter MUST enforce: `max(train_date) + label_lookahead_days < min(val_date)`. Pinned by `tests/test_walk_forward_splits.py::TestEmbargo`. Two splitters with divergent semantics = §5.13.5 violation — use canonical `kernel.purged_cv.PurgedKFold` and delete the parallel impl. Specific symptoms of an embargo-missing splitter:
- `assign_split_column(panel, cut)` has no `embargo_days` parameter
- Train rows in the last `lookahead_days` of train have label windows that reach into val
- IC numbers are inflated by leakage; different model capacities exploit leakage asymmetrically (simpler models often benefit more)

**5.13.17 After-session code audit.** (2026-05-20 audit found 17 P0 items I should have caught at write time.) Before declaring any refactor/new-feature session "done":
1. `git diff --name-only main..HEAD` enumerates touched files
2. For each touched .py, run the 6-category audit (§5.12, dead code, logic bugs, logging gaps, BUGS, missed parallelism)
3. For each new eval/training driver, audit its data dependencies (splitter, label, calibrator) end-to-end against known invariants (§5.2a)
4. Any code that REPORTS A NUMBER must have its data pipeline audited before the number is trusted in a status report
Do not wait for the user to request a deep audit — self-trigger at session end. User-triggered audit = the safety net you should not have needed.

### 5.14 — Design of Experiments (DOE) for parameter sweeps

**5.14.1 Use canonical DOE, not ad-hoc one-knob-at-a-time.** When tuning ≥2 numeric knobs that can interact (stop_loss × trailing × halt × σ-aware; learning_rate × depth × subsample; etc.), one-knob-at-a-time is **forbidden** — it misses 2-way interactions and over-counts main effects under confounding. Pick the methodology by stage:

| Stage | Question | Method | Runs | Reference |
|---|---|---|---|---|
| Screening | "Which knobs matter at all?" | Plackett-Burman | ~k+4 | Plackett & Burman 1946 *Biometrika* 33:305 |
| Screening (with 2-way) | "Which knobs + which pairs matter?" | 2-level Fractional Factorial 2^(k-p), Resolution IV+ | 8-32 | Box-Hunter-Hunter 2005 ch.6 |
| Optimization | "Where is the optimum?" | **Box-Behnken** or Central Composite Design | 13-46 | Box & Behnken 1960 *Technometrics* 2:455; Box & Wilson 1951 *J. Royal Stat. Soc.* 13:1 |
| Confirmation | "Does the predicted optimum actually win?" | 2-3 runs at predicted optimum + DSR/PBO | 3-5 | §5.13.4; Bailey-López de Prado 2014 *J. Portfolio Mgmt* 40(5):94 |

Default to **Box-Behnken at 3 levels** for ≤6 knobs in continuous-tunable regions inside the legal config domain. Quadratic response surface `y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²` captures main effects + 2-way interactions + curvature with ≈25 runs (4 knobs) or ≈46 runs (6 knobs).

**5.14.2 Always include center points.** 3-5 replicates at the design center serve two purposes: (a) lack-of-fit test for the quadratic surface, (b) `σ²_pure` estimate for proper response-surface inference. Single-seed-deterministic sims (RenQuant case) still need ≥1 center to anchor the surface, even if σ²_pure = 0.

**5.14.3 Cite the DOE library, not the reinvention.** Default tooling:
- `pyDOE2.bbdesign(k, center=N)` — Box-Behnken matrix
- `pyDOE2.ccdesign(k, alpha='r')` — rotatable CCD
- `pyDOE2.fracfact("a b c ab")` — fractional factorial
- `sklearn.preprocessing.PolynomialFeatures(degree=2)` + `LinearRegression` — fit surface
- `scipy.optimize.minimize(..., bounds=...)` — optimum on fitted surface

Re-deriving the design matrix by hand = §5.12 violation.

**5.14.4 Multiple-comparison correction is mandatory.** With N design points the winner's headline Sharpe is **inflated** by selection-bias. Apply both:
1. **Deflated Sharpe Ratio** (Bailey-López de Prado 2014). Sim runner already computes DSR with `n_trials = N_design_points`. Winner gates on DSR > 0 (not raw SR).
2. **Probability of Backtest Overfitting via CSCV** (Bailey, Borwein, López de Prado & Zhu 2015 *J. Comp. Finance* 14(1)). PBO > 50% → winner is overfit; reject. RenQuant code: `sim/runner.py::run_backtest_multi_seed` returns PBO when ≥2 seeds.

Quote multiple-tested numbers as `Sharpe_raw=X / DSR=Y / PBO=Z%` in any report — never just raw SR.

**5.14.5 Knob bounds come from baseline distribution analysis, not author preference.** Before picking levels, **extract the relevant percentile from baseline** — e.g., for `max_single_day_loss_pct`, pull the p90/p95/p99 of observed worst single-day-position-drops; for `stop_loss_pct`, pull the cumulative-loss percentile of trades that exited via stop. Bounds at empirically meaningful breakpoints, not round numbers. Round-number bounds (5%, 10%, 15%) are §5.12 violations — "exploratory, will tune via A/B" gets re-tuned forever.

**5.14.6 Interaction-aware reporting.** Every DOE report must show:
1. Main-effects plot (each knob's β coefficient with confidence interval)
2. 2-way interaction plots for any significant βᵢⱼ
3. Contour / heatmap of the fitted surface in the top-2 knob plane
4. Pareto frontier across competing objectives (APY vs MaxDD, etc.)
5. The optimum is the point on the surface, **not** the best of the N evaluated runs (the surface is the inference; the runs are samples).

---

## Documentation Index

**Foundation**: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/arch/indicators.md`](doc/arch/indicators.md), [`doc/arch/models.md`](doc/arch/models.md)

**Components**: [`panel-ltr`](doc/components/panel-ltr.md) · [`buy-logic`](doc/components/buy-logic.md) · [`sell-logic`](doc/components/sell-logic.md) · [`calibration`](doc/components/calibration.md) · [`rotation`](doc/components/rotation.md) · [`portfolio-qp`](doc/components/portfolio-qp.md) (cvxpy CLARABEL) · [`databases`](doc/components/databases.md) · [`training-pipeline`](doc/components/training-pipeline.md)

**Operations**: [`golden-config`](doc/ops/golden-config.md) · [`usage`](doc/ops/usage.md) · [`setup`](doc/ops/setup.md) · [`environment`](doc/ops/environment.md) · [`schedule`](doc/ops/schedule.md) — single source of truth for cron / weekly / monthly cadence

**Research**: [`papers-implemented`](doc/research/papers-implemented.md) · [`scoring`](doc/research/scoring-research.md) · [`rotation`](doc/research/rotation-research.md) · [`watchlist-100`](doc/research/watchlist-100.md) · [`ic-evaluation-methodology`](doc/research/ic-evaluation-methodology.md) · [`failed-experiments-log`](doc/research/failed-experiments-log.md)

**Experiments**: [`ab-journal`](doc/experiments/ab-journal.md) · [`panel-training-runs`](doc/experiments/panel-training-runs.md) · [`sim-ab-results`](doc/experiments/sim-ab-results.md)

**Roadmap + audit**: [`doc/roadmap.md`](doc/roadmap.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md), [`doc/AUDIT_2026-05-09_SUMMARY.md`](doc/AUDIT_2026-05-09_SUMMARY.md)

**History**: `doc/archives/sessions/`, `doc/archives/audits/` — `git log --follow` for provenance.

---

## General Coding Guidelines

**Bias toward caution. Use judgment for trivial tasks.**

**Think before coding.** State assumptions explicitly. Multiple interpretations? Present them. Simpler approach exists? Say so. Unclear? Stop and ask.

**Simplicity first.** No features beyond what was asked. No abstractions for single-use code. No flexibility/configurability that wasn't requested. No error handling for impossible scenarios. 200 lines that should be 50 → rewrite.

**Surgical changes.** Don't "improve" adjacent code. Don't refactor what isn't broken. Match existing style. Mention unrelated dead code; don't delete it. Remove imports/variables/functions that YOUR changes made unused.

**Goal-driven execution.** Transform tasks into verifiable goals: "add validation" → "tests for invalid inputs, then make them pass". For multi-step tasks, state a brief plan with verify steps before starting.
