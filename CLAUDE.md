# CLAUDE.md

Authoritative agent guidance for the RenQuant codebase. Cross-repo canon —
all 13 renquant repos inherit the workflow rules in §3.

**Table of contents**
- §1 — [PRIME DIRECTIVE](#1--prime-directive--renquant-is-regime-conditional)
- §2 — [What RenQuant is, current state, quick navigation](#2--what-renquant-is)
- §3 — [Workflow rules (cross-repo canon)](#3--workflow-rules-cross-repo-canon)
- §4 — [Environment & invocation modes](#4--environment--invocation-modes)
- §5 — [Architecture (Task / Job / Pipeline)](#5--architecture)
- §6 — [Execution discipline](#6--execution-discipline)
- §7 — [Engineering principles](#7--engineering-principles)
- §8 — [Status report style](#8--status-report-style)
- §9 — [Design of Experiments](#9--design-of-experiments-doe)
- §10 — [General coding guidelines](#10--general-coding-guidelines)
- [Appendix · Historical source incidents](#appendix--historical-source-incidents)

---

## 1 · PRIME DIRECTIVE — RenQuant is regime-conditional

🔴 **Non-negotiable** (user mandate 2026-05-14). Every feature, knob, and
experiment is designed and evaluated through a regime-conditional lens.
Pooled-mean metrics across regimes are MISLEADING and produce false NEITHER
verdicts.

| Rule | Concrete shape |
|---|---|
| **1.1** Per-regime config | Every numeric knob lives under `regime_params.<REGIME>.<knob>` — never as a global scalar. |
| **1.2** Per-regime experiment design | Every experiment starts with "which regime does this thesis apply to?" Regime-neutral theses are red flags. |
| **1.3** Per-regime first evaluation | Reports show per-regime numbers FIRST, pooled-mean SECOND. Regime-stratified test is the PRIMARY signal. |
| **1.4** Detector quality is P0 | If the detector mislabels (2022 Q2 bear → BULL_CALM bug), all regime-conditional logic is theatrical. Regression-test the detector whenever it changes. |
| **1.5** Promotion is regime-conditional | A change that wins in regimes A+B and loses in C is promotable ONLY as a regime-conditional config edit (enable in A+B, default in C). NEVER a global flip. |

**Knobs that should be per-regime** (most are not yet — wire progressively):
`long_short.*`, `stop_loss_pct`, `trailing_stop_*`, `max_holding_days`,
`take_profit_pct`, `drawdown_*_pct`, `kappa`, `qp_dw_max`, `qp_turnover_penalty`,
`vol_target`, `kelly_scale`, `cash_reserve_pct`, `min_model_score`,
`bear_defensive_*`, `defensive_tickers`, `entry_mode`, `min_price_move_pct`.

**Where to read first**:
- Regime detector: `kernel/regime.py` + `kernel/pipeline/task_regime.py` (keep in sync)
- Regime params: `strategy_config.golden.json::regime_params.{BULL_CALM,BULL_VOLATILE,BEAR,CHOPPY}`
- Promotion methodology: `doc/research/promotion-methodology.md` (3-tier)
- Source: `memory/feedback_regime_conditional_strategy.md` + `doc/research/2026-05-14-longshort-clean-FINAL.md`

---

## 2 · What RenQuant is

Personal quantitative trading workstation for Apple Silicon. Glass-box
pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live
trading (Alpaca / IBKR). Statistically interpretable, strictly decoupled.

**Active strategy**: `renquant_104` (panel-LTR cross-sectional ranking, HF
PatchTST production primary since 2026-06-05; previous XGBoost
`rank:pairwise` scorer retained as readonly shadow / rollback).
`renquant_103` archived; rollback only.

**Live mode**: Alpaca PAPER for cron schedules (2026-05-11 safety mandate).
Explicit `"e2e"` → LIVE Alpaca real money (§4.1).

**NAV invariant** (Bug C regression guard, pinned by
`tests/test_sim_nav_t2_settlement.py`):
```
NAV ≡ free_cash + pending_settle + Σ(shares × price)
```

### 2.1 · Mainline memory + recent sessions

**Resume any RenQuant 104 continuation with**
[`doc/research/2026-05-23-mainline-memory.md`](doc/research/2026-05-23-mainline-memory.md) —
the live mainline for the current repair campaign.

Day-to-day status: `git fetch origin -q && git log --oneline -30 origin/main`.
Read `origin/main` (not local `main`) so the result reflects what other agents
shipped even if you're on a feature branch or your local `main` is stale.

### 2.2 · Quick navigation

| You want to | Start at |
|---|---|
| Resume mainline work | `doc/research/2026-05-23-mainline-memory.md` |
| Understand the prod path | `kernel/pipeline/pp_inference.py::InferencePipeline.run` |
| Add a new Task | §5 → reference `kernel/portfolio_qp/{tasks.py, job_qp.py}` or `kernel/preflight_pipeline/` |
| Open a PR | §3.1 |
| Pull latest before working | §3.2 |
| Run e2e | §4.1 |
| Run daily full | §4.2 |
| Promote a model | §7.4 (3-tier promotion) |
| Evaluate an A/B | §7.2 (sanity triad) + §9 (DOE) |
| See what other agents shipped | `git fetch origin -q && git log --oneline -30 origin/main` |
| Look up old sessions | `git log --oneline --all -- doc/archives/` |

**Documentation index**:
- Foundation: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/arch/indicators.md`](doc/arch/indicators.md), [`doc/arch/models.md`](doc/arch/models.md)
- Components: [`panel-ltr`](doc/components/panel-ltr.md), [`buy-logic`](doc/components/buy-logic.md), [`sell-logic`](doc/components/sell-logic.md), [`calibration`](doc/components/calibration.md), [`rotation`](doc/components/rotation.md), [`portfolio-qp`](doc/components/portfolio-qp.md), [`databases`](doc/components/databases.md), [`training-pipeline`](doc/components/training-pipeline.md)
- Operations: [`golden-config`](doc/ops/golden-config.md), [`usage`](doc/ops/usage.md), [`setup`](doc/ops/setup.md), [`environment`](doc/ops/environment.md), [`schedule`](doc/ops/schedule.md)
- Research: [`papers-implemented`](doc/research/papers-implemented.md), [`scoring`](doc/research/scoring-research.md), [`rotation`](doc/research/rotation-research.md), [`ic-evaluation-methodology`](doc/research/ic-evaluation-methodology.md), [`failed-experiments-log`](doc/research/failed-experiments-log.md)
- Experiments: [`ab-journal`](doc/experiments/ab-journal.md), [`panel-training-runs`](doc/experiments/panel-training-runs.md), [`sim-ab-results`](doc/experiments/sim-ab-results.md)
- Roadmap + audits: [`doc/roadmap.md`](doc/roadmap.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md)
- History: `git log --oneline --all -- doc/archives/`

---

## 3 · Workflow rules (cross-repo canon)

Applies to ALL 13 renquant repos: umbrella `RenQuant` + 12 subrepos under
`hallovorld/`.

### 3.0 · Agent control contract (read first)

The recurring systemic agent failure is contained by **external controls**, defined once in
`renquant-orchestrator` and inherited cross-repo:
- [`doc/AGENT-RETROSPECTIVE.md`](https://github.com/hallovorld/renquant-orchestrator/blob/main/doc/AGENT-RETROSPECTIVE.md)
  — controls C1–C5 + §7.1, the per-PR Codex review checklist.
- [`doc/memory/`](https://github.com/hallovorld/renquant-orchestrator/tree/main/doc/memory)
  — externalised executive memory: **LONG** (binding ledger) · **MID** (plan) · **SHORT** (state); read LONG→MID→SHORT before acting.

Non-negotiable for every agent: **report bottom-line-first** (conclusion + the decision needed
+ one number + `[VERIFIED]`/`[GUESS]`); **no "X works/fails" without the §4(b) evidence block**;
**never write production paths** (data/parquet, strategy configs, live artifacts/state, committed
WF corpora) — experiments in isolated worktrees only; **every PR carries `doc/progress/<date>-<slug>.md`**;
**never self-merge**; Codex review approval is the *intended* merge gate for agent PRs (an
`APPROVED` review by the *other* agent — consistent with §3.1 #5; today rulesets require only
1 approval from any actor, so "Codex specifically" is convention until a required-reviewer rule
lands). Reviewers submit findings only; the PR owner applies fixes. A reviewer
must never commit or push to a peer-owned PR branch. A PR branch has exactly
one GitHub commit identity, the PR creator; any extra attribution, including a
`Co-Authored-By` trailer, blocks merge until the owner rebuilds the branch from
the target base as clean single-identity history. Enforcement = Codex review +
mechanical hooks, **not** this doc.

### 3.1 · PR-based workflow — STRICT

**User mandate** (2026-05-30): *"future code changes should be PR based! this
is a strict rule for all repos! enforce it! main branch of all repos should
be protected from now!"*

**Repos covered**:
- 11 public, server-side protected: `RenQuant`, `renquant-artifacts`, `renquant-backtesting`, `renquant-base-data`, `renquant-common`, `renquant-execution`, `renquant-model-gbdt`, `renquant-model-patchtst`, `renquant-orchestrator`, `renquant-pipeline`, `renquant-strategy-104`.
- 2 private, agent-rule + pre-push hook enforcement: `renquant-model`, `renquant-state-backup`.
- Future `renquant-*` repos covered by the agent rule; apply server-side protection at creation time + `bash scripts/install_pr_hook.sh <repo-path>` if private.

**Hard rule**:
1. NEVER commit directly to `main`. Always feature-branch first (`feat/`, `fix/`, `chore/`, `docs/`, `bug/` prefix).
2. Open a PR via `gh pr create --base main --head <branch>`. Body must include: change summary, test evidence, rollback notes if production-touching.
3. Self-merge allowed (solo dev) but the PR IS the audit surface. Use `gh pr merge --merge|--squash <PR#>`.
4. NEVER `git push origin main` from a branch checkout. Server-side blocks public repos; agent rule + pre-push hook block private.
5. **Auto-merge (v2 Phase B, 2026-06-03)**: agent-authored PRs (`agent:claude` / `agent:codex` label) MAY auto-merge via the `agent-auto-merge` workflow if EITHER the PR carries the `agent:auto-merge` label OR the repo variable `AGENT_AUTO_MERGE_DEFAULT=true`. The workflow gates on 8 conditions (per [`doc/ops/agent-automation-v2-design.md §3.4`](doc/ops/agent-automation-v2-design.md#34--gap-4--no-auto-approve-no-auto-merge)) — `APPROVED` review on latest head, zero `CHANGES_REQUESTED`, all required checks green, branch up to date with base (auto-rebased), no stop labels (`agent:manual-hold` / `agent:rebase-conflict` / `agent:cost-cap`), no `agent:fix:*:attempt-3` retry exhaustion, paired-mirror sister merged if applicable. PR still exists; reviews still happen; `APPROVED` is the audit trail replacement for "verbal approval". Human-authored PRs and any PR without the opt-in label still merge manually.

**Server-side protection settings** (applied 2026-05-30 to 11 public repos):
`enforce_admins=true`, `required_pull_request_reviews.required_approving_review_count=0`,
`allow_force_pushes=false`, `allow_deletions=false`.

**Pre-push hook**: `bash scripts/install_pr_hook.sh --all` on every local clone.
Re-run after any fresh `git clone` of a renquant repo.

**Workflow template** (includes §3.2 sync at every required boundary):
```bash
# 0. Sync first (§3.2). Branch off latest origin/main.
git fetch origin
git checkout main && git pull --ff-only origin main
git checkout -b feat/foo-bar

# 1. Edit, commit.
git add -A && git commit -m "..."

# 2. BEFORE pushing / opening the PR — rebase on latest origin/main
#    so the PR isn't stale before review starts (§3.2).
git fetch origin
git status --short                # confirm clean working tree
git rebase origin/main

# 3. Push + open PR.
git push -u origin feat/foo-bar
gh pr create --base main --head feat/foo-bar --title "..." --body "..."

# 4. BEFORE declaring merge-ready, re-sync in case another agent merged
#    while the PR was being reviewed (§3.2).
#    NOTE: the `agent-pre-merge-rebase` workflow now enforces this
#    server-side for any agent-authored PR; manual rebase is still
#    recommended but no longer load-bearing for the auto-merge path.
git fetch origin
git rebase origin/main && git push --force-with-lease

# 5. Merge — either path:
#    (a) Auto: APPROVED review + 8-gate workflow merges (per §3.1 point 5).
#        Add the `agent:auto-merge` label for per-PR opt-in.
#    (b) Manual: `gh pr merge --merge --delete-branch` after verbal approval.
```

The two rebase steps (before-PR, before-merge-ready) are NOT optional —
they're the same §3.2 mandate that requires re-sync at each task boundary.
Skipping them is how stale PRs land.

### 3.2 · Sync-from-remote before every task — STRICT

**User mandate** (2026-05-30): *"告诉所有repo，记得经常sync from remote！现在是多agent互相协作！"*

Multi-agent collaboration (Claude main + codex + user + future agents) means
each agent starts cold and assumes nothing about its local clone's freshness.
Without an explicit pull-from-remote step at every task boundary: stale
state, duplicate work, missed in-flight fixes, silent merge conflicts.

**Hard rule** for every repo covered by §3.1:

```bash
# At the start of EVERY task, before any edit/commit/PR in this repo:
git fetch origin

# If already on a clean main, fast-forward it.
if [ "$(git branch --show-current)" = "main" ] && [ -z "$(git status --porcelain)" ]; then
    git pull --ff-only origin main
else
    git status --short --branch
fi

# Before opening a PR or declaring merge-ready from a feature branch:
git fetch origin
git status --short
git rebase origin/main
```

**Mandatory sync points**:
1. Starting any new task (even resuming after a 10-minute break)
2. Before editing any file you haven't touched in the current task
3. Before opening any PR (from a clean feature branch, rebase on latest `origin/main`)
4. After receiving a task-notification or codex-merged event
5. Before declaring a PR ready for verbal-approve + merge

**Sync ALL repos when work touches multiple** (branch-safe, never pulls into a feature branch):
```bash
for d in /Users/renhao/git/github/RenQuant /Users/renhao/git/github/renquant-*; do
    [ -d "$d/.git" ] || continue
    (
        cd "$d" || exit
        git fetch origin -q
        branch="$(git branch --show-current)"
        dirty="$(git status --porcelain)"
        if [ "$branch" = "main" ] && [ -z "$dirty" ]; then
            git pull --ff-only origin main 2>&1 | grep -v "^Already up to date"
        else
            printf '%s: fetched only (branch=%s dirty=%s)\n' \
                "$d" "${branch:-detached}" "$([ -n "$dirty" ] && echo yes || echo no)"
        fi
    )
done
```

**Cross-repo aware**: When you `pull` one repo, also check sibling renquant
repos for recent merges. Even if your current task only touches one repo,
Phase 1 byte-equivalence tests assert state across multiple — drift on one
repo can break tests on another.

**Propagation to subrepos**: This rule, like §3.1, is renquant-wide canon.
Subrepo CLAUDE.md files may point here rather than duplicating the full text,
but agents working from a subrepo must still execute this sync protocol
before editing or reviewing.

### 3.3 · Commits & secrets

Commit and push all changed files. Before every commit: `git status` first.
If any file contains sensitive data, gitignore it FIRST then commit
`.gitignore` THEN handle the file. Currently gitignored: `.env`, `live/logs/`,
`data/`, `backtesting/data/`, `backtesting/*/backtests/`. If it's not
gitignored, it should be in remote.

**Agent GitHub tokens** (Claude / Codex PATs) follow
[`doc/ops/agent-token-storage.md`](doc/ops/agent-token-storage.md): stored in
the OS Keychain (never `.env`, never a file, never pasted into chat), loaded per
agent via `scripts/agent_gh_env.sh`, and guarded by the pre-push secret-scan in
`scripts/install_pr_hook.sh`. A token exposed anywhere (including a transcript)
is rotated immediately.

### 3.4 · PR review protocol

Treat review findings as first-class PR artifacts.
- When a review finds a blocking or material issue, leave a PR comment with
  issue, risk, and intended fix before or alongside pushing code.
- If you directly patch a reviewed PR branch, add a follow-up PR comment
  with the fix commit, tests run, merge decision. Don't leave rationale only
  in chat.
- For paired cross-repo PRs (umbrella + subrepo same change), comment on
  EACH affected PR with the matching upstream/downstream commit and
  verification evidence before merging.

### 3.5 · Multi-repo code placement

New code goes in the repo that OWNS the subject (model research →
`renquant-model`; pipeline kernels → `renquant-pipeline`; etc.). **Never the
umbrella baseline** as the canonical source. The umbrella is the integration
layer; production execution currently flows through umbrella's `kernel.*`
imports because Phase 1 invariant is text-only byte-equivalent mirror.
Phase 5 will retire that bridge.

When the same file lives in both umbrella and a subrepo (Phase 1
byte-equivalent state), changes must land in BOTH — paired PRs — and tests
verify MD5 equivalence (`tests/test_c211_panel_pipeline_lift.py`, etc.).

### 3.6 · Keep docs current

After any non-trivial change, sync: [`doc/arch/overview.md`](doc/arch/overview.md),
[`doc/arch/strategy-104.md`](doc/arch/strategy-104.md),
[`doc/ops/schedule.md`](doc/ops/schedule.md), [`doc/roadmap.md`](doc/roadmap.md),
and this file. Code wins on conflict.

### 3.7 · Agent attribution + auto-review + auto-fix

Cross-repo agent automation (Claude, Codex, future) — branch identity,
PR attribution, auto-review of non-agent PRs, auto-fix on reviewer
findings — is canonicalized in
[`doc/ops/agent-automation.md`](doc/ops/agent-automation.md). That doc
is the single source of truth; subrepo `CLAUDE.md` (or future
`AGENTS.md`) MUST point here rather than duplicate the design.

Bottom-line rules every agent-opened branch / commit / PR must obey:

1. **Authorship label** — exactly ONE of `agent:claude` / `agent:codex` set at PR open ([§2.1](doc/ops/agent-automation.md#21--canonical-labels)).
2. **PR body footer** — ends with `Agent-Origin: <Name>` + the standard `🤖 Generated with [<Agent Name> Code]` line.
3. **Commit identity** — every PR branch has commit attribution only from the
   GitHub account that created the PR. Do not add `Co-Authored-By` trailers for
   agent provenance; use the PR body footer and visible review/fix/merge
   comments instead.
4. **Cross-agent invites** use `agent:fix:<name>` (executor-permission label) — NEVER add a second authorship label to invite another agent's G3.
5. **Stop knobs**: `agent:manual-hold` halts ALL agent automation on a PR; `agent:fix:<name>:attempt-3` halts that one agent's G3 retries.

Where the machinery actually lives:

- Reusable workflow templates: `RenQuant/.github/workflows/_agent-*-template.yml` (umbrella canon — referenced via `uses: hallovorld/RenQuant/...@main`)
- Per-repo wrappers (~25 lines): `.github/workflows/agent-{review,autofix,attribution-check}.yml` in each renquant repo
- Repo-local agent context: future per-subrepo `AGENTS.md` for tests / layout / forbidden-imports — distinct from this cross-repo canon

---

## 4 · Environment & invocation modes

Activate the project venv: `source .venv/bin/activate` (NOT conda). Docker
≥ 16 GB for LEAN. Alpaca creds in `.env` (gitignored). Full setup:
[`doc/ops/environment.md`](doc/ops/environment.md).

### 4.1 · "e2e" = LIVE Alpaca real money

When the user says **"e2e"**, **"daily e2e"**, **"run with alpaca account"**,
or any variant, run against the **LIVE Alpaca account with real money**. Do NOT
propose paper variants. Do NOT switch to `--broker alpaca-paper` or
`--broker paper` silently or via a clarifying question. `.env` only has LIVE
creds — paper API calls 401.

This overrides the 2026-05-11 PAPER safety mandate for explicit e2e
invocations only. PAPER mandate still applies to launchd cron schedules.
Locked in 2026-05-17 after user verbatim said:
*"我他妈的说了一万遍了！live account！写进claude.md！"*

```bash
nohup bash -c 'set -a; source .env; set +a; .venv/bin/python -m live.runner \
    --strategy renquant_104 --broker alpaca --once' \
  > logs/live_e2e/e2e_alpaca_live_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

### 4.2 · "daily full" / "full daily" = run both legs

When the user says **"daily full"**, **"full daily"**, **"跑 daily full"**, do
not treat a wrapper holiday / calendar / cadence skip as completion. Try the
wrapper if useful, but if `scripts/daily_104.sh` skips, immediately run the
direct full paths and report the complete decision-tree:

```bash
set -a; source .env; set +a
.venv/bin/python -m live.runner --strategy renquant_104 --broker alpaca --once
.venv/bin/python -m live.runner --strategy renquant_104 --broker readonly-alpaca \
    --once --strategy-config-name strategy_config.shadow.json
```

This means execute and surface hard-gate outcomes. It does NOT authorize
disabling WF / model / preflight hard gates, ignoring failed artifact
contracts, or inventing discretionary live orders. Explicit manual orders
require exact ticker, quantity/notional, order type.

### 4.3 · Workflow modes (other)

| Mode | Command | When |
|---|---|---|
| Research | Open notebook | Train + iterate, no Docker |
| Validation | `lean backtest .` (after `export_lean_watchlist.py --strategy X`) | Final OOS check |
| Analysis | `python scripts/analyze_backtest.py --strategy X` | Visualize finished backtest |
| Live (one-shot) | `python -m live.runner --strategy X --broker {paper,alpaca-paper,alpaca,ibkr} --once` | One-shot trade |
| Scheduled | macOS launchd via 5 plists for renquant_104 | Daily cron |

**LEAN data isolation**: LEAN reads `backtesting/data/equity/usa/daily/{sym}.zip`,
NOT `data/ohlcv/`. Run `export_lean_watchlist.py` before backtesting.
CLI + plist details: [`doc/ops/usage.md`](doc/ops/usage.md).

### 4.4 · Adding a new strategy

```bash
python scripts/new_strategy.py --name foo --symbol AAPL --type classification
cd backtesting/foo && lean backtest .
python -m live.runner --strategy foo --broker paper --once
```

---

## 5 · Architecture

Three pipelines own ALL decision logic. **Code is the source of truth** —
when code and doc conflict, code wins, doc gets corrected. For renquant_104,
start at `kernel/pipeline/pp_inference.py::InferencePipeline.run`.

| Pipeline | File | Phases / Jobs |
|---|---|---|
| `InferencePipeline` / `SellOnlyPipeline` | `kernel/pipeline/` | regime → drawdown → buy gates → sell (∥) → buy candidates (∥) → ranking → rotation → selection. Used by LEAN, live, sim. |
| `FullTrainingPipeline` | `kernel/pipeline/pp_training_full.py` | `BaselineTournamentJob → PanelTrainingJob → RecalibrationJob` |
| `PanelTrainingPipeline` | `training_panel/pp_panel_training.py` | `PanelDataJob → PanelFeatureJob → PanelAssemblyJob → PanelModelJob → PanelNGBoostJob → RefreshPanelCalibratorJob` |

LEAN / live / sim all enter through `InferencePipeline` via `LeanAdapter` /
`RunnerAdapter` / `SimAdapter`. Universe admission:
`kernel/pipeline/job_universe.py::LoadUniverseJob`.

### 5.1 · Every logical unit is a Task, Job, or Pipeline

- **Task** — atomic step; reads/writes `InferenceContext`; returns `False` to short-circuit the Job.
- **Job** — sequential Task chain with `should_skip(ctx)` gate. May run serially or via `run_parallel()` for per-ticker work.
- **Pipeline** — orders Jobs into phases.

New decision logic always wires as Task → Job → Pipeline. No hand-written
loops bypassing the orchestration. Add paired alignment tests in
`tests/test_panel_alignment.py` or `tests/test_policy_alignment.py`.

### 5.2 · Decompose by single responsibility

Each Task ≤ 50 lines (soft target), single responsibility — one of
`{extract, validate, compute, transform, persist, emit}`. Its own unit test
asserting only its ctx mutations. Tasks communicate via documented ctx fields
(`ctx.X` public; `ctx._job_*` private).

Reference split patterns:
- `kernel/portfolio_qp/{tasks.py, job_qp.py, task_joint_qp.py}` — 5-Task QP refactor
- `kernel/preflight_pipeline/` — 16-Task preflight refactor (2026-05-30)

Never start with a monolith intending to "split later". Promote to
`run_parallel()` only when independent rows AND measured speedup ≥ 1.5×.

---

## 6 · Execution discipline

### 6.1 · Execute now, never defer

**User mandate** (2026-05-18, re-emphasized 2026-05-20): *"任何 planned job
马上开工, 不要等下个 session"*, *"我说过很多次了！不要推迟到下个 session！"*

When a job is planned (in the task list, `roadmap.md`, or just verbally
agreed), START IT NOW. Sessions are MY implementation detail, not the user's
project unit. A task too big for one session doesn't get split into "today" +
"tomorrow" — it runs as far as it can, then the user resumes when ready.

**Forbidden phrases**: "next session", "tomorrow", "later this week", "明天",
"下次", "wait for verdict then ship Y", "schedule wakeup for tomorrow",
"planned for later sprint", "will start when prerequisites ready".

**What TO do**:
- Concurrent BG jobs when ROI permits (multiple trainings, sims, data fetches).
- Foreground tasks that don't compete for the BG resource (CPU vs MPS GPU).
- Maximize the user's session time — their attention is the rate-limiting resource.
- Long compute → BG, report estimated time + monitor.
- When asking for triage, frame as "which order should I execute" NOT "should we save X for later".

**Self-check before any "later" / "next" word**: would the user be happier
if this work were running BG right now? If yes — start it.

**Caveat**: still respect risk gates (don't ship live without §7.2 sanity,
don't promote without §7.4 Tier 3, don't skip preflight). But "respect gates"
≠ "defer to next session" — gates fail fast, then execute the next plan.

### 6.2 · Never idle behind a running experiment

Once a long experiment is running, immediately switch to non-competing work:
code review, data-flow audit, decision-trace audit, cleanup, regression tests,
docs, or the next bounded feature/fix. Waiting is allowed only when the next
action truly depends on the result or would contend for the same scarce
resource (MPS / GPU memory). Record what was done while the run was active.

### 6.3 · No-run path first

Before launching any new experiment, explicitly ask: *"can code review, data
audit, procedure audit, an existing trace, or a focused unit/integration test
answer this?"* If yes, do that first. New experiments are for unresolved
empirical questions, not for debugging invariants that can be proven from
code/data already on disk.

### 6.4 · Reuse existing evidence before spending compute

Before re-running a sim / WF / training job, inspect prior artifacts / logs /
traces and write down why they are insufficient (stale code, missing fields,
failed contract, wrong regime split). If the only gap is missing analysis,
write the analyzer and use existing data. Re-running without first checking
available evidence is a process bug.

### 6.5 · Saturate hardware

Hardware: **M4 Pro / 14 cores (10P + 4E) / 48 GB / 20 GPU cores**. Before
any long compute job: `OMP_NUM_THREADS=14`, `MKL_NUM_THREADS=14`,
`OPENBLAS_NUM_THREADS=14`; XGBoost `nthread=14` / `n_jobs=-1`; sklearn
`n_jobs=-1`. Datasets < 5 GB → fully into RAM. **Verify** ≥80% user CPU via
`top -l 1 | grep CPU` after dispatch. Fix bottleneck BEFORE letting it eat hours.

---

## 7 · Engineering principles

Each principle names the bug-class it prevents. Source incidents archived in
the Appendix.

### 7.1 · Test invariants

| Principle | Rule |
|---|---|
| Every feature + bug has a paired test | `pytest tests/ -v` before commit. Bug-fix workflow: reproduce with failing test → land fix → keep both in same commit. |
| Test through real adapters, not fixtures | Every fix has at least one test that calls through actual `SimAdapter` / `RunnerAdapter` / `Pipeline` — not hand-constructed fixtures. (Source: 124 σ-aware-stop-loss tests passed using `HoldingState(sigma=0.30)` while production had `state.sigma = None`.) |
| Each fix = regression-guard class | Name it `class Test<Bug>RegressionGuard` or include "AUDIT REGRESSION GUARD" in docstring, pinning the invariant. Reference: `tests/test_qp_wash_sale_cost_aware.py`. |
| Pytest snapshot in commit message | Every push touching `kernel/`, `adapters/`, `training_panel/`, `live/`, `scripts/{train,daily,weekly,monthly}_*` includes a fresh `pytest tests/ --tb=no -q` snapshot. ≥ 5 failures = broken commit, investigate before merge. |
| Splitter embargo invariant | Every train/val splitter enforces `max(train_date) + label_lookahead_days < min(val_date)`. Pinned by `tests/test_walk_forward_splits.py::TestEmbargo`. |

### 7.2 · Sanity discipline for numbers

Every new number ships with at least one sanity check from the **mandatory triad**:
- **A/A** (re-split → does the lift persist?)
- **Shuffled label** (IC ≈ 0)
- **Time-shift placebo** (IC ≈ 0)

Without one, the number is a guess. (Source: Track F's +98 bp "win" was
regime-persistence fitting the placebo would have caught.)

**Audit the data-pipeline foundation BEFORE reporting numbers from new
training code** (source: 2026-05-20 walk-forward leakage incident,
`doc/research/2026-05-20-jiantaoshu.md`). Any new training/eval script
depending on a splitter, label, or calibrator MUST audit those dependencies
at WRITE time:
1. **Splitter invariant** (§7.1) — default to absent = bug. Use `kernel.purged_cv.PurgedKFold` or pin embargo explicitly.
2. **Label causality** — every feature at timestamp `t` uses only data from `t+1` to `t+lookahead_days`. Grep for `iloc[-1]` / `tail(1)` / `.last(...)` broadcast-to-history patterns.
3. **Calibrator scope** — fit on train rows ONLY (NOT val OR test). Verify the script reads `split_label == "train"` (NOT `!= "val"` which includes test).
4. **Triad executed** on the new code path before the FIRST log line reporting an IC/Sharpe number. If `--label-shift-days 10` doesn't drop IC ≈ 0, there is leakage — STOP, audit, do not report.

#### 7.2.1 · Mandatory rules R1–R5 (post 2026-06-02 audit)

Five operating rules installed after the B_tuned leak audit
([`doc/research/2026-06-02-experiment-validity-audit.md`](doc/research/2026-06-02-experiment-validity-audit.md))
caught five distinct §7.2 / §6.4 / §7.7 / §7.13 violations across two
weeks. Every agent must observe these — codex PR review is empowered to
mark non-compliance as a HIGH blocker:

- **R1** — Marking a task `completed` requires reading the verdict file
  (`verdict.json`, `invalid_experiment.json`, `placebo_gate.json`, …) and
  reproducing the key numbers in the TaskUpdate description or commit
  message. "Background job exited 0" is NOT a verdict.
- **R2** — Any PR / commit / status report quoting an IC / Sharpe / APY
  number MUST include a companion placebo verdict block: at least one of
  `{shuffle_placebo, timeshift_placebo, a/a split}` with concrete values
  and the gate threshold. Reviewers (codex / agents) mark IC-without-
  placebo as a HIGH blocker.
- **R3** — Once a leakage gate (e.g., the G3 / G4 / G5 gates in the
  2026-06-01 leakage architecture, PR #43 v12) lands, all PRE-EXISTING
  artifacts must be retrofitted within 30 days. A gate that only fires
  on new artifacts is §7.7 "decoration".
- **R4** — When a leak / failure has a hypothesis list, every hypothesis
  must get a written audit memo (`ruled_in` / `ruled_out` /
  `inconclusive` + evidence + commit SHAs) BEFORE any new experiment is
  launched. "Fix one and re-run" is forbidden — re-runs are §6.4
  compute waste until the audit log is closed.
- **R5** — Memory files (`memory/*.md`) that contradict an earlier memory
  must explicitly `invalidate` the older entry OR resolve the conflict
  in the body. Silent coexistence of contradictory findings is the
  failure mode that let the 2026-05-29 "+0.066 placebo-clean" claim
  survive next to the 2026-05-31 leak verdict for two weeks.

### 7.3 · Multi-measurement requirement

**Single performance number = unverified claim.** Any APY / Sharpe / IC
quoted in commit / doc / roadmap MUST be `mean ± std` from ≥ 5 runs
(different seeds OR bar-orderings). Single-measurement claims are forbidden.

For DOE / parameter sweeps: quote multi-tested numbers as
`Sharpe_raw=X / DSR=Y / PBO=Z%` — never just raw SR (see §9 DOE methodology).

### 7.4 · Promotion gating — 3-tier

See [`doc/research/promotion-methodology.md`](doc/research/promotion-methodology.md):

| Tier | Criterion | Action |
|---|---|---|
| 1 (REJECT) | mean ΔAPY < 0 ∧ mean ΔSharpe < 0 | Discard |
| 2 (SCREEN, NOT live-promotable) | mean ΔAPY > 0 ∧ mean ΔSharpe ≥ 0 ∧ ≥ 4/N consistent ∧ ΔSPY-α ≥ 0 | Continue research |
| 3 (LIVE-PROMOTABLE) | Tier 2 + (DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 with t > 3.0) | Promote |

No live config flip without Tier 3. Run `python scripts/analyze_experiments.py`
at the end of every multi-config batch.

**Promotion thresholds aren't floors for theoretically-sound wins.** Default:
APY ≥ +2 pts on 27-mo OOS = promote. Exceptions (variables rigorously
controlled): live/sim parity fixes, theory-aligned wins with predicted
magnitude, mechanism-clean changes with positive margin. **Not exceptions**:
new strategies, hyperparameter sweeps, panel retrains.

### 7.5 · Single source of truth

**One business decision = one function.** Wash-sale, position cap, drawdown
halt, post-stop cooldown, earnings blackout — exactly ONE implementation; all
callers route through it. Adding a parallel impl requires DELETING the
original first.

### 7.6 · Data-flow safety

| Failure mode | Guard | Source |
|---|---|---|
| Code change ≠ data update | Commit modifying a data-pipeline script must include `⚠️ requires data regen: <command>`. Until regen runs and artifact mtime updates, the fix is NOT in production. | BUG #5 |
| `NaN > 0` is `False` | Any `>` / `<` on a value that COULD be NaN (broker fills, config floats, OHLCV closes, fund features, calibrator output) must pair with `math.isfinite(x)`. | 2026-05-09 audit |
| Unbounded calibrator output | `np.clip(out, lo, hi)` AT THE TRAIN SITE so the artifact stores sane bounds. Defense in depth: also clip at consumer site. | Calibrator's `expected_return.y` ranged −0.30 to +4.01 until `training_panel/global_calibrator.py:362-388`. |
| Hardcoded artifact filenames | Every load resolves through `cfg["ranking"]["panel_scoring"]["artifact_path"]`. | alpha158 promotion: `panel-ltr.json` became 21-feat stub while inference loaded 169-feat artifact. |
| Side configs are loaded weapons | Any `strategy_config.<label>.json` (label != "") must alias all `artifact_path` keys to side-paths containing the label. | Pinned by `tests/test_side_config_artifact_paths.py`. |

### 7.7 · Anti-decoration

| Failure mode | Guard |
|---|---|
| Dead module | Before declaring shipped: `grep -rn '<module>' backtesting/<strategy>/{adapters,kernel,live,scripts}/`. Tests-only references = orphaned. (`smart_orders.py` was 156 lines + 42 tests with zero prod imports.) |
| Dead `if optional_field is not None` branch | `grep -r 'optional_field = '` and show ≥ 1 prod write site that fires under current config. Otherwise the path is dead — revert or guard by a documented config flag. |
| Safety gate ≠ enforced safety gate | Every safety gate ships TWO artifacts: (a) gate function + tests, (b) a scheduled cron (plist + `.sh`) that invokes it WITHOUT override. If only (a), the gate is decoration. (`_check_wf_gate` was committed with tests but every daily promote set `RQ_ALLOW_NO_WF=1` — theatrical.) |

### 7.8 · Audit discipline

**"Fixed" = full 24h audit clean.** A bug is fixed only when (a) patched,
(b) regression test green, (c) every file touched in last 24h re-audited
end-to-end, (d) every up/downstream consumer re-audited, (e) zero open
issues across all of them. Show the audit log.

**"Audit subsystem X" = 4-step protocol**:
1. List X's data-flow inputs.
2. List X's outputs.
3. For each input → output edge, find or write an integration test.
4. For each `if X is not None` branch, prove via grep that X has a non-None path in production.

**After-session code audit** — before declaring any refactor / new-feature
session "done":
1. `git diff --name-only main..HEAD` enumerates touched files.
2. For each touched .py: 6-category audit (§7.10 canonical refs, dead code, logic bugs, logging gaps, BUGS, missed parallelism).
3. For each new eval / training driver: audit data dependencies (splitter, label, calibrator) end-to-end against §7.2.
4. Any code that REPORTS A NUMBER must have its data pipeline audited before the number is trusted in a status report.

Self-trigger at session end — don't wait for the user to request a deep audit.

### 7.9 · Cron cadence is info-theoretic

Any new cron's docstring answers: "this frequency adds N% new
training-relevant info per tick vs the next-coarsest alternative". < 5%/tick
= wrong cadence. Daily retrain on a `fwd_60d`-label model adds ~0.014%/day —
cargo cult.

### 7.10 · Default to canonical references

First question for any algorithmic / data / model decision: *what does Qlib /
cvxportfolio / scikit-learn / PyPortfolioOpt / the canonical paper do?*

Reference set:
- Cross-sectional ranking & features: `microsoft/qlib` (Alpha158, LinearModel, LGBModel, TransformerModel)
- Convex portfolio optimization: `cvxportfolio` (Boyd) + CVXPY
- Asset-pricing factors: Kelly-Gu-Xiu RFS 2020 firm-characteristics list
- Risk + transaction cost: Almgren-Chriss 2000, Ledoit-Wolf 2004
- Time-series ML: PatchTST (Nie 2023), TFT (Lim 2021), Qlib `pytorch_*_ts.py`

**"Backed by literature" means READ, not name-dropped.** Mandatory checklist for non-trivial design:
1. Cite specific source (paper title + year, OR `org/repo:file.py` + commit SHA).
2. Read it.
3. Confirm the decision matches the source — same hyperparameters, preprocessing, evaluation. List intentional divergences with reason.
4. If no source — mark "exploratory, will tune via A/B".

Applies to: model architecture, loss, optimizer, preprocessing, label
construction, train/val/test split, evaluation metrics, sanity-test design,
feature engineering. (Decoration-citing burned ~2 h on `alpha158_lite` — 8
substantive deviations from canonical Qlib alpha158; faithful replication
tripled test IC.)

### 7.11 · Experiment design

**Decision tree before any multi-hour run**:
0. **No-run path first** (§6.3).
1. **Range-finding** ("does X work at all?") → top-down single endpoint, 30 min wallclock.
2. **Optimization** ("find best subset/hyperparam") → ONLY after range-finding shows X works.
3. **Diagnostic** ("why did Y fail?") → §7.2 sanity sequence FIRST, then mechanism.

Sunk-cost guard: kill the experiment when mid-run evidence already answers the question.

### 7.12 · Unexpected result → audit before accepting

When theory predicts X and the result is ¬X, the first hypothesis is "my
implementation has a bug or my assumptions were wrong" — not "the theory is
wrong". Checklist before shipping a negative finding:
1. Per-bar log of new Task's inputs on ≥ 3 sample bars — sane?
2. Reason through every input/output independently: with all inputs correct, what would we *expect*?
3. Re-read commit defaults — does the GOLDEN variant in the A/B preserve baseline?
4. Check if any other task / config reads the same data — interaction possible?

### 7.13 · Process safety

| Rule | Why |
|---|---|
| Run relevant tests before every commit/push | Even "obviously correct" scripts get `python -c "import the_module"` smoke before overnight runs. |
| Every "fix" names the invariant that prevents the entire bug class | A patch fixes the symptom; a fix names "what invariant would have made this impossible". Example: NGBoost feature drift → `max_feature_drift_pct` hard guard (architectural impossibility), not "retrain the head". |
| Don't edit running scripts | Mid-run edits to chain scripts are silent failures. Stop+restart, or use `ScheduleWakeup`. |
| Production-touching changes rehearse rollback | "We have an auto-revert script" ≠ "rollback works". Manually trigger the rollback path on a non-prod copy and verify post-rollback state is fully self-consistent (config + model + state files all aligned). Production-touching = anything live runner / launchd reads, plus models in `artifacts/`. |
| Document every failed experiment | [`doc/research/failed-experiments-log.md`](doc/research/failed-experiments-log.md): hypothesis, implementation, exact numbers, sanity check, conclusion, reproduction recipe — all required, written same day. |

---

## 8 · Status report style

### 8.1 · Use concepts, not code labels

Status reports use plain-language concept names, not commit hashes / file
paths / variable names / task IDs / bug labels. *"面板排序模型"* not
`panel-ltr.json`. *"PR-based workflow"* not `§3.1`. Code-level identifiers
belong in commit messages and direct technical answers only.

### 8.2 · Multi-track plans ship as ASCII timeline

Tracks on rows, time on X-axis, dependencies as arrows. Apply when ≥ 2
sequential phases OR ≥ 2 parallel work streams OR a stop-gate followed by
branching:

```
[现在]              [+2h]               [+3h]                [+并行]
P0 fixes ─────────→ retrain ──────────→ sim ──────────────→ promote/reject
                     ↓
                    P1 fixes 同时跑
```

---

## 9 · Design of Experiments (DOE)

For parameter sweeps with ≥ 2 interacting knobs, one-knob-at-a-time is
**forbidden** — it misses 2-way interactions and over-counts main effects
under confounding.

| Stage | Question | Method | Runs | Reference |
|---|---|---|---|---|
| Screening | Which knobs matter? | Plackett-Burman | ~k+4 | Plackett & Burman 1946 *Biometrika* 33:305 |
| Screening w/ 2-way | Which knobs + pairs? | 2-level Fractional Factorial 2^(k-p), Resolution IV+ | 8-32 | Box-Hunter-Hunter 2005 ch.6 |
| Optimization | Where is the optimum? | **Box-Behnken** or CCD | 13-46 | Box & Behnken 1960 *Technometrics* 2:455 |
| Confirmation | Does predicted optimum win? | 2-3 runs at optimum + DSR/PBO | 3-5 | §7.3; Bailey-López de Prado 2014 *J. Portfolio Mgmt* 40(5):94 |

**Default**: Box-Behnken at 3 levels for ≤ 6 knobs in continuous-tunable
regions. Quadratic response surface `y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²`
captures main effects + 2-way + curvature with ≈ 25 runs (4 knobs) or ≈ 46
runs (6 knobs).

**Center points**: 3-5 replicates at the design center for lack-of-fit test
+ `σ²_pure` estimate. Single-seed-deterministic sims still need ≥ 1 center.

**Cite the DOE library, not the reinvention**:
- `pyDOE2.bbdesign(k, center=N)` — Box-Behnken matrix
- `pyDOE2.ccdesign(k, alpha='r')` — rotatable CCD
- `pyDOE2.fracfact("a b c ab")` — fractional factorial
- `sklearn.preprocessing.PolynomialFeatures(degree=2)` + `LinearRegression` — fit surface
- `scipy.optimize.minimize(..., bounds=...)` — optimum on fitted surface

Re-deriving the design matrix by hand = §7.10 violation.

**Multiple-comparison correction is mandatory** (§7.3): Deflated Sharpe Ratio
(Bailey-López de Prado 2014) + PBO via CSCV (Bailey, Borwein, López de Prado
& Zhu 2015). Quote `Sharpe_raw=X / DSR=Y / PBO=Z%` — never just raw SR.

**Knob bounds come from baseline distribution**, not author preference.
Extract the relevant percentile from baseline before picking levels
(p90 / p95 / p99 of observed worst single-day-position-drops; cumulative-loss
percentile of stop-exited trades; etc.). Round-number bounds (5%, 10%, 15%)
are §7.10 violations.

**Interaction-aware reporting**:
1. Main-effects plot (each knob's β with confidence interval)
2. 2-way interaction plots for significant βᵢⱼ
3. Contour / heatmap of fitted surface in top-2 knob plane
4. Pareto frontier across competing objectives (APY vs MaxDD)
5. The optimum is the surface point, NOT the best of N evaluated runs

---

## 10 · General coding guidelines

**Bias toward caution. Use judgment for trivial tasks.**

**Think before coding.** State assumptions explicitly. Multiple
interpretations? Present them. Simpler approach exists? Say so. Unclear?
Stop and ask.

**Simplicity first.** No features beyond what was asked. No abstractions for
single-use code. No flexibility / configurability that wasn't requested. No
error handling for impossible scenarios. 200 lines that should be 50 → rewrite.

**Surgical changes.** Don't "improve" adjacent code. Don't refactor what
isn't broken. Match existing style. Mention unrelated dead code; don't
delete it. Remove imports / variables / functions that YOUR changes made unused.

**Goal-driven execution.** Transform tasks into verifiable goals: "add
validation" → "tests for invalid inputs, then make them pass". For multi-step
tasks, state a brief plan with verify steps before starting.

---

## Appendix · Historical source incidents

Source incidents that produced each rule live in `memory/` (auto-loaded into
agent context per session), `doc/archives/sessions/`, and
`doc/archives/audits/`. Notable anchors:

| Rule | Source |
|---|---|
| §1 PRIME DIRECTIVE | `memory/feedback_regime_conditional_strategy.md` + `doc/research/2026-05-14-longshort-clean-FINAL.md` (long-short ON pooled NEITHER but per-regime: BEAR +22pt / CHOPPY +14pt / BULL_VOL +13pt wins vs BULL_CALM −8pt / BULL_STRONG −2pt losses) |
| §1.4 detector quality | 2022 Q2 bear labeled BULL_CALM 100% because Hurst > 0.65 routes to BULL_CALM regardless of direction. Fixed via `3925c0d` (SPY < MA50 direction signal). |
| §3.1 PR-based workflow | `memory/feedback_pr_based_workflow.md` (2026-05-30 user mandate) |
| §3.2 sync mandate | 2026-05-30 multi-agent incidents: codex pushed `9982de8` + `7f3cd14` to in-flight PRs while local main was unaware; cross-repo recorder lift landed 7 PRs in 1 hour. |
| §3.4 PR review protocol | `renquant-backtesting/CLAUDE.md` (PR #7, 2026-05-30) — codex hardened after silent-fix push pattern in PRs #2 + #9 |
| §6.1 execute-now mandate | 2026-05-19 14:30 PT shipped HF Trainer + eval drivers without §7.2 sanity. 2026-05-20 11:18 PT proposed "明天重跑 ..." after partial P0 fixes. |
| §7.x engineering principles | `doc/archives/audits/2026-04-28-deep-audit.md`, `2026-04-28-nvts-buy-postmortem.md`, `doc/AUDIT_2026-05-09.md` (the 17 anti-patterns audit) |
| §7.2 splitter embargo | `doc/research/2026-05-20-jiantaoshu.md` (walk-forward leakage incident) |
| §7.6 calibrator clip | Production calibrator's `expected_return.y` ranged −0.30 to +4.01 until `training_panel/global_calibrator.py:362-388` clipped at train site |
| §7.7 dead code | `smart_orders.py` (156 lines + 42 tests, zero prod imports). `_check_wf_gate` (committed with tests but every daily promote set `RQ_ALLOW_NO_WF=1`). |
| §7.10 canonical refs | `alpha158_lite` (8 deviations from canonical Qlib; faithful replication tripled test IC) |
| Track H 16-Task preflight refactor | 2026-05-30 (umbrella PR #7, #8 + subrepo pipeline #2) |
| (d) `RQ_SIM_BYPASS_BUY_FLOOR` env flag | 2026-05-30 sim methodology audit — `memory/project_wf_sim_unfair_to_compressed_models_2026-05-30.md`. Hardened by codex to sim-only via `ctx._run_type` check. |

For chronological context of any rule, `git log --follow CLAUDE.md` traces
when it was added.
