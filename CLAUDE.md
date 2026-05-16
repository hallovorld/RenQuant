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

## 🗂 Status (2026-05-15 EVENING — calibrator P0 + NGBoost CONFIRMED + regime-aware)

> **EVENING UPDATE (2026-05-15) — 6 LOAD-BEARING SESSION OUTCOMES:**
> 1. **Calibrator P0 closed** — prod artifact had `expected_return.y` up
>    to +1.0 (= +100% predicted) corrupting Kelly μ. Refit with
>    train-site ±0.20 clip + load-time guard + saturation warning +
>    G12 preflight check. Post-refit: pool_IC=+0.094 (preserved),
>    er.y in [-0.105, +0.200], n_unique_prob_y=81. (commits b16e2a1,
>    00f94ff, 342309e, 5b78ffe, d1624b3)
> 2. **Phase 3 μ/σ wiring + Upgrades A+B (gates) ACTIVATED in golden** —
>    `use_calibrator_mu`, `use_realized_vol_fallback`, `regime_momentum`,
>    `deep_drawdown_veto` all flipped ON. Kelly is no longer impotent
>    (`mu_none=N`).
> 3. **NGBoost SUSPECT → ✅ CONFIRMED** — 5-seed Duan 2020 §4 config
>    val_IC=+0.0351 ± 0.0036, σ-calib=+0.271 ± 0.005, t=+2.76 vs
>    XGB-quantile baseline (95% significant). E55's −20.6pt was
>    misconfig + broken σ wire, NOT theory failure. NGB has real signal
>    AND calibrated σ. Re-test pending after prod retrain + σ-aware
>    Kelly wire.
> 4. **p0activated 16-window sim → REGIME-CONDITIONAL pattern
>    confirmed** — Pooled +0.52pp NEITHER (p=0.90), but stratified
>    via SPY-return-based regime label:
>    * BEAR (n=1): +0.00 (no firing)
>    * CHOPPY (n=4): mean +1pp
>    * BULL_VOLATILE (n=8): **+6.88pp WIN** (Q05 -19 outlier)
>    * BULL_STRONG (n=3): -1.68pp lose
>    * The 4 new flags help in vol-spike regimes, hurt in trending bull
>    rallies (mean-revert mega-caps ARE the winners). New variant
>    `sim_p0activated_regime_aware` (gates disabled in
>    BULL_CALM/BULL_STRONG) queued — running now.
> 5. **HMM regime detector still mis-labels 2022 deep-bear as
>    BULL_CALM** — analyzer now uses data-driven SPY-return classification
>    via yfinance, bypassing the buggy in-sim HMM (CLAUDE.md PRIME
>    DIRECTIVE). Detector fix still TODO.
> 6. **Regime-conditional re-evaluation queue** running sequentially:
>    `re_stop007` ✓ done → `p0activated_regime_aware` (running) →
>    re_sdl_n2 → re_trail015 → re_cvar025 → re_cvar050 → re_kelly_t1_035.
>    ETA ~3.5h. Expected: 2-3 of 6 will surface conditional WIN regimes
>    that were hidden by pooled-mean rejection (per Explore agent
>    survey of failed-experiments-log).
>
> **3 audit-mandated regression tests SHIPPED** (doc/AUDIT_2026-05-12
> requirement that was deferred):
> `test_vol_target_scales_qp_upper.py` /
> `test_dd_kelly_scales_qp_upper.py` /
> `test_vol_target_independent_of_ngb.py` — 11/11 pass.
>
> **META trade canceled on LIVE Alpaca account** — Friday 17:11 ET queue
> would have fired at Monday open with broken gates. New gates +
> conditional opt-out will re-evaluate cleanly Monday.
>
> **Alpaca-shorts paper account WIRED** — broker label isolation works
> end-to-end (placed + canceled test short, position_intent=sell_to_open).
> Paper-only; live broker untouched. Phase 2D (cover stop + tax §1233)
> queued for next session.
>
> Detailed re-eval plan: [`doc/research/2026-05-15-regime-reeval-plan.md`](doc/research/2026-05-15-regime-reeval-plan.md).
> Audit doc updated: [`doc/AUDIT_2026-05-12_dead_paths.md`](doc/AUDIT_2026-05-12_dead_paths.md) §"2026-05-15 RESOLUTION UPDATE".

---

## 🗂 Status (2026-05-12 EVENING — methodology rebuild + regime-conditional finding)

> **EVENING UPDATE (2026-05-12 22:30 PT) — 3 NEW LOAD-BEARING FINDINGS:**
> 1. **Prior 6-window mean-APY method was STATISTICALLY INVALID** (mixed window lengths, heavy overlap, regime variance dominating). All 36 "TIER 1 REJECT" verdicts from the morning batch are now flagged INCONCLUSIVE.
> 2. **Built industry-grade evaluation** — paired daily returns + Newey-West HAC + stationary block bootstrap (statsmodels + arch). 8 non-overlapping 3-month windows, 496 paired daily observations. See [`doc/research/evaluation-protocol.md`](doc/research/evaluation-protocol.md).
> 3. **Discovered regime-conditional structure** — Grinold-Kahn α→μ transform (commit `7bc9b56`) wins +18%/yr in SPY-HIGH_CALM (n=123, t=+1.67) but loses −32%/yr in SPY-HIGH_SPIKED (n=53, t=−1.95). Pooled NEITHER hides the structure. Conditional deployment blocked on regime detector fix (currently labels 95% of days BULL_CALM) — see [`doc/research/2026-05-12-findings-and-next.md`](doc/research/2026-05-12-findings-and-next.md).
>
> **Strategic top-line (NEW industry-grade method, 8-window panel):**
> Pooled paired t-stat:
> - vt15 vs baseline:  +0.75  mean Δ +0.84%/yr  95% CI [−1.0%, +3.4%]  → NEITHER
> - GK094 vs baseline: +0.50  mean Δ +2.66%/yr  95% CI [−7.6%, +13.0%] → NEITHER (regime-conditional!)
> - GK15 vs baseline:  pending (running 22:30)
>
> **PRIOR STATUS (kept for context):** 6-window walk-forward post-Bug-C: Baseline mean APY +15.2% / Sharpe 0.41 / MaxDD 8.4%; strategy LOSES TO SPY by −2.3pt mean alpha. These numbers are from the OLD methodology and may not survive under the new 8-window paired analysis. Baseline still unchanged; no prod flips today.
>
> **Rejected this autonomous run (after Bug-C fix):**
> - CVaR sweep (λ ∈ {0.15, 0.25, 0.35, 0.50}) — all within noise
> - vol-target / trend-overlay / DD-Kelly (Phase 1 config-only) — Kelly path is dead (μ=None when NGB off)
> - NGBoost on (E55) — destroys −20.6 pt APY decisively
> - 5-knob stop-loss sweep (stop07/12, trail15, maxh250, sdl2) — confirms pre-Bug-C verdicts
> - multi-horizon (fwd5d/20d/60d static) — fwd60d static +3.3pt but fails consistency gate; fwd5d/20d both lose
>
> **Deferred to future sessions** (require model retraining ≥3h each):
> - E26 wl183 universe expansion
> - E41 R1K universe (full Russell 1000)
> - B1/B2 stop-loss revival (need code patch for `stop_decay_days` + `sdl_skip_if_unrealized_above`)
>
> Further alpha gains require STRUCTURAL changes (new universe, new feature set, new model class), not parameter tuning. Single-knob optimization is exhausted.



> Detailed audit: [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md). Roadmap: [`doc/roadmap.md`](doc/roadmap.md). Closed tracks + failed experiments: [`doc/research/failed-experiments-log.md`](doc/research/failed-experiments-log.md). Session history: `doc/archives/sessions/`.

- **🔴 BIG FINDING: Bug C (commit `29e34b0`)** — SimAdapter._portfolio_value omitted T+2 pending balance from NAV. Phantom ±sale_amount returns inflated Vol by ~75× and MaxDD by ~10×, AND distorted compound APY downward via path-dependent vol drag. **This bug corrupted every sim metric this session.** Fix invariant: NAV ≡ free_cash + pending_settle + Σ(shares × price). 5 regression tests pinned.
- **Strategy reality after Bug C fix (3 windows, post-fix):**
    Baseline mean APY = **+11.6%**, Sharpe = **0.77**, MaxDD = **8.2%**
    Pre-fix this was reported as APY = −7.2%, Sharpe = 0.60, MaxDD = 46.4% — ALL artifacts of Bug C. **The strategy is credible long-only at the ranking level.**
- **Corrected prior verdicts:**
  - Meta-label "E63 −9.2 pt active harm" → post-fix actual −1.5 pt (within noise). Classifier AUC = 0.55 still random, so disabled decision stands on theory not measurement. Prod commit `0cf758d` (meta off + drawdown resume halt/2) is unchanged.
  - CVaR "E65 +7.3 pt ± 16 σ noise" → post-fix +0.1 pt ± 7.6. Still regime-dependent, still rejected.
  - max_position_pct "E66 8% winner +7.2 pt" → post-fix **8% LOSES by 8.2 pt** vs baseline 20%. Vol-drag hypothesis was based on inflated pre-fix Vol; real Vol is ~10-20%, no drag to fix. Keep current 20% caps.
  - A1 calibrator recent-12mo → post-fix mean −1.7 pt with high σ. Not worth flipping.
- **🟡 YELLOW open: Bug D** — `ctx.cash = self._cash` is settled-only; Alpaca margin allows T+2 unsettled cash as buying power. Sim under-trades after recent sells vs live. Direction: sim shows LOWER returns than live would achieve.
- **🟢 GREEN: 3 cron scripts shipped** (monthly meta-label retrain — currently retraining a disabled artifact; weekly SEC fund refresh; monthly calibrator already in place). User installs via `launchctl load`.
- **All 3-window numbers are still n=3** — wider CI than σ implies. No DSR/PBO yet. But the picture is now PROFITABLE strategy with measurement noise on the optimizations, not "no-edge strategy" as feared.
- **Production model:** XGBoost rank:pairwise, alpha158 + 5 fund + 3 PEAD + 3 SUE = 169 features, artifact `panel-ltr.alpha158_fund.json` (trained 2026-05-09 03:44). NGBoost OFF. **Meta-label OFF (2026-05-11).** Calibrator pool_ic = +0.094.
- **Live:** Alpaca PAPER mode (per user safety mandate 2026-05-11). **Watchlists:** 103 runtime / 292 training / wl162 quality-first selected pending evaluation.
- **Active strategy:** `renquant_104` (panel-LTR cross-sectional ranking). `renquant_103` archived for rollback only.

**Sanity-test triad** (mandatory before any "+IC" / "+APY" claim — see §5.2):
1. A/A test (same config × multiple seeds → σ)
2. Shuffled-label (`panel_ltr.label_shuffle_seed=N` → IC ≈ 0)
3. Time-shift placebo (`panel_ltr.label_shift_days=N` → IC ≈ 0)

**IC evaluation** (read [`doc/research/ic-evaluation-methodology.md`](doc/research/ic-evaluation-methodology.md)): walk-forward ≥5 cuts, paired vs Linear/Ridge/XGB baseline. Current 7-cut WF (2026-05-08, fwd_20d, 291-ticker + fundamental): OLS +0.029±0.038 / Ridge +0.030 / XGB +0.039±0.046. Param/sample > 1/100 ⇒ forbidden (overfitting guaranteed).

---

## Project

RenQuant — personal quantitative trading workstation for Apple Silicon. Glass-box pipeline: data ingestion → ML signal generation → backtesting (LEAN) → live trading (Alpaca/IBKR). Statistically interpretable, strictly decoupled.

## Environment

Activate the project venv: `source .venv/bin/activate` (NOT conda). Docker ≥ 16GB for LEAN. Alpaca creds in `.env` (gitignored). Full setup: [`doc/ops/environment.md`](doc/ops/environment.md).

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

### 4. Keep docs current
After any non-trivial change, sync: [`doc/arch/overview.md`](doc/arch/overview.md), [`doc/arch/strategy-104.md`](doc/arch/strategy-104.md), [`doc/ops/schedule.md`](doc/ops/schedule.md), [`doc/roadmap.md`](doc/roadmap.md), and this file. Code wins on conflict.

### 5. Engineering Principles

Each rule names the bug-class it prevents. Source incidents: [`doc/archives/audits/2026-04-28-deep-audit.md`](doc/archives/audits/2026-04-28-deep-audit.md), [`2026-04-28-nvts-buy-postmortem.md`](doc/archives/audits/2026-04-28-nvts-buy-postmortem.md), [`doc/AUDIT_2026-05-09.md`](doc/AUDIT_2026-05-09.md).

**5.1 Run relevant tests before every commit/push.** Even "obviously correct" scripts get a 1-second `python -c "import the_module"` check before overnight runs.

**5.2 Every new number ships with at least one sanity check.** Mandatory triad — A/A (resplit → does lift persist?), shuffled-label (IC ≈ 0), time-shift placebo (IC ≈ 0). Without one, the number is a guess. Track F's +98bp "win" was regime-persistence fitting that the placebo would have caught.

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

**5.10 Saturate hardware.** M2 Pro = 10 cores + 32 GB RAM. Before any long compute job: `OMP_NUM_THREADS=10`, `MKL_NUM_THREADS=10`, `OPENBLAS_NUM_THREADS=10`; XGBoost `nthread=10` / `n_jobs=-1`; sklearn `n_jobs=-1`. Datasets < 5 GB load fully into RAM. **Verify** ≥80% user CPU via `top -l 1 | grep CPU` after dispatch — fix bottleneck BEFORE letting it eat hours.

**5.11 Experiment design optimizes for time-to-answer.** Decision tree before any multi-hour run:
1. **Range-finding** ("does X work at all?") → top-down single endpoint, 30 min wallclock.
2. **Optimization** ("find best subset/hyperparam") → ONLY after range-finding shows X works.
3. **Diagnostic** ("why did Y fail?") → §5.2 sanity sequence FIRST, THEN mechanism.

Sunk-cost guard: kill the experiment when mid-run evidence already answers the question.

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
