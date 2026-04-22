# renquant Maintenance Workflow (applies to 103 and 104)

## Purpose

This is the reusable workflow for the kind of work done in prior review passes:

- deep review of `renquant_103` or `renquant_104`
- notebook / sim vs LEAN vs live parity checks
- canonical-semantics decisions
- test and replay-harness updates
- doc synchronization
- validation, commit, and push
- optional performance-improvement follow-up

Use this when you want the repo brought back to a known-good, aligned state after strategy changes. Both strategies share the same inference trunk (`kernel/pipeline/`, `InferencePipeline`), so the workflow is identical — only the strategy-specific artifacts differ (per-ticker tournament for 103; per-ticker tournament **plus** panel-LTR artifact for 104).

## Recommended Prompt

Use one of these prompts when you want this workflow run again:

- `Run the renquant maintenance workflow for 104.`
- `Do the renquant_104 review/alignment workflow, then validate, commit, and push.`
- `Run the workflow from doc/renquant_103_maintenance_workflow.md against renquant_104.`

If you want only part of it, say which phase to skip. When targeting 103 (reference/rollback) instead of 104, swap every `_104` path for its `_103` counterpart.

## Scope

Primary files and areas touched by this workflow (swap `_104` ↔ `_103` depending on which strategy you are maintaining):

- `backtesting/renquant_{103,104}/renquant_{103,104}.ipynb`
- `backtesting/renquant_{103,104}/main.py`
- `backtesting/renquant_{103,104}/kernel/` — regime, indicators, models, exits, selection, sizing, scoring, rotation
- `backtesting/renquant_{103,104}/kernel/pipeline/` — `pp_inference.py`, `pp_training.py`, and (104 only) `pp_training_full.py`
- `backtesting/renquant_104/kernel/panel_pipeline/` — `panel_scorer.py`, `feature_matrix.py`, `job_panel_scoring.py`
- `backtesting/renquant_{103,104}/training/` — features.py, tournament.py, export.py
- `backtesting/renquant_104/sim/runner.py` — hand-written sim loop; every decision added to `InferencePipeline` must be mirrored here until it is refactored through the pipeline
- `live/runner.py`, `live/adapters/lean.py`, `live/adapters/runner.py`
- `scripts/train_104.py`, `scripts/train_panel_model.py`, `scripts/recalibrate_scores.py`
- `tests/test_policy_alignment.py`, `tests/test_panel_alignment.py`, `tests/test_simulation_policies.py`, `tests/test_strategy_ledger_parity.py`, `tests/test_runner_ranking.py`, `tests/test_training_modules.py`, `tests/test_panel_scoring_job.py`, `tests/test_panel_training_pipeline.py`
- `README.md`, `CLAUDE.md`
- `doc/architecture.md`, `doc/logic_graph_103.md` (shared trunk), `doc/renquant_103_design.md`, `doc/renquant_104_design.md`
- `doc/deep_review_YYYY-MM-DD.md` when a formal audit/report is requested

## Definition Of Done

The workflow is complete when all of the following are true:

1. The current strategy semantics are explicit and internally consistent.
2. Notebook, LEAN, and live behavior are aligned where they are intended to match.
3. Parity and runner tests cover the changed behavior.
4. Docs match the actual code.
5. Relevant tests pass.
6. Changes are committed and pushed if requested.
7. A short summary of what changed, what was verified, and what remains open is provided.

## Workflow

### Phase 1: Preflight

1. Read `CLAUDE.md` and any directly relevant docs.
2. Check git status and current branch.
3. Confirm the active Python environment is the `renquant` interpreter.
4. Identify whether the request is:
   - full maintenance workflow
   - targeted bug fix inside the workflow
   - audit/report only
   - performance pass only

### Phase 2: Review And Diff Mapping

1. Inspect the active `renquant_103` notebook, LEAN strategy, live runner, tests, and docs.
2. Identify semantic drift in areas such as:
   - raw score vs calibrated score usage
   - ranking weights
   - tier thresholds
   - wash-sale checks (scan phase AND selection loop)
   - min-hold behavior
   - regime-confidence sizing
   - sell ordering
   - data fetch set (notebook fetches WATCHLIST ∪ sector_etf_map.values() ∪ {SPY}; runner must match)
3. If a deep audit is requested, write or update a dated review file in `doc/`.

### Phase 3: Choose Canonical Semantics

Before editing, make the intended rules explicit.

Canonical choices from this conversation were:

- raw score drives upstream buy/sell action eligibility
- calibrated `rank_score` drives filtering, ranking, and tier thresholds
- ranking blend weights come from config
- `max_position_pct` and `cash_reserve_pct` scale by regime confidence
- wash-sale is checked in both candidate scan AND the selection loop (all three components)
- live runner fetches `WATCHLIST ∪ sector_etf_map.values() ∪ {SPY}` so RS scores use real ETF data
- RS ranking uses 20-day lookback in all three components (notebook pct_change(20) = runner iloc[-21])

If any of these change in a future pass, update notebook, LEAN, live, tests, and docs together.

### Phase 4: Implement Parity Changes

1. Update the notebook simulation so its daily decision flow matches the canonical rules.
2. Update LEAN to match the same scoring, selection, and sizing semantics.
3. Update the live runner when live behavior, diagnostics, or trade logging should follow the same semantics.
4. Keep fixes minimal and root-cause oriented rather than adding workaround branches.

### Phase 5: Update Test Coverage

Touch the smallest relevant suites first:

- `tests/test_policy_alignment.py` for policy-by-policy notebook/LEAN parity
- `tests/test_simulation_policies.py` for notebook-like simulation behavior
- `tests/test_strategy_ledger_parity.py` for replay-style selection ledger parity
- `tests/test_runner_ranking.py` for live ranking, calibration, and logging

Add tests whenever a new rule or branch is introduced.

### Phase 6: Sync Docs

At minimum, review and update these when semantics changed:

- `README.md`
- `CLAUDE.md`
- `doc/architecture.md`
- `doc/logic_graph_103.md` (shared trunk for 103 and 104)
- `doc/models.md` when scoring/calibration/sizing semantics changed

Update `doc/renquant_103_design.md` if renquant_103 semantics changed, or `doc/renquant_104_design.md` if any panel-LTR layer (training, scoring, veto, conviction sizing, rotation advantage) changed.

### Phase 7: Validate

Run targeted tests first, then broader suites if the change was structural.

Typical commands:

```bash
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest tests/test_policy_alignment.py -q
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest tests/test_simulation_policies.py -q
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest tests/test_strategy_ledger_parity.py -q
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest tests/test_runner_ranking.py -q
```

If the request is broad, run:

```bash
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest tests/ -v
```

### Phase 8: Review Generated Artifacts

If notebook export was rerun, review generated outputs before committing, especially:

- `backtesting/renquant_103/models/**`
- chart PNGs under `backtesting/renquant_103/`
- notebook output/state changes

Only treat these as noise if they are truly unrelated to the requested workflow. Otherwise keep them with the semantic change they belong to.

### Phase 9: Commit And Push

If asked to commit:

1. Stage reviewed files.
2. Use a descriptive commit message.
3. Push the current branch, usually `main`.

If not asked to commit, leave the tree ready and summarize what remains unstaged.

### Phase 10: Optional Performance Pass

Only do this after parity is stable.

Focus areas:

- simplify ranking if a signal is no longer earning its complexity
- compare calibrated-score-only vs blended ranking behavior
- test whether confidence scaling improves outcomes or only adds noise
- remove layers that are hard to justify in replay or backtest evidence

This phase should end with a concrete change, a validation result, and a short explanation of the tradeoff.

## Output Format For Future Runs

When reporting completion, include:

1. what changed
2. what was validated
3. whether anything remains open
4. whether commit/push was completed

## Current Notes

This workflow was derived from the 2026-04-16 conversation that produced:

- canonical score/ranking semantics for `renquant_103`
- replay-style parity coverage
- doc alignment updates
- a pushed parity commit on `main`
- a follow-up live-runner log improvement to print model type and calibrated score explicitly

Updated 2026-04-17 maintenance pass findings:

- **Sector ETF data bug (fixed)**: Runner was only fetching `watchlist + [SPY]`. XLK and XLI are
  not in the watchlist, so all 12 tech/industrial stocks had RS = 0.0. Fixed: runner now fetches
  `WATCHLIST ∪ sector_etf_map.values() ∪ {SPY}` to match notebook behavior.
- **Wash-sale selection re-check (fixed)**: Runner's selection loop was missing the re-check for
  wash-sale violations that both notebook and LEAN perform. Fixed: added re-check before sector
  guard in runner's selection loop.
- **False positive parity gaps**: RS timeframe (both 20-day), defensive_tickers (from config, all 4),
  consecutive_sell_signals (from config, both 3), min-hold (both 20d from config) — all confirmed
  aligned. Earlier analysis was comparing defaults, not actual config values.
- **Improvement plan**: `doc/improvement_plan_2026-04-17.md` — 7 improvements ranked by priority.
  Top items: verify RS non-zero after retrain; CHOPPY max_concurrent_positions → 4; GMM confidence
  veto (<55% = no buys).

Updated 2026-04-19 refactor pass:

- **Notebook training cells extracted**: Cells 6 (85 lines), 7 (167 lines), and 8 (114 lines) were
  extracted into `training/features.py`, `training/tournament.py`, and `training/export.py`. The
  notebook is now a thin orchestrator (~15 lines per training step).
- **LEAN parity fix**: `_build_exit_params()` in `main.py` was missing `lt_hold_gate_days` and
  `lt_hold_min_gain` params. Fixed: these are now read from `CONFIG` and passed to `compute_exits()`.
- **New tests (18)**: `tests/test_training_modules.py` covers `training/features.py`,
  `training/tournament.py`, and `training/export.py` — 8 feature tests, 3 Sharpe tests, 7 export/
  tournament tests.
- **Current test count**: 560 passed, 2 skipped (562 total collected).