# renquant_104 — Panel-LTR Cross-Sectional Ranking

**Status**: Implemented — ready to replace renquant_103 as the active daily strategy.
**Author**: Ren Hao
**Last updated**: 2026-04-21
**Based on**: renquant_103 (adaptive regime multi-stock)

---

## 1. What's different from renquant_103

renquant_104 inherits the entire renquant_103 decision graph — regime detection, sell
priority, buy gates, sector/wash-sale guards, rotation — and adds a
**cross-sectional panel-LTR ranker** on top of it. Every other node in the
logic graph is unchanged.

| Concern | renquant_103 | renquant_104 |
|---|---|---|
| Per-ticker model | Champion from tournament (Classification / QLearning / XGBoost / Manual) | Same |
| Candidate rank | Per-ticker `rank_score` (calibrated Platt/isotonic) | **Cross-sectional panel-LTR `rank_score`** replaces per-ticker when `panel_scoring.enabled=true` |
| Feature scope | Per-ticker indicators only | Per-ticker indicators + panel-level neutralized factors (sector momentum, size z-score, beta-residuals) |
| Training driver | `Notebooks/renquant_103.ipynb` | **`scripts/train_104.py`** (no notebook — `FullTrainingPipeline` Job/Task chain) |
| History lookback (inference) | 60 daily bars | **520 bars** when panel scoring is enabled (neutralization + factor windows need ≥504) |

Everything else — exits, selection ledger, tiered thresholds, rotation — is
identical. The logic graph in `doc/logic_graph_103.md` continues to apply
after inserting a single node between CandidateScan and Ranking:

```
… → CandidateJob → PanelScoringJob → RankingJob → RotationJob → SelectionJob
                       ↑
                       only runs when ranking.panel_scoring.enabled=true
                       otherwise skipped via should_skip()
```

---

## 2. Panel-LTR design

The panel-LTR model is a single XGBoost learning-to-rank model fitted on the
cross-section of all watchlist tickers per day. Labels are forward
excess-return ranks neutralized by:

- Sector (via sector ETF returns)
- Size (log market-cap proxy via price × volume moving average)
- Beta-residuals vs SPY

The artifact written by training (`artifacts/panel-ltr.json`) contains:

- `booster_raw_json` — serialized XGBoost model
- `feature_cols` — exact column order used at inference
- `oos_mean_ic` — mean information coefficient across CV folds
- `trained_date`

At inference time `PanelScoringJob` performs three atomic tasks:

1. **LoadScorerTask** — deserialize the booster, resolve `artifact_path` against the
   strategy dir if relative. Short-circuits the chain if disabled or missing.
2. **BuildFeatureMatrixTask** — stack today's row from each candidate's
   neutralized feature frame + factor frame into a single matrix keyed by ticker.
3. **ApplyScoresTask** — predict and overwrite each `CandidateResult.rank_score`.

When the flag is off, `PanelScoringJob.should_skip()` returns True and the
per-ticker `rank_score` set by `CandidateJob` is used as-is (identical to 103).

---

## 3. FullTrainingPipeline (`pp_training_full.py`)

`scripts/train_104.py` is a thin CLI wrapper. All orchestration lives in
`kernel/pipeline/pp_training_full.py`:

```
FullTrainingPipeline
  ├─ BaselineTournamentJob     wraps TrainingPipeline (per-ticker champion)
  │    └─ RunBaselineTask
  │
  ├─ PanelTrainingJob          wraps PanelTrainingPipeline
  │    ├─ FetchPanelDataTask         OHLCV for watchlist ∪ SPY ∪ sector ETFs
  │    ├─ BuildPanelFeatureFramesTask  per-ticker labelled feature frames
  │    └─ RunPanelTrainingTask       panel-LTR model → artifacts/panel-ltr.json
  │
  └─ RecalibrationJob          wraps scripts.recalibrate_scores.recalibrate
       └─ RunRecalibrationTask refresh blend weights + per-symbol calibrations
```

Every phase is skippable via CLI flag (`--skip-baseline`, `--skip-panel`,
`--skip-recalibrate`) — each Job's `should_skip(ctx)` reads the corresponding
bool on `FullTrainingContext`. Tasks short-circuit the enclosing Job's chain
by returning False, matching the convention in `pp_inference.py`.

---

## 4. Runtime wiring

Three runtime entry points must set the same panel flag. All of them read
`ranking.panel_scoring.enabled` from `strategy_config.json`:

| Entry point | Responsibility |
|---|---|
| `main.py` (LEAN) | Uses `LeanAdapter` which pulls **520 bars** from LEAN History when the flag is on, then calls `prepare_inference_panel_frames` before `InferencePipeline.run()` |
| `live/runner.py` | Uses `RunnerAdapter` — identical prep, but fetches OHLCV from parquet cache via `common.fetch_ohlcv` |
| `sim/runner.py` | Notebook-style simulation; caller uses `swap_in_panel_scores()` helper to replace baseline `oos_raw_scores` before running selection |

The lazy import pattern in `pp_inference.py` (`from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob` inside `run()`) is load-bearing: `kernel.panel_pipeline.__init__` pulls in `job_panel_scoring`, which imports from `kernel.pipeline.context`, which triggers `kernel.pipeline.__init__` → `pp_inference`. Without the deferral we have a cycle. See `tests/test_panel_alignment.py::TestPipelineOrdering::test_panel_job_imported_lazily_inside_run` for the guard.

---

## 5. Test coverage

All renquant_103 alignment + policy tests ported to renquant_104 paths. Plus
panel-specific coverage:

| Test file | What it covers |
|---|---|
| `tests/test_panel_scoring_job.py` | 24 tests — Load / BuildMatrix / ApplyScores / Job wiring |
| `tests/test_panel_training_pipeline.py` | 33 tests — PanelTrainingPipeline end-to-end with Job/Task ABCs |
| `tests/test_panel_pipeline_e2e.py` | 4 tests — `prepare_inference_panel_frames` path |
| `tests/test_panel_inference.py` | 18 tests — inference-time feature / factor flows |
| `tests/test_panel_alignment.py` | **7 tests** — panel flag parity across LeanAdapter / RunnerAdapter / PanelScoringJob, pipeline ordering invariant |
| `tests/test_panel_*` (frame, labels, neutralization, imputation, factors, purged_cv, ltr_model, feature_matrix) | 100+ tests for the underlying building blocks |

Total test count after migration: **748 passing, 2 skipped**.

---

## 6. Scheduled runs

Daily automation mirrors renquant_103 schedule but drives the 104 scripts:

| Run | Time (PT) | Script | What it does |
|---|---|---|---|
| Market open | 6:32 AM | `live_only_104.sh --sell-only` | Exit stop-loss / gap-down positions using today's opening price |
| Pre-close | 12:44 PM | `live_only_104.sh --sell-only` | Exit intraday stop breaches before close |
| After close | 1:55 PM | `daily_104.sh` | `FullTrainingPipeline` → `export_lean_watchlist` → `live.runner --broker alpaca --once` |

LaunchAgents installed at `~/Library/LaunchAgents/com.renquant.{open,preclose,daily}104.plist`. Log paths: `logs/daily_104/`, `logs/live_104/`. Lock files under `/tmp/renquant_104_*` prevent concurrent runs. NYSE holiday guard and already-ran-today guard are identical to 103.
