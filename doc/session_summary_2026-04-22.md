# Session Summary — 2026-04-22

Working through `doc/improvement_roadmap.md`. Session goal: drive Panel-LTR OOS IC from 0.025 → ≥ 0.04, ship σ-aware ranking + σ-sizing, consolidate decision traces into SQL, and expand inference infra for intraday price-sensitive sells.

## Items completed (7 of 8)

| # | Item | Key result |
|---|---|---|
| 1 | **Run 3 — Panel-LTR lookahead=10d + regularization sweep** | OOS mean-IC **+0.0403** (from +0.0250), all 5 folds positive, train/OOS ratio 8× (from 16×). Top-5 feature gain concentration 30.1%. |
| 2 | **Global calibrator on panel** | Single isotonic fit on pooled 89,633 rows (38×) instead of 38 per-ticker calibrators. Pool IC 0.071. Enabled live via `ranking.panel_scoring.global_calibration`. |
| 3 | **SQLite decision-trace database** | 5 tables (pipeline_runs / candidate_scores / trades / rotations / training_runs) + 6 canned queries + hooks in SimAdapter / RunnerAdapter. **Enabled live** — every pipeline run now writes to `data/runs.db`. |
| 4 | **BaselineTournament winner by IC** | New `oos_single_ticker_ic` metric + `ranking.tournament.winner_metric` flag (default `"sharpe"` preserves current behavior). |
| 5 | **Alpaca minute bars + intraday sell check** | `fetch_intraday_bars` with IEX feed (bypasses SIP free-tier block). `--sell-only --intraday` CLI. launchd plist loaded — 20 slots between 07:00–12:30 PT Mon-Fri. Smoke-tested live: 44/44 symbols overlaid successfully. |
| 6 | **CPCV + regime-conditional calibration** | `CombinatorialPurgedCV` with n_test_groups=k → C(N,k) splits + IC quantiles. Plugged into `CrossValidateTask` via `panel_ltr.cv_method: "cpcv"`. Regime-conditional calibration deferred (requires DB evidence). |
| 7 | **LightGBM LambdaRank backend** | `PanelLGBMModel` + `PanelLGBMScorer` with NDCG@10 objective. Dispatcher in `PanelScorer.load` + `FinalFitTask`. Tests + infra shipped; not benchmarked vs XGBoost yet. |
| 8 | **Hourly-bar features** | Scope revised from minute → hourly (~80% of value, ~20% of cost). Plan updated in roadmap; data ingestion infra from #5 is reusable. Deferred to future session. |

## Training run comparison (Panel-LTR)

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Config | fundamentals on, 400 rounds, depth=6 | fundamentals off, 150 rounds, depth=4 | lookahead=10d, 300 rounds, depth=3, strong reg |
| OOS mean-IC | +0.0039 | +0.0250 | **+0.0403** |
| Min fold IC | −0.0102 | +0.0158 | **+0.0253** |
| All folds ≥ 0 | ✗ | ✓ | ✓ |
| Train IC | +0.833 | +0.399 | **+0.326** |
| Train/OOS ratio | 213× | 16× | **8.1×** |

## Config deltas applied live

```json
{
  "training": {
    "cadence": "custom",
    "allowed_weekdays": [1, 3, 6]       // Tue/Thu/Sun
  },
  "panel_ltr": {
    "lookahead_days": 10,
    "cv_embargo_days": 10,
    "num_boost_round": 300,
    "xgb_params": {
      "eta": 0.02, "max_depth": 3, "min_child_weight": 60,
      "subsample": 0.5, "colsample_bytree": 0.5,
      "lambda": 5.0, "alpha": 2.0
    },
    "fundamentals": {"enabled": false},   // deferred until time-series data
    "ngboost": {"enabled": true, "score_mode": "mu_minus_lambda_sigma"},
    "backend": "xgboost"                  // can flip to "lightgbm"
  },
  "ranking": {
    "panel_scoring": {
      "enabled": true,
      "global_calibration": {"enabled": true},
      "ngboost": {"enabled": true, "lambda_sigma": 1.0},
      "sigma_sizing": {"enabled": true, "floor": 0.3, "ceiling": 1.0}
    },
    "tournament": {
      "winner_metric": "sharpe"            // kept default; can flip to "ic"
    }
  },
  "persistence": {
    "enabled": true,
    "db_path": "data/runs.db"
  }
}
```

## Schedule (updated)

| Time (PT) | Script | Purpose |
|---|---|---|
| Mon-Fri 06:32 | `live_only_104.sh --sell-only` | Open-bar exit check |
| Mon-Fri 07:00–12:30 every 30min | `intraday_sell_104.sh` (NEW) | Intraday SDL / trailing-stop triggers |
| Mon-Fri 12:44 | `live_only_104.sh --sell-only` | Pre-close exit check |
| Mon-Fri 13:55 | `daily_104.sh` | Daily trade + Tue/Thu retrain |
| Sun 10:00 | `retrain_panel.sh` | Forced weekly retrain (NEW) |

launchd plists loaded: `com.renquant.{open,preclose,daily,retrain-panel,intraday}104`

## Tests

904 passed, 2 skipped (+45 since start of session).

New test files:
- `test_persistence.py` (11 tests)
- `test_global_calibrator.py` (12 tests)
- `test_panel_lgbm.py` (8 tests)
- CPCV extension to `test_panel_purged_cv.py` (+6 tests)
- Tournament IC metric in `test_training_modules.py` (+4 tests)
- Sim adapter smoke in `test_sim_pipeline_smoke.py` (+6 tests)

## Known issues

1. **Notebook panel-sim bug (fixed)** — `swap_in_panel_scores` `continue` statement skipped `out[ticker] = new_r` when `global_calibration` was passed, so the panel `results` dict dropped every ticker and the sim saw zero candidates. Fixed (replaced `continue` with `if/else` branch).
2. **Sim path not yet refactored** to use `InferencePipeline` via `SimAdapter`. The SimAdapter infrastructure is built (Item #3 from prior session + this one's persistence hooks) and gated behind `sim.use_pipeline: false` to preserve current notebook behavior. Formal parity test + migration is a dedicated next-session item.
3. **NGBoost `score_mode=mu_minus_lambda_sigma` in sim** — writes μ-λσ (range ~[-0.05, +0.05]) into `oos_raw_scores`, which then goes through the global calibrator fit on raw panel scores (range ~[-0.23, +0.04]). Most candidates map below the 0.10 min_score. Workaround: sim uses `score_mode="additive"` so ranking stays on the panel+calibrator path and σ-sizing still applies. Live/LEAN doesn't have this issue because BlendScoresTask renormalizes rank_score per-bar after NGBoost. A proper fix for sim would be to either mirror the renormalization in the legacy path or migrate to the pipeline-driven path.

## Next-session priorities

1. **Re-run training + sim with the bug fix** → confirm Panel-new sim numbers match expectations
2. **Item #8 hourly microstructure** (now 2-3d instead of 1wk+)
3. **Regime-conditional calibrators** (requires Item #3 DB to accumulate enough real-world evidence first)
4. **Finish sim/runner.py → InferencePipeline refactor** with parity tests
5. **LightGBM A/B** — retrain with `backend: "lightgbm"` and compare IC vs XGBoost
