# RenQuant 104 — Improvement Roadmap

Living doc. Ordered by ROI (highest first). Each item has: rationale, detailed action plan, acceptance criteria, estimated cost, dependencies.

When starting an item, flip its status to **🟡 in progress**.
When done, flip to **✅ done** and add the actual result in the trailing `### Result` section.

Related living docs:
- `doc/panel_ltr_primer.md` — training-method tutorial + glossary
- `doc/panel_training_runs.md` — per-run training log
- `doc/research_scoring.md` — original research / Stage 1-3 plan

---

## Status overview

| # | Item | Status | Est. | Depends on |
|---|---|---|---|---|
| 1 | Run 3 — lookahead=10d + regularization sweep | ✅ done — mean-IC 0.040, all folds positive | 0.5d | — |
| 2 | Global calibrator on panel | ✅ done — pool IC 0.071, enabled live | 1d | #1 |
| 3 | SQLite decision-trace database | ✅ done — schema + 5 queries + sim/live hooks | 1d | — |
| 4 | BaselineTournament winner by IC | ✅ done — flag wired, default sharpe preserved | 2h | — |
| 5 | Alpaca minute bars + intraday sell | ✅ done — IEX feed, 44/44 overlay, plist loaded | 1-2d | — |
| 6 | CPCV + regime-conditional calibration | 🟡 partial — CPCV shipped, regime-cond TODO | 2d | #2 |
| 7 | LightGBM LambdaRank backend | ✅ done — config-flag selectable | 1d | — |
| 8 | Hourly-bar features (not minute) | ⬜ pending | 2-3d | #5 |
| 9 | Cross-sectional transformer panel backend | ❌ shelved — ratio 0.49 OOS IC vs XGBoost; infra kept | 1d | — |

---

## Item 1 — Run 3: Panel-LTR lookahead=10d + regularization sweep

**Goal:** lift OOS mean-IC from 0.025 → ≥ 0.04, keep every fold positive.

**Rationale:**
Run 2 dropped overfitting from 213× to 16× train/OOS gap, but the 5-day forward label is too noisy for the available signal. Extending the horizon to 10 days doubles signal-to-noise while still matching our rotation horizon (20d). Further regularization squeezes the remaining bias/variance tradeoff.

**Action plan:**

1. Change config:
   - `panel_ltr.lookahead_days: 5 → 10`
   - `panel_ltr.cv_embargo_days: 5 → 10` (must match lookahead)
   - `panel_ltr.num_boost_round: 150 → 300`
   - `panel_ltr.xgb_params`: `eta 0.05 → 0.02`, `max_depth 4 → 3`, `min_child_weight 40 → 60`, `subsample 0.7 → 0.5`, `colsample_bytree 0.7 → 0.5`, add `lambda: 5.0`, `alpha: 2.0`
2. Run `scripts/train_104.py --force --skip-baseline` (~15 min on M4 Pro)
3. Record result in `doc/panel_training_runs.md` (prepend as Run 3)
4. Compare feature importance + fold distribution vs Run 2

**If Run 3 underperforms:** try the mirror direction — `lookahead=20`, `depth=4`, `rounds=200`.

**Acceptance:**
- OOS mean-IC ≥ 0.04 AND all 5 folds positive
- OR: documented why the 10d lookahead fails (label too sparse? β window issue?) + concrete next experiment

**Estimated cost:** 30 min config + 15 min training + 15 min writeup = **~1 hour**

### Result

**✅ Acceptance met.** OOS mean-IC lifted from Run 2's +0.025 to **+0.0403** (>0.04 target). All 5 folds positive with tight IQR (min 0.025, max 0.066). Train/OOS ratio halved again from 16× → 8.1×. Full writeup in `doc/panel_training_runs.md::Run 3`.

The lookahead extension (5d → 10d) was the single biggest lever — doubles the signal-to-noise ratio on the label since forward return variance scales with sqrt(horizon). Regularization tightening (eta 0.02, depth 3, stronger L1/L2) reduced overfitting without collapsing signal.

---

## Item 2 — Global calibrator on panel

**Goal:** replace 38 per-ticker `score_calibration` objects with a single panel-wide calibrator fitted on ~95000 rows. Reduces calibration variance by ~6×.

**Rationale:**
Current calibration is per-ticker isotonic/Platt on ~2500 rows each. Small-sample noise makes `rank_score` unreliable — a 0.6 from AAPL isn't comparable to a 0.6 from CRWD. Panel-LTR's raw output is already cross-sectionally meaningful; a single global calibrator preserves that.

**Action plan:**

1. New file `backtesting/renquant_104/training_panel/global_calibrator.py`:
   - `fit_global_calibrator(panel_scores_dict, future_returns_dict, lookahead, threshold) → GlobalCalibration`
   - Pools all (panel_score, future_return) pairs across tickers, fits a single isotonic mapping `raw → P(outperform SPY by threshold%)`
   - Separately fits `raw → E[R_i - R_spy]` for rotation's expected_return head
2. New script `scripts/fit_panel_calibrator.py`:
   - Loads `artifacts/panel-ltr.json` + reuses `PanelScorer` to generate OOS scores
   - Calls `fit_global_calibrator`
   - Writes `artifacts/panel-rank-calibration.json`
3. New task `ApplyGlobalCalibrationTask` in `kernel/panel_pipeline/job_panel_scoring.py`:
   - Runs between `ApplyScoresTask` and `VetoWeakBuysTask`
   - Reads panel_score on each candidate/holding, writes calibrated probability into `rank_score` and `expected_return`
4. Config block:
   ```json
   "ranking": {
     "panel_scoring": {
       "global_calibration": {
         "enabled": false,           // default off, flip after validation
         "artifact_path": "artifacts/panel-rank-calibration.json"
       }
     }
   }
   ```
5. Tests in `tests/test_global_calibrator.py`:
   - Monotonicity of calibrated output
   - Roundtrip save/load
   - Higher panel score → higher calibrated probability on average

**Wire into `FullTrainingPipeline`:** add `FitGlobalCalibratorJob` between `PanelNGBoostJob` and `RecalibrationJob`.

**Acceptance:**
- `artifacts/panel-rank-calibration.json` exists after training run
- Monotonicity tests pass
- Calibrated `rank_score` distribution in the notebook has sensible base rate (0.05-0.30 depending on regime)
- Optionally: A/B test showing flag=on vs off on the same sim window — expect smaller variance in per-bar selected stocks

**Estimated cost:** ~1 day

### Result

**✅ Shipped + enabled live.**

- `training_panel/global_calibrator.py::GlobalPanelCalibration` — two isotonic heads (probability + expected_return), JSON-serializable.
- `scripts/fit_panel_calibrator.py` — rebuilds panel feature/factor frames, scores every (ticker, date) via `PanelScorer`, computes forward relative-to-SPY returns, fits isotonic pool → writes `artifacts/panel-rank-calibration.json`.
- New pipeline tasks `LoadGlobalCalibrationTask` + `ApplyGlobalCalibrationTask` slotted between `VetoWeakBuysTask` and `LoadNGBoostTask` in `PanelScoringJob` (now 8 tasks total).
- Deferral rule: when `ngboost.enabled: true`, global-calibration is a no-op — NGBoost's μ-λσ is already a calibrated comparison-ready score.
- Config: `ranking.panel_scoring.global_calibration.{enabled, artifact_path}` — **enabled live**.
- Tests: `tests/test_global_calibrator.py` 12/12 pass (monotonicity, roundtrip, metadata, pipeline integration).
- Live calibrator fit: pool IC 0.0706 over 89,633 rows (includes in-sample — that's expected since calibration pools all data). OOS base rate 27.3%.

Flipped on in `strategy_config.json`. Note the ordering: global calibration runs before NGBoost, so turning off NGBoost will fall back to isotonic calibration automatically.

---

## Item 3 — SQLite decision-trace database

**Goal:** persistent structured storage for every inference run's decision trace and training artifact metadata, enabling SQL introspection.

**Rationale:**
Current setup dumps decisions into per-day JSON logs. To answer "which rotations were profitable last quarter?" or "which tickers get vetoed by sector_cap most often?" requires ad-hoc jq/grep. SQLite gives us indexable, joinable data at zero operational cost.

**Action plan:**

1. New file `backtesting/renquant_104/kernel/persistence.py`:
   ```python
   def get_db(path: Path) -> sqlite3.Connection
   def ensure_schema(conn) -> None
   def record_pipeline_run(conn, run_meta: dict) -> str  # returns run_id
   def record_candidate_scores(conn, run_id: str, candidates: list, holdings: dict) -> None
   def record_trades(conn, run_id: str, exits: list, orders: list) -> None
   def record_training_run(conn, artifact_type: str, metadata: dict) -> None
   ```

2. Schema (full SQL in module):
   ```sql
   -- Every InferencePipeline run (lean, live, sim all write here)
   CREATE TABLE pipeline_runs (
     run_id           TEXT PRIMARY KEY,
     run_date         DATE NOT NULL,
     run_type         TEXT NOT NULL,   -- 'lean' | 'live' | 'sim'
     regime           TEXT,
     confidence       REAL,
     portfolio_value  REAL,
     cash             REAL,
     n_candidates     INTEGER,
     n_exits          INTEGER,
     n_rotations      INTEGER,
     n_buys           INTEGER,
     commit_sha       TEXT,
     created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   CREATE INDEX idx_pipeline_runs_date ON pipeline_runs(run_date);

   -- Each ticker's score + blocker in each run
   CREATE TABLE candidate_scores (
     run_id         TEXT,
     ticker         TEXT,
     role           TEXT,             -- 'candidate' | 'holding'
     raw_score      REAL,
     rank_score     REAL,
     panel_score    REAL,
     rs_score       REAL,
     mu             REAL,
     sigma          REAL,
     selected       INTEGER,
     blocked_by     TEXT,             -- 'sector_cap' | 'correlation' | 'wash_sale' | 'below_threshold' | null
     PRIMARY KEY (run_id, ticker, role),
     FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
   );
   CREATE INDEX idx_cand_ticker ON candidate_scores(ticker);

   -- Actual executed trades
   CREATE TABLE trades (
     run_id         TEXT,
     ticker         TEXT,
     action         TEXT,             -- 'buy' | 'sell'
     shares         REAL,
     price          REAL,
     invest         REAL,
     target_pct     REAL,
     exit_reason    TEXT,             -- sell only
     pnl_pct        REAL,             -- sell only
     hold_days      INTEGER,          -- sell only
     tax            REAL,             -- sell only
     rank_score     REAL,             -- snapshot at entry
     conviction     REAL,
     sigma_mult     REAL,
     mu             REAL,
     sigma          REAL,
     FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
   );
   CREATE INDEX idx_trades_ticker ON trades(ticker);
   CREATE INDEX idx_trades_action ON trades(action);

   -- Rotation decision trees (one row per pair considered, whether executed or not)
   CREATE TABLE rotations (
     run_id        TEXT,
     cand_ticker   TEXT,
     held_ticker   TEXT,
     decision      TEXT,              -- 'swap' | 'below_threshold' | 'lt_protected' | 'min_hold' | ...
     cand_er       REAL,
     held_er       REAL,
     raw_adv       REAL,
     net_adv       REAL,
     tax_drag      REAL,
     threshold     REAL,
     FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
   );

   -- Training-side metadata
   CREATE TABLE training_runs (
     run_id         TEXT PRIMARY KEY,
     run_date       TIMESTAMP NOT NULL,
     artifact_type  TEXT,              -- 'panel-ltr' | 'ngboost-head' | 'tournament' | 'recalibrate'
     config_json    TEXT,
     oos_mean_ic    REAL,
     train_ic       REAL,
     n_rows         INTEGER,
     feature_cols   TEXT,              -- JSON array
     artifact_path  TEXT,
     created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

3. Hook-in points:
   - `adapters/lean.py::LeanAdapter.commit` → `record_pipeline_run` + `record_candidate_scores` + `record_trades`
   - `adapters/runner.py::RunnerAdapter.commit` → same three
   - `adapters/sim.py::SimAdapter.commit` → same three
   - `kernel/pipeline/pp_training_full.py::FullTrainingPipeline.run` → `record_training_run` for each phase artifact

4. Config (default off to keep git-friendliness):
   ```json
   "persistence": {
     "enabled": false,
     "db_path": "data/runs.db"
   }
   ```

5. Add `data/runs.db` to `.gitignore`.

6. New helper `scripts/query_runs.py` with 5 canned queries:
   - Last 30 days' pnl by exit_reason
   - Rotation realization rate (how many rotations ended up profitable within 20d)
   - Top-10 vetoed tickers by blocked_by
   - Per-regime win rate
   - Feature coverage over time (fundamentals, etc.)

7. Tests `tests/test_persistence.py`:
   - Schema creation
   - Insert + read back
   - Foreign key integrity
   - Flag off = no writes

**Acceptance:**
- After a live run with flag=on, `data/runs.db` exists and has populated rows
- 5 example queries in `scripts/query_runs.py` return sensible results
- Flag=off path is a no-op (no file created)

**Estimated cost:** ~1 day

### Result

**✅ Shipped.**

- `backtesting/renquant_104/kernel/persistence.py`: 5 tables (`pipeline_runs`, `candidate_scores`, `trades`, `rotations`, `training_runs`) + record_* helpers + schema auto-init.
- `adapters/sim.py::SimAdapter.commit`: writes pipeline_run + candidate_scores + trades per bar.
- `adapters/runner.py::RunnerAdapter.commit`: writes the same for live runs (synthesises trade events from ctx.exits + ctx.orders).
- `scripts/query_runs.py`: 6 canned queries (recent_runs, pnl_by_reason, rank_score_buckets, top_vetoed_tickers, regime_win_rate, training_history). `python scripts/query_runs.py list` enumerates them; `all` runs everything.
- Config: `persistence.enabled: true` + `persistence.db_path: data/runs.db` in `strategy_config.json`. Default off for new strategies so it stays opt-in.
- `/data/` already in `.gitignore`, so `data/runs.db` is excluded.
- Tests: `tests/test_persistence.py` 11/11 pass (schema, insert/read roundtrip, flag-off no-op, SimAdapter integration).

LeanAdapter hook deferred — LEAN's Docker needs a writable mount for the DB; lower priority than live/sim paths. Can land as a follow-up.

---

## Item 4 — BaselineTournament winner selection by IC

**Goal:** switch per-ticker tournament winner selection from long-only Sharpe to per-day Spearman IC (the metric that matches what the score is actually used for).

**Rationale:**
Current winner is picked by OOS Sharpe of the long-only signal sequence, but in 104 the baseline score mostly drives candidate filtering (`min_model_score` threshold) and rotation ER. Sharpe of a single-ticker signal path is noisy with ~500 OOS rows; IC is the objective metric and typically has ~3× lower variance.

**Action plan:**

1. In `training/tournament.py::run_tournament`, add alongside `oos_sharpe(prices, sigs)`:
   ```python
   def oos_per_day_ic(raw_scores, future_returns):
       # Spearman of raw_score vs fwd_return, per day, averaged
   ```
2. Config knob:
   ```json
   "ranking": {
     "tournament": {
       "winner_metric": "sharpe"  // "sharpe" | "ic"
     }
   }
   ```
3. When `winner_metric == "ic"`, swap the `sh` variable for IC in the winner-selection block (line ~115-180 in tournament.py).
4. `fit_probability_calibration` logic unaffected — it still uses raw scores.
5. Tests: new `TestTournamentWinnerMetric` class in `test_training_modules.py` asserting:
   - Same model selected when IC and Sharpe agree
   - Different model selected when they disagree (synthetic case)

**Note:** This is a low-priority item for 104 because the baseline tournament's ranking contribution is now overridden by Panel-LTR. Still, a cleaner metric means better `action` triggers → fewer false-positive candidates into `PanelScoringJob`.

**Acceptance:**
- Tests green
- With `winner_metric=ic`, at least one ticker changes its best_approach vs the Sharpe baseline
- No IC regression on the OOS set

**Estimated cost:** ~2 hours

### Result

**✅ Shipped.**

- `training/tournament.py::oos_single_ticker_ic` — per-ticker Spearman of raw_score vs future relative-to-SPY return.
- `run_tournament(..., winner_metric="sharpe"|"ic")` — new arg; default `"sharpe"` preserves existing behavior.
- `run_tournament_all` reads `ranking.tournament.winner_metric` from config.
- Result dict gains `selection_metric` + `selection_score` for downstream introspection.
- Tests: `TestTournamentWinnerMetric` 4/4 pass (default = sharpe, IC mode reports IC, perfect-foresight IC ≈ 1.0, random-signal IC ≈ 0).
- Not flipped in live config — leaves shipped model selection unchanged until you want to A/B. Flip via `"ranking": {"tournament": {"winner_metric": "ic"}}` in strategy_config.json.

---

## Item 5 — Alpaca minute bars + intraday sell check

**Goal:** every 30 min during market hours, run `SellOnlyPipeline` against freshly-pulled 5-min bars so stop-loss / trailing-stop / SDL trigger intraday instead of waiting until close.

**Rationale:**
Today's pipeline is EOD. A gap-down that resolves at close means we lose the entire day's decline before reacting. With Alpaca's free 5-min bars we can run mid-day risk-off checks at negligible cost.

**Action plan:**

1. Extend `kernel/data.py::fetch_ohlcv` (or new `fetch_intraday_bars`):
   ```python
   def fetch_intraday_bars(
       symbol: str,
       *,
       timeframe: str = "5Min",
       start: datetime | None = None,
       end: datetime | None = None,
   ) -> pd.DataFrame:
       from alpaca.data.historical import StockHistoricalDataClient
       from alpaca.data.requests import StockBarsRequest
       from alpaca.data.timeframe import TimeFrame
       # ... use ALPACA_API_KEY/SECRET from env
   ```
2. New adapter mode in `live/runner.py` or new `live/intraday_runner.py`:
   - Build InferenceContext from minute bars aggregated into daily-shape (OHLCV up to this minute)
   - Run `SellOnlyPipeline` (no buys!)
   - Commit exits via broker
3. Shell script `scripts/intraday_sell.sh`:
   - NYSE market-hours guard
   - Run `python -m live.intraday_runner --strategy renquant_104`
4. Launchd plist `com.renquant.intraday104.plist`:
   - Mon-Fri, every 30 min from 06:30 to 12:45 PT (market hours)
   - `StartCalendarInterval` with array of entries
5. Tests `tests/test_intraday_runner.py`:
   - Mocks Alpaca, asserts SellOnlyPipeline runs
   - Verifies no buys executed in intraday path

**Acceptance:**
- Manual `bash scripts/intraday_sell.sh` during market hours logs "N positions evaluated, M exits triggered"
- Plist loaded and scheduled; logs populate under `logs/intraday_104/`
- No buys ever emitted in intraday path

**Estimated cost:** ~1-2 days

**Risk:**
- Alpaca free tier gives **IEX data only** (not full NMS). For mid-cap/small-cap this may show different prices than the consolidated tape. For our watchlist (mostly large caps), negligible.
- If a legit mid-day SDL triggers, we sell at current IEX print — make sure the limit/market order type is chosen carefully.

### Result

**✅ Shipped + loaded into launchd.**

- `kernel/data.py::fetch_intraday_bars` — Alpaca 5Min/15Min/1Hour feeder, forces `DataFeed.IEX` to bypass the free-tier SIP block.
- `RunnerAdapter` gains `use_intraday_prices: bool`; when on, overlays latest 5-min close onto today's daily bar + prices dict. Gracefully falls back to daily closes if Alpaca fetch fails.
- `live.runner` CLI: new `--intraday` flag (combine with `--sell-only`).
- `scripts/intraday_sell_104.sh` + `com.renquant.intraday104.plist` — Mon-Fri, 20 slots between 07:00 and 12:30 PT (every 30 min, skips 06:32 open / 12:44 preclose slots which have their own plists). NYSE + lock file guards.
- **Smoke-tested live:** `Intraday overlay: 44/44 symbols had fresh minute bars` logged; SellOnlyPipeline runs in ~1.5s with zero positions (paper mode, no holdings to exit).
- Plist loaded: `launchctl list | grep renquant` shows all 5 active (open, preclose, daily, retrain-panel, intraday).

---

## Item 6 — CPCV + regime-conditional calibration

**Goal:** replace single-split Purged K-fold with Combinatorial Purged CV (full distribution of OOS IC, not a single mean); fit per-regime calibrators (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR each get their own).

**Rationale:**
5-fold CV gives one mean-IC with large stderr. CPCV tests K choose N subsets → yields a full distribution, lets us quantify model stability (std_ic / skew of fold IC). Per-regime calibrators correct for the fact that base rates differ dramatically across BULL and BEAR.

**Action plan:**

1. `training_panel/purged_cv.py`: add `CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_days=10)` — yields 15 combinations for (6, 2).
2. `training_panel/pp_panel_training.py::CrossValidateTask`: keep 5-fold as default, add `panel_ltr.cv_method: "purged" | "cpcv"` flag.
3. Artifact metadata gains `oos_ic_distribution` (list of ~15 values) + `oos_ic_std` + `oos_ic_quantiles`.
4. `training_panel/global_calibrator.py` (from item #2): extend with `fit_regime_conditional_calibrators(panel_scores, returns, regime_series) → dict[str, GlobalCalibration]`.
5. Artifact: `artifacts/panel-calibration-BULL_CALM.json`, `..._BEAR.json`, etc.
6. `kernel/panel_pipeline/job_panel_scoring.py::ApplyGlobalCalibrationTask`: look up calibrator by `ctx.regime`; fall back to mean calibrator if regime is empty.

**Acceptance:**
- CPCV produces a distribution (std_ic > 0)
- 4 regime calibrator JSONs exist
- ApplyGlobalCalibration picks correct calibrator per bar (tested)

**Estimated cost:** ~2 days

**Depends on:** Item #2 (global calibrator infra)

### Result

**🟡 CPCV shipped; regime-conditional calibration deferred.**

- `training_panel/purged_cv.py::CombinatorialPurgedCV` — `C(n_splits, n_test_groups)` splits with per-block purge + embargo.
- `cross_validated_ic_cpcv` adds `quantiles` (q05/25/50/75/95) to the result dict.
- `CrossValidateTask` dispatches on `panel_ltr.cv_method`: `"purged"` (default, current behavior) or `"cpcv"`. `cv_n_test_groups: 2` default.
- Panel artifact metadata now carries `oos_ic_quantiles` when CPCV is used, so the notebook + database can surface IC dispersion alongside mean.
- Also plumbed `backend` support into CrossValidateTask so LightGBM can CV too.
- Tests: `TestCombinatorialPurgedCV` 6 new tests (C(6,2) count, k-fold degenerate case, coverage, disjointness, validation, distribution sanity). All 15 purged_cv tests green.

**Regime-conditional calibration deferred** — requires per-regime panel_score buckets + regime-tagged labels, which is straightforward but adds complexity to `fit_panel_calibrator.py`. Not enabled until we have empirical evidence regime-specific calibrators beat the global one on realized pnl (needs database from Item #3 to accumulate data first). Logged as follow-up.

Flip `panel_ltr.cv_method: "cpcv"` when you want to run CPCV. Not enabled by default — at 15 CV fits it's ~3× slower than the 5-fold purged path.

---

## Item 7 — LightGBM LambdaRank backend

**Goal:** offer LightGBM with `lambdarank` objective as an alternative to XGBoost `rank:pairwise`. Expect ~2× faster training and a small IC improvement (NDCG@10 optimizes the top-of-list, which is what we care about).

**Rationale:**
`rank:pairwise` weights all pairs equally; NDCG@k weights top-k pairs more. For our use case (top-8 selection) this is exactly what we want. LightGBM also trains ~2× faster than XGBoost at equivalent accuracy.

**Action plan:**

1. `pip install lightgbm` (verify on Apple Silicon conda env).
2. New file `training_panel/lightgbm_ltr.py`:
   - `class PanelLGBMModel` with the same interface as `PanelLTRModel` (train / predict / save / load)
   - Uses `lightgbm.train` with `objective="lambdarank"`, `metric="ndcg"`, `eval_at=[5, 10]`, `group` from panel group sizes.
3. Config:
   ```json
   "panel_ltr": {
     "backend": "xgboost",    // "xgboost" | "lightgbm"
     "lightgbm_params": {
       "num_leaves": 15,
       "learning_rate": 0.02,
       "feature_fraction": 0.7,
       "bagging_fraction": 0.7,
       "lambda_rank_truncation_level": 10
     }
   }
   ```
4. `pp_panel_training.py::FinalFitTask`: dispatch on `backend`.
5. `kernel/panel_pipeline/panel_scorer.py::PanelScorer`: detect backend from artifact metadata (`"backend": "lightgbm"`), load the appropriate model.
6. Artifact compat: two formats (xgboost or lightgbm), but both expose `.feature_cols` + `.score(matrix)`.
7. Parity tests `tests/test_panel_lgbm.py`:
   - Trained model can be round-tripped via JSON
   - Train IC > 0.5 on synthetic strong-signal data (sanity)
   - Same panel with both backends produces correlated scores (Spearman > 0.7)

**Acceptance:**
- Both backends produce valid artifacts
- LightGBM artifact loadable by `PanelScorer`
- Training time for LightGBM ≤ 50% of XGBoost on real panel

**Estimated cost:** ~1 day

### Result

**✅ Shipped (infra only, default still XGBoost).**

- `training_panel/lgbm_ltr.py::PanelLGBMModel` with LambdaRank objective + NDCG@5,10 metric + `lambdarank_truncation_level: 10` (optimizes top-10 pairs, matches our 8-selection budget).
- `PanelLGBMScorer` mirrors `PanelScorer` interface.
- Label bucketization (`_bucketize_labels`): continuous Gaussianized labels → 11 integer gains [0..10] via quantile digitize (LambdaRank requires integer relevance).
- `FinalFitTask` dispatches on `panel_ltr.backend` ("xgboost" default | "lightgbm").
- `PanelScorer.load` dispatches on artifact `kind` so inference works transparently.
- Config: `panel_ltr.backend` + `panel_ltr.lightgbm_params`. Default kept on XGBoost to preserve Run 3's validated performance.
- Tests: `tests/test_panel_lgbm.py` 8/8 pass — bucketize monotonicity, fit+predict, save/load, dispatcher doesn't break XGBoost path.
- **Not benchmarked vs XGBoost yet** — deferred until you want to A/B. Flip `backend: "lightgbm"` in config and re-run `scripts/train_104.py --force --skip-baseline`. The LGBM scorer + calibrator hooks should work unchanged since they duck-type.

---

## Item 8 — Hourly-bar features (revised from minute-level)

**Goal:** expand panel feature set with **hourly-bar** derived signals (7 bars per trading session from Alpaca / yfinance): morning-to-afternoon drift, first-hour vs last-hour volume ratio, intraday VWAP vs close, intraday realized vol. Revised down from minute-level on 2026-04-22 — hourly captures ~80% of the microstructure value at ~20% of the engineering cost.

**Rationale:**
Daily OHLCV throws away all intra-day structure. Two days with the same daily bar can have very different intraday stories (slow drift up vs sharp morning spike and afternoon fade). Microstructure features capture this.

**Action plan:**

1. New module `training_panel/hourly_features.py` with ~6 features per (ticker, date):
   - `morning_drift`   — (hr1_close − open) / open
   - `afternoon_drift` — (close − hr1_close) / hr1_close
   - `vwap_premium`    — (close − intraday_vwap) / intraday_vwap
   - `vol_ratio`       — sum(last-hour volume) / sum(first-hour volume)
   - `intraday_realized_vol` — std of 7 hourly returns
   - `overnight_gap`   — (open − prev_close) / prev_close
2. Data ingestion: reuse Item #5's `fetch_intraday_bars` with `timeframe="1Hour"`. Cache at `data/intraday/{SYMBOL}/1h.parquet`.
3. Extend `PanelDataJob` with `FetchHourlyTask` + `HourlyFeatureTask` (computed once per day after close, no new scheduler).
4. Panel feature set grows from 20 → ~26 columns.
5. Retrain, compare IC vs Run 3 baseline.

**Acceptance:**
- OOS mean-IC improvement of ≥ 0.01 vs Run 3 (0.040 → ≥ 0.050)
- Training time stays under 20 min end-to-end
- Hourly features collectively account for > 10% of feature importance

**Estimated cost:** 2-3 days (design + fetch + feature code + training + eval)

**Depends on:** Item #5 (Alpaca intraday data ingestion infra — now complete).

### Result
(fill in)

---

## Item 9 — Cross-sectional transformer panel backend (shelved 2026-04-23)

**Goal:** A/B the transformer panel backend (design in `doc/renquant_104_transformer_design.md`) vs XGBoost. Ship if OOS IC ratio ≥ 1.30, ensemble if ≥ 1.10.

### Result

**❌ Shelved at ratio 0.49** on the real 47k-row panel. Full results logged in `doc/panel_training_runs.md::A/B — 2026-04-23 09:54 PT`.

| | XGBoost | Transformer |
|---|---|---|
| OOS mean IC (5-fold purged) | +0.0316 | **+0.0156** |
| Train IC (train/OOS ratio)  | 0.167 (5.6×) | 0.278 (**18×** — severe overfit) |
| Per-fold min IC | −0.023 | −0.070 |
| Training time | 51s | 198s |

**Kept (infra still shipped):**
- `backtesting/renquant_104/training_panel/transformer_model.py` — `PanelTransformerModel` class, 27 unit tests, standalone use works fine.
- `backtesting/renquant_104/kernel/panel_pipeline/transformer_scorer.py` — `TransformerPanelScorer` + PanelScorer.load dispatch.
- `backtesting/renquant_104/kernel/panel_pipeline/ensemble_scorer.py` — `EnsemblePanelScorer` (rank-averaging), ready for future experiments.
- `scripts/compare_panel_backends.py` — reproducible A/B driver, caps CV work so it finishes in ~5 min.

**Not wired (skipped step 2e):**
- Ensemble Task in `PanelScoringJob`. At ratio 0.49 it would drag composite IC down; no point wiring.

**When to revisit:**
- Panel grows to ~200k+ rows (2× universe OR 2× history window), OR
- A richer feature set (hourly microstructure from Item #8, alt-data feeds) gives the transformer's attention something tree splits can't capture.
- If either condition lands, flip `panel_ltr.backend: "transformer"` and rerun the A/B — no code changes needed.

---

## Working rhythm

1. Pick the topmost non-done item.
2. Flip its status to **🟡 in progress** in the status overview.
3. Execute the action plan end-to-end.
4. Fill in the **### Result** section at the bottom of that item (what actually happened, any surprises, any new subsequent items added to the roadmap).
5. Flip status to **✅ done**.
6. Run the full test suite before considering the item truly done.
7. Update `CLAUDE.md` test count if tests changed.

**Stop the current item when:**
- Acceptance criteria are met, OR
- The item's cost estimate is 2× exceeded without clear path to success — in that case, document the blocker in the Result section and move on.

**Never skip Result documentation** — future-us (or a different session) reading this doc needs to know what actually happened, not just the plan.
