# Database Design — Decision Factors & Model Factors

**Per user spec 2026-04-26 round-5**: 每天所有股票的 decision factor 都要
记到数据库里。

This doc enumerates EVERY field that needs to be recorded per bar +
per ticker, plus the model-side metadata. Tables already exist for
some of these; this doc is the SPEC for what should be there.

## Scope

Recording goal: post-hoc analyst can reconstruct **why** any given
ticker did or didn't trade on any given bar, AND under what model
version this decision was made.

## Two-tier design

### Tier 1: per-(date, ticker) decision context
Captures full state for every watchlist ticker at decision time.

### Tier 2: per-model-artifact registry
Captures model versions + training metadata. Bar-level rows reference
this via FK (commit_sha + artifact paths).

---

## Tier 1: Decision Factors (per-bar per-ticker)

### 1.1 Bar-level context
Stored in `pipeline_runs` (one row per bar, FK from per-ticker rows).

| Field | Type | Description |
|---|---|---|
| run_id | TEXT PK | unique per bar (date + run_type + uuid) |
| run_date | DATE | bar date |
| run_type | TEXT | 'live' / 'sim' / 'lean' |
| strategy | TEXT | 'renquant_104' |
| regime | TEXT | 'BULL_CALM' etc |
| confidence | REAL | regime detector confidence 0..1 |
| portfolio_value | REAL | account equity |
| cash | REAL | available cash |
| n_candidates | INTEGER | post-filter cand count |
| n_exits | INTEGER | total exits this bar |
| n_rotations | INTEGER | rotation pairs emitted |
| n_buys | INTEGER | buys placed |
| commit_sha | TEXT | git commit at run time |
| created_at | TIMESTAMP | row write time |

✅ All fields exist in `pipeline_runs`.

### 1.2 Per-(date, ticker) FULL state
Stored in `ticker_daily_state` (NEW — per user spec round-5).

| Field | Type | Description | Status |
|---|---|---|---|
| date | TEXT NOT NULL | bar date (PK with ticker) | NEW |
| ticker | TEXT NOT NULL | symbol | NEW |
| regime | TEXT | denormalized for query convenience | NEW |
| confidence | REAL | denormalized | NEW |
| in_watchlist | INTEGER | 1 if in strategy_config.watchlist | NEW |
| in_universe | INTEGER | 1 if passed universe floor (Sharpe ≥ 1.0) | NEW |
| pending_at_broker | INTEGER | 1 if BROKER-PRECHECK excluded | NEW |
| has_position | INTEGER | 1 if currently held | NEW |
| position_qty | REAL | shares held | NEW |
| position_pct | REAL | pct of portfolio | NEW |
| model_type | TEXT | per-ticker model class | NEW |
| model_action | TEXT | 'buy' / 'hold' / 'sell' | NEW |
| sell_streak | INTEGER | consecutive sell days (only when has_position=1) | NEW |
| panel_score | REAL | raw panel-LTR score | NEW |
| rank_score | REAL | calibrated probability | NEW |
| expected_return | REAL | calibrated ER | NEW |
| kelly_target_pct | REAL | μ/σ² Kelly target | NEW |
| mu | REAL | NGBoost μ | NEW |
| sigma | REAL | NGBoost σ | NEW |
| in_candidates | INTEGER | 1 if entered ctx.candidates (per-ticker model said BUY) | NEW |
| selected | INTEGER | 1 if BUY order placed this bar | NEW |
| blocked_by | TEXT | reason: 'sector_cap'/'corr'/'wash_sale'/'tier'/'universe_floor'/'broker_pending'/'no_model_signal' | NEW |
| sector | TEXT | from sector_map | NEW |
| PRIMARY KEY | (date, ticker) | | |

**This is the table user explicitly asked for.** Every watchlist ticker
gets a row per bar — including those filtered out at every gate.

### 1.3 Per-bar trade events (executed)
Stored in `trades`. ✅ already exists with all relevant fields.

### 1.4 Per-bar rotation pairs (proposed + outcome)
Stored in `rotations`. ✅ already exists.

### 1.5 Score distribution (Phase 1, just shipped)
Stored in `score_distribution` + `score_percentiles_daily`. ✅ commit `c75b611`.

---

## Tier 2: Model Factors (per-model-version registry)

### 2.1 Per-ticker model registry
Stored in `training_runs`. ✅ exists with:
- run_id, run_date, strategy, artifact_type, config_json
- oos_mean_ic, train_ic, n_rows, feature_cols, artifact_path
- commit_sha, elapsed_sec, trigger, n_tickers, n_dates, n_features
- device, deterministic, training_window_years, notes

**Missing for per-ticker context**:
- `oos_sharpe` (currently in policy-metadata.json but not training_runs)
- `train_end_date` (last in-sample date)
- `model_class` (Manual / XGBoost / QLearning / Classification)
- `feature_versions` (which indicator definitions)

### 2.2 Panel-LTR model registry
Same `training_runs` table can hold panel-LTR rows (artifact_type='panel_ltr').
Already includes:
- backend (xgboost / lightgbm / transformer) — implicit in artifact name
- oos_mean_ic, train_ic
- monotone_constraints — in config_json
- feature_cols (the panel features)

Adding `panel_buy_floor`, `panel_buy_top_n` snapshot: should be in
config_json BLOB.

### 2.3 Calibrator registry
Currently NOT in training_runs as a separate row. Calibrator metadata
lives only in panel-rank-calibration.json.

**Recommended**: add a new `calibrator_runs` table or use
`training_runs` with `artifact_type='calibrator'`:

| Field | Type | Description |
|---|---|---|
| run_id | TEXT PK | |
| run_date | DATE | |
| pool_ic | REAL | |
| scorer_oos_mean_ic | REAL | |
| n_rows | INTEGER | calibrator eval rows |
| base_rate | REAL | |
| threshold | REAL | (e.g. 0.03 forward return) |
| method | TEXT | 'isotonic' / 'platt' / 'constant' |
| artifact_path | TEXT | |
| commit_sha | TEXT | |

Use existing `training_runs` table with an additional `kind` column or
a new dedicated table.

### 2.4 NGBoost head registry
Same approach: `training_runs` row with `artifact_type='ngboost_head'`.
Already feasible with current schema; just need consistent writer.

---

## Implementation status (round-5)

| Tier | Item | Status | Commit |
|---|---|---|---|
| 1.1 | pipeline_runs | ✅ exists, commit_sha included | (pre-existing) |
| 1.2 | candidate_scores extended (model_type, sector, expected_return, kelly_target, panel_artifact) | ✅ shipped | (this session) |
| 1.2 | **ticker_daily_state full coverage** | 🟠 SCHEMA ADDED, writer not yet wired | (this session) |
| 1.3 | trades | ✅ pre-existing | |
| 1.4 | rotations | ✅ pre-existing | |
| 1.5 | score_distribution | ✅ shipped | `c75b611` |
| 2.1-2.4 | training_runs | 🟡 partial — needs writer for calibrator + ngboost rows | future |

## What's NOT yet wired but defined

1. **Writer for ticker_daily_state** (writes EVERY watchlist ticker, not
   just ctx.candidates). This is the user's primary ask. ETA 30 min:
   - In adapters/runner.py commit(): build per-ticker row from
     watchlist + universe + broker + holdings + ctx.candidates
   - Insert via `record_ticker_daily_state(conn, run_id, rows)`
2. **calibrator_runs row** on each calibrator refresh (script-side).
3. **NGBoost head row** on each NGBoost retrain.

## Recommended call order in adapters/runner.py.commit()

```python
run_id = record_pipeline_run(...)                 # bar-level
record_candidate_scores(...)                      # cands + holdings (row-rich)
record_ticker_daily_state(conn, run_id, ...)      # NEW: every watchlist ticker
record_trades(...)                                # actual orders
# rotations recorded separately by EmitRotationsTask
record_live_state_snapshot(...)                   # state JSON archive
```

## Disk size estimate

- ticker_daily_state: 100 tickers × 250 trading days × 10 years = 250k rows
  × ~25 bytes/row (mostly small integers + score floats) ≈ 6.25 MB.
  Trivial.
- score_distribution: similar scale.
- pipeline_runs: 250 days × 10 years × 4 runs/day (intraday + daily) = 10k rows × ~100 bytes = 1 MB.

Total DB growth: ~10-15 MB/year. Negligible.
