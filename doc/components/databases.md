# RenQuant Database — `data/runs.db` + `data/sim_runs.db`

**The database is a core asset.** Every pipeline decision, every trade, every retrain, every held-position snapshot is logged so future analysis can introspect *why* the system did what it did without re-running anything.

> **2026-05-20 update**:
> - Universe grew 103 → wl200 (142 tickers, 2026-05-18). Projection: ~3.4× more
>   candidate_scores per day → ~2.4 GB/yr growth (was 700 MB/yr at 42-ticker projection)
> - `training_runs.artifact_type` enum extended for model_registry kinds:
>   `xgb`, `patchtst`, `hf_patchtst`, `regime_router`
> - `pending_orders` / `pending_broker_tickers` tracking (2026-05-17 State-EXT-SELL
>   fix) — needed so Sunday-queued unfilled BUY isn't misclassified as
>   "externally sold". Lives in `live_state.alpaca.json`
> - HF PatchTST shadow scoring outputs persist via `mlruns/renquant_104_shadow/`
>   (MLflow), not in runs.db directly; cross-ref via `pipeline_run_id`

> **2026-05-07 addition**: `experiment_configs` table (Task #38) —
> side-config storage as DB rows. Replaces the file-system
> `strategy_config.*.json` proliferation. Schema + helpers in
> `scripts/migrate_experiment_configs_to_db.py`; inflate at runtime
> via `holdout_backtest.py --experiment-label <name>`. Test coverage:
> `tests/test_experiment_configs_db.py` (11 green).
>
> ```sql
> CREATE TABLE experiment_configs (
>     label             TEXT PRIMARY KEY,    -- e.g. 'alpha158_linear'
>     base_config_name  TEXT NOT NULL,       -- usually 'strategy_config.json'
>     overrides_json    TEXT NOT NULL,       -- dotted-key dict
>     audit_label       TEXT,                -- side-config invariant test key
>     created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
>     updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
>     notes             TEXT
> );
> CREATE INDEX idx_experiment_configs_audit_label ON experiment_configs(audit_label);
> ```

Keep it:
- **Flexible** — schema migrations are idempotent (`ALTER TABLE ADD COLUMN` guarded by `PRAGMA table_info`); JSON blob columns absorb ad-hoc fields without schema churn.
- **Clean** — live/LEAN authoritative data is separated from ephemeral notebook experimentation.
- **Reproducible** — every row has a `commit_sha` pointing to the exact code that produced it.

---

## Two files, two roles (architecture 2026-04-24)

| File | Role | Writers | Lifetime | Read-only consumers |
|---|---|---|---|---|
| **`data/runs.db`** | Production / authoritative | `RunnerAdapter` (live trades), `LeanAdapter` (future, see roadmap) | **Permanent** — accumulate forever | Analyzers, retraining audits |
| **`data/sim_runs.db`** | Notebook experimentation | `SimAdapter` (notebook `run_backtest`) | **Ephemeral** — TRUNCATEd at the start of every `run_backtest()` so the 100th sim of the day is the only one whose rows survive | Ad-hoc notebook analysis of the CURRENT sim |

**Why the split:** the model is evolving. Yesterday's sim decisions are not meaningful ground truth for analyzing today's live decisions. Running the notebook 100 times while iterating should not pollute AA statistics. Live → authoritative; sim → overwritable.

### Role routing

`kernel.persistence.get_connection(config, strategy_dir, role="live" | "sim")`:
- `role="live"` (default) → reads `persistence.db_path` (default `data/runs.db`)
- `role="sim"` → reads `persistence.sim_db_path` (default `data/sim_runs.db`)

Config example:
```json
{
  "persistence": {
    "enabled":     true,
    "db_path":     "data/runs.db",
    "sim_db_path": "data/sim_runs.db"
  }
}
```

### Sim truncation

`clear_sim_tables(conn)` wipes these tables before each fresh sim:
- `pipeline_runs`
- `candidate_scores`
- `trades`
- `rotations`
- `live_state_snapshots`

**Not wiped** (treated as persistent derived data):
- `ticker_forward_returns` — historical OHLCV-derived, reusable across sim sessions
- `training_runs` — retrain audit log, accumulates across sessions

---

## Tables — 11 tables, 1 DB per role

| # | Table | Purpose | Owner |
|---|---|---|---|
| 1 | `pipeline_runs` | One row per `InferencePipeline.run()` invocation | `RunnerAdapter`, `LeanAdapter`, `SimAdapter` |
| 2 | `candidate_scores` | Per-(run, ticker, role) decision telemetry | `RankingJob`, `CandidateJob` |
| 3 | `trades` | Per-fill execution log | `RunnerAdapter` |
| 4 | `rotations` | Sell-paired-with-buy rotation events | `RotationJob` |
| 5 | `training_runs` | One row per `FullTrainingPipeline.run()` | `record_training_run()` |
| 6 | `ticker_forward_returns` | as_of_date × ticker fwd_1d/5d/10d/20d | `scripts/backfill_forward_returns.py` |
| 7 | `live_state_snapshots` | Per-bar `live_state.json` audit | `record_live_state_snapshot()` |
| 8 | `ticker_daily_state` | Daily per-ticker streak/HWM/has_position | `RunnerAdapter` (Bug #144 migration) |
| 9 | `score_distribution` + `score_percentiles_daily` + `score_distribution_meta` | Calibrator drift tracking — 1 row per training run + per-day distribution shapes | `RefreshPanelCalibratorJob` |
| 10 | `challenger_decisions` | **Phase 4a (2026-04-26)** — shadow-mode challenger vs live decision log | `kernel.challenger.log_decision()` (Phase 4b will wire in pp_inference) |
| 11 | _planned_ `training_run_gates` | **Per-gate verdict per training run** (deferred — see [`metadata-db-and-backup-plan.md`](metadata-db-and-backup-plan.md)) | `ModelAcceptanceGate.evaluate()` |

Both DBs (`runs.db` + `sim_runs.db`) have the same schema. Rows differ.

### 1. `pipeline_runs`

One row per `InferencePipeline.run()` invocation.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | `{run_date}-{run_type}-{uuid8}` |
| `run_date` | DATE | Trading day this pipeline ran for |
| `run_type` | TEXT | `"live"` / `"sim"` / `"lean"` (future) |
| `strategy` | TEXT | e.g. `"renquant_104"` |
| `regime` | TEXT | `BULL_CALM` / `BULL_VOLATILE` / `CHOPPY` / `BEAR` |
| `confidence` | REAL | Regime confidence ∈ [0, 1] |
| `portfolio_value` | REAL | NAV at pipeline entry |
| `cash` | REAL | Free cash at pipeline entry |
| `n_candidates` | INTEGER | Candidates surviving all filters |
| `n_exits` | INTEGER | ExitSignals emitted |
| `n_rotations` | INTEGER | Rotation pairs emitted |
| `n_buys` | INTEGER | New/top-up buy orders emitted |
| `commit_sha` | TEXT | **Git SHA at run time** — reproducibility key |
| `created_at` | TIMESTAMP | Wall-clock insert time |

Indexes: `(run_date)`, `(strategy)`.

### 2. `candidate_scores`

Per-(run, ticker, role) decision telemetry.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT | FK → `pipeline_runs` |
| `ticker` | TEXT | |
| `role` | TEXT | `"candidate"` or `"holding"` |
| `raw_score` | REAL | Per-ticker tournament raw output |
| `rank_score` | REAL | Calibrated probability (cross-model comparable) |
| `panel_score` | REAL | Panel-LTR cross-sectional score |
| `rs_score` | REAL | Relative-strength vs sector ETF |
| `mu` | REAL | NGBoost μ (excess return prediction) |
| `sigma` | REAL | NGBoost σ (std prediction) |
| `selected` | INTEGER | 1 if ended up in `ctx.orders` that bar |
| `blocked_by` | TEXT | Rejection reason (Plan P): `sector_guard` / `wash_sale` / `correlation_guard` / `tier_threshold` / `defensive_non_bear` — nullable |

PK `(run_id, ticker, role)`. Index `(ticker)`.

### 3. `trades`

Executed buys + sells.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT | FK |
| `ticker` | TEXT | |
| `action` | TEXT | `"buy"` / `"sell"` |
| `shares` | REAL | |
| `price` | REAL | |
| `invest` | REAL | Dollar amount (buys) |
| `target_pct` | REAL | Sized fraction of portfolio |
| `exit_reason` | TEXT | `model_sell` / `stop_loss` / `trailing_stop` / `max_hold` / `rotation` / `kelly_trim` / ... |
| `pnl_pct` | REAL | Realised pnl on sells |
| `hold_days` | INTEGER | |
| `tax` | REAL | Tax cost (sells) |
| `rank_score` | REAL | Score at decision time |
| `conviction` | REAL | conviction_multiplier value |
| `sigma_mult` | REAL | sigma_multiplier value |
| `mu` | REAL | NGBoost μ at trade time |
| `sigma` | REAL | NGBoost σ at trade time |

Indexes: `(ticker)`, `(action)`.

### 4. `rotations`

Per-pair rotation swap diagnostics (what was considered, what fired, what was rejected).

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT | FK |
| `cand_ticker` | TEXT | Buy side of swap |
| `held_ticker` | TEXT | Sell side of swap |
| `decision` | TEXT | `"swap"` / `"below_threshold"` / `"available"` / `"no_score"` / ... |
| `cand_er` | REAL | Candidate E[R-SPY] over horizon |
| `held_er` | REAL | Held E[R-SPY] over horizon |
| `raw_adv` | REAL | `cand_er - held_er` |
| `net_adv` | REAL | `raw_adv - tax_drag - txn_cost` |
| `tax_drag` | REAL | |
| `threshold` | REAL | `min_expected_advantage_pct` |

Index `(cand_ticker, held_ticker)`.

### 5. `training_runs`

Every `FullTrainingPipeline` (panel + tournament + NGBoost) run — a permanent retrain audit log.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | |
| `run_date` | TIMESTAMP | |
| `strategy` | TEXT | |
| `artifact_type` | TEXT | `panel-ltr` / `ngboost-head` / `tournament` / `panel-transformer` |
| `config_json` | TEXT | Full config snapshot |
| `oos_mean_ic` | REAL | Panel-LTR CPCV OOS mean IC |
| `train_ic` | REAL | Training IC |
| `n_rows` | INTEGER | Panel row count |
| `feature_cols` | TEXT (JSON) | Feature names |
| `artifact_path` | TEXT | Disk location of saved model |
| `commit_sha` | TEXT | Git SHA at retrain time |
| `created_at` | TIMESTAMP | |
| `elapsed_sec` | REAL | Retrain wall-time |
| `trigger` | TEXT | `scheduled_weekly` / `anomaly_spy_2pct` / `anomaly_vix_5pct` / `manual` / `cadence_daily` / `backtest` |
| `n_tickers` | INTEGER | |
| `n_dates` | INTEGER | |
| `n_features` | INTEGER | |
| `device` | TEXT | `mps` / `cuda` / `cpu` |
| `deterministic` | INTEGER | 0/1 — bit-reproducible mode |
| `training_window_years` | REAL | e.g. 5.0 when restricted |
| `notes` | TEXT | Freeform |

Index `(run_date)`.

### 6. `ticker_forward_returns`

Per-(date, ticker) forward returns — THE ground-truth layer for AA analysis.

| Column | Type | Notes |
|---|---|---|
| `as_of_date` | DATE | Decision date |
| `ticker` | TEXT | |
| `close_price` | REAL | Close on `as_of_date` |
| `fwd_1d` | REAL | `close[t+1] / close[t] - 1` |
| `fwd_5d` | REAL | |
| `fwd_10d` | REAL | Primary horizon for tier realization analysis |
| `fwd_20d` | REAL | Primary horizon for rotation analysis |
| `updated_at` | TIMESTAMP | |

PK `(as_of_date, ticker)`. Index `(ticker)`.

Populated by `scripts/backfill_forward_returns.py`:
```bash
# Backfill live DB (the usual case)
python scripts/backfill_forward_returns.py                 # → data/runs.db
python scripts/backfill_forward_returns.py --source sim    # → data/sim_runs.db
```

Upsert-with-COALESCE so running the script repeatedly merges partial data (e.g. when only `fwd_1d` is available the first day, `fwd_20d` fills in 20 trading days later).

### 7. `live_state_snapshots`

Per-bar snapshot of `live_state.json` — an append-only audit of what state the live runner wrote to disk each bar.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT PK | FK → `pipeline_runs` |
| `run_date` | DATE | |
| `strategy` | TEXT | |
| `regime` | TEXT | |
| `confidence` | REAL | |
| `high_water_mark` | REAL | |
| `cash` | REAL | |
| `portfolio_value` | REAL | |
| `n_holdings` | INTEGER | |
| `state_json` | TEXT | **Full JSON blob** — entry_dates / entry_signals / regime_state / monitor_state / etc. |
| `created_at` | TIMESTAMP | |

Indexes: `(run_date)`, `(strategy)`.

Answers historical queries like "what was `high_water_mark` on 2026-04-20?" without grepping logs:
```sql
SELECT high_water_mark FROM live_state_snapshots
WHERE run_date = '2026-04-20' AND strategy = 'renquant_104';
```

### 8. `ticker_daily_state`

Per-(run_date, ticker, strategy) snapshot. Bug #144 migration moved `sell_streak` from `live_state.json` into the DB.

| Column | Type | Notes |
|---|---|---|
| `run_date` | DATE | Trading day |
| `ticker` | TEXT | |
| `strategy` | TEXT | |
| `has_position` | INTEGER | 0/1 |
| `sell_streak` | INTEGER | Consecutive sell-signal days (only meaningful when has_position=1) |
| `entry_date` | DATE | When current position was opened |
| `entry_signal` | TEXT | Signal type that opened the position |
| `…` | | Other per-ticker daily state |
| `created_at` | TIMESTAMP | |

Indexes: `(run_date, strategy)`, `(ticker)`.

### 9. Calibrator drift tracking — `score_distribution` + `score_percentiles_daily` + `score_distribution_meta`

Trio of tables populated by `RefreshPanelCalibratorJob`:
- `score_distribution_meta` (1 row per training run): `date`, `calibrator_pool_ic`, `scorer_oos_ic`, `base_rate`, `threshold`, `n_features`, `artifact_path`
- `score_distribution`: histogram bins of raw scores
- `score_percentiles_daily`: per-day p05/p50/p95 of raw scores for drift dashboard

### 10. `challenger_decisions`

**Phase 4a (2026-04-26).** One row per (run_id, ticker, decision_date) when a challenger artifact is enabled via `acceptance.challenger.enabled=true`. Stores both the challenger's hypothetical decision and the live runner's actual decision so `compare_window()` can compute agreement / score correlation / disagreement-on-buy after a shadow window.

| Column | Type | Notes |
|---|---|---|
| `decision_id` | INTEGER PK AUTOINCREMENT | |
| `run_id` | TEXT | FK → `pipeline_runs.run_id` |
| `decision_date` | TEXT (ISO) | Trading day this decision applied to |
| `ticker` | TEXT | |
| `challenger_name` | TEXT | e.g. `"macro-enabled"`, `"transformer-v6"` |
| `challenger_score` | REAL | Raw panel score from challenger model |
| `challenger_rank_score` | REAL | Post-calibration rank score |
| `challenger_action` | TEXT | `"BUY"` / `"SELL"` / `"HOLD"` / `"PASS"` |
| `actual_score` | REAL | Live model's score for the same (date, ticker) |
| `actual_action` | TEXT | What live actually did |
| `created_at` | TIMESTAMP | |

Indexes: `idx_challenger_run (run_id)`, `idx_challenger_window (challenger_name, decision_date)`.

Live wiring (per-bar challenger scoring + log_decision call) is **Phase 4b** — schema landed in Phase 4a so the production DB is migration-ready.

### 11. _planned_ `training_run_gates`

Per-gate verdict per training run — see [`metadata-db-and-backup-plan.md`](metadata-db-and-backup-plan.md). Schema designed; implementation deferred.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT FK | → `training_runs` |
| `gate_name` | TEXT | `G1_schema`, `G4_oos_ic_vs_prior`, … |
| `severity` | TEXT | `"hard"` / `"soft"` |
| `passed` | INTEGER | 0/1 |
| `metric` | REAL | Computed value |
| `threshold` | REAL | Configured floor |
| `detail` | TEXT | Free-form explanation from gate |

PK `(run_id, gate_name)`.

---

## Schema evolution — migrations are idempotent

`ensure_schema(conn)` runs:
1. `CREATE TABLE IF NOT EXISTS` for every table (fresh DBs get the full schema).
2. `_apply_column_migrations()` runs idempotent `ALTER TABLE ADD COLUMN` for any column listed in `_COLUMN_MIGRATIONS` that's missing from an existing table.

**To add a new column:**
1. Add it to the `_SCHEMA_SQL` CREATE statement (for fresh DBs).
2. Add it to `_COLUMN_MIGRATIONS["table_name"]` list (for existing DBs).
3. Tests: add a case to `tests/test_training_run_audit.py::TestLegacySchemaMigration` (or parallel) verifying legacy-DB + new-column → migrated-DB works.

`PRAGMA foreign_keys = ON;` is enforced at connection time. `PRAGMA journal_mode = WAL;` for concurrent reads.

---

## Common queries

### What was the decision trace for 2026-04-20?
```sql
SELECT ticker, rank_score, selected, blocked_by
FROM candidate_scores cs
JOIN pipeline_runs ps ON ps.run_id = cs.run_id
WHERE ps.run_date = '2026-04-20' AND ps.run_type = 'live'
ORDER BY rank_score DESC;
```

### Why wasn't X bought on 2026-04-20?
```sql
SELECT blocked_by
FROM candidate_scores cs
JOIN pipeline_runs ps ON ps.run_id = cs.run_id
WHERE ps.run_date = '2026-04-20' AND cs.ticker = 'NVDA' AND ps.run_type = 'live';
```

### What was the panel retrain IC on the day we shipped v4?
```sql
SELECT run_date, oos_mean_ic FROM training_runs
WHERE commit_sha = 'eb8fab5' AND artifact_type = 'panel-ltr';
```

### Cross-regime IC on live data only
```bash
python scripts/analyze_decision_factors.py --source live --horizon 10
```

---

## Retention + hygiene (future)

- **sim DB** — TRUNCATEd every `run_backtest()`, so bounded.
- **live DB** — grows forever. Projected growth: ~1000 decisions × 42 tickers × ~50 fields ≈ 2 MB/day. After 1 year: ~700 MB. SQLite can handle GB-scale reads with the existing indexes. Monitor.
- If live DB exceeds ~5 GB: add `scripts/archive_runs.py` to cold-storage anything older than 2 years into a read-only attached DB.

---

## DB is a core asset — change discipline

- **Never** drop a column; add new columns via `_COLUMN_MIGRATIONS`.
- **Never** manually edit `data/runs.db` — use `record_*` helpers so every insert is audit-traceable (run_id, commit_sha, created_at).
- **Always** add a regression test when adding a column or a new table.
