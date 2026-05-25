"""SQLite-backed decision-trace persistence.

Every InferencePipeline run (LEAN, live, sim) writes a structured trace
to `data/runs.db` so future analysis can introspect *why* a decision was
made without grepping JSON logs.

Schema:

  pipeline_runs      — one row per InferencePipeline.run() invocation
  candidate_scores   — per-(run, ticker) score + blocker telemetry
  trades             — executed buys/sells with pnl + exit reason + tax
  rotations          — rotation pairs considered (swap/rejected + diagnostics)
  training_runs      — FullTrainingPipeline artifact metadata

All writes go through `record_*` functions. If `persistence.enabled = false`
in config, every record_* becomes a no-op — nothing is written and no DB
file is created. Default is off.

Kept `common/`-free (self-contained stdlib + sqlite3) so it runs inside
LEAN's Docker too.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("kernel.persistence")
_ECON_ABS_TOL = 1e-6
_ECON_REL_TOL = 1e-6


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    run_date         DATE NOT NULL,
    run_type         TEXT NOT NULL,
    strategy         TEXT,
    regime           TEXT,
    confidence       REAL,
    portfolio_value  REAL,
    cash             REAL,
    n_candidates     INTEGER,
    n_exits          INTEGER,
    n_rotations      INTEGER,
    n_buys           INTEGER,
    buy_blocked      INTEGER,
    skip_buys        INTEGER,
    bear_only        INTEGER,
    counters_json    TEXT,
    run_bundle_json  TEXT,
    commit_sha       TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date ON pipeline_runs(run_date);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_strategy ON pipeline_runs(strategy);

CREATE TABLE IF NOT EXISTS candidate_scores (
    run_id         TEXT,
    ticker         TEXT,
    role           TEXT,
    raw_score      REAL,
    rank_score     REAL,
    panel_score    REAL,
    rs_score       REAL,
    mu             REAL,
    sigma          REAL,
    selected       INTEGER,
    blocked_by     TEXT,
    -- Audit fix DB-DECISION-FACTORS (2026-04-26 round-5): per user spec
    -- "每天的所有股票的 decision factor 都要记到数据库里". Capture
    -- additional factors that drove this bar's decision so post-hoc
    -- analysis can reconstruct WHY each ticker was selected/blocked.
    expected_return    REAL,        -- calibrated ER (drives rotation)
    expected_return_horizon_days INTEGER,
    kelly_target_pct   REAL,        -- Kelly sizing target (μ/σ²)
    model_type         TEXT,        -- per-ticker model: 'Manual' | 'XGBoost' | 'QLearning' | 'Classification'
    sector             TEXT,        -- from sector_map, easier than join
    panel_ltr_artifact TEXT,        -- 'panel-ltr.json' filename or full path
    mu_horizon_days    INTEGER,
    qp_delta_w         REAL,        -- QP optimized delta weight for this ticker
    qp_target_w        REAL,        -- QP optimized target weight for this ticker
    qp_status          TEXT,        -- optimizer status attached to this row
    PRIMARY KEY (run_id, ticker, role),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_cand_ticker ON candidate_scores(ticker);

CREATE TABLE IF NOT EXISTS trades (
    run_id         TEXT,
    trade_date     DATE,
    ticker         TEXT,
    action         TEXT,
    shares         REAL,
    price          REAL,
    invest         REAL,
    target_pct     REAL,
    exit_reason    TEXT,
    pnl_pct        REAL,
    hold_days      INTEGER,
    tax            REAL,
    gross_pnl      REAL,
    proceeds_basis REAL,
    net_pnl_after_tax REAL,
    tax_cash_debited REAL,
    tax_cash_debit_mode TEXT,
    tax_lot_method TEXT,
    rank_score     REAL,
    conviction     REAL,
    sigma_mult     REAL,
    mu             REAL,
    mu_horizon_days INTEGER,
    sigma          REAL,
    panel_score    REAL,
    rs_score       REAL,
    expected_return REAL,
    expected_return_horizon_days INTEGER,
    kelly_target_pct REAL,
    model_type     TEXT,
    sector         TEXT,
    blocked_by     TEXT,
    qp_delta_w     REAL,
    qp_target_w    REAL,
    qp_status      TEXT,
    regime         TEXT,
    confidence     REAL,
    order_type     TEXT,
    source         TEXT,
    source_job     TEXT,
    source_task    TEXT,
    order_source   TEXT,
    attribution_version TEXT,
    score_snapshot_json  TEXT,
    decision_inputs_json TEXT,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action);

CREATE TABLE IF NOT EXISTS rotations (
    run_id        TEXT,
    cand_ticker   TEXT,
    held_ticker   TEXT,
    decision      TEXT,
    cand_er       REAL,
    held_er       REAL,
    raw_adv       REAL,
    net_adv       REAL,
    tax_drag      REAL,
    threshold     REAL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_rotations_swap ON rotations(cand_ticker, held_ticker);

-- Daily portfolio risk metrics — computed from pipeline_runs.portfolio_value
-- time series. The user's goal is Sharpe=2.0 on the golden config; without
-- tracking Sharpe over time we can't measure progress. Backfilled + updated
-- by scripts/compute_portfolio_metrics.py. Supports both live + sim roles.
CREATE TABLE IF NOT EXISTS portfolio_daily_metrics (
    as_of_date      DATE NOT NULL,
    run_type        TEXT NOT NULL,    -- 'live' | 'sim' | 'lean'
    strategy        TEXT,
    portfolio_value REAL,
    daily_return    REAL,             -- one-day simple return
    -- Rolling windows (trading days)
    sharpe_21d      REAL,             -- 1-month rolling Sharpe (annualized)
    sharpe_63d      REAL,             -- 3-month rolling Sharpe (annualized)
    sharpe_252d     REAL,             -- 1-year rolling Sharpe (annualized)
    realized_vol_21d REAL,            -- annualized stdev of daily returns, 21d
    realized_vol_252d REAL,           -- annualized stdev, 252d
    max_drawdown_252d REAL,           -- max peak-to-trough drawdown, 252d window
    var_95_21d      REAL,             -- 95%-VaR (1-day), 21-day empirical
    var_99_21d      REAL,             -- 99%-VaR
    beta_spy_252d   REAL,             -- regression beta vs SPY over 252d
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, run_type, strategy)
);
CREATE INDEX IF NOT EXISTS idx_pdm_date ON portfolio_daily_metrics(as_of_date);

-- Plan S — per-bar snapshots of live_state.json for historical audit.
-- The JSON file is the source of truth for live state (fast bootstrap, human-
-- editable). These rows are an append-only audit trail: "what did live_state
-- look like at the close of run R?". Indexed fields allow quick queries
-- without parsing the blob.
CREATE TABLE IF NOT EXISTS live_state_snapshots (
    run_id          TEXT PRIMARY KEY,    -- FK to pipeline_runs.run_id
    run_date        DATE NOT NULL,
    strategy        TEXT,
    regime          TEXT,
    confidence      REAL,
    high_water_mark REAL,
    cash            REAL,
    portfolio_value REAL,
    n_holdings      INTEGER,
    state_json      TEXT NOT NULL,       -- full state blob for later introspection
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_lss_date ON live_state_snapshots(run_date);
CREATE INDEX IF NOT EXISTS idx_lss_strategy ON live_state_snapshots(strategy);

-- Plan AA — forward returns keyed by (date, ticker). Decoupled from the
-- candidate_scores row so we can backfill out-of-band once N days have
-- elapsed since the decision. Populated by `scripts/backfill_forward_returns.py`.
CREATE TABLE IF NOT EXISTS ticker_forward_returns (
    as_of_date  DATE NOT NULL,   -- decision date (matches pipeline_runs.run_date)
    ticker      TEXT NOT NULL,
    close_price REAL,            -- close on as_of_date (base for the % changes)
    fwd_1d      REAL,            -- close[t+1]/close[t] - 1
    fwd_5d      REAL,
    fwd_10d     REAL,
    fwd_20d     REAL,
    fwd_60d     REAL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_tfr_ticker ON ticker_forward_returns(ticker);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id         TEXT PRIMARY KEY,
    run_date       TIMESTAMP NOT NULL,
    strategy       TEXT,
    artifact_type  TEXT,
    config_json    TEXT,
    oos_mean_ic    REAL,
    train_ic       REAL,
    n_rows         INTEGER,
    feature_cols   TEXT,
    artifact_path  TEXT,
    commit_sha     TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Round 5 additions: explicit training-time metadata per user spec
    elapsed_sec    REAL,
    trigger        TEXT,         -- 'scheduled_weekly' | 'anomaly_spy_2pct' | 'anomaly_vix_5pct' | 'manual' | 'cadence_daily' | 'backtest'
    n_tickers      INTEGER,
    n_dates        INTEGER,
    n_features     INTEGER,
    device         TEXT,         -- 'mps' | 'cuda' | 'cpu' | 'n/a'
    deterministic  INTEGER,      -- 0 = non-det, 1 = deterministic (determinstic mode is slower but bit-reproducible)
    training_window_years REAL,  -- e.g. 5.0 when restricted to last-5-year window
    notes          TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_runs_date ON training_runs(run_date);

-- Calibration score database (2026-04-26 round-5).
-- Per user spec: "建立 calibrate 数据库, 这样才知道什么 score value 是 top 5%"
-- Phase 1: collect score distribution per bar; phase 2 will use these
-- to drive percentile-based admission in JointActionTask.
--
-- Each row is one (date, ticker) candidate scored by the panel scorer
-- in PanelScoringJob. Holdings ARE included (they have rank_score too).
-- Per-(date, ticker) DAILY DECISION SNAPSHOT for ALL watchlist tickers.
-- Per user spec 2026-04-26 round-5: "每天所有股票的 decision factor
-- 都要记到数据库里". Unlike `candidate_scores` (only cands + holdings),
-- this table covers EVERY watchlist ticker per bar with its FULL
-- context — even those filtered out by universe/broker-precheck/etc.
-- Goal: post-hoc analysis can answer "what did we KNOW about XYZ on
-- 2026-04-26 and WHY didn't we trade it?".
CREATE TABLE IF NOT EXISTS ticker_daily_state (
    run_id            TEXT NOT NULL,
    date              TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    -- Bar-level context (joined for query convenience, denormalized)
    regime            TEXT,
    confidence        REAL,
    -- Universe / broker membership
    in_watchlist      INTEGER,        -- 1 if ticker in strategy_config.watchlist
    in_universe       INTEGER,        -- 1 if model passed universe floor (Sharpe etc.)
    pending_at_broker INTEGER,        -- 1 if BROKER-PRECHECK excluded this bar
    -- Position state
    has_position      INTEGER,        -- 1 if currently held
    position_qty      REAL,           -- shares held (NULL if not held)
    position_pct      REAL,           -- pct of portfolio (NULL if not held)
    -- Per-ticker model output
    model_type        TEXT,           -- 'Manual' | 'XGBoost' | 'QLearning' | 'Classification'
    model_action      TEXT,           -- 'buy' | 'hold' | 'sell'
    sell_streak       INTEGER,        -- only meaningful when has_position=1
    -- Panel scores (when computed)
    panel_score       REAL,
    rank_score        REAL,
    expected_return   REAL,
    expected_return_horizon_days INTEGER,
    kelly_target_pct  REAL,
    mu                REAL,
    mu_horizon_days   INTEGER,
    sigma             REAL,
    -- Final decision
    in_candidates     INTEGER,        -- 1 if reached ctx.candidates (per-ticker model said buy)
    selected          INTEGER,        -- 1 if BUY order placed this bar
    blocked_by        TEXT,           -- reason if blocked: 'sector_cap'|'corr'|'wash_sale'|'tier'|'universe_floor'|'broker_pending'|'no_model_signal'
    sector            TEXT,
    qp_delta_w        REAL,           -- QP optimized delta weight for this ticker
    qp_target_w       REAL,           -- QP optimized target weight for this ticker
    qp_status         TEXT,           -- optimizer status attached to this row
    PRIMARY KEY (run_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_tds_date ON ticker_daily_state(date);
CREATE INDEX IF NOT EXISTS idx_tds_ticker ON ticker_daily_state(ticker);

CREATE TABLE IF NOT EXISTS score_distribution (
    run_id        TEXT NOT NULL,
    date          TEXT NOT NULL,        -- YYYY-MM-DD (string for sqlite friendliness)
    run_type      TEXT,                 -- live | sim | lean
    ticker        TEXT NOT NULL,
    raw_panel     REAL,                 -- pre-calibration scorer output (panel_score)
    rank_score    REAL,                 -- post-calibration probability
    expected_return_horizon_days INTEGER,
    mu            REAL,                 -- NGBoost μ if active
    mu_horizon_days INTEGER,
    sigma         REAL,                 -- NGBoost σ if active
    regime        TEXT,                 -- BULL_CALM / etc.
    is_holding    INTEGER DEFAULT 0,    -- 0 = candidate, 1 = held
    model_type    TEXT,
    sector        TEXT,
    blocked_by    TEXT,
    PRIMARY KEY (run_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_score_dist_date ON score_distribution(date);

-- Daily aggregated percentiles for fast threshold lookup.
-- Computed at end of each bar from that day's score_distribution rows.
-- Phase 2 JointAction reads this to convert "top X%" → absolute threshold.
CREATE TABLE IF NOT EXISTS score_percentiles_daily (
    run_id        TEXT PRIMARY KEY,
    date          TEXT NOT NULL,
    run_type      TEXT,
    n_cands       INTEGER NOT NULL,
    p01           REAL,
    p05           REAL,
    p10           REAL,
    p25           REAL,
    p50           REAL,
    p75           REAL,
    p85           REAL,                 -- "top 15%"
    p90           REAL,                 -- "top 10%"
    p95           REAL,                 -- "top 5%"
    p99           REAL,
    score_min     REAL,
    score_max     REAL,
    score_mean    REAL,
    score_std     REAL,
    regime        TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pctiles_date ON score_percentiles_daily(date);

-- Calibrator drift tracking — 1 row per training run.
-- Operator dashboard can plot pool_ic / scorer_oos_ic over time.
CREATE TABLE IF NOT EXISTS score_distribution_meta (
    date              TEXT PRIMARY KEY,
    calibrator_pool_ic REAL,
    scorer_oos_ic     REAL,
    base_rate         REAL,
    threshold         REAL,
    n_features        INTEGER,
    artifact_path     TEXT
);

-- Phase 4 (model-selection 2026-04-26): challenger / shadow-mode log.
-- A row per (run_id, ticker, decision_date) when a challenger artifact
-- is enabled. Stores both the challenger's hypothetical decision and the
-- live runner's actual decision so `compare_window()` can compute
-- agreement / score correlation / disagreement-on-buy after a shadow
-- period. Wired into live runner / sim in Phase 4b — schema first so
-- the production DB is ready.
CREATE TABLE IF NOT EXISTS challenger_decisions (
    decision_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 TEXT    NOT NULL,
    decision_date          TEXT    NOT NULL,
    ticker                 TEXT    NOT NULL,
    challenger_name        TEXT    NOT NULL,
    challenger_score       REAL,
    challenger_rank_score  REAL,
    challenger_action      TEXT,
    actual_score           REAL,
    actual_action          TEXT,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_challenger_run    ON challenger_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_challenger_window ON challenger_decisions(challenger_name, decision_date);

-- Trade-evaluation table (roadmap §2026-04-26 Phase 1, shipped 2026-05-02).
-- Re-evaluates every trade at multiple horizons (1d, 5d, 7d, 14d, 28d).
-- Joining trades × ticker_forward_returns at horizon h gives the realized
-- forward return; comparing against SPY's same-horizon return gives the
-- benchmark-relative outcome. Populated by:
--   * scripts/backfill_trade_evaluations.py (nightly cron, Phase 2)
--   * record_trade_evaluations() helper for ad-hoc evaluation
-- Each (run_id, ticker, action, horizon_days) is unique — primary-key
-- guarantee prevents double-counting on backfill re-runs.
CREATE TABLE IF NOT EXISTS trade_evaluations (
    run_id           TEXT    NOT NULL,   -- FK to pipeline_runs.run_id (= trades.run_id)
    ticker           TEXT    NOT NULL,
    action           TEXT    NOT NULL,   -- 'buy' or 'sell'
    horizon_days     INTEGER NOT NULL,   -- 1, 5, 7, 14, 28 (or any positive int)
    fwd_return       REAL,                -- ticker's realized forward return at horizon
    fwd_return_spy   REAL,                -- SPY's forward return on same date+horizon
    relative_return  REAL,                -- fwd_return - fwd_return_spy (excess)
    is_winner        INTEGER,             -- 1 if relative_return > 0 else 0; NULL if missing
    n_trade_rows     INTEGER,             -- multiplicity in case of partial sells / top-ups
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, ticker, action, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_trade_eval_ticker  ON trade_evaluations(ticker);
CREATE INDEX IF NOT EXISTS idx_trade_eval_horizon ON trade_evaluations(horizon_days);
CREATE INDEX IF NOT EXISTS idx_trade_eval_run     ON trade_evaluations(run_id);
"""


# Idempotent column migrations for tables created before a column was added.
# SQLite's `CREATE TABLE IF NOT EXISTS` is a no-op on pre-existing tables, so
# any column added to _SCHEMA_SQL after first creation must also be listed here.
_COLUMN_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "pipeline_runs": [
        ("buy_blocked",   "INTEGER"),
        ("skip_buys",     "INTEGER"),
        ("bear_only",     "INTEGER"),
        ("counters_json", "TEXT"),
        ("run_bundle_json", "TEXT"),
    ],
    "training_runs": [
        ("elapsed_sec",           "REAL"),
        ("trigger",               "TEXT"),
        ("n_tickers",             "INTEGER"),
        ("n_dates",               "INTEGER"),
        ("n_features",            "INTEGER"),
        ("device",                "TEXT"),
        ("deterministic",         "INTEGER"),
        ("training_window_years", "REAL"),
        ("notes",                 "TEXT"),
    ],
    # Audit fix DB-DECISION-FACTORS (2026-04-26 round-5): migrate
    # existing candidate_scores tables to add the new factor columns.
    "candidate_scores": [
        ("expected_return",    "REAL"),
        ("expected_return_horizon_days", "INTEGER"),
        ("kelly_target_pct",   "REAL"),
        ("model_type",         "TEXT"),
        ("sector",             "TEXT"),
        ("panel_ltr_artifact", "TEXT"),
        ("mu_horizon_days",    "INTEGER"),
        ("qp_delta_w",         "REAL"),
        ("qp_target_w",        "REAL"),
        ("qp_status",          "TEXT"),
    ],
    "ticker_daily_state": [
        ("qp_delta_w",         "REAL"),
        ("qp_target_w",        "REAL"),
        ("qp_status",          "TEXT"),
        ("expected_return_horizon_days", "INTEGER"),
        ("mu_horizon_days",    "INTEGER"),
    ],
    "score_distribution": [
        ("run_type",                    "TEXT"),
        ("expected_return_horizon_days", "INTEGER"),
        ("mu_horizon_days",             "INTEGER"),
        ("model_type",                  "TEXT"),
        ("sector",                      "TEXT"),
        ("blocked_by",                  "TEXT"),
    ],
    "score_percentiles_daily": [
        ("run_type",                    "TEXT"),
    ],
    # 2026-05-22: persist the per-trade decision tree, not just scalar P&L.
    # Buy orders already carry order-attribution fields at emission time;
    # sell events carry exit diagnostics. These columns make the executed
    # trade table replayable without joining back to JSON logs.
    "trades": [
        ("trade_date",            "DATE"),
        ("order_type",            "TEXT"),
        ("source",                "TEXT"),
        ("source_job",            "TEXT"),
        ("source_task",           "TEXT"),
        ("order_source",          "TEXT"),
        ("attribution_version",   "TEXT"),
        ("score_snapshot_json",   "TEXT"),
        ("decision_inputs_json",  "TEXT"),
        ("gross_pnl",             "REAL"),
        ("proceeds_basis",        "REAL"),
        ("net_pnl_after_tax",     "REAL"),
        ("tax_cash_debited",      "REAL"),
        ("tax_cash_debit_mode",   "TEXT"),
        ("tax_lot_method",        "TEXT"),
        ("mu_horizon_days",       "INTEGER"),
        ("panel_score",           "REAL"),
        ("rs_score",              "REAL"),
        ("expected_return",       "REAL"),
        ("expected_return_horizon_days", "INTEGER"),
        ("kelly_target_pct",      "REAL"),
        ("model_type",            "TEXT"),
        ("sector",                "TEXT"),
        ("blocked_by",            "TEXT"),
        ("qp_delta_w",            "REAL"),
        ("qp_target_w",           "REAL"),
        ("qp_status",             "TEXT"),
        ("regime",                "TEXT"),
        ("confidence",            "REAL"),
    ],
    "ticker_forward_returns": [
        ("fwd_60d",               "REAL"),
    ],
}


def _pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in sorted((r for r in rows if int(r[5] or 0) > 0),
                                 key=lambda r: int(r[5]))]


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sell_economics_are_valid(
    gross_pnl: Any,
    tax: Any,
    net_pnl_after_tax: Any,
) -> bool:
    """Validate realized sell accounting invariants.

    The trace DB is an audit surface, so sell rows must satisfy basic
    accounting identities before downstream analysis trusts them:

    ``net_pnl_after_tax == gross_pnl - tax``; tax is finite/non-negative;
    losses do not carry positive tax; and tax cannot exceed positive gross
    P&L. Annual netting may lower the eventual tax bill, but it cannot make an
    event-level row internally inconsistent.
    """
    gross = _finite_float(gross_pnl)
    tax_v = _finite_float(tax)
    net = _finite_float(net_pnl_after_tax)
    if gross is None or tax_v is None or net is None:
        return False
    scale = max(abs(gross), abs(tax_v), abs(net), 1.0)
    tol = max(_ECON_ABS_TOL, _ECON_REL_TOL * scale)
    if tax_v < -tol:
        return False
    if gross <= tol and tax_v > tol:
        return False
    if gross > tol and tax_v - gross > tol:
        return False
    return math.isclose(net, gross - tax_v, rel_tol=_ECON_REL_TOL, abs_tol=tol)


def _legacy_run_id_expr(table: str) -> str:
    cols = {r[1] for r in table_info_cache.get(table, [])}
    if "run_id" in cols:
        return "COALESCE(run_id, 'legacy-' || date)"
    return "'legacy-' || date"


table_info_cache: dict[str, list] = {}


def _rebuild_ticker_daily_state_if_needed(conn: sqlite3.Connection) -> None:
    """Migrate date-keyed ticker_daily_state to append-only run scope.

    Pre-2026-05-21 schema keyed by (date, ticker), so a same-day e2e/gate
    replay overwrote the real daily decision tree. Rebuild the table when the
    primary key is not (run_id, ticker). Existing rows are preserved under
    synthetic run_id='legacy-{date}'.
    """
    if not _has_table(conn, "ticker_daily_state"):
        return
    if _pk_columns(conn, "ticker_daily_state") == ["run_id", "ticker"]:
        return

    tmp = f"ticker_daily_state__old_{uuid.uuid4().hex[:8]}"
    table_info_cache["ticker_daily_state"] = conn.execute(
        "PRAGMA table_info(ticker_daily_state)"
    ).fetchall()
    old_cols = {r[1] for r in table_info_cache["ticker_daily_state"]}
    def _old_col(name: str, default: str = "NULL") -> str:
        return name if name in old_cols else default

    run_expr = _legacy_run_id_expr("ticker_daily_state")
    conn.execute(f"ALTER TABLE ticker_daily_state RENAME TO {tmp}")
    conn.execute(
        """CREATE TABLE ticker_daily_state (
            run_id            TEXT NOT NULL,
            date              TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            regime            TEXT,
            confidence        REAL,
            in_watchlist      INTEGER,
            in_universe       INTEGER,
            pending_at_broker INTEGER,
            has_position      INTEGER,
            position_qty      REAL,
            position_pct      REAL,
            model_type        TEXT,
            model_action      TEXT,
            sell_streak       INTEGER,
            panel_score       REAL,
            rank_score        REAL,
            expected_return   REAL,
            expected_return_horizon_days INTEGER,
            kelly_target_pct  REAL,
            mu                REAL,
            mu_horizon_days    INTEGER,
            sigma             REAL,
            in_candidates     INTEGER,
            selected          INTEGER,
            blocked_by        TEXT,
            sector            TEXT,
            qp_delta_w        REAL,
            qp_target_w       REAL,
            qp_status         TEXT,
            PRIMARY KEY (run_id, ticker)
        )"""
    )
    conn.execute(
        f"""INSERT OR REPLACE INTO ticker_daily_state
              (run_id, date, ticker, regime, confidence,
               in_watchlist, in_universe, pending_at_broker,
               has_position, position_qty, position_pct,
               model_type, model_action, sell_streak,
               panel_score, rank_score, expected_return,
               expected_return_horizon_days, kelly_target_pct,
               mu, mu_horizon_days, sigma,
               in_candidates, selected, blocked_by, sector,
               qp_delta_w, qp_target_w, qp_status)
            SELECT {run_expr}, date, ticker, regime, confidence,
                   in_watchlist, in_universe, pending_at_broker,
                   has_position, position_qty, position_pct,
                   model_type, model_action, sell_streak,
                   panel_score, rank_score, expected_return,
                   {_old_col("expected_return_horizon_days")},
                   kelly_target_pct,
                   mu, {_old_col("mu_horizon_days")}, sigma,
                   in_candidates, selected, blocked_by, sector,
                   {_old_col("qp_delta_w")}, {_old_col("qp_target_w")},
                   {_old_col("qp_status")}
              FROM {tmp}"""
    )
    conn.execute(f"DROP TABLE {tmp}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tds_date ON ticker_daily_state(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tds_ticker ON ticker_daily_state(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tds_run ON ticker_daily_state(run_id)")


def _rebuild_score_tables_if_needed(conn: sqlite3.Connection) -> None:
    """Migrate score distribution tables from date scope to run scope."""
    if _has_table(conn, "score_distribution") and _pk_columns(conn, "score_distribution") != ["run_id", "ticker"]:
        tmp = f"score_distribution__old_{uuid.uuid4().hex[:8]}"
        table_info_cache["score_distribution"] = conn.execute(
            "PRAGMA table_info(score_distribution)"
        ).fetchall()
        old_cols = {r[1] for r in table_info_cache["score_distribution"]}
        def _old_col(name: str, default: str = "NULL") -> str:
            return name if name in old_cols else default
        run_expr = _legacy_run_id_expr("score_distribution")
        conn.execute(f"ALTER TABLE score_distribution RENAME TO {tmp}")
        conn.execute(
            """CREATE TABLE score_distribution (
                run_id        TEXT NOT NULL,
                date          TEXT NOT NULL,
                run_type      TEXT,
                ticker        TEXT NOT NULL,
                raw_panel     REAL,
                rank_score    REAL,
                expected_return_horizon_days INTEGER,
                mu            REAL,
                mu_horizon_days INTEGER,
                sigma         REAL,
                regime        TEXT,
                is_holding    INTEGER DEFAULT 0,
                model_type    TEXT,
                sector        TEXT,
                blocked_by    TEXT,
                PRIMARY KEY (run_id, ticker)
            )"""
        )
        conn.execute(
            f"""INSERT OR REPLACE INTO score_distribution
                  (run_id, date, run_type, ticker, raw_panel, rank_score,
                   expected_return_horizon_days, mu, mu_horizon_days, sigma,
                   regime, is_holding, model_type, sector, blocked_by)
                SELECT {run_expr}, date, {_old_col("run_type")}, ticker,
                       raw_panel, rank_score,
                       {_old_col("expected_return_horizon_days")},
                       mu, {_old_col("mu_horizon_days")}, sigma, regime,
                       is_holding, {_old_col("model_type")},
                       {_old_col("sector")}, {_old_col("blocked_by")}
                  FROM {tmp}"""
        )
        conn.execute(f"DROP TABLE {tmp}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score_dist_date ON score_distribution(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score_dist_run ON score_distribution(run_id)")

    if _has_table(conn, "score_percentiles_daily") and _pk_columns(conn, "score_percentiles_daily") != ["run_id"]:
        tmp = f"score_percentiles_daily__old_{uuid.uuid4().hex[:8]}"
        table_info_cache["score_percentiles_daily"] = conn.execute(
            "PRAGMA table_info(score_percentiles_daily)"
        ).fetchall()
        old_cols = {r[1] for r in table_info_cache["score_percentiles_daily"]}
        def _old_col(name: str, default: str = "NULL") -> str:
            return name if name in old_cols else default
        run_expr = _legacy_run_id_expr("score_percentiles_daily")
        conn.execute(f"ALTER TABLE score_percentiles_daily RENAME TO {tmp}")
        conn.execute(
            """CREATE TABLE score_percentiles_daily (
                run_id        TEXT PRIMARY KEY,
                date          TEXT NOT NULL,
                run_type      TEXT,
                n_cands       INTEGER NOT NULL,
                p01           REAL,
                p05           REAL,
                p10           REAL,
                p25           REAL,
                p50           REAL,
                p75           REAL,
                p85           REAL,
                p90           REAL,
                p95           REAL,
                p99           REAL,
                score_min     REAL,
                score_max     REAL,
                score_mean    REAL,
                score_std     REAL,
                regime        TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            f"""INSERT OR REPLACE INTO score_percentiles_daily
                  (run_id, date, run_type, n_cands, p01, p05, p10, p25, p50, p75,
                   p85, p90, p95, p99, score_min, score_max, score_mean,
                   score_std, regime)
                SELECT {run_expr}, date, {_old_col("run_type")},
                       n_cands, p01, p05, p10, p25, p50, p75,
                       p85, p90, p95, p99, score_min, score_max, score_mean,
                       score_std, regime
                  FROM {tmp}"""
        )
        conn.execute(f"DROP TABLE {tmp}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pctiles_date ON score_percentiles_daily(date)")


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue   # CREATE TABLE IF NOT EXISTS just handled the fresh case
        for name, typ in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    _apply_column_migrations(conn)
    _rebuild_ticker_daily_state_if_needed(conn)
    _rebuild_score_tables_if_needed(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tds_run ON ticker_daily_state(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_score_dist_run ON score_distribution(run_id)")
    conn.commit()


# ── Connection management ─────────────────────────────────────────────────────

def _is_enabled(config: dict) -> bool:
    return bool(config.get("persistence", {}).get("enabled", False))


def _db_path(
    config:       dict,
    strategy_dir: Path | None = None,
    role:         str = "live",
) -> Path:
    """Resolve the SQLite file path for this adapter role.

    Roles (user-driven architecture 2026-04-24):
      * ``"live"``  — permanent production data (live runner + LEAN).
                       Path: ``persistence.db_path`` (default ``data/runs.db``).
      * ``"sim"``   — ephemeral notebook sim data. TRUNCATEd at start
                       of every ``run_backtest()`` via ``clear_sim_tables()``,
                       so the 100th sim of the day is the only one that
                       remains.
                       Path: ``persistence.sim_db_path`` (default
                       ``data/sim_runs.db``).

    The split prevents notebook experimentation from polluting live
    decision-audit statistics: AA analysis defaults to reading the
    live DB as the source-of-truth.
    """
    persistence = config.get("persistence", {})
    if role == "sim":
        raw = persistence.get("sim_db_path", "data/sim_runs.db")
    else:
        raw = persistence.get("db_path", "data/runs.db")
    p = Path(raw)
    if not p.is_absolute():
        if strategy_dir is not None:
            # Resolve relative to repo root: strategy_dir = backtesting/renquant_104 → .../../
            repo_root = Path(strategy_dir).parent.parent
            p = repo_root / p
        else:
            p = Path.cwd() / p
    return p


def get_connection(
    config:       dict,
    strategy_dir: Path | None = None,
    *,
    role:         str = "live",
) -> sqlite3.Connection | None:
    """Open (or create) the SQLite DB configured in config. Returns None when disabled.

    See :func:`_db_path` for the live-vs-sim role semantics.
    """
    if not _is_enabled(config):
        return None
    path = _db_path(config, strategy_dir, role=role)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)   # autocommit
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    ensure_schema(conn)
    return conn


# Tables that carry per-run decision state — wiped at start of each
# notebook sim via `clear_sim_tables`. Forward returns and training
# audit are DERIVED data (historical prices, retrain metadata) and
# survive the reset.
_SIM_RESET_TABLES = [
    "candidate_scores",
    "score_distribution",
    "score_percentiles_daily",
    "ticker_daily_state",
    "trades",
    "rotations",
    "live_state_snapshots",
    "pipeline_runs",   # last — has FKs into the other tables above
]


def clear_sim_tables(conn: sqlite3.Connection | None) -> int:
    """TRUNCATE the decision-trace tables on a sim DB.

    Called from :func:`sim.runner.run_backtest` before a fresh notebook
    sim populates its rows. Leaves derived tables (`ticker_forward_returns`,
    `training_runs`) intact — they're reused across sim sessions.

    Returns the total number of rows deleted.
    """
    if conn is None:
        return 0
    deleted = 0
    for table in _SIM_RESET_TABLES:
        cur = conn.execute(f"DELETE FROM {table}")
        deleted += cur.rowcount
    conn.commit()
    return deleted


# ── Commit SHA helper (for provenance) ────────────────────────────────────────

# Round-3 audit (#R3-82 #R3-52): cache once per process so a 700-bar sim
# doesn't fork git 700 times. Resolved at first call; survives the
# process. (If the user rewrites history mid-sim, the cached SHA is
# slightly stale — acceptable.)
_COMMIT_SHA_RESOLVED: bool = False
_COMMIT_SHA_VALUE: "str | None" = None


def _commit_sha() -> str | None:
    global _COMMIT_SHA_RESOLVED, _COMMIT_SHA_VALUE
    if _COMMIT_SHA_RESOLVED:
        return _COMMIT_SHA_VALUE
    try:
        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        _COMMIT_SHA_VALUE = sha or None
    except Exception:
        _COMMIT_SHA_VALUE = None
    _COMMIT_SHA_RESOLVED = True
    return _COMMIT_SHA_VALUE


def _default_training_jsonl_dir(conn: sqlite3.Connection | None) -> Path:
    """Resolve the plain-text training audit path from the SQLite DB path.

    Production writes to ``data/runs.db`` and should mirror to the repo-level
    ``logs/training`` directory. Temporary/test DBs should keep JSONL output
    beside the temp DB, so test runs cannot pollute the operator audit stream.
    """
    if conn is None:
        return Path("logs/training")
    try:
        db_rows = conn.execute("PRAGMA database_list").fetchall()
        main_path = next((r[2] for r in db_rows if r[1] == "main"), "")
    except Exception:
        main_path = ""
    if not main_path:
        return Path("logs/training")
    db_path = Path(main_path)
    root = db_path.parent.parent if db_path.parent.name == "data" else db_path.parent
    return root / "logs" / "training"


# ── Recording helpers ─────────────────────────────────────────────────────────

def record_pipeline_run(
    conn: sqlite3.Connection | None,
    *,
    run_type: str,                      # "lean" | "live" | "sim"
    run_date: datetime.date,
    strategy: str = "",
    regime: str | None = None,
    confidence: float | None = None,
    portfolio_value: float | None = None,
    cash: float | None = None,
    n_candidates: int = 0,
    n_exits: int = 0,
    n_rotations: int = 0,
    n_buys: int = 0,
    buy_blocked: bool | None = None,
    skip_buys: bool | None = None,
    bear_only: bool | None = None,
    counters: dict[str, Any] | None = None,
    run_bundle: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str | None:
    """Insert a pipeline_runs row and return the generated run_id."""
    if conn is None:
        return None
    run_id = run_id or f"{run_date.isoformat()}-{run_type}-{uuid.uuid4().hex[:8]}"
    conn.execute(
        """INSERT OR REPLACE INTO pipeline_runs
              (run_id, run_date, run_type, strategy, regime, confidence,
               portfolio_value, cash, n_candidates, n_exits, n_rotations, n_buys,
               buy_blocked, skip_buys, bear_only, counters_json, run_bundle_json,
               commit_sha)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, run_date.isoformat(), run_type, strategy, regime, confidence,
         portfolio_value, cash, n_candidates, n_exits, n_rotations, n_buys,
         None if buy_blocked is None else int(bool(buy_blocked)),
         None if skip_buys is None else int(bool(skip_buys)),
         None if bear_only is None else int(bool(bear_only)),
         json.dumps(counters or {}, sort_keys=True, default=str),
         json.dumps(run_bundle or {}, sort_keys=True, default=str),
         _commit_sha()),
    )
    return run_id


def record_candidate_scores(
    conn: sqlite3.Connection | None,
    run_id: str | None,
    candidates: Iterable[Any],
    holdings: dict[str, Any],
    selected_tickers: set[str],
    blocked_map: dict[str, str] | None = None,
    *,
    sector_map:    dict[str, str] | None = None,
    model_types:   dict[str, str] | None = None,
    panel_artifact: str | None = None,
    qp_delta_by_ticker: dict[str, float] | None = None,
    qp_target_by_ticker: dict[str, float] | None = None,
    qp_status: str | None = None,
    excluded_holding_tickers: set[str] | None = None,
) -> None:
    """Insert one row per candidate + one per holding.

    `candidates`:  iterable of CandidateResult-like objects (must have
                   .ticker, .raw_score, .rank_score, .rs_score, .panel_score,
                   .mu, .sigma, and optional horizon-day fields)
    `holdings`:    dict of ticker → HoldingState (only rank_score / panel_score /
                   mu / sigma attributes are read; other fields ignored)
    `selected_tickers`: set of candidate tickers that ended up in orders this run
    `blocked_map`: optional dict of ticker → reason ("sector_cap", "correlation",
                   "wash_sale", "below_threshold", etc.)
    """
    if conn is None or run_id is None:
        return
    blocked_map = blocked_map or {}
    rows = []
    # Audit fix PR-1/PR-2 (Round 9, 2026-04-25): pre-fix, raw_score /
    # rank_score / rs_score went through `float(... or 0.0)` which
    # preserved NaN (Python `bool(NaN) = True`, so `NaN or 0.0` = NaN),
    # while panel_score/mu/sigma already used `_none_or_float`. NaN
    # raw_scores got persisted into a numeric DB column while NaN
    # μ/σ was stored as NULL — inconsistent, and analytics queries
    # (median, percentile) silently broke on the rows with NaN raw_score.
    # Now: every numeric score uses `_none_or_float` which returns None
    # on missing/NaN/inf, persisting as SQL NULL. Analytics that want a
    # display default must COALESCE explicitly so missing evidence is never
    # confused with a real zero score.
    sector_map = sector_map or {}
    model_types = model_types or {}
    qp_delta_by_ticker = qp_delta_by_ticker or {}
    qp_target_by_ticker = qp_target_by_ticker or {}
    excluded_holding_keys = {str(t).upper() for t in (excluded_holding_tickers or set())}

    def _sector_for(ticker: str) -> str | None:
        value = sector_map.get(ticker)
        if isinstance(value, str) and value:
            return value
        value = sector_map.get(str(ticker).upper())
        return value if isinstance(value, str) and value else None

    for c in candidates:
        selected = c.ticker in selected_tickers
        rows.append((
            run_id, c.ticker, "candidate",
            _none_or_float(getattr(c, "raw_score",  None)),
            _none_or_float(getattr(c, "rank_score", None)),
            _none_or_float(getattr(c, "panel_score", None)),
            _none_or_float(getattr(c, "rs_score",   None)),
            _none_or_float(getattr(c, "mu",    None)),
            _none_or_float(getattr(c, "sigma", None)),
            1 if selected else 0,
            None if selected else blocked_map.get(c.ticker, "candidate_not_selected"),
            # New decision-factor columns
            _none_or_float(getattr(c, "expected_return", None)),
            _none_or_int(getattr(c, "expected_return_horizon_days", None)),
            _none_or_float(getattr(c, "kelly_target_pct", None)),
            model_types.get(c.ticker),
            _sector_for(c.ticker),
            panel_artifact,
            _none_or_int(getattr(c, "mu_horizon_days", None)),
            _none_or_float(qp_delta_by_ticker.get(c.ticker)),
            _none_or_float(qp_target_by_ticker.get(c.ticker)),
            qp_status,
        ))
    for ticker, hs in holdings.items():
        if str(ticker).upper() in excluded_holding_keys:
            continue
        rows.append((
            run_id, ticker, "holding",
            None,
            _none_or_float(getattr(hs, "rank_score", None)),
            _none_or_float(getattr(hs, "panel_score", None)),
            None,
            _none_or_float(getattr(hs, "mu",    None)),
            _none_or_float(getattr(hs, "sigma", None)),
            0,
            None,
            _none_or_float(getattr(hs, "expected_return", None)),
            _none_or_int(getattr(hs, "expected_return_horizon_days", None)),
            _none_or_float(getattr(hs, "kelly_target_pct", None)),
            model_types.get(ticker),
            _sector_for(ticker),
            panel_artifact,
            _none_or_int(getattr(hs, "mu_horizon_days", None)),
            _none_or_float(qp_delta_by_ticker.get(ticker)),
            _none_or_float(qp_target_by_ticker.get(ticker)),
            qp_status,
        ))
    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO candidate_scores
                  (run_id, ticker, role, raw_score, rank_score, panel_score, rs_score,
                   mu, sigma, selected, blocked_by,
                   expected_return, expected_return_horizon_days,
                   kelly_target_pct, model_type, sector,
                   panel_ltr_artifact, mu_horizon_days,
                   qp_delta_w, qp_target_w, qp_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def record_trades(
    conn: sqlite3.Connection | None,
    run_id: str | None,
    trade_events: Iterable[dict],
) -> None:
    """Insert rows into trades from a list of trade dicts.

    Expected keys (all optional except ticker + action):
      ticker, action ('buy'|'sell'), shares, price, invest, target_pct,
      exit_reason, pnl_pct, hold_days, tax, gross_pnl, proceeds_basis,
      net_pnl_after_tax, tax_cash_debited, tax_cash_debit_mode,
      tax_lot_method, rank_score, conviction, sigma_mult, mu,
      mu_horizon_days, sigma, panel_score, rs_score, expected_return,
      expected_return_horizon_days, kelly_target_pct, model_type, sector,
      blocked_by, qp_delta_w, qp_target_w, qp_status, regime, confidence,
      order_type/source/source_job/source_task/
      order_source/attribution_version, score_snapshot, decision_inputs.
    """
    if conn is None or run_id is None:
        return

    def _json_or_none(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, default=str)

    def _score_snapshot_or_none(t: dict) -> dict[str, Any] | None:
        snap = t.get("score_snapshot")
        if isinstance(snap, dict):
            return snap
        keys = (
            "rank_score", "panel_score", "rs_score", "mu", "sigma",
            "mu_horizon_days", "kelly_target_pct", "expected_return",
            "expected_return_horizon_days", "confidence", "regime",
            "model_type", "sector", "blocked_by",
        )
        fallback = {k: t.get(k) for k in keys if k in t and t.get(k) is not None}
        if fallback:
            return fallback
        if t.get("ticker") or t.get("action"):
            return {
                "attribution_missing": True,
                "ticker": t.get("ticker"),
                "action": t.get("action"),
            }
        return None

    def _decision_inputs_or_none(t: dict) -> dict[str, Any] | None:
        raw = t.get("decision_inputs")
        if isinstance(raw, dict) and raw:
            return raw
        fallback = {
            k: t.get(k)
            for k in (
                "ticker", "action", "order_type", "order_source",
                "source_job", "source_task", "exit_reason", "target_pct",
                "shares", "price", "invest",
            )
            if t.get(k) is not None
        }
        if fallback:
            fallback.setdefault(
                "acceptance_reason",
                t.get("exit_reason")
                or t.get("order_source")
                or t.get("order_type")
                or "recorded_trade",
            )
            if raw == {}:
                fallback["attribution_missing"] = True
            return fallback
        return None

    rows = []
    for t in trade_events:
        rows.append((
            run_id,
            str(t.get("date") or t.get("trade_date") or "") or None,
            t.get("ticker"),
            t.get("action"),
            _none_or_float(t.get("shares")),
            _none_or_float(t.get("price")),
            _none_or_float(t.get("invest")),
            _none_or_float(t.get("target_pct")),
            t.get("exit_reason"),
            _none_or_float(t.get("pnl_pct")),
            _none_or_int(t.get("hold_days")),
            _none_or_float(t.get("tax")),
            _none_or_float(t.get("gross_pnl")),
            _none_or_float(t.get("proceeds_basis")),
            _none_or_float(t.get("net_pnl_after_tax")),
            _none_or_float(t.get("tax_cash_debited")),
            t.get("tax_cash_debit_mode"),
            t.get("tax_lot_method"),
            _none_or_float(t.get("rank_score")),
            _none_or_float(t.get("conviction")),
            _none_or_float(t.get("sigma_mult")),
            _none_or_float(t.get("mu")),
            _none_or_int(t.get("mu_horizon_days")),
            _none_or_float(t.get("sigma")),
            _none_or_float(t.get("panel_score")),
            _none_or_float(t.get("rs_score")),
            _none_or_float(t.get("expected_return")),
            _none_or_int(t.get("expected_return_horizon_days")),
            _none_or_float(t.get("kelly_target_pct")),
            t.get("model_type"),
            t.get("sector"),
            t.get("blocked_by"),
            _none_or_float(t.get("qp_delta_w")),
            _none_or_float(t.get("qp_target_w")),
            t.get("qp_status"),
            t.get("regime"),
            _none_or_float(t.get("confidence")),
            t.get("order_type"),
            t.get("source"),
            t.get("source_job"),
            t.get("source_task"),
            t.get("order_source"),
            t.get("attribution_version"),
            _json_or_none(_score_snapshot_or_none(t)),
            _json_or_none(_decision_inputs_or_none(t)),
        ))
    if rows:
        conn.executemany(
            """INSERT INTO trades
                  (run_id, trade_date, ticker, action, shares, price, invest, target_pct,
                   exit_reason, pnl_pct, hold_days, tax,
                   gross_pnl, proceeds_basis, net_pnl_after_tax,
                   tax_cash_debited, tax_cash_debit_mode, tax_lot_method,
                   rank_score, conviction, sigma_mult, mu, mu_horizon_days, sigma,
                   panel_score, rs_score, expected_return,
                   expected_return_horizon_days, kelly_target_pct,
                   model_type, sector, blocked_by,
                   qp_delta_w, qp_target_w, qp_status,
                   regime, confidence,
                   order_type, source, source_job, source_task, order_source,
                   attribution_version, score_snapshot_json, decision_inputs_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def record_training_run(
    conn: sqlite3.Connection | None,
    *,
    run_date: datetime.datetime | None = None,
    strategy: str = "",
    artifact_type: str = "",              # 'panel-ltr' | 'ngboost-head' | 'tournament' | 'panel-transformer'
    config_snapshot: dict | None = None,
    oos_mean_ic: float | None = None,
    train_ic: float | None = None,
    n_rows: int | None = None,
    feature_cols: list[str] | None = None,
    artifact_path: str | None = None,
    # Round 5 additions
    elapsed_sec: float | None = None,
    trigger: str | None = None,
    n_tickers: int | None = None,
    n_dates: int | None = None,
    n_features: int | None = None,
    device: str | None = None,
    deterministic: bool | None = None,
    training_window_years: float | None = None,
    notes: str | None = None,
    also_log_jsonl: bool = True,
    jsonl_dir: Path | None = None,
) -> str | None:
    """Record a training run to SQLite and (by default) append a line to
    `logs/training/{YYYY-MM-DD}.jsonl` mirroring the same fields.

    The JSONL log exists for operators who want to grep training history
    without opening the SQLite DB — a symmetric plain-text audit trail.
    """
    rd = run_date or datetime.datetime.utcnow()
    run_id = f"{rd.strftime('%Y%m%d%H%M%S')}-{artifact_type}-{uuid.uuid4().hex[:6]}"

    # Audit #71: subprocess to git was being invoked twice (row dict + SQL
    # VALUES). Resolve once and reuse.
    sha = _commit_sha()

    row = {
        "run_id":                run_id,
        "run_date":              rd.isoformat(),
        "strategy":              strategy,
        "artifact_type":         artifact_type,
        "oos_mean_ic":           oos_mean_ic,
        "train_ic":              train_ic,
        "n_rows":                n_rows,
        "n_tickers":             n_tickers,
        "n_dates":               n_dates,
        "n_features":            n_features,
        "elapsed_sec":           elapsed_sec,
        "trigger":               trigger,
        "device":                device,
        "deterministic":         deterministic,
        "training_window_years": training_window_years,
        "artifact_path":         artifact_path,
        "commit_sha":            sha,
        "notes":                 notes,
    }

    if conn is not None:
        conn.execute(
            """INSERT INTO training_runs
                  (run_id, run_date, strategy, artifact_type, config_json,
                   oos_mean_ic, train_ic, n_rows, feature_cols, artifact_path,
                   commit_sha, elapsed_sec, trigger, n_tickers, n_dates,
                   n_features, device, deterministic, training_window_years,
                   notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, rd.isoformat(), strategy, artifact_type,
             json.dumps(config_snapshot, default=str) if config_snapshot else None,
             oos_mean_ic, train_ic, n_rows,
             json.dumps(feature_cols) if feature_cols is not None else None,
             artifact_path, sha,
             elapsed_sec, trigger, n_tickers, n_dates, n_features, device,
             int(deterministic) if deterministic is not None else None,
             training_window_years, notes),
        )
        conn.commit()

    # JSONL log (operator-friendly audit trail)
    if also_log_jsonl:
        log_dir = jsonl_dir or _default_training_jsonl_dir(conn)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{rd.strftime('%Y-%m-%d')}.jsonl"
        with log_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    return run_id


def record_live_state_snapshot(
    conn: sqlite3.Connection | None,
    run_id: str | None,
    *,
    run_date: datetime.date,
    strategy: str = "",
    state: dict | None = None,
    cash: float | None = None,
    portfolio_value: float | None = None,
    n_holdings: int | None = None,
) -> None:
    """Append one row to live_state_snapshots (Plan S).

    `state` is the full dict serialised to `live_state.json`. Common
    query fields (regime / confidence / high_water_mark) are denormalized
    into columns; the full blob is stored as JSON for later introspection.
    """
    if conn is None or run_id is None:
        return
    state = state or {}
    conn.execute(
        """INSERT OR REPLACE INTO live_state_snapshots
              (run_id, run_date, strategy, regime, confidence,
               high_water_mark, cash, portfolio_value, n_holdings, state_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            run_date.isoformat(),
            strategy,
            state.get("regime"),
            _none_or_float(state.get("regime_confidence")),
            _none_or_float(state.get("high_water_mark")),
            _none_or_float(cash),
            _none_or_float(portfolio_value),
            _none_or_int(n_holdings),
            json.dumps(state, default=str),
        ),
    )


def load_latest_live_state(
    conn: sqlite3.Connection | None,
    *,
    strategy: str = "",
    max_age_days: int | None = None,
) -> dict | None:
    """Load the most recent live_state_snapshots row as a dict (#144).

    Returns the JSON-decoded `state_json` blob (suitable for writing
    back to live_state.json) or None if no snapshot exists / db is
    missing / snapshot is older than `max_age_days`.

    Per user spec 2026-04-26: "live state json应该至少备份在db里" —
    db is now the canonical store; live_state.json is a fast cache.
    On startup, if the JSON file is missing or stale, the runner
    falls back to db via this function.

    Args:
        conn: open sqlite connection (may be None — degrades to None return)
        strategy: filter by strategy name (e.g. "renquant_104"); empty
            means any
        max_age_days: if set, return None when snapshot is older than
            this many days (defensive — prevents resurrecting ancient
            state). None = no age check.

    Returns:
        dict from state_json column, OR None.
    """
    if conn is None:
        return None
    try:
        if strategy:
            row = conn.execute(
                """SELECT state_json, run_date FROM live_state_snapshots
                   WHERE strategy = ? ORDER BY run_date DESC, created_at DESC
                   LIMIT 1""",
                (strategy,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT state_json, run_date FROM live_state_snapshots
                   ORDER BY run_date DESC, created_at DESC LIMIT 1""",
            ).fetchone()
    except sqlite3.OperationalError:
        # Table missing (fresh db) → no snapshot to load.
        return None
    if row is None:
        return None
    state_json, run_date_str = row
    if max_age_days is not None and run_date_str:
        try:
            run_date = datetime.date.fromisoformat(str(run_date_str))
            age = (datetime.date.today() - run_date).days
            if age > max_age_days:
                return None
        except (ValueError, TypeError):
            pass   # bad date format → fail open (return what we have)
    try:
        return json.loads(state_json)
    except (json.JSONDecodeError, TypeError):
        return None


def record_portfolio_metrics(
    conn: sqlite3.Connection | None,
    rows: Iterable[dict],
) -> int:
    """Upsert portfolio_daily_metrics rows (APY=1.41/Sharpe=2 goal tracker).

    Each row: `{as_of_date, run_type, strategy, portfolio_value, daily_return,
    sharpe_21d, sharpe_63d, sharpe_252d, realized_vol_21d, realized_vol_252d,
    max_drawdown_252d, var_95_21d, var_99_21d, beta_spy_252d}`.
    """
    if conn is None:
        return 0
    payload = []
    for r in rows:
        payload.append((
            r["as_of_date"] if isinstance(r["as_of_date"], str)
            else r["as_of_date"].isoformat(),
            r.get("run_type", "sim"),
            r.get("strategy", ""),
            _none_or_float(r.get("portfolio_value")),
            _none_or_float(r.get("daily_return")),
            _none_or_float(r.get("sharpe_21d")),
            _none_or_float(r.get("sharpe_63d")),
            _none_or_float(r.get("sharpe_252d")),
            _none_or_float(r.get("realized_vol_21d")),
            _none_or_float(r.get("realized_vol_252d")),
            _none_or_float(r.get("max_drawdown_252d")),
            _none_or_float(r.get("var_95_21d")),
            _none_or_float(r.get("var_99_21d")),
            _none_or_float(r.get("beta_spy_252d")),
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT INTO portfolio_daily_metrics
              (as_of_date, run_type, strategy, portfolio_value, daily_return,
               sharpe_21d, sharpe_63d, sharpe_252d,
               realized_vol_21d, realized_vol_252d, max_drawdown_252d,
               var_95_21d, var_99_21d, beta_spy_252d)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(as_of_date, run_type, strategy) DO UPDATE SET
              portfolio_value    = COALESCE(excluded.portfolio_value,    portfolio_value),
              daily_return       = COALESCE(excluded.daily_return,       daily_return),
              sharpe_21d         = COALESCE(excluded.sharpe_21d,         sharpe_21d),
              sharpe_63d         = COALESCE(excluded.sharpe_63d,         sharpe_63d),
              sharpe_252d        = COALESCE(excluded.sharpe_252d,        sharpe_252d),
              realized_vol_21d   = COALESCE(excluded.realized_vol_21d,   realized_vol_21d),
              realized_vol_252d  = COALESCE(excluded.realized_vol_252d,  realized_vol_252d),
              max_drawdown_252d  = COALESCE(excluded.max_drawdown_252d,  max_drawdown_252d),
              var_95_21d         = COALESCE(excluded.var_95_21d,         var_95_21d),
              var_99_21d         = COALESCE(excluded.var_99_21d,         var_99_21d),
              beta_spy_252d      = COALESCE(excluded.beta_spy_252d,      beta_spy_252d),
              computed_at        = CURRENT_TIMESTAMP""",
        payload,
    )
    return len(payload)


def record_ticker_daily_state(
    conn: sqlite3.Connection | None,
    *,
    run_date: datetime.date,
    rows: Iterable[dict],
    run_id: str | None = None,
) -> int:
    """Upsert ticker_daily_state rows — one per watchlist ticker per bar.

    Per user spec round-5 (2026-04-26): every watchlist ticker gets a row
    every bar, including those filtered by universe floor / broker
    pre-check / no-model-signal — so post-hoc analysis can answer "what
    did we KNOW about XYZ on this date and WHY didn't we trade it?".

    Each row dict supports: regime, confidence, in_watchlist, in_universe,
    pending_at_broker, has_position, position_qty, position_pct,
    model_type, model_action, sell_streak, panel_score, rank_score,
    expected_return, expected_return_horizon_days, kelly_target_pct,
    mu, mu_horizon_days, sigma, in_candidates, selected, blocked_by,
    sector, qp_delta_w, qp_target_w, qp_status.
    `ticker` required.
    """
    if conn is None:
        return 0
    payload = []
    rd_str = run_date.isoformat() if hasattr(run_date, "isoformat") else str(run_date)
    rid = run_id or f"{rd_str}-unscoped"
    for r in rows:
        if not r.get("ticker"):
            continue
        selected = bool(_none_or_int(r.get("selected")))
        payload.append((
            rid,
            rd_str,
            r["ticker"],
            r.get("regime"),
            _none_or_float(r.get("confidence")),
            _none_or_int(r.get("in_watchlist")),
            _none_or_int(r.get("in_universe")),
            _none_or_int(r.get("pending_at_broker")),
            _none_or_int(r.get("has_position")),
            _none_or_float(r.get("position_qty")),
            _none_or_float(r.get("position_pct")),
            r.get("model_type"),
            r.get("model_action"),
            _none_or_int(r.get("sell_streak")),
            _none_or_float(r.get("panel_score")),
            _none_or_float(r.get("rank_score")),
            _none_or_float(r.get("expected_return")),
            _none_or_int(r.get("expected_return_horizon_days")),
            _none_or_float(r.get("kelly_target_pct")),
            _none_or_float(r.get("mu")),
            _none_or_int(r.get("mu_horizon_days")),
            _none_or_float(r.get("sigma")),
            _none_or_int(r.get("in_candidates")),
            _none_or_int(r.get("selected")),
            None if selected else r.get("blocked_by"),
            r.get("sector"),
            _none_or_float(r.get("qp_delta_w")),
            _none_or_float(r.get("qp_target_w")),
            r.get("qp_status"),
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT OR REPLACE INTO ticker_daily_state
              (run_id, date, ticker, regime, confidence,
               in_watchlist, in_universe, pending_at_broker,
               has_position, position_qty, position_pct,
               model_type, model_action, sell_streak,
               panel_score, rank_score, expected_return,
               expected_return_horizon_days, kelly_target_pct,
               mu, mu_horizon_days, sigma,
               in_candidates, selected, blocked_by, sector,
               qp_delta_w, qp_target_w, qp_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        payload,
    )
    return len(payload)


def decision_trace_integrity_report(
    conn: sqlite3.Connection | None,
    run_id: str,
    *,
    expected_watchlist: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return per-run decision-trace invariant counts for audit gates."""
    if conn is None:
        return {}
    expected = {str(t) for t in (expected_watchlist or [])}
    rows = conn.execute(
        "SELECT ticker FROM ticker_daily_state WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    recorded = {str(r[0]) for r in rows}
    selected_blockers = conn.execute(
        """SELECT COUNT(*) FROM ticker_daily_state
           WHERE run_id = ? AND selected = 1 AND blocked_by IS NOT NULL""",
        (run_id,),
    ).fetchone()[0]
    candidate_selected_blockers = conn.execute(
        """SELECT COUNT(*) FROM candidate_scores
           WHERE run_id = ? AND selected = 1 AND blocked_by IS NOT NULL""",
        (run_id,),
    ).fetchone()[0]
    candidate_reason_gaps = conn.execute(
        """SELECT COUNT(*) FROM candidate_scores
           WHERE run_id = ?
             AND role = 'candidate'
             AND COALESCE(selected, 0) = 0
             AND blocked_by IS NULL""",
        (run_id,),
    ).fetchone()[0]
    decision_reason_gaps = conn.execute(
        """SELECT COUNT(*) FROM ticker_daily_state
           WHERE run_id = ?
             AND COALESCE(selected, 0) = 0
             AND blocked_by IS NULL""",
        (run_id,),
    ).fetchone()[0]
    trade_payload_gaps = conn.execute(
        """SELECT COUNT(*) FROM trades
           WHERE run_id = ?
             AND (
               score_snapshot_json IS NULL
               OR decision_inputs_json IS NULL
               OR score_snapshot_json IN ('{}', 'null')
               OR decision_inputs_json IN ('{}', 'null')
             )""",
        (run_id,),
    ).fetchone()[0]
    model_type_gaps = conn.execute(
        """SELECT COUNT(*) FROM ticker_daily_state
           WHERE run_id = ?
             AND COALESCE(in_universe, 0) = 1
             AND model_type IS NULL""",
        (run_id,),
    ).fetchone()[0]
    selected_sector_gaps = conn.execute(
        """SELECT COUNT(*) FROM ticker_daily_state
           WHERE run_id = ?
             AND COALESCE(selected, 0) = 1
             AND (sector IS NULL OR TRIM(sector) = '')""",
        (run_id,),
    ).fetchone()[0]
    candidate_selected_sector_gaps = conn.execute(
        """SELECT COUNT(*) FROM candidate_scores
           WHERE run_id = ?
             AND role = 'candidate'
             AND COALESCE(selected, 0) = 1
             AND (sector IS NULL OR TRIM(sector) = '')""",
        (run_id,),
    ).fetchone()[0]
    candidate_horizon_gaps = conn.execute(
        """SELECT COUNT(*) FROM candidate_scores
           WHERE run_id = ?
             AND (
               (expected_return IS NOT NULL AND expected_return_horizon_days IS NULL)
               OR (mu IS NOT NULL AND mu_horizon_days IS NULL)
             )""",
        (run_id,),
    ).fetchone()[0]
    decision_horizon_gaps = conn.execute(
        """SELECT COUNT(*) FROM ticker_daily_state
           WHERE run_id = ?
             AND (
               (expected_return IS NOT NULL AND expected_return_horizon_days IS NULL)
               OR (mu IS NOT NULL AND mu_horizon_days IS NULL)
             )""",
        (run_id,),
    ).fetchone()[0]
    trade_horizon_gaps = conn.execute(
        """SELECT COUNT(*) FROM trades
           WHERE run_id = ?
             AND (
               (expected_return IS NOT NULL AND expected_return_horizon_days IS NULL)
               OR (mu IS NOT NULL AND mu_horizon_days IS NULL)
             )""",
        (run_id,),
    ).fetchone()[0]

    trade_rows = conn.execute(
        """SELECT action, shares, gross_pnl, tax, net_pnl_after_tax,
                  source_job, source_task, order_source,
                  score_snapshot_json, decision_inputs_json
             FROM trades
            WHERE run_id = ?""",
        (run_id,),
    ).fetchall()
    sell_share_gaps = 0
    sell_economic_gaps = 0
    fallback_trade_attribution_gaps = 0
    qp_trade_attribution_gaps = 0
    qp_buy_horizon_gaps = 0
    for (
        action,
        shares,
        gross_pnl,
        tax,
        net_pnl_after_tax,
        source_job,
        source_task,
        order_source,
        score_json,
        inputs_json,
    ) in trade_rows:
        if str(action or "").lower() == "sell":
            try:
                sh = float(shares)
            except (TypeError, ValueError):
                sh = 0.0
            if sh <= 0:
                sell_share_gaps += 1
            if not _sell_economics_are_valid(gross_pnl, tax, net_pnl_after_tax):
                sell_economic_gaps += 1
        try:
            score_snapshot = json.loads(score_json) if score_json else {}
        except Exception:  # noqa: BLE001
            score_snapshot = {}
        try:
            decision_inputs = json.loads(inputs_json) if inputs_json else {}
        except Exception:  # noqa: BLE001
            decision_inputs = {}
        if isinstance(score_snapshot, dict) and score_snapshot.get("attribution_missing"):
            fallback_trade_attribution_gaps += 1
        if isinstance(decision_inputs, dict) and decision_inputs.get("attribution_missing"):
            fallback_trade_attribution_gaps += 1

        merged_source = " ".join(
            str(v or "")
            for v in (
                source_job,
                source_task,
                order_source,
                decision_inputs.get("source_job") if isinstance(decision_inputs, dict) else "",
                decision_inputs.get("source_task") if isinstance(decision_inputs, dict) else "",
                decision_inputs.get("order_source") if isinstance(decision_inputs, dict) else "",
                decision_inputs.get("acceptance_reason") if isinstance(decision_inputs, dict) else "",
            )
        ).lower()
        if "qp" in merged_source or "jointportfolioqpjob" in merged_source:
            required = ("delta_w", "target_w", "solver_status")
            if not isinstance(decision_inputs, dict) or any(
                decision_inputs.get(k) is None for k in required
            ):
                qp_trade_attribution_gaps += 1
            if str(action or "").lower() == "buy":
                if (
                    not isinstance(score_snapshot, dict)
                    or not _json_finite(score_snapshot.get("expected_return"))
                    or not _json_positive_int(
                        score_snapshot.get("expected_return_horizon_days")
                    )
                    or not _json_positive_int(score_snapshot.get("mu_horizon_days"))
                    or not isinstance(decision_inputs, dict)
                    or not _json_positive_int(
                        decision_inputs.get("expected_return_horizon_days")
                    )
                    or not _json_positive_int(decision_inputs.get("mu_horizon_days"))
                ):
                    qp_buy_horizon_gaps += 1
    return {
        "run_id": run_id,
        "ticker_daily_state_rows": len(recorded),
        "expected_watchlist_rows": len(expected) if expected else None,
        "missing_watchlist_tickers": sorted(expected - recorded),
        "extra_tickers": sorted(recorded - expected) if expected else [],
        "selected_blocked_rows": int(selected_blockers or 0),
        "candidate_selected_blocked_rows": int(candidate_selected_blockers or 0),
        "candidate_reason_gaps": int(candidate_reason_gaps or 0),
        "decision_reason_gaps": int(decision_reason_gaps or 0),
        "trade_payload_gaps": int(trade_payload_gaps or 0),
        "fallback_trade_attribution_gaps": int(fallback_trade_attribution_gaps),
        "sell_share_gaps": int(sell_share_gaps),
        "sell_economic_gaps": int(sell_economic_gaps),
        "qp_trade_attribution_gaps": int(qp_trade_attribution_gaps),
        "qp_buy_horizon_gaps": int(qp_buy_horizon_gaps),
        "model_type_gaps": int(model_type_gaps or 0),
        "selected_sector_gaps": int(selected_sector_gaps or 0),
        "candidate_selected_sector_gaps": int(candidate_selected_sector_gaps or 0),
        "candidate_horizon_gaps": int(candidate_horizon_gaps or 0),
        "decision_horizon_gaps": int(decision_horizon_gaps or 0),
        "trade_horizon_gaps": int(trade_horizon_gaps or 0),
        "ok": (
            (not expected or recorded == expected)
            and int(selected_blockers or 0) == 0
            and int(candidate_selected_blockers or 0) == 0
            and int(candidate_reason_gaps or 0) == 0
            and int(decision_reason_gaps or 0) == 0
            and int(trade_payload_gaps or 0) == 0
            and int(fallback_trade_attribution_gaps) == 0
            and int(sell_share_gaps) == 0
            and int(sell_economic_gaps) == 0
            and int(qp_trade_attribution_gaps) == 0
            and int(qp_buy_horizon_gaps) == 0
            and int(model_type_gaps or 0) == 0
            and int(selected_sector_gaps or 0) == 0
            and int(candidate_selected_sector_gaps or 0) == 0
            and int(candidate_horizon_gaps or 0) == 0
            and int(decision_horizon_gaps or 0) == 0
            and int(trade_horizon_gaps or 0) == 0
        ),
    }


def _json_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def validate_decision_trace_integrity(
    conn: sqlite3.Connection | None,
    run_id: str | None,
    config: dict,
    *,
    context: str = "decision_trace",
) -> dict[str, Any]:
    """Validate the just-written decision trace and optionally fail closed.

    The persistence layer is only useful if every run can be replayed from DB
    rows. This helper is called by sim/live/LEAN adapters immediately after
    they write candidate_scores, trades, and ticker_daily_state.
    """
    if conn is None or run_id is None:
        return {}
    from kernel.pipeline.task_benchmark_sleeve import decision_trace_tickers  # noqa: PLC0415

    report = decision_trace_integrity_report(
        conn,
        run_id,
        expected_watchlist=decision_trace_tickers(config),
    )
    if report.get("ok", False):
        log.info("%s: decision trace integrity OK (run_id=%s)", context, run_id)
        return report

    msg = (
        f"{context}: decision trace integrity failed for run_id={run_id}: "
        f"{json.dumps(report, sort_keys=True, default=str)}"
    )
    strict = bool(
        (config.get("persistence", {}) or {})
        .get("strict_decision_trace_integrity", True)
    )
    if strict:
        raise RuntimeError(msg)
    log.warning(msg)
    return report


def record_forward_returns(
    conn: sqlite3.Connection | None,
    rows: Iterable[dict],
) -> int:
    """Upsert ticker_forward_returns rows (Plan AA).

    Each row: `{as_of_date, ticker, close_price, fwd_1d, fwd_5d,
    fwd_10d, fwd_20d, fwd_60d}`.
    Any field except (as_of_date, ticker) can be None. Returns number of rows written.
    """
    if conn is None:
        return 0
    payload = []
    for r in rows:
        payload.append((
            r["as_of_date"] if isinstance(r["as_of_date"], str)
            else r["as_of_date"].isoformat(),
            r["ticker"],
            _none_or_float(r.get("close_price")),
            _none_or_float(r.get("fwd_1d")),
            _none_or_float(r.get("fwd_5d")),
            _none_or_float(r.get("fwd_10d")),
            _none_or_float(r.get("fwd_20d")),
            _none_or_float(r.get("fwd_60d")),
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT INTO ticker_forward_returns
              (as_of_date, ticker, close_price, fwd_1d, fwd_5d, fwd_10d, fwd_20d, fwd_60d)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(as_of_date, ticker) DO UPDATE SET
              close_price = COALESCE(excluded.close_price, close_price),
              fwd_1d      = COALESCE(excluded.fwd_1d, fwd_1d),
              fwd_5d      = COALESCE(excluded.fwd_5d, fwd_5d),
              fwd_10d     = COALESCE(excluded.fwd_10d, fwd_10d),
              fwd_20d     = COALESCE(excluded.fwd_20d, fwd_20d),
              fwd_60d     = COALESCE(excluded.fwd_60d, fwd_60d),
              updated_at  = CURRENT_TIMESTAMP""",
        payload,
    )
    return len(payload)


def record_trade_evaluations(
    conn: sqlite3.Connection | None,
    rows: Iterable[dict],
) -> int:
    """Upsert trade_evaluations rows (roadmap §2026-04-26 Phase 1).

    Each row: ``{run_id, ticker, action, horizon_days, fwd_return,
    fwd_return_spy, relative_return, is_winner, n_trade_rows}``.

    `is_winner` is computed by the caller (1 / 0 / None) so we don't
    silently re-derive it from `relative_return` here — the caller's
    intent (e.g. relative-to-benchmark vs absolute) stays explicit.

    On conflict we REPLACE the row — backfill is idempotent. The
    primary-key (run_id, ticker, action, horizon_days) prevents
    double-counting when the same (trade, horizon) pair gets re-evaluated.

    Returns the number of rows attempted (not necessarily inserted —
    SQLite's INSERT OR REPLACE returns 1 for both insert + update).
    """
    if conn is None:
        return 0
    payload = []
    for r in rows:
        try:
            run_id  = str(r["run_id"])
            ticker  = str(r["ticker"])
            action  = str(r["action"])
            horizon = int(r["horizon_days"])
        except (KeyError, ValueError, TypeError) as exc:
            log.warning(
                "record_trade_evaluations: skipping row missing required "
                "key (run_id/ticker/action/horizon_days): %s — %s", r, exc,
            )
            continue
        if action not in ("buy", "sell"):
            log.warning(
                "record_trade_evaluations: skipping row with invalid action=%r"
                " (must be 'buy' or 'sell')", action,
            )
            continue
        if horizon <= 0:
            log.warning(
                "record_trade_evaluations: skipping row with non-positive "
                "horizon_days=%r", horizon,
            )
            continue
        payload.append((
            run_id, ticker, action, horizon,
            _none_or_float(r.get("fwd_return")),
            _none_or_float(r.get("fwd_return_spy")),
            _none_or_float(r.get("relative_return")),
            _none_or_int(r.get("is_winner")),
            _none_or_int(r.get("n_trade_rows")),
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT INTO trade_evaluations
              (run_id, ticker, action, horizon_days,
               fwd_return, fwd_return_spy, relative_return,
               is_winner, n_trade_rows)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_id, ticker, action, horizon_days) DO UPDATE SET
              fwd_return       = COALESCE(excluded.fwd_return,       fwd_return),
              fwd_return_spy   = COALESCE(excluded.fwd_return_spy,   fwd_return_spy),
              relative_return  = COALESCE(excluded.relative_return,  relative_return),
              is_winner        = COALESCE(excluded.is_winner,        is_winner),
              n_trade_rows     = COALESCE(excluded.n_trade_rows,     n_trade_rows),
              created_at       = CURRENT_TIMESTAMP""",
        payload,
    )
    return len(payload)


# ── Small helpers ─────────────────────────────────────────────────────────────

def _none_or_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        # Round-3 audit (#R3-45): also filter ±inf. SQLite stores them as
        # REAL but later analytics queries (median, percentile) silently
        # break. Treat as missing.
        import math
        if not math.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _none_or_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def lookup_candidate_scores_on_date(
    conn,
    tickers: "list[str]",
    as_of: "datetime.date",
    role: str = "candidate",
) -> dict[str, dict]:
    """Return {ticker: {rank_score, panel_score, mu, sigma}} for the
    snapshot recorded on `as_of` with the given `role`.

    Used by Rotation V4 (thesis_symmetric) to look up B's score on A's
    entry date. Joins candidate_scores × pipeline_runs to find the run
    that executed on `as_of` and pulls each ticker's scores.

    Round-3 audit (#R3-46): added `role` filter (default "candidate") so
    the lookup doesn't accidentally pick up a holding-side snapshot when
    both exist for the same ticker on the same date. Holdings have
    `raw_score=NULL` (line 418 in record_candidate_scores), which would
    silently mis-rank rotation pairs.

    Returns an empty dict if no run landed on that date (sim hasn't
    processed it yet, or it was pre-sim-start). Callers should treat
    absence as "skip this pair" rather than "signal=0".
    """
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    cur = conn.execute(
        f"""
        SELECT cs.ticker, cs.rank_score, cs.panel_score, cs.mu, cs.sigma
        FROM candidate_scores cs
        JOIN pipeline_runs pr ON cs.run_id = pr.run_id
        WHERE pr.run_date = ?
          AND cs.role     = ?
          AND cs.ticker IN ({placeholders})
        """,
        (str(as_of), role, *tickers),
    )
    out: dict[str, dict] = {}
    for row in cur:
        out[row[0]] = {
            "rank_score":  row[1],
            "panel_score": row[2],
            "mu":          row[3],
            "sigma":       row[4],
        }
    return out


__all__ = [
    "ensure_schema",
    "get_connection",
    "clear_sim_tables",
    "record_pipeline_run",
    "record_candidate_scores",
    "record_trades",
    "record_training_run",
    "record_forward_returns",
    "record_live_state_snapshot",
    "load_latest_live_state",
    "record_portfolio_metrics",
    "record_ticker_daily_state",
    "decision_trace_integrity_report",
    "validate_decision_trace_integrity",
    "lookup_candidate_scores_on_date",
]
