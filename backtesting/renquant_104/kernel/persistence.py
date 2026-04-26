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
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("kernel.persistence")


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
    kelly_target_pct   REAL,        -- Kelly sizing target (μ/σ²)
    model_type         TEXT,        -- per-ticker model: 'Manual' | 'XGBoost' | 'QLearning' | 'Classification'
    sector             TEXT,        -- from sector_map, easier than join
    panel_ltr_artifact TEXT,        -- 'panel-ltr.json' filename or full path
    PRIMARY KEY (run_id, ticker, role),
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_cand_ticker ON candidate_scores(ticker);

CREATE TABLE IF NOT EXISTS trades (
    run_id         TEXT,
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
    rank_score     REAL,
    conviction     REAL,
    sigma_mult     REAL,
    mu             REAL,
    sigma          REAL,
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
    kelly_target_pct  REAL,
    mu                REAL,
    sigma             REAL,
    -- Final decision
    in_candidates     INTEGER,        -- 1 if reached ctx.candidates (per-ticker model said buy)
    selected          INTEGER,        -- 1 if BUY order placed this bar
    blocked_by        TEXT,           -- reason if blocked: 'sector_cap'|'corr'|'wash_sale'|'tier'|'universe_floor'|'broker_pending'|'no_model_signal'
    sector            TEXT,
    PRIMARY KEY (date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_tds_date ON ticker_daily_state(date);
CREATE INDEX IF NOT EXISTS idx_tds_ticker ON ticker_daily_state(ticker);

CREATE TABLE IF NOT EXISTS score_distribution (
    date          TEXT NOT NULL,        -- YYYY-MM-DD (string for sqlite friendliness)
    ticker        TEXT NOT NULL,
    raw_panel     REAL,                 -- pre-calibration scorer output (panel_score)
    rank_score    REAL,                 -- post-calibration probability
    mu            REAL,                 -- NGBoost μ if active
    sigma         REAL,                 -- NGBoost σ if active
    regime        TEXT,                 -- BULL_CALM / etc.
    is_holding    INTEGER DEFAULT 0,    -- 0 = candidate, 1 = held
    PRIMARY KEY (date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_score_dist_date ON score_distribution(date);

-- Daily aggregated percentiles for fast threshold lookup.
-- Computed at end of each bar from that day's score_distribution rows.
-- Phase 2 JointAction reads this to convert "top X%" → absolute threshold.
CREATE TABLE IF NOT EXISTS score_percentiles_daily (
    date          TEXT PRIMARY KEY,
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
    regime        TEXT
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
"""


# Idempotent column migrations for tables created before a column was added.
# SQLite's `CREATE TABLE IF NOT EXISTS` is a no-op on pre-existing tables, so
# any column added to _SCHEMA_SQL after first creation must also be listed here.
_COLUMN_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
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
        ("kelly_target_pct",   "REAL"),
        ("model_type",         "TEXT"),
        ("sector",             "TEXT"),
        ("panel_ltr_artifact", "TEXT"),
    ],
}


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
) -> str | None:
    """Insert a pipeline_runs row and return the generated run_id."""
    if conn is None:
        return None
    run_id = f"{run_date.isoformat()}-{run_type}-{uuid.uuid4().hex[:8]}"
    conn.execute(
        """INSERT INTO pipeline_runs
              (run_id, run_date, run_type, strategy, regime, confidence,
               portfolio_value, cash, n_candidates, n_exits, n_rotations, n_buys,
               commit_sha)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, run_date.isoformat(), run_type, strategy, regime, confidence,
         portfolio_value, cash, n_candidates, n_exits, n_rotations, n_buys,
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
) -> None:
    """Insert one row per candidate + one per holding.

    `candidates`:  iterable of CandidateResult-like objects (must have
                   .ticker, .raw_score, .rank_score, .rs_score, .panel_score,
                   .mu, .sigma)
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
    # on NaN/inf, persisting as SQL NULL.
    def _safe_float_or_default(v: Any, default: float = 0.0) -> float:
        f = _none_or_float(v)
        return default if f is None else f
    sector_map = sector_map or {}
    model_types = model_types or {}
    for c in candidates:
        rows.append((
            run_id, c.ticker, "candidate",
            _safe_float_or_default(getattr(c, "raw_score",  None)),
            _safe_float_or_default(getattr(c, "rank_score", None)),
            _none_or_float(getattr(c, "panel_score", None)),
            _safe_float_or_default(getattr(c, "rs_score",   None)),
            _none_or_float(getattr(c, "mu",    None)),
            _none_or_float(getattr(c, "sigma", None)),
            1 if c.ticker in selected_tickers else 0,
            blocked_map.get(c.ticker),
            # New decision-factor columns
            _none_or_float(getattr(c, "expected_return", None)),
            _none_or_float(getattr(c, "kelly_target_pct", None)),
            model_types.get(c.ticker),
            sector_map.get(c.ticker),
            panel_artifact,
        ))
    for ticker, hs in holdings.items():
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
            _none_or_float(getattr(hs, "kelly_target_pct", None)),
            model_types.get(ticker),
            sector_map.get(ticker),
            panel_artifact,
        ))
    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO candidate_scores
                  (run_id, ticker, role, raw_score, rank_score, panel_score, rs_score,
                   mu, sigma, selected, blocked_by,
                   expected_return, kelly_target_pct, model_type, sector,
                   panel_ltr_artifact)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
      exit_reason, pnl_pct, hold_days, tax, rank_score, conviction,
      sigma_mult, mu, sigma
    """
    if conn is None or run_id is None:
        return
    rows = []
    for t in trade_events:
        rows.append((
            run_id,
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
            _none_or_float(t.get("rank_score")),
            _none_or_float(t.get("conviction")),
            _none_or_float(t.get("sigma_mult")),
            _none_or_float(t.get("mu")),
            _none_or_float(t.get("sigma")),
        ))
    if rows:
        conn.executemany(
            """INSERT INTO trades
                  (run_id, ticker, action, shares, price, invest, target_pct,
                   exit_reason, pnl_pct, hold_days, tax,
                   rank_score, conviction, sigma_mult, mu, sigma)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        log_dir = jsonl_dir or Path("logs/training")
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
) -> int:
    """Upsert ticker_daily_state rows — one per watchlist ticker per bar.

    Per user spec round-5 (2026-04-26): every watchlist ticker gets a row
    every bar, including those filtered by universe floor / broker
    pre-check / no-model-signal — so post-hoc analysis can answer "what
    did we KNOW about XYZ on this date and WHY didn't we trade it?".

    Each row dict supports: regime, confidence, in_watchlist, in_universe,
    pending_at_broker, has_position, position_qty, position_pct,
    model_type, model_action, sell_streak, panel_score, rank_score,
    expected_return, kelly_target_pct, mu, sigma, in_candidates,
    selected, blocked_by, sector. `ticker` required.
    """
    if conn is None:
        return 0
    payload = []
    rd_str = run_date.isoformat() if hasattr(run_date, "isoformat") else str(run_date)
    for r in rows:
        if not r.get("ticker"):
            continue
        payload.append((
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
            _none_or_float(r.get("kelly_target_pct")),
            _none_or_float(r.get("mu")),
            _none_or_float(r.get("sigma")),
            _none_or_int(r.get("in_candidates")),
            _none_or_int(r.get("selected")),
            r.get("blocked_by"),
            r.get("sector"),
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT OR REPLACE INTO ticker_daily_state
              (date, ticker, regime, confidence,
               in_watchlist, in_universe, pending_at_broker,
               has_position, position_qty, position_pct,
               model_type, model_action, sell_streak,
               panel_score, rank_score, expected_return, kelly_target_pct,
               mu, sigma, in_candidates, selected, blocked_by, sector)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        payload,
    )
    return len(payload)


def record_forward_returns(
    conn: sqlite3.Connection | None,
    rows: Iterable[dict],
) -> int:
    """Upsert ticker_forward_returns rows (Plan AA).

    Each row: `{as_of_date, ticker, close_price, fwd_1d, fwd_5d, fwd_10d, fwd_20d}`.
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
        ))
    if not payload:
        return 0
    conn.executemany(
        """INSERT INTO ticker_forward_returns
              (as_of_date, ticker, close_price, fwd_1d, fwd_5d, fwd_10d, fwd_20d)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(as_of_date, ticker) DO UPDATE SET
              close_price = COALESCE(excluded.close_price, close_price),
              fwd_1d      = COALESCE(excluded.fwd_1d, fwd_1d),
              fwd_5d      = COALESCE(excluded.fwd_5d, fwd_5d),
              fwd_10d     = COALESCE(excluded.fwd_10d, fwd_10d),
              fwd_20d     = COALESCE(excluded.fwd_20d, fwd_20d),
              updated_at  = CURRENT_TIMESTAMP""",
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
    "record_portfolio_metrics",
    "record_ticker_daily_state",
    "lookup_candidate_scores_on_date",
]
