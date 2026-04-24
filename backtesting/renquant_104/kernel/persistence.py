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

def _commit_sha() -> str | None:
    try:
        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return sha or None
    except Exception:
        return None


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
    for c in candidates:
        rows.append((
            run_id, c.ticker, "candidate",
            float(getattr(c, "raw_score",  0.0) or 0.0),
            float(getattr(c, "rank_score", 0.0) or 0.0),
            _none_or_float(getattr(c, "panel_score", None)),
            float(getattr(c, "rs_score",   0.0) or 0.0),
            _none_or_float(getattr(c, "mu",    None)),
            _none_or_float(getattr(c, "sigma", None)),
            1 if c.ticker in selected_tickers else 0,
            blocked_map.get(c.ticker),
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
        ))
    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO candidate_scores
                  (run_id, ticker, role, raw_score, rank_score, panel_score, rs_score,
                   mu, sigma, selected, blocked_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        "commit_sha":            _commit_sha(),
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
             artifact_path, _commit_sha(),
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
        return f if f == f else None   # filter NaN
    except (TypeError, ValueError):
        return None


def _none_or_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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
]
