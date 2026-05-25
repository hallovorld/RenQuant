#!/usr/bin/env python
"""Plan AA — compute forward returns for every (date, ticker) in candidate_scores.

Joins each candidate decision day with the parquet OHLCV cache and
writes close_price + fwd_{1,5,10,20,60}d into the ticker_forward_returns
table. Idempotent upsert: skips rows where all 4 horizons are already
populated.

Usage::

    python scripts/backfill_forward_returns.py
    python scripts/backfill_forward_returns.py --strategy renquant_104
    python scripts/backfill_forward_returns.py --db data/runs.db
    python scripts/backfill_forward_returns.py --since 2026-01-01

Designed to run daily (cheap after first pass) — most rows are already
filled, only the tail from the last 60 trading days needs updating as
new bars arrive.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backfill-forward-returns")


HORIZONS = [1, 5, 10, 20, 60]


def _available_fwd_cols(conn) -> list[str]:
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(ticker_forward_returns)").fetchall()
    }
    return [f"fwd_{h}d" for h in HORIZONS if f"fwd_{h}d" in cols]


def _missing_forward_return_predicate(conn, alias: str = "tfr") -> str:
    fwd_cols = _available_fwd_cols(conn)
    if not fwd_cols:
        return f"{alias}.as_of_date IS NULL"
    checks = " OR ".join(f"{alias}.{c} IS NULL" for c in fwd_cols)
    return f"({alias}.as_of_date IS NULL OR {checks})"


def _load_ohlcv(ticker: str, cache_root: Path) -> "pd.DataFrame | None":
    """Read the per-ticker parquet cache; return None when missing."""
    import pandas as pd  # noqa: PLC0415
    path = cache_root / ticker / "1d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Parquet cache indexes on Date already; normalise if not
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def _compute_row(
    date: datetime.date,
    ticker: str,
    df: "pd.DataFrame",
) -> dict | None:
    """Return an upsert payload dict, or None if close_price unavailable."""
    import pandas as pd  # noqa: PLC0415
    ts = pd.Timestamp(date)
    if ts not in df.index:
        return None

    close = float(df.loc[ts, "close"])
    idx   = df.index.get_loc(ts)

    out: dict = {
        "as_of_date":  date,
        "ticker":      ticker,
        "close_price": close,
    }
    for h in HORIZONS:
        tgt_idx = idx + h
        if tgt_idx < len(df):
            tgt_close = float(df.iloc[tgt_idx]["close"])
            out[f"fwd_{h}d"] = (tgt_close / close) - 1.0
        else:
            out[f"fwd_{h}d"] = None
    return out


def _rows_needing_backfill(
    conn, since: datetime.date | None,
) -> list[tuple[str, str]]:
    """Return (as_of_date, ticker) pairs where any fwd_* is NULL.

    Skips the very recent tail where some horizons can't be filled yet
    (e.g. fwd_20d needs 20 trading days in the future) — those show up
    tomorrow and get picked up by the next run.
    """
    missing_predicate = _missing_forward_return_predicate(conn, "tfr")
    q = f"""
        SELECT DISTINCT ps.run_date, cs.ticker
          FROM candidate_scores cs
          JOIN pipeline_runs    ps ON ps.run_id = cs.run_id
     LEFT JOIN ticker_forward_returns tfr
            ON tfr.as_of_date = ps.run_date AND tfr.ticker = cs.ticker
         WHERE {missing_predicate}
    """
    params: list = []
    if since is not None:
        q += " AND ps.run_date >= ?"
        params.append(since.isoformat())
    q += " ORDER BY ps.run_date, cs.ticker"
    return conn.execute(q, params).fetchall()


def _benchmark_pairs(
    conn, benchmarks: list[str], since: datetime.date | None,
) -> list[tuple[str, str]]:
    """Return (run_date, benchmark_ticker) pairs missing forward returns.

    Benchmarks (SPY, sector ETFs) are not stored in candidate_scores —
    they're the *reference* against which candidates are evaluated. But
    downstream consumers (M3 conformal Gate B fit; trade-eval DB
    relative-return labels) JOIN forward returns by benchmark too, so the
    backfill must cover them. Pre-fix the LEFT JOIN nulled out the entire
    fit input → fit_conformal_gate_b.py reported "0 valid rows" while
    74k otherwise-valid candidate rows existed.

    Audit fix L1 (2026-05-01): pre-fix this fn ran one SELECT per
    (date, benchmark) pair — for our 567-date DB that's 567 queries,
    fine but O(D × B). Now: a single LEFT JOIN per benchmark,
    O(B) queries total. ~50× speedup at current scale; matters when
    the trade-eval DB grows or new benchmarks are added.
    """
    if not benchmarks:
        return []
    out: list[tuple[str, str]] = []
    for bench in benchmarks:
        # Single LEFT JOIN per benchmark — emit (run_date, bench) when
        # the ticker_forward_returns row is missing OR any of the four
        # forward-return columns is NULL.
        missing_predicate = _missing_forward_return_predicate(conn, "tfr")
        q = f"""
            SELECT DISTINCT ps.run_date
              FROM pipeline_runs ps
              LEFT JOIN ticker_forward_returns tfr
                     ON tfr.as_of_date = ps.run_date
                    AND tfr.ticker     = ?
             WHERE {missing_predicate}
        """
        params: list = [bench]
        if since is not None:
            q += " AND ps.run_date >= ?"
            params.append(since.isoformat())
        q += " ORDER BY ps.run_date"
        for (date_str,) in conn.execute(q, params).fetchall():
            out.append((date_str, bench))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--source", choices=["live", "sim"], default="live",
                   help="Backfill the live DB (data/runs.db, default) or the "
                        "ephemeral notebook-sim DB (data/sim_runs.db). Live is "
                        "the common case; sim is only useful to analyze a "
                        "specific notebook session's decisions.")
    p.add_argument("--db", default=None,
                   help="Explicit path; bypasses --source mapping.")
    p.add_argument("--broker", default="alpaca",
                   choices=["alpaca", "alpaca_paper", "paper", "ibkr"],
                   help="Broker tag for live runs.db lookup. Live RenQuant 104 "
                        "uses data/runs.{broker}.db, not legacy data/runs.db.")
    p.add_argument("--since", type=lambda s: datetime.date.fromisoformat(s),
                   default=None,
                   help="Only backfill rows at or after this date (YYYY-MM-DD).")
    p.add_argument("--cache-root", default="data/ohlcv",
                   help="Root of per-ticker parquet cache.")
    p.add_argument(
        "--benchmarks", default="SPY",
        help="Comma-separated benchmarks to also backfill (default: SPY). "
             "Empty string disables benchmark backfill. Required for "
             "fit_conformal_gate_b.py — without SPY in the table the "
             "conformal-fit JOIN nulls every candidate row.",
    )
    args = p.parse_args()
    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    if args.db is None:
        if args.source == "sim":
            args.db = "data/sim_runs.db"
        else:
            from kernel.state_paths import runs_db_path  # noqa: PLC0415
            p_db = runs_db_path("data/runs.db", args.broker)
            args.db = str(
                p_db.relative_to(REPO_ROOT) if p_db.is_absolute() else p_db
            )

    from kernel.persistence import (  # noqa: PLC0415
        get_connection, record_forward_returns,
    )
    conn = get_connection(
        {"persistence": {"enabled": True, "db_path": str(REPO_ROOT / args.db)}},
    )
    if conn is None:
        log.error("Could not open DB at %s", args.db)
        sys.exit(1)

    cache_root = REPO_ROOT / args.cache_root
    if not cache_root.exists():
        log.error("OHLCV cache missing: %s", cache_root)
        sys.exit(1)

    pairs = _rows_needing_backfill(conn, args.since)

    benchmarks = [b.strip().upper() for b in args.benchmarks.split(",") if b.strip()]
    bench_pairs = _benchmark_pairs(conn, benchmarks, args.since)
    if bench_pairs:
        log.info("Benchmark backfill: %d (date, benchmark) pair(s) for %s",
                 len(bench_pairs), benchmarks)
        pairs = pairs + bench_pairs

    if not pairs:
        log.info("Nothing to backfill — every candidate + benchmark row already has forward returns.")
        return

    # Group by ticker to amortise parquet load
    by_ticker: dict[str, list[str]] = {}
    for date_str, ticker in pairs:
        by_ticker.setdefault(ticker, []).append(date_str)

    log.info("Backfilling %d (date, ticker) pairs across %d tickers",
             len(pairs), len(by_ticker))

    total_written = 0
    for ticker, dates in sorted(by_ticker.items()):
        df = _load_ohlcv(ticker, cache_root)
        if df is None:
            log.warning("  %-6s — no parquet at %s/%s/1d.parquet, skipping %d rows",
                        ticker, cache_root.name, ticker, len(dates))
            continue
        payload = []
        for d in dates:
            row = _compute_row(datetime.date.fromisoformat(d), ticker, df)
            if row is not None:
                payload.append(row)
        if payload:
            written = record_forward_returns(conn, payload)
            total_written += written
            log.info("  %-6s — wrote %d rows", ticker, written)

    conn.commit()
    log.info("Done. %d rows upserted into ticker_forward_returns.", total_written)


if __name__ == "__main__":
    main()
