#!/usr/bin/env python
"""Plan AA — compute forward returns for every (date, ticker) in candidate_scores.

Joins each candidate decision day with the parquet OHLCV cache and
writes close_price + fwd_{1,5,10,20}d into the ticker_forward_returns
table. Idempotent upsert: skips rows where all 4 horizons are already
populated.

Usage::

    python scripts/backfill_forward_returns.py
    python scripts/backfill_forward_returns.py --strategy renquant_104
    python scripts/backfill_forward_returns.py --db data/runs.db
    python scripts/backfill_forward_returns.py --since 2026-01-01

Designed to run daily (cheap after first pass) — most rows are already
filled, only the tail from the last 20 trading days needs updating as
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


HORIZONS = [1, 5, 10, 20]


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
    q = """
        SELECT DISTINCT ps.run_date, cs.ticker
          FROM candidate_scores cs
          JOIN pipeline_runs    ps ON ps.run_id = cs.run_id
     LEFT JOIN ticker_forward_returns tfr
            ON tfr.as_of_date = ps.run_date AND tfr.ticker = cs.ticker
         WHERE (tfr.as_of_date IS NULL
                OR tfr.fwd_1d  IS NULL OR tfr.fwd_5d  IS NULL
                OR tfr.fwd_10d IS NULL OR tfr.fwd_20d IS NULL)
    """
    params: list = []
    if since is not None:
        q += " AND ps.run_date >= ?"
        params.append(since.isoformat())
    q += " ORDER BY ps.run_date, cs.ticker"
    return conn.execute(q, params).fetchall()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--db", default="data/runs.db")
    p.add_argument("--since", type=lambda s: datetime.date.fromisoformat(s),
                   default=None,
                   help="Only backfill rows at or after this date (YYYY-MM-DD).")
    p.add_argument("--cache-root", default="data/ohlcv",
                   help="Root of per-ticker parquet cache.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

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
    if not pairs:
        log.info("Nothing to backfill — every candidate row already has forward returns.")
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
