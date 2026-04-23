#!/usr/bin/env python
"""Canned introspection queries over data/runs.db.

Usage::

    python scripts/query_runs.py                  # runs all canned queries
    python scripts/query_runs.py pnl_by_reason    # specific query

New queries: add a function `def q_<name>(conn) -> pd.DataFrame` at the
top of this module and they auto-register.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "runs.db"


def q_recent_runs(conn) -> pd.DataFrame:
    """Last 20 pipeline_runs with portfolio value + activity counts."""
    return pd.read_sql_query(
        """SELECT run_date, run_type, strategy, regime, confidence,
                  portfolio_value, n_candidates, n_exits, n_rotations, n_buys
           FROM pipeline_runs
           ORDER BY run_date DESC, created_at DESC
           LIMIT 20""",
        conn,
    )


def q_pnl_by_reason(conn) -> pd.DataFrame:
    """Per-exit-reason summary (n, mean pnl %, median hold days, tax)."""
    return pd.read_sql_query(
        """SELECT exit_reason,
                  COUNT(*)              AS n,
                  AVG(pnl_pct)          AS avg_pnl_pct,
                  AVG(hold_days)        AS avg_hold_days,
                  SUM(tax)              AS total_tax
           FROM trades
           WHERE action = 'sell' AND exit_reason IS NOT NULL
           GROUP BY exit_reason
           ORDER BY n DESC""",
        conn,
    )


def q_rank_score_buckets(conn) -> pd.DataFrame:
    """Realized sell-pnl bucketed by entry rank_score (in 0.1 bins).

    Joins trades (sell) with the matching same-ticker buy to recover the
    entry rank_score. Useful for calibration-quality introspection.
    """
    return pd.read_sql_query(
        """WITH sells AS (
             SELECT ticker, pnl_pct, hold_days
             FROM trades WHERE action = 'sell' AND pnl_pct IS NOT NULL
           ),
           buys AS (
             SELECT ticker, rank_score
             FROM trades WHERE action = 'buy' AND rank_score IS NOT NULL
           )
           SELECT ROUND(buys.rank_score, 1) AS rank_bucket,
                  COUNT(*)                  AS n,
                  AVG(sells.pnl_pct)        AS avg_pnl,
                  AVG(sells.hold_days)      AS avg_hold_days
           FROM sells JOIN buys ON sells.ticker = buys.ticker
           GROUP BY rank_bucket ORDER BY rank_bucket""",
        conn,
    )


def q_top_vetoed_tickers(conn) -> pd.DataFrame:
    """Tickers most often blocked by a guard, grouped by blocker type."""
    return pd.read_sql_query(
        """SELECT ticker, blocked_by, COUNT(*) AS n_vetoed
           FROM candidate_scores
           WHERE blocked_by IS NOT NULL
           GROUP BY ticker, blocked_by
           ORDER BY n_vetoed DESC LIMIT 30""",
        conn,
    )


def q_regime_win_rate(conn) -> pd.DataFrame:
    """Realized win rate per regime (% of profitable sells)."""
    return pd.read_sql_query(
        """WITH s AS (
             SELECT t.pnl_pct, p.regime
             FROM trades t JOIN pipeline_runs p USING (run_id)
             WHERE t.action = 'sell' AND t.pnl_pct IS NOT NULL
           )
           SELECT regime,
                  COUNT(*)                                                 AS n_sells,
                  AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                  AVG(pnl_pct)                                            AS avg_pnl
           FROM s GROUP BY regime ORDER BY n_sells DESC""",
        conn,
    )


def q_training_history(conn) -> pd.DataFrame:
    """Every training run + its IC."""
    return pd.read_sql_query(
        """SELECT run_date, strategy, artifact_type, oos_mean_ic, train_ic,
                  n_rows, artifact_path
           FROM training_runs
           ORDER BY run_date DESC""",
        conn,
    )


# Registry of queries (auto-discovered)
QUERIES = {name[2:]: fn for name, fn in list(globals().items())
           if name.startswith("q_") and callable(fn)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="?", default="all",
                   choices=list(QUERIES.keys()) + ["all", "list"])
    p.add_argument("--db", default=str(DEFAULT_DB))
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        print("Run a strategy with `persistence.enabled: true` first.", file=sys.stderr)
        sys.exit(1)

    if args.query == "list":
        for name, fn in QUERIES.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:<24s} {doc}")
        return

    conn = sqlite3.connect(db_path)
    to_run = list(QUERIES.keys()) if args.query == "all" else [args.query]
    for name in to_run:
        print(f"\n── {name} ──")
        df = QUERIES[name](conn)
        if df.empty:
            print("  (no rows)")
        else:
            print(df.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
