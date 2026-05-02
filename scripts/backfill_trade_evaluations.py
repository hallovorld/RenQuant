#!/usr/bin/env python
"""Backfill trade_evaluations — re-evaluate every trade at multiple horizons.

Roadmap §2026-04-26 Phase 2. Joins ``trades`` × ``ticker_forward_returns``
to compute realized forward returns at horizons (1, 5, 7, 14, 28 days
default) for each trade, plus benchmark-relative excess return.

Auto-surfaces "we lost money on these trades" patterns ~14 days after
trade — the post-mortem cycle that today only happens when an operator
manually asks (e.g. CAT 2026-05-01 sell-after-earnings-rip / FTNT
2026-04-29 cross-earnings buy were caught by hand).

This script is the cron job. Phase 3 will add ntfy on >1σ degradation.

Production safety: read trades + ticker_forward_returns; write to
trade_evaluations. Idempotent (PRIMARY KEY enforces no double-write).
Live state files untouched.

Usage::

    python scripts/backfill_trade_evaluations.py
    python scripts/backfill_trade_evaluations.py --since 2026-04-01
    python scripts/backfill_trade_evaluations.py --horizons 1 5 7 14 28
    python scripts/backfill_trade_evaluations.py --dry-run   # show what would write

Exit codes
----------
  0  — backfill completed (n_written reported in stdout)
  1  — invalid args / DB missing
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backfill-trade-eval")

DEFAULT_HORIZONS = [1, 5, 7, 14, 28]


def _supported_fwd_col(horizon_days: int) -> str | None:
    """Map horizon_days → ticker_forward_returns column name.

    The schema only carries fwd_1d / fwd_5d / fwd_10d / fwd_20d. For
    horizons not in that set (7, 14, 28), fall back to closest.
    Caller can opt to fill missing horizons with NULL instead.
    """
    direct = {1: "fwd_1d", 5: "fwd_5d", 10: "fwd_10d", 20: "fwd_20d"}
    if horizon_days in direct:
        return direct[horizon_days]
    return None   # horizon not directly supported by current DB schema


def _build_eval_rows(
    conn: sqlite3.Connection,
    horizons: list[int],
    since: str | None,
    benchmark: str = "SPY",
) -> list[dict]:
    """Query trades + ticker_forward_returns and compose evaluation rows.

    For horizons not directly supported by the DB schema (7, 14, 28),
    we currently SKIP — would require adding fwd_7d/14d/28d columns
    + extending the backfill_forward_returns.py script. That's a
    follow-up task. Direct-supported horizons (1, 5, 10, 20) work today.
    """
    cur = conn.cursor()
    where = ""
    params: list = []
    if since:
        where = "WHERE pr.run_date >= ?"
        params.append(since)

    rows: list[dict] = []
    n_skipped_unsupported = 0
    for h in horizons:
        col = _supported_fwd_col(h)
        if col is None:
            log.info("Horizon %dd: not directly in DB schema (only "
                     "1/5/10/20 supported) — skipping", h)
            n_skipped_unsupported += 1
            continue

        sql = f"""
            SELECT
                t.run_id        AS run_id,
                t.ticker        AS ticker,
                t.action        AS action,
                pr.run_date     AS run_date,
                tfr.{col}       AS fwd_return,
                bfr.{col}       AS fwd_return_spy,
                COUNT(*)        AS n_trade_rows
            FROM trades t
            JOIN pipeline_runs pr ON pr.run_id = t.run_id
            LEFT JOIN ticker_forward_returns tfr
                   ON tfr.ticker     = t.ticker
                  AND tfr.as_of_date = pr.run_date
            LEFT JOIN ticker_forward_returns bfr
                   ON bfr.ticker     = ?
                  AND bfr.as_of_date = pr.run_date
            {where}
            GROUP BY t.run_id, t.ticker, t.action, pr.run_date
        """
        full_params = [benchmark] + params
        for r in cur.execute(sql, full_params).fetchall():
            run_id, ticker, action, _run_date, fwd_t, fwd_b, n = r
            if fwd_t is None:
                # No forward return available yet (too recent). Skip; the
                # nightly cron will pick it up in a few days.
                continue
            rel = (fwd_t - fwd_b) if (fwd_t is not None and fwd_b is not None) else None
            is_winner = None
            if rel is not None:
                is_winner = 1 if rel > 0 else 0
            rows.append({
                "run_id":          run_id,
                "ticker":          ticker,
                "action":          action,
                "horizon_days":    h,
                "fwd_return":      fwd_t,
                "fwd_return_spy":  fwd_b,
                "relative_return": rel,
                "is_winner":       is_winner,
                "n_trade_rows":    int(n),
            })

    if n_skipped_unsupported:
        log.warning(
            "Skipped %d horizon(s) not in DB schema. To support 7d/14d/28d, "
            "extend ticker_forward_returns with fwd_7d/14d/28d columns and "
            "the backfill_forward_returns.py script.", n_skipped_unsupported,
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/runs.alpaca.db")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                   help="Evaluation horizons in days. Note that the current DB "
                        "schema only has fwd_1d/5d/10d/20d — others are skipped "
                        "with a warning.")
    p.add_argument("--since", default=None,
                   help="Only evaluate trades whose run_date is on or after this "
                        "ISO date (e.g. '2026-04-01'). Default: all.")
    p.add_argument("--benchmark", default="SPY",
                   help="Benchmark ticker for relative-return computation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute rows but don't write to DB.")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("DB not found: %s", db_path); return 1

    log.info("Backfilling trade_evaluations (db=%s, horizons=%s, since=%s)",
             db_path, args.horizons, args.since or "all")

    started = _dt.datetime.now(_dt.timezone.utc)
    conn = sqlite3.connect(db_path)
    try:
        rows = _build_eval_rows(
            conn, horizons=args.horizons, since=args.since,
            benchmark=args.benchmark,
        )
        log.info("Computed %d evaluation rows", len(rows))

        if args.dry_run:
            log.info("--dry-run: NOT writing to DB. Sample (up to 5):")
            for r in rows[:5]:
                log.info("  %s", r)
            return 0

        from kernel.persistence import record_trade_evaluations  # noqa: PLC0415
        n_written = record_trade_evaluations(conn, rows)
        conn.commit()
    finally:
        conn.close()

    finished = _dt.datetime.now(_dt.timezone.utc)
    print()
    print("=" * 60)
    print(f"  TRADE EVALUATIONS BACKFILL")
    print("=" * 60)
    print(f"  rows written         {n_written}")
    print(f"  horizons             {args.horizons}")
    print(f"  since                {args.since or 'all'}")
    print(f"  wall seconds         {(finished - started).total_seconds():.1f}")
    print(f"  db                   {db_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
