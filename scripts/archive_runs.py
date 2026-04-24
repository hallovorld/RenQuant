#!/usr/bin/env python
"""Move old pipeline_runs to an archive DB — prep for when runs.db > 5 GB.

The live `data/runs.db` accumulates forever. Projected growth: ~2 MB/day
(1000 decisions × 42 tickers × 50 fields). At 5 GB (~7 years) queries
start slowing. This script archives rows older than N days to a
separate read-only archive DB, freeing the main DB for recent queries.

Dry-run by default — prints counts, doesn't move anything. Use
`--execute` to actually move. Always creates a `.backup` of the source
before destructive operations.

Tables archived (all foreign-keyed by run_id):
  * pipeline_runs       (master table — drives the join)
  * candidate_scores
  * trades
  * rotations
  * live_state_snapshots

NOT archived (independent):
  * ticker_forward_returns   (price-derived, global, small)
  * training_runs            (retrain audit log, small, queryable by date)
  * portfolio_daily_metrics  (daily summary, small)

Usage::

    python scripts/archive_runs.py                       # dry-run default
    python scripts/archive_runs.py --older-than-days 730 # 2-year window
    python scripts/archive_runs.py --execute             # actually move
    python scripts/archive_runs.py --source data/runs.db \\
                                   --archive data/runs_archive.db
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("archive-runs")


# Tables that carry per-run state (deleted from main after archival).
_PER_RUN_TABLES = [
    "candidate_scores",
    "trades",
    "rotations",
    "live_state_snapshots",
    "pipeline_runs",   # LAST — FKs point to it from the others
]


def _count_rows_older_than(conn: sqlite3.Connection, cutoff_date: str) -> dict[str, int]:
    """Count rows eligible for archival, broken down by table."""
    counts = {}
    cur = conn.execute(
        "SELECT COUNT(*) FROM pipeline_runs WHERE run_date < ?", (cutoff_date,),
    )
    counts["pipeline_runs"] = cur.fetchone()[0]

    for table in ("candidate_scores", "trades", "rotations", "live_state_snapshots"):
        cur = conn.execute(
            f"""SELECT COUNT(*) FROM {table}
                 WHERE run_id IN (
                     SELECT run_id FROM pipeline_runs WHERE run_date < ?
                 )""",
            (cutoff_date,),
        )
        counts[table] = cur.fetchone()[0]
    return counts


def _ensure_archive_schema(archive_path: Path) -> None:
    """Open the archive DB and run ensure_schema on it."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
    from kernel.persistence import get_connection  # noqa: PLC0415
    conn = get_connection(
        {"persistence": {"enabled": True, "db_path": str(archive_path)}},
    )
    if conn is None:
        raise RuntimeError("Could not open archive DB")
    return conn


def _copy_old_rows(src: sqlite3.Connection, dst: sqlite3.Connection,
                   cutoff_date: str) -> dict[str, int]:
    """Copy rows older than cutoff from src into dst (without deleting).

    Returns number of rows copied per table.
    """
    copied: dict[str, int] = {}

    # Get list of run_ids to archive
    rows = src.execute(
        "SELECT run_id FROM pipeline_runs WHERE run_date < ?", (cutoff_date,),
    ).fetchall()
    run_ids = [r[0] for r in rows]

    if not run_ids:
        return copied

    # Copy by run_id batches
    BATCH = 500
    for table in ("pipeline_runs", "candidate_scores", "trades",
                  "rotations", "live_state_snapshots"):
        # Get column list
        pragma = src.execute(f"PRAGMA table_info({table})").fetchall()
        cols = [r[1] for r in pragma]
        col_list = ", ".join(cols)
        placeholders = ", ".join(["?"] * len(cols))
        total = 0
        for i in range(0, len(run_ids), BATCH):
            batch = run_ids[i:i + BATCH]
            if table == "pipeline_runs":
                q = f"SELECT {col_list} FROM {table} WHERE run_id IN ({','.join('?' * len(batch))})"
            else:
                q = f"SELECT {col_list} FROM {table} WHERE run_id IN ({','.join('?' * len(batch))})"
            rows_batch = src.execute(q, batch).fetchall()
            if rows_batch:
                dst.executemany(
                    f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
                    rows_batch,
                )
                total += len(rows_batch)
        copied[table] = total
    return copied


def _delete_old_rows(src: sqlite3.Connection, cutoff_date: str) -> dict[str, int]:
    deleted: dict[str, int] = {}
    # Delete FK-dependents FIRST, then pipeline_runs
    for table in ("candidate_scores", "trades", "rotations", "live_state_snapshots"):
        cur = src.execute(
            f"""DELETE FROM {table}
                 WHERE run_id IN (
                     SELECT run_id FROM pipeline_runs WHERE run_date < ?
                 )""",
            (cutoff_date,),
        )
        deleted[table] = cur.rowcount
    cur = src.execute("DELETE FROM pipeline_runs WHERE run_date < ?", (cutoff_date,))
    deleted["pipeline_runs"] = cur.rowcount
    return deleted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",  default="data/runs.db",
                   help="Main DB to archive FROM")
    p.add_argument("--archive", default="data/runs_archive.db",
                   help="Archive DB to move rows TO")
    p.add_argument("--older-than-days", type=int, default=730,
                   help="Rows with run_date older than this are archived (default 730 = 2yr)")
    p.add_argument("--execute", action="store_true",
                   help="Perform the move. Without this flag, script is dry-run.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip source DB backup (not recommended).")
    args = p.parse_args()

    src_path = REPO_ROOT / args.source
    arc_path = REPO_ROOT / args.archive
    if not src_path.exists():
        log.error("Source DB missing: %s", src_path)
        return 1

    cutoff = (datetime.date.today() - datetime.timedelta(days=args.older_than_days))
    cutoff_str = cutoff.isoformat()
    log.info("Cutoff date: %s (rows older than this will be archived)", cutoff_str)

    src_conn = sqlite3.connect(src_path, isolation_level=None)
    try:
        counts = _count_rows_older_than(src_conn, cutoff_str)
    finally:
        src_conn.close()

    log.info("Archive candidates:")
    total = 0
    for table, n in counts.items():
        log.info("  %-22s %8d rows", table, n)
        total += n

    if total == 0:
        log.info("Nothing to archive — all data is within %d days.", args.older_than_days)
        return 0

    if not args.execute:
        log.info("[dry-run] %d total rows would be archived.", total)
        log.info("[dry-run] Use --execute to actually perform the move.")
        return 0

    # Real run
    if not args.no_backup:
        backup = src_path.with_suffix(src_path.suffix + ".backup")
        log.info("Backing up %s → %s", src_path, backup)
        shutil.copy2(src_path, backup)

    log.info("Copying to archive DB ...")
    dst_conn = _ensure_archive_schema(arc_path)
    src_conn = sqlite3.connect(src_path, isolation_level=None)
    try:
        copied = _copy_old_rows(src_conn, dst_conn, cutoff_str)
        for table, n in copied.items():
            log.info("  copied %-22s %d rows", table, n)
        dst_conn.commit()

        log.info("Deleting from source ...")
        deleted = _delete_old_rows(src_conn, cutoff_str)
        for table, n in deleted.items():
            log.info("  deleted %-22s %d rows", table, n)
        src_conn.commit()

        # Reclaim space
        log.info("Running VACUUM on source ...")
        src_conn.execute("VACUUM;")
    finally:
        src_conn.close()
        dst_conn.close()

    log.info("Archive complete. Source: %s, Archive: %s", src_path, arc_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
