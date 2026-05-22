#!/usr/bin/env python3
"""Repair/check decision-trace invariants in RenQuant SQLite DBs.

Invariant:
    selected = 1 => blocked_by IS NULL

This is a data-hygiene repair for historical rows written before the
2026-05-21 persistence fix. It never deletes rows; it only clears stale
blocker labels from rows that represent executed/selected outcomes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DECISION_TABLES = ("candidate_scores", "ticker_daily_state")


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def count_selected_blockers(conn: sqlite3.Connection) -> dict[str, int]:
    """Return selected+blocked violation counts by decision table."""
    out: dict[str, int] = {}
    for table in DECISION_TABLES:
        if not _has_table(conn, table):
            out[table] = 0
            continue
        row = conn.execute(
            f"""SELECT COUNT(*)
                FROM {table}
                WHERE selected = 1 AND blocked_by IS NOT NULL"""
        ).fetchone()
        out[table] = int(row[0] if row else 0)
    return out


def clear_selected_blockers(conn: sqlite3.Connection) -> dict[str, int]:
    """Clear stale blocker labels and return rows updated by table."""
    updated: dict[str, int] = {}
    for table in DECISION_TABLES:
        if not _has_table(conn, table):
            updated[table] = 0
            continue
        cur = conn.execute(
            f"""UPDATE {table}
                SET blocked_by = NULL
                WHERE selected = 1 AND blocked_by IS NOT NULL"""
        )
        updated[table] = int(cur.rowcount if cur.rowcount is not None else 0)
    return updated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("db", nargs="?", default="data/sim_runs.db",
                   help="SQLite DB path. Default: data/sim_runs.db")
    p.add_argument("--apply", action="store_true",
                   help="Apply the repair. Without this, only reports counts.")
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if violations remain after the run.")
    p.add_argument("--timeout-sec", type=float, default=30.0,
                   help="SQLite lock timeout in seconds. Default: 30.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.db)
    if not path.exists():
        raise FileNotFoundError(path)

    conn = sqlite3.connect(path, timeout=float(args.timeout_sec))
    try:
        before = count_selected_blockers(conn)
        updated = {table: 0 for table in DECISION_TABLES}
        if args.apply:
            updated = clear_selected_blockers(conn)
            conn.commit()
        after = count_selected_blockers(conn)
    finally:
        conn.close()

    payload = {
        "db": str(path),
        "applied": bool(args.apply),
        "before": before,
        "updated": updated,
        "after": after,
        "remaining": int(sum(after.values())),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if args.check and payload["remaining"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
