#!/usr/bin/env python3
"""DRPH capture/verify CLI — build golden replay cases from persisted runs.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §IV + S2
item 5; core substrate: backtesting/renquant_104/kernel/drph.py.

Capture reads ONLY what persistence already wrote (pipeline_runs,
ticker_daily_state, live_state_snapshots) — zero runtime risk. The
decision snapshot is the per-ticker verdict surface (selected /
blocked_by / scores / sizing) plus the book-level flags and counters:
exactly the surface a refactor must reproduce bit-identically.

Usage:
  capture a golden case (e.g. the 2026-06-11 false-BEAR run):
    python scripts/drph_capture.py capture \
      --db data/runs.alpaca.db --run-id 2026-06-11-live-f68231b0 \
      --out tests/drph_corpus/2026-06-11_false_bear

  verify a later re-run of the same frozen day against the case:
    python scripts/drph_capture.py verify \
      --db data/sim_runs.db --run-id <replay-run-id> \
      --case tests/drph_corpus/2026-06-11_false_bear

  integrity-check the corpus (CI):
    python scripts/drph_capture.py check --case tests/drph_corpus/<case>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.drph import ReplayCase, canonical_json, sha  # noqa: E402

# The per-ticker decision surface. run_id deliberately excluded (volatile
# across replays of the same frozen day); column order is the canonical
# order of the snapshot.
TICKER_DECISION_COLS = [
    "ticker", "date", "regime", "confidence", "in_watchlist", "in_universe",
    "has_position", "position_qty", "model_action", "panel_score",
    "rank_score", "expected_return", "kelly_target_pct", "mu", "sigma",
    "in_candidates", "selected", "blocked_by", "sector", "qp_delta_w",
    "qp_target_w", "qp_status", "model_admission_ok",
    "model_admission_reason", "active_scorer",
]


def extract_decisions(conn: sqlite3.Connection, run_id: str) -> dict:
    """The canonical decision snapshot for one persisted run."""
    run = conn.execute(
        """SELECT run_date, run_type, regime, confidence, buy_blocked,
                  skip_buys, bear_only, n_candidates, n_exits, n_buys,
                  counters_json
             FROM pipeline_runs WHERE run_id = ?""", (run_id,)).fetchone()
    if run is None:
        raise SystemExit(f"run_id not found in pipeline_runs: {run_id!r}")
    (run_date, run_type, regime, confidence, buy_blocked, skip_buys,
     bear_only, n_candidates, n_exits, n_buys, counters_json) = run

    # Schema-vintage resilience: older DDLs (fresh sim dbs) lack columns
    # that live dbs gained via migration. The snapshot SURFACE is fixed
    # (TICKER_DECISION_COLS); columns absent from this db read as None,
    # so capture and verify stay comparable across db vintages.
    present = {r[1] for r in conn.execute("PRAGMA table_info(ticker_daily_state)")}
    cols = ", ".join(
        c if c in present else f"NULL AS {c}" for c in TICKER_DECISION_COLS)
    rows = conn.execute(
        f"SELECT {cols} FROM ticker_daily_state WHERE run_id = ? "
        f"ORDER BY ticker", (run_id,)).fetchall()
    tickers = [dict(zip(TICKER_DECISION_COLS, r)) for r in rows]

    return {
        "book": {
            "run_date": run_date,
            "run_type": run_type,
            "regime": regime,
            "confidence": confidence,
            "buy_blocked": buy_blocked,
            "skip_buys": skip_buys,
            "bear_only": bear_only,
            "n_candidates": n_candidates,
            "n_exits": n_exits,
            "n_buys": n_buys,
            "counters": json.loads(counters_json) if counters_json else {},
        },
        "tickers": tickers,
    }


def extract_inputs(conn: sqlite3.Connection, run_id: str) -> dict:
    """Frozen-input payloads: provenance bundle + state snapshot."""
    inputs: dict = {}
    bundle = conn.execute(
        "SELECT run_bundle_json FROM pipeline_runs WHERE run_id = ?",
        (run_id,)).fetchone()
    if bundle and bundle[0]:
        inputs["run_bundle"] = json.loads(bundle[0])
    snap = conn.execute(
        "SELECT state_json FROM live_state_snapshots WHERE run_id = ?",
        (run_id,)).fetchone()
    if snap and snap[0]:
        inputs["live_state"] = json.loads(snap[0])
    if not inputs:
        raise SystemExit(
            f"no run_bundle/live_state persisted for {run_id!r} — refusing "
            f"to freeze a case without provenance (no-silent-continue)")
    return inputs


def cmd_capture(args) -> int:
    conn = sqlite3.connect(args.db)
    decisions = extract_decisions(conn, args.run_id)
    inputs = extract_inputs(conn, args.run_id)
    inputs["capture_meta"] = {
        "source_db": str(args.db),
        "source_run_id": args.run_id,
        "decision_cols": TICKER_DECISION_COLS,
    }
    case = ReplayCase(Path(args.out))
    case_id = case.write(inputs=inputs, expected_decisions=decisions)
    print(f"captured case id={case_id} → {args.out}")
    print(f"  tickers={len(decisions['tickers'])} "
          f"regime={decisions['book']['regime']} "
          f"decisions_sha={sha(canonical_json(decisions))}")
    return 0


def cmd_verify(args) -> int:
    case = ReplayCase(Path(args.case))
    problems = case.check_integrity()
    if problems:
        print("CORPUS INTEGRITY FAILED (fix the corpus before trusting "
              "any verify):")
        for p in problems:
            print(f"  ✗ {p}")
        return 2
    conn = sqlite3.connect(args.db)
    actual = extract_decisions(conn, args.run_id)
    ok, diffs = case.verify(actual)
    if ok:
        print(f"PARITY OK — {args.run_id} reproduces {args.case} "
              f"bit-identically")
        return 0
    print(f"PARITY FAILED — {len(diffs)} diverging path(s) (first 20):")
    for d in diffs:
        print(f"  ✗ {d}")
    return 1


def cmd_check(args) -> int:
    case = ReplayCase(Path(args.case))
    problems = case.check_integrity()
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        return 2
    print(f"corpus integrity OK: {args.case}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="freeze a persisted run as a golden case")
    cap.add_argument("--db", required=True)
    cap.add_argument("--run-id", required=True)
    cap.add_argument("--out", required=True)
    cap.set_defaults(fn=cmd_capture)
    ver = sub.add_parser("verify", help="byte-compare a re-run against a case")
    ver.add_argument("--db", required=True)
    ver.add_argument("--run-id", required=True)
    ver.add_argument("--case", required=True)
    ver.set_defaults(fn=cmd_verify)
    chk = sub.add_parser("check", help="corpus integrity check (CI)")
    chk.add_argument("--case", required=True)
    chk.set_defaults(fn=cmd_check)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
