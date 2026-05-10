#!/usr/bin/env python
"""CLI: replay live fills through the sim and emit a daily reconciliation report.

Usage:
    python scripts/reconcile_live_sim.py \\
        --broker alpaca \\
        --start-date 2026-05-09 --end-date 2026-05-09 \\
        --output reports/recon_2026-05-09.md

Defaults: broker=alpaca, both dates=yesterday (NY business day approx),
output=reports/recon_<end_date>.md.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.reconciliation import (  # noqa: E402
    compute_decision_divergence,
    compute_rolling_ic,
    compute_slippage,
    emit_report,
    load_live_fills,
    load_sim_decisions,
    replay_through_sim,
)
from kernel.reconciliation.live_sim_reconcile import (  # noqa: E402
    build_per_day_breakdown,
)


def _yesterday_iso() -> str:
    return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--broker", choices=["alpaca", "paper", "ibkr"], default="alpaca")
    p.add_argument("--start-date", default=_yesterday_iso())
    p.add_argument("--end-date",   default=_yesterday_iso())
    p.add_argument("--live-db", default=None,
                   help="Override live runs.<broker>.db path.")
    p.add_argument("--sim-db", default=None,
                   help="Override sim_runs.db path.")
    p.add_argument("--output", default=None,
                   help="Markdown output path. Defaults to "
                        "reports/recon_<end_date>.md.")
    return p.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    live_db = (Path(args.live_db) if args.live_db
               else REPO_ROOT / "data" / f"runs.{args.broker}.db")
    sim_db = (Path(args.sim_db) if args.sim_db
              else REPO_ROOT / "data" / "sim_runs.db")
    output = (Path(args.output) if args.output
              else REPO_ROOT / "reports" / f"recon_{args.end_date}.md")
    return live_db, sim_db, output


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    live_db, sim_db, output = _resolve_paths(args)

    if not live_db.exists():
        print(f"WARN: live DB not found at {live_db}; emitting empty report.")
        fills, sims = [], []
    else:
        fills = load_live_fills(live_db, args.start_date, args.end_date)
        sims  = load_sim_decisions(sim_db, args.start_date, args.end_date) \
                if sim_db.exists() else []

    matched = replay_through_sim(fills, sim_decisions=sims)
    metrics = {
        "broker":       args.broker,
        "start_date":   args.start_date,
        "end_date":     args.end_date,
        "n_fills":      len(fills),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "slippage":     compute_slippage(fills, matched),
        "divergence":   compute_decision_divergence(fills, matched),
        "per_day":      build_per_day_breakdown(fills, matched),
        "rolling_ic":   compute_rolling_ic(live_db, args.start_date, args.end_date),
    }
    written = emit_report(metrics, output)
    print(f"Wrote {written}  fills={len(fills)}  "
          f"divergence_rate={metrics['divergence']['divergence_rate']:.2%}  "
          f"slip_p95={metrics['slippage']['p95_bps']:+.2f}bps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
