#!/usr/bin/env python
"""Analyze execution-level score quality from round-trip CSV output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.trade_score_diagnostics import (  # noqa: E402
    compute_score_diagnostics,
    render_markdown,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-trips-csv", required=True,
                   help="CSV produced by scripts/run_sim_104.py --round-trips-csv")
    p.add_argument("--outcome-col", default="pnl_pct",
                   help="Outcome column to evaluate against, default pnl_pct")
    p.add_argument("--include-open", action="store_true",
                   help="Include open lots; default evaluates closed trips only")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    args = p.parse_args()

    path = Path(args.round_trips_csv)
    df = pd.read_csv(path)
    payload = compute_score_diagnostics(
        df,
        outcome_col=args.outcome_col,
        closed_only=not args.include_open,
    )
    payload["source"] = str(path)
    text = render_markdown(payload)

    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()

