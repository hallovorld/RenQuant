#!/usr/bin/env python
"""Verify that daily's preflight will let buys through, given current artifact + config.

Runs the same ``run_preflight`` invocation that ``live.runner`` does on the
``full``/``buy`` run_mode, and reports which gates pass/fail. Use after any of:

  * stamping wf_gate_metadata onto the prod GBDT artifact
  * stamping P-PANEL-CONTRACT fields
  * editing strategy_config.json::wf_gate flags

Exit code:
  0 → no hard fails → daily would create orders (subject to live broker checks)
  1 → at least one hard fail → daily would BUY-BLOCK
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy-config", type=Path,
                    default=STRATEGY_DIR / "strategy_config.json")
    ap.add_argument("--run-mode", default="full",
                    choices=["full", "sell_only", "shadow"])
    ap.add_argument("--quiet", action="store_true",
                    help="Print only the summary line, not per-check detail.")
    args = ap.parse_args()

    sys.path.insert(0, str(STRATEGY_DIR))
    from kernel import preflight  # noqa: PLC0415

    config = json.loads(args.strategy_config.read_text())
    try:
        checks = preflight.run_preflight(
            config,
            strategy_dir=STRATEGY_DIR,
            run_mode=args.run_mode,
        )
    except preflight.PreflightFailed as exc:
        # The preflight raised; that's the same behaviour live.runner sees
        print(f"PREFLIGHT-FAIL (hard): {exc}")
        return 1

    hard_fails = [c for c in checks if c.severity == "hard" and not c.ok]
    soft_fails = [c for c in checks if c.severity == "soft" and not c.ok]
    if not args.quiet:
        for c in checks:
            status = "✓" if c.ok else "✗"
            print(f"  {status} {c.name:25s} [{c.severity}] {c.message[:100]}")
    print(
        f"SUMMARY: {len(checks)} checks; {len(hard_fails)} hard FAIL; "
        f"{len(soft_fails)} soft FAIL"
    )
    if hard_fails:
        print("HARD FAILS (would BUY-BLOCK):")
        for c in hard_fails:
            print(f"  ✗ {c.name}: {c.message[:200]}")
        return 1
    print("OK — daily would create orders (subject to broker live checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
