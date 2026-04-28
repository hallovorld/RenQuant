#!/usr/bin/env python
"""A/B sim: panel_scoring.bypass_ticker_gate=false vs true.

Why
---
The 2026-04-27 incident showed that even after the NGBoost feature drift
was fixed, NVDA/AMD remained un-investable because their per-ticker
tournament models (NVDA: QLearning, AMD: Manual rules) emit "hold". With
the default `bypass_ticker_gate=false`, that veto blocks Panel-LTR from
ever scoring them as buy candidates.

If `bypass_ticker_gate=true`, Panel-LTR (which is global / cross-sectional)
ranks all admissible-universe tickers regardless of per-ticker tournament
output. Quality is still enforced downstream by Gate B (edge_sharpe ≥ τ),
selection-loop tiered thresholds, and Kelly sizing.

This script measures the APY impact in sim before flipping the live flag.

Usage
-----
    python scripts/ab_bypass_ticker_gate.py
    python scripts/ab_bypass_ticker_gate.py --strategy renquant_104

Decision rule (per `feedback_sharpe_floor` + `feedback_golden_tracks_best`):
  - Promote bypass=true ONLY if APY delta ≥ +2 pts on 27-mo OOS AND
    portfolio-level Sharpe stays ≥ 1.0.
  - bypass=true is mechanism-clean (single-flag toggle, no hyperparameter
    drift, no panel retrain), so any positive non-trivial margin should
    be carefully evaluated even below +2 pts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ab_harness import run_ab  # noqa: PLC0415

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ab-bypass-ticker-gate")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    args = p.parse_args()

    def set_bypass_true(c: dict) -> None:
        c.setdefault("ranking", {}).setdefault("panel_scoring", {})["bypass_ticker_gate"] = True

    variants = [
        ("A_GOLDEN_bypass_false", lambda c: None),
        ("B_bypass_ticker_gate_true", set_bypass_true),
    ]

    log.info("Running A/B: bypass_ticker_gate false vs true on %s", args.strategy)
    results = run_ab(
        strategy=args.strategy,
        variants=variants,
        initial_cash=args.initial_cash,
    )

    print()
    print("══ A/B Results — bypass_ticker_gate ══")
    print(f"{'label':40s}  {'APY':>8s}  {'final':>10s}  {'sharpe':>8s}  {'buys':>5s}  {'sells':>5s}  {'rotations':>9s}")
    for r in results:
        sharpe = r.get("sharpe", float("nan"))
        print(f"{r['label']:40s}  {r['apy']:>+8.2f}  {r['final']:>10.0f}  {sharpe:>+8.3f}  "
              f"{r.get('buys',0):>5d}  {r.get('sells',0):>5d}  {r.get('rotations',0):>9d}")

    if len(results) == 2:
        a, b = results
        delta_apy = b["apy"] - a["apy"]
        delta_sharpe = b.get("sharpe", float("nan")) - a.get("sharpe", float("nan"))
        delta_buys = b.get("buys", 0) - a.get("buys", 0)
        print()
        print(f"  Δ APY    = {delta_apy:+.2f} pts")
        print(f"  Δ Sharpe = {delta_sharpe:+.3f}")
        print(f"  Δ buys   = {delta_buys:+d}  (we expect bypass=true to admit MORE buy candidates)")
        promote = (delta_apy >= 2.0) and (b.get("sharpe", 0.0) >= 1.0)
        print(f"  decision = {'PROMOTE' if promote else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
