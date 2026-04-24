#!/usr/bin/env python
"""Flag-drift detection — guard against default-off flags quietly becoming on.

The 2026-04-24 AB-trim incident exposed this class of bug: a flag was
added with "default off" semantics in code, but the strategy_config.json
was shipped with the flag true. Result: golden APY silently regressed
12.7 pts before A/B caught it.

This script compares two configs and flags any boolean that flipped from
false → true (or absent → true), or numeric field that shifted by ≥ N%
(default 10%). Print diff + exit 1 if drift found.

Default comparison: `strategy_config.json` (live) vs
`strategy_config.golden.json` (frozen baseline). Use as a pre-commit
hook OR a nightly sanity check.

Usage::

    python scripts/check_config_drift.py
    python scripts/check_config_drift.py --strategy renquant_104
    python scripts/check_config_drift.py --numeric-tolerance 0.05
    python scripts/check_config_drift.py --ignore-path ranking.kelly_sizing.enabled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _walk(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into `{dotted.path: value}`. Lists stay as values."""
    flat: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_walk(v, path))
            else:
                flat[path] = v
    return flat


def _bool_drift(baseline: bool | None, live: bool | None) -> str | None:
    """Return a drift description or None."""
    if baseline == live:
        return None
    if baseline is False and live is True:
        return "false → true (flag quietly enabled!)"
    if baseline is None and live is True:
        return "(absent) → true (new flag enabled)"
    return f"{baseline} → {live}"


def _numeric_drift(
    baseline: float | None,
    live: float | None,
    tolerance: float,
) -> str | None:
    if baseline is None or live is None:
        return None
    if baseline == live:
        return None
    if baseline == 0:
        # Avoid div by zero — flag any non-zero change
        return f"{baseline} → {live}" if live != 0 else None
    pct = abs((live - baseline) / baseline)
    if pct >= tolerance:
        return f"{baseline} → {live}  ({pct*100:+.1f}%)"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--numeric-tolerance", type=float, default=0.10,
                   help="Flag numeric changes ≥ this fraction (default 0.10 = 10%%)")
    p.add_argument("--ignore-path", action="append", default=[],
                   help="Dotted config path to skip (can repeat)")
    # Paths that legitimately change between runs (auto-written by daily
    # recalibration or training), so they shouldn't trigger drift alerts.
    p.add_argument("--no-default-ignores", action="store_true",
                   help="Disable the built-in ignore list (diagnostic only)")
    p.add_argument("--baseline", default="strategy_config.golden.json")
    p.add_argument("--live",     default="strategy_config.json")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    baseline_path = strategy_dir / args.baseline
    live_path     = strategy_dir / args.live

    if not baseline_path.exists():
        print(f"ERROR: baseline missing: {baseline_path}", file=sys.stderr)
        return 1
    if not live_path.exists():
        print(f"ERROR: live missing: {live_path}", file=sys.stderr)
        return 1

    baseline = json.loads(baseline_path.read_text())
    live     = json.loads(live_path.read_text())

    b_flat = _walk(baseline)
    l_flat = _walk(live)

    # Default ignore list — fields auto-written by recalibrate_scores.py /
    # panel trainer / ngboost trainer. These shift daily and aren't real
    # configuration drift.
    DEFAULT_IGNORES = {
        "ranking.blend_n_symbols",       # recalibrate_scores.py (symbol count)
        "ranking.blend_weights.rank",    # recalibrate_scores.py (weight fit)
        "ranking.blend_weights.rs",      # recalibrate_scores.py (weight fit)
    }
    ignored = set(args.ignore_path)
    if not args.no_default_ignores:
        ignored |= DEFAULT_IGNORES
    all_keys = sorted(set(b_flat) | set(l_flat))

    bool_drifts: list[tuple[str, str]] = []
    num_drifts:  list[tuple[str, str]] = []
    for key in all_keys:
        if key in ignored:
            continue
        b = b_flat.get(key)
        l = l_flat.get(key)
        if isinstance(b, bool) or isinstance(l, bool):
            d = _bool_drift(b, l)
            if d:
                bool_drifts.append((key, d))
        elif isinstance(b, (int, float)) and isinstance(l, (int, float)):
            d = _numeric_drift(float(b), float(l), args.numeric_tolerance)
            if d:
                num_drifts.append((key, d))

    if not bool_drifts and not num_drifts:
        print(f"✅ Config drift check OK — {args.live} matches {args.baseline}.")
        return 0

    print(f"⚠️  Config drift detected in {live_path.name} vs {baseline_path.name}:\n")
    if bool_drifts:
        print("  Boolean flag changes:")
        for key, desc in bool_drifts:
            print(f"    {key}: {desc}")
        print()
    if num_drifts:
        print(f"  Numeric changes (tolerance {args.numeric_tolerance*100:.0f}%):")
        for key, desc in num_drifts:
            print(f"    {key}: {desc}")
        print()
    print("If these changes are intentional:")
    print("  1. Update strategy_config.golden.json (promote) OR")
    print("  2. Revert strategy_config.json if drift is unintended")
    return 1


if __name__ == "__main__":
    sys.exit(main())
