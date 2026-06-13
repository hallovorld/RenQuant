#!/usr/bin/env python3
"""Clock/timezone lint — ratchet on naive time sources in session paths.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §III.4 +
the P0.3 session-clock work (live/clock.py). The box runs
America/Los_Angeles; the market runs America/New_York. A naive local
clock used for TRADING-day semantics rolls at midnight PT, not with the
exchange (the trading-date bugs; next DST transition 2026-11-01).

P0.3 fixed the live-path offenders; this lint RATCHETS the count so new
naive time sources can't creep back in. AST-based (no comment/string
false-positives). Flags:
  datetime.now()      naive local datetime
  date.today()        naive local date
  datetime.utcnow()   naive UTC (deprecated)
NOT flagged: time.time() (epoch, tz-agnostic) and any *.now(tz=...) /
datetime.now(NY) call that passes a tzinfo (already timezone-aware).

Scopes are session-critical only: live/ and the umbrella adapters. The
clock authority module (live/clock.py) is exempt — it is where aware
time lives.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCOPES = [REPO / "live", REPO / "backtesting" / "renquant_104" / "adapters"]
EXEMPT = {REPO / "live" / "clock.py"}
RATCHET_FILE = REPO / "scripts" / "clock_ratchet.json"


def _is_naive_call(node: ast.AST) -> str | None:
    """Return a label if `node` is a naive time call, else None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    # datetime.now() / date.today() / datetime.utcnow() — attribute calls
    if isinstance(f, ast.Attribute):
        attr = f.attr
        if attr == "utcnow":
            return "datetime.utcnow()"
        if attr == "now":
            # now(tz=...) or now(SOMETZ) is aware → not flagged
            if node.args or any(kw.arg == "tz" for kw in node.keywords):
                return None
            return "datetime.now()"
        if attr == "today":
            return "date.today()"
    return None


def scan_file(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return []
    hits = []
    for node in ast.walk(tree):
        label = _is_naive_call(node)
        if label is not None:
            hits.append((node.lineno, label))
    return hits


def scan(scopes=SCOPES, exempt=EXEMPT) -> list[dict]:
    out = []
    for scope in scopes:
        if not scope.exists():
            continue
        for py in sorted(scope.rglob("*.py")):
            if py in exempt or "test" in py.name:
                continue
            for lineno, label in scan_file(py):
                out.append({"file": str(py.relative_to(REPO)),
                            "line": lineno, "kind": label})
    return out


def main() -> int:
    hits = scan()
    ratchet = json.loads(RATCHET_FILE.read_text())
    cap = int(ratchet["max_naive_time_sources"])
    n = len(hits)
    print(f"naive time sources in session paths: {n} (ratchet cap {cap})")
    for h in hits:
        print(f"  {h['file']}:{h['line']}  {h['kind']}")
    if n > cap:
        print(f"\nFAIL: {n} > {cap} — new naive time source(s) added. Use "
              f"live.clock.trading_date()/ny_now() for trading-day semantics.")
        return 1
    if n < cap:
        print(f"\nNOTE: {n} < {cap} — naive sources were migrated; tighten "
              f"the ratchet to {n} in {RATCHET_FILE.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
