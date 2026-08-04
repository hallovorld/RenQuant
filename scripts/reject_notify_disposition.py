#!/usr/bin/env python3
"""Classify the RFC#210 fallback verdict for the weekly reject notification.

Operator directive 2026-08-04: a WF-gate reject while the SERVED model is fresh
is the healthy steady state of the freshness governance (the prod recipe is
chronically placebo-dominated, so the gate rejects ~every run; RFC#210 bounds
staleness at 28d). That state kept being reported with a failure tone and a
failure exit, which reads as "the model is broken" when nothing needs doing.

This helper decides which tone the wrapper's reject branch uses. It prints ONE
line to stdout and always exits 0 (it is a disposition, not a gate — the CALLER
maps the category to notify tone + exit code):

    CALM_FRESH|<staleness_days>|<prod_trained>
        The verdict PROVES the healthy shape: decision is exactly "REFUSE",
        refused on exactly the prod_stale check, that check's ok is exactly
        False (bool), and it carries an int staleness_days <= its SLA plus a
        non-empty prod_trained date.

    ALARM|<reason>
        Everything else: missing/unreadable/malformed verdict, refusal on any
        other check, prod actually stale, unexpected decision value. Fail
        closed toward attention — an unproven "healthy" must alarm.

The verdict shape is the one freshness_fallback --stamp writes (see
20260804T200020Z.fallback_verdict.json for a live example): top-level
"decision"/"refused_on"/"checks", each check an object with "check"/"ok".
The test-harness stub writes {"verdict": ...} — a different shape, which lands
in ALARM by design (never guess on an unproven verdict).
"""
from __future__ import annotations

import json
import sys

SLA_DAYS = 28


def dispose(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        return f"ALARM|verdict unreadable ({exc.__class__.__name__})"
    if not isinstance(d, dict):
        return "ALARM|verdict is not an object"
    if d.get("decision") != "REFUSE":
        return f"ALARM|unexpected decision {d.get('decision')!r}"
    if d.get("refused_on") != "prod_stale":
        return f"ALARM|refused on {d.get('refused_on')!r}, not prod_stale"
    checks = d.get("checks")
    if not isinstance(checks, list):
        return "ALARM|checks missing"
    prod = next((c for c in checks if isinstance(c, dict)
                 and c.get("check") == "prod_stale"), None)
    if prod is None:
        return "ALARM|prod_stale check absent from checks"
    # Explicit-sentinel rule: ok must be the bool False, not merely falsy.
    if prod.get("ok") is not False:
        return f"ALARM|prod_stale.ok is {prod.get('ok')!r}, not False"
    age = prod.get("staleness_days")
    trained = prod.get("prod_trained")
    if not isinstance(age, int) or isinstance(age, bool):
        return f"ALARM|staleness_days is {age!r}, not an int"
    if age > SLA_DAYS:
        return f"ALARM|staleness_days {age} exceeds the {SLA_DAYS}d SLA"
    if not isinstance(trained, str) or not trained.strip():
        return f"ALARM|prod_trained is {trained!r}"
    return f"CALM_FRESH|{age}|{trained}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("ALARM|usage: reject_notify_disposition.py FALLBACK_JSON")
        return 0
    print(dispose(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
