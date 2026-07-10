"""Test-support: resolve the renquant-execution-owned cash-cap delegate.

The cash-cap sizing math is OWNED by renquant-execution
``order_math.cap_affordable_qty`` (execution#25 — ownership moved there per
the RenQuant#454 review); the umbrella ``adapters/runner_execmath.py::
cap_buy_order_to_cash`` is a time-bounded compatibility call-site that
delegates to it and fails closed to the legacy whole-share truncation when
the pinned renquant-execution predates ``order_math``.

This helper makes the sibling pinned checkouts importable
(``subrepos.lock.json`` ``local_path`` — the same resolution pattern as
``test_live_multirepo_entrypoints``) and returns the owner function, or
``None`` when unavailable, so tests can force EITHER wiring
deterministically (monkeypatch the module global): delegate-dependent tests
skip on an older pin; the fail-closed fallback tests always run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _inject_sibling_src_paths() -> None:
    lock = json.loads((REPO / "subrepos.lock.json").read_text())
    # renquant-common: renquant_execution's package __init__ imports it.
    for name in ("renquant-execution", "renquant-common"):
        entry = next((e for e in lock["subrepos"] if e["name"] == name), None)
        if not entry:
            continue
        src = Path(entry["local_path"]) / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.append(str(src))


def owner_cap_affordable_qty():
    """The owner implementation, or None when the pin predates execution#25."""
    _inject_sibling_src_paths()
    try:
        from renquant_execution.order_math import cap_affordable_qty
    except ImportError:
        return None
    return cap_affordable_qty
