"""Contract test — every site that appends to ctx.orders MUST honor
ctx.buy_blocked / ctx.skip_buys (or be the QP path that handles
cross-asset suppression in EmitOrdersFromQPSolutionTask).

This catches the bug class introduced as bugs 3 + 4 in the wl183
incident: a new emit path is added but its author forgets to gate on
buy_blocked. Past examples:
  - QP top-ups during DrawdownGate firing (bug 3, 2026-05-05)
  - QP top-ups within ±N days of earnings (bug 4, 2026-05-05)

The test runs at source level — fast, no sim required, catches
regressions before they ship.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "backtesting" / "renquant_104" / "kernel"

# Files that emit BUY orders (append to ctx.orders).
EMIT_FILES = [
    KERNEL / "pipeline" / "task_topup.py",       # TopUpHeldTask
    KERNEL / "pipeline" / "task_benchmark_sleeve.py",  # BenchmarkSleeveTask
    KERNEL / "pipeline" / "task_selection.py",   # SizeAndEmitTask (greedy)
    KERNEL / "pipeline" / "task_rotation.py",    # EmitRotationsTask (rotation buys)
    KERNEL / "pipeline" / "task_joint_actions.py",  # JointActionTask (greedy QP fallback)
    KERNEL / "portfolio_qp" / "tasks.py",        # EmitOrdersFromQPSolutionTask (QP)
]


def _emit_class_bodies(path: Path) -> dict[str, str]:
    """Return {class_name: source_body_str} for classes containing
    `ctx.orders.append`."""
    src = path.read_text()
    bodies = {}
    # Find class boundaries
    class_starts = [(m.start(), m.group(1))
                    for m in re.finditer(r"^class (\w+)", src, re.M)]
    for i, (start, name) in enumerate(class_starts):
        end = class_starts[i + 1][0] if i + 1 < len(class_starts) else len(src)
        body = src[start:end]
        if "ctx.orders.append" in body:
            bodies[name] = body
    return bodies


def test_every_buy_emitter_checks_buy_blocked_or_skip_buys():
    """Each class that emits BUY orders must either:
      (a) check buy_blocked OR skip_buys explicitly, OR
      (b) be implicitly gated (cls inherits from a Task that's only
          run when those flags are False — verifiable in source)
    """
    violations = []
    for path in EMIT_FILES:
        if not path.exists():
            continue
        for class_name, body in _emit_class_bodies(path).items():
            has_explicit_gate = "buy_blocked" in body and "skip_buys" in body
            if not has_explicit_gate:
                violations.append((path.name, class_name))

    assert not violations, (
        f"Unexpected buy-emit class lacks both buy_blocked and skip_buys "
        f"check: {violations}. Add the gate before the emitter can ship."
    )


def test_qp_emit_task_has_both_gate_and_earnings_check():
    """Specific assertions for the QP path (production-active).
    Pin the bug-3 + bug-4 fixes in source so they can't silently
    regress.

    2026-05-06 §1c split moved the earnings-check call out of the class
    body into the `_gate_buy_or_block` helper. The Task now invokes
    that helper, so we check the WHOLE module — earnings logic is still
    load-bearing, just refactored.
    """
    src = (KERNEL / "portfolio_qp" / "tasks.py").read_text()
    # Class body still owns the orchestration + counter names.
    idx = src.find("class EmitOrdersFromQPSolutionTask")
    next_class = src.find("class ", idx + 1)
    body = src[idx:next_class] if next_class > 0 else src[idx:]
    assert "buy_blocked" in body
    assert "skip_buys" in body
    assert "buys_gated" in body
    # Earnings gate now lives in the _gate_buy_or_block helper (module
    # scope) — the class body delegates to it. Check whole-module text:
    assert "is_earnings_blocked" in src, (
        "Earnings blackout helper missing from portfolio_qp.tasks — "
        "Bug 4 (wl183 2026-05-05) regressed."
    )
    assert "_gate_buy_or_block" in body, (
        "Class body must delegate to _gate_buy_or_block helper — "
        "otherwise the earnings/bear_only/buys_gated checks aren't wired."
    )
