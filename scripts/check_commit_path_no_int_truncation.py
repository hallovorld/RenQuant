#!/usr/bin/env python
"""S-FRAC stage 0 truncation audit — no int() cast on fill quantities.

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
§2.3 ("Truncation audit"): a static check asserting that no ``int(...)``
cast is applied to fill quantities anywhere on the RunnerAdapter commit
path — the modules the design enumerates (runner.py, runner_execmath.py,
runner_ext_sell.py, broker_sync.py) plus the stop-emission module
(z9_stops.py) and the contract module itself (commit_contract.py) —
unless it is the ONE sanctioned whole-share branch
(``commit_contract.normalize_fill_qty``, which snaps an eps-integral fill
to int for byte-identical flag-off behavior).

Why static AST, not grep: the v1 blocker was a single expression —
``int(execution["filled_qty"] or shares)`` — that silently turned a
0.435578 broker fill into 0 shares in orders_placed, live_state, the
trade journal, cash accounting, and the Z9 stop quantity. Any future
reintroduction under a different spelling (``int(fill["qty"])``,
``int(sell_qty)``, ...) must fail this audit, not wait for a live no-buy
forensic. Same check-script pattern as the orchestrator's
``scripts/check_model_bundle_consistency.py``.

Exit codes: 0 = clean, 1 = violation(s) found, 2 = audit could not run.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = REPO_ROOT / "backtesting" / "renquant_104" / "adapters"

# The commit-path modules the design §2.3 enumerates, plus the stop
# emission + contract modules that carry quantities on the same path.
AUDITED_MODULES = (
    ADAPTERS / "runner.py",
    ADAPTERS / "runner_execmath.py",
    ADAPTERS / "runner_ext_sell.py",
    ADAPTERS / "broker_sync.py",
    ADAPTERS / "z9_stops.py",
    ADAPTERS / "commit_contract.py",
)

# Identifiers / dict keys / attribute names that denote a fill-derived
# quantity. An int() call whose argument subtree mentions any of these is
# a truncation of a fill quantity.
FILL_QTY_TOKENS = frozenset({
    "filled_qty",
    "fill_qty",
    "sell_qty",
    "shares_sold",
    "held_now",
    "qty",
    "qty_avail",
    "qty_available",
})

# The sanctioned whole-share branches (§2.2.1), both eps-guarded by
# is_integral_qty/round so a fractional quantity can never reach their
# int() cast:
#   * normalize_fill_qty — snaps an eps-integral fill to int so flag-off
#     whole-share behavior is byte-identical to the killed legacy cast;
#   * fmt_qty — display-only: renders an eps-integral qty as "5" (legacy
#     %d/%.0f parity) and a fractional qty verbatim.
# Nothing else may int() a fill quantity.
ALLOWED_FUNCTIONS = frozenset({"normalize_fill_qty", "fmt_qty"})


def _subtree_tokens(node: ast.AST) -> set[str]:
    """Collect identifiers, attribute names, and string keys in a subtree."""
    tokens: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            tokens.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            tokens.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            tokens.add(sub.value)
        elif isinstance(sub, ast.keyword) and sub.arg:
            tokens.add(sub.arg)
    return tokens


class _IntCastAuditor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.func_stack: list[str] = []
        self.violations: list[tuple[int, str, str]] = []

    def _visit_func(self, node: ast.AST) -> None:
        self.func_stack.append(getattr(node, "name", "<lambda>"))
        self.generic_visit(node)
        self.func_stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "int":
            hits = set()
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                hits |= _subtree_tokens(arg) & FILL_QTY_TOKENS
            if hits and not (set(self.func_stack) & ALLOWED_FUNCTIONS):
                snippet = ast.get_source_segment(self.source, node) or "int(...)"
                self.violations.append(
                    (node.lineno, ",".join(sorted(hits)), snippet.strip()),
                )
        self.generic_visit(node)


def audit_module(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    auditor = _IntCastAuditor(path, source)
    auditor.visit(tree)
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:  # audited path outside the repo (test fixtures)
        rel = path
    return [
        f"{rel}:{lineno}: int() cast on fill "
        f"quantity ({tokens}): {snippet}"
        for lineno, tokens, snippet in auditor.violations
    ]


def run_audit(paths: tuple[Path, ...] = AUDITED_MODULES) -> list[str]:
    violations: list[str] = []
    for path in paths:
        if not path.exists():
            violations.append(f"{path}: audited commit-path module MISSING")
            continue
        violations.extend(audit_module(path))
    return violations


def main() -> int:
    try:
        violations = run_audit()
    except SyntaxError as exc:  # pragma: no cover — unparseable source
        print(f"TRUNCATION-AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    if violations:
        print(
            "TRUNCATION-AUDIT FAIL — int() casts on fill quantities on the "
            "commit path (S-FRAC stage 0, design §2.3):",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "Fill quantities are broker-authoritative floats; the only "
            "sanctioned whole-share branch is "
            "adapters/commit_contract.py::normalize_fill_qty.",
            file=sys.stderr,
        )
        return 1
    print(
        f"TRUNCATION-AUDIT OK — {len(AUDITED_MODULES)} commit-path modules, "
        "no int() cast on fill quantities.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
