"""Regression guard: RunnerAdapter.commit() must pass `self._config` (not a bare
`config`) to save_live_state_atomic().

2026-06-26 P0 (live trading). Commit 64f6b46 (2026-06-16, "wire config flag
through save path") changed the live-state save call to::

    save_live_state_atomic(state_file, self._state, config)

but `config` is undefined in commit()'s scope — every other reference in the
method uses `self._config`. The argument is evaluated at the call site, so the
NameError fires on EVERY commit, AFTER orders are already placed at the broker —
the state file is never written. Live `last_activity_date` froze, and ~10 days
later the dormancy preflight (P-BROKER-FILL-FRESHNESS, 20-trading-day cap)
tripped and fail-closed the intraday cron. The bug ran in production undetected
because no test reaches commit()'s save section: the only existing commit() test
uses SimAdapter (a different commit path), and any test that DID reach line 1692
on the buggy code would itself NameError before save_live_state_atomic ran.

This guard pins the invariant by AST — no heavyweight RunnerAdapter context is
needed (commit() reads ~all of self/ctx). The save call's config argument must
be the attribute access ``self._config``, never a bare name.
"""
from __future__ import annotations

import ast
from pathlib import Path

RUNNER = (Path(__file__).resolve().parent.parent
          / "backtesting" / "renquant_104" / "adapters" / "runner.py")


def _commit_funcdef() -> ast.FunctionDef:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RunnerAdapter":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "commit":
                    return item
    raise AssertionError("RunnerAdapter.commit not found in adapters/runner.py")


def _save_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "save_live_state_atomic"]


def test_commit_saves_state_with_self_config_not_bare_config():
    calls = _save_calls(_commit_funcdef())
    assert len(calls) == 1, f"expected exactly one save_live_state_atomic call, got {len(calls)}"
    call = calls[0]
    assert len(call.args) >= 3, "save call must pass the config argument (3rd positional)"
    cfg = call.args[2]
    # The exact regression: a bare ast.Name('config') is the 2026-06-16 bug.
    assert not (isinstance(cfg, ast.Name) and cfg.id == "config"), (
        "save_live_state_atomic must NOT receive a bare `config` (undefined in "
        "commit() → NameError after orders placed); use self._config")
    assert (isinstance(cfg, ast.Attribute)
            and cfg.attr == "_config"
            and isinstance(cfg.value, ast.Name)
            and cfg.value.id == "self"), (
        "save_live_state_atomic's config argument must be `self._config`, "
        f"got: {ast.dump(cfg)}")
