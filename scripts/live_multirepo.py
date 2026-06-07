#!/usr/bin/env python
"""Compatibility entrypoint for the orchestrator-owned live bridge."""
from __future__ import annotations

import importlib  # noqa: F401 - tests monkeypatch this module for fail-closed import checks.
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from orchestrator_bridge_bootstrap import resolve_orchestrator_src  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
LOCK_FILE = REPO / "subrepos.lock.json"
ORCH_SRC = resolve_orchestrator_src(REPO, SIBLINGS)
if str(ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(ORCH_SRC))

from renquant_orchestrator import live_bridge as _bridge  # noqa: E402


_PIN_SRCS = list(_bridge.DEFAULT_PIN_SRCS)


def _arg_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    return _bridge._arg_value(argv, flag, default)


def _without_arg(argv: list[str], flag: str) -> list[str]:
    return _bridge._without_arg(argv, flag)


def _strategy_config_name(argv: list[str]) -> str:
    return _bridge._strategy_config_name(argv)


def _with_pinned_strategy_config(argv: list[str]) -> list[str]:
    return _bridge._with_pinned_strategy_config(argv, repo_root=REPO)


def _subrepo_src_roots() -> tuple[list[Path], list[str]]:
    return _bridge._subrepo_src_roots(
        repo_root=REPO,
        lock_file=LOCK_FILE,
        siblings=SIBLINGS,
        pin_srcs=_PIN_SRCS,
    )


def _force_alias(alias: str, target: str, aliased: list[str]) -> None:
    return _bridge._force_alias(alias, target, aliased)


def _bootstrap_multirepo() -> list[str]:
    return _bridge.bootstrap_multirepo(
        repo_root=REPO,
        lock_file=LOCK_FILE,
        siblings=SIBLINGS,
        pin_srcs=_PIN_SRCS,
    )


def main() -> int:
    return _bridge.main(mode="live", repo_root=REPO)


if __name__ == "__main__":
    raise SystemExit(main())
