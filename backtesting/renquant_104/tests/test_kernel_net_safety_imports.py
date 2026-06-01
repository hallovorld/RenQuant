"""Guards against the renquant_common.net_safety lift regressing in the
strategy-104 kernel modules.

Ports the test pattern from renquant-base-data PR #2:
`tests/test_net_safety_imports.py`. Catches the class of miss the
reviewer flagged on PR #26: top-level docstring/import sweeps fix the
visible call sites, but lazy `from .net_safety import ...` lines buried
in function bodies get left behind and point at a module that no
longer exists (now in renquant-common).

Two layers:

1. AST scan — fails fast if any kernel module body contains a relative
   import of `.net_safety`. Pure static check; no network.
2. Live import — imports each kernel module that historically used
   net_safety, confirming no ModuleNotFoundError at top level.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

# Strategy 104 kernel modules that historically called net_safety.
# Updated when new kernel.* modules learn to use the network.
KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernel"

# Each member is a module FILE in the kernel dir (no .py extension).
MODULES_TO_SCAN = [
    "fundamentals",
    "earnings_surprise",
    "insider_trades",
    "data",
    "data_cache",
]


def _module_path(name: str) -> Path:
    return KERNEL_DIR / f"{name}.py"


@pytest.mark.parametrize("module_name", MODULES_TO_SCAN)
def test_no_relative_net_safety_import(module_name: str) -> None:
    """No `from .net_safety import ...` anywhere — must be
    `from renquant_common.net_safety import ...`. Catches the lazy-
    import-in-function-body miss class.
    """
    path = _module_path(module_name)
    if not path.exists():
        pytest.skip(f"kernel module not present: {module_name}")
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1 and (node.module or "").endswith("net_safety"):
                names = ", ".join(alias.name for alias in node.names)
                offenders.append(
                    f"{path.name}:{node.lineno}: from .{node.module} import {names}"
                )
    assert not offenders, (
        f"Stale relative net_safety imports in kernel.{module_name}:\n  "
        + "\n  ".join(offenders)
        + "\nFix: replace with `from renquant_common.net_safety import ...`."
    )


@pytest.mark.parametrize("module_name", MODULES_TO_SCAN)
def test_no_absolute_kernel_net_safety_import(module_name: str) -> None:
    """No `from kernel.net_safety import ...` either. The lift completed
    end-to-end; only `from renquant_common.net_safety import ...` is
    the right path going forward."""
    path = _module_path(module_name)
    if not path.exists():
        pytest.skip(f"kernel module not present: {module_name}")
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "kernel.net_safety":
                names = ", ".join(alias.name for alias in node.names)
                offenders.append(
                    f"{path.name}:{node.lineno}: from kernel.net_safety import {names}"
                )
    assert not offenders, (
        f"Stale absolute kernel.net_safety imports in kernel.{module_name}:\n  "
        + "\n  ".join(offenders)
        + "\nFix: replace with `from renquant_common.net_safety import ...`."
    )


def test_kernel_modules_import_clean() -> None:
    """Top-level import of each module must succeed without
    ModuleNotFoundError on net_safety — covers any top-level import
    that points at the wrong path.
    """
    # Put strategy-104 dir on sys.path so `from kernel.X` resolves.
    strategy_dir = KERNEL_DIR.parent
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    for module_name in MODULES_TO_SCAN:
        if not _module_path(module_name).exists():
            continue
        try:
            importlib.import_module(f"kernel.{module_name}")
        except ModuleNotFoundError as exc:
            if "net_safety" in str(exc):
                pytest.fail(
                    f"kernel.{module_name} import fails on net_safety: {exc}. "
                    f"Likely a stale `from .net_safety import ...` or "
                    f"`from kernel.net_safety import ...` still present."
                )
            # Other ModuleNotFoundError (e.g., yfinance not installed)
            # is not our concern — skip the test.
            pytest.skip(f"kernel.{module_name}: unrelated missing dep: {exc}")
