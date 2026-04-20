"""CI enforcement: kernel/ must not import from common/.

Scans every .py file in backtesting/renquant_103/kernel/ and asserts
no 'import common' or 'from common' statement exists.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_103" / "kernel"
KERNEL_FILES = sorted(KERNEL_DIR.glob("*.py"))


@pytest.mark.parametrize("py_file", KERNEL_FILES, ids=lambda f: f.name)
def test_no_common_import(py_file: Path) -> None:
    """kernel/*.py must not import from common/."""
    source = py_file.read_text()
    # Quick regex check first (catches comments too, but fast)
    forbidden = re.compile(r"\bfrom\s+common\b|\bimport\s+common\b")
    if not forbidden.search(source):
        return  # fast path

    # Parse AST for precise check (ignores comments / strings)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("common"), (
                    f"{py_file.name} imports 'common' at line {node.lineno}"
                )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("common"), (
                f"{py_file.name} imports from 'common' at line {node.lineno}"
            )


def test_kernel_files_exist() -> None:
    """All expected kernel modules must be present."""
    expected = {"config.py", "regime.py", "indicators.py", "models.py",
                "exits.py", "selection.py", "sizing.py",
                "market_gates.py", "portfolio.py", "__init__.py"}
    found = {f.name for f in KERNEL_FILES} | {"__init__.py"}
    missing = expected - found
    assert not missing, f"Missing kernel modules: {missing}"
