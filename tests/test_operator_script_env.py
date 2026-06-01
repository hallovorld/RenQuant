"""Operator scripts should use the project venv, not stale conda paths."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _non_comment(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_challenger_window_uses_project_venv() -> None:
    src = (REPO / "scripts" / "check_challenger_window.sh").read_text(encoding="utf-8")
    non_comment = _non_comment(REPO / "scripts" / "check_challenger_window.sh")

    assert "miniconda" not in non_comment
    assert "CONDA_PREFIX" not in non_comment
    assert 'PYTHON="${REPO_ROOT}/.venv/bin/python"' in src
    assert '"$PYTHON" "${REPO_ROOT}/scripts/finalize_challenger.py"' in src


def test_manual_promote_uses_project_venv() -> None:
    src = (REPO / "scripts" / "manual_promote.sh").read_text(encoding="utf-8")
    non_comment = _non_comment(REPO / "scripts" / "manual_promote.sh")

    assert "miniconda" not in non_comment
    assert "CONDA_PREFIX" not in non_comment
    assert 'PYTHON="$REPO_DIR/.venv/bin/python"' in src
