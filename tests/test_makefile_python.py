from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_makefile_prefers_repo_venv_python() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)" in text
