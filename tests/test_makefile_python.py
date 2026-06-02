from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_makefile_prefers_repo_venv_python() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)" in text


def test_makefile_has_safe_launchagent_install_target() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "ops-preinstall-ready:" in text
    assert "$(PYTHON) scripts/check_ops_deployment_ready.py --skip-launchagents" in text
    assert "ops-install-launchagents: subrepo-runtime-root" in text
    assert "PYTHON=$(PYTHON) bash scripts/install_launchagents.sh" in text
