"""Regression guards for legacy scheduled wrappers."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_daily_102_requires_explicit_legacy_enable() -> None:
    src = (REPO / "scripts" / "daily_102.sh").read_text()
    assert "RQ_LEGACY_102_DAILY_ENABLED" in src
    assert "legacy rollback-only" in src
    assert "renquant_102 notebook" in src


def test_daily_103_requires_explicit_legacy_enable() -> None:
    src = (REPO / "scripts" / "daily_103.sh").read_text()
    assert "RQ_LEGACY_103_DAILY_ENABLED" in src
    assert "legacy rollback-only" in src
    assert "renquant_103 notebook" in src


def test_live_only_103_requires_explicit_legacy_enable() -> None:
    src = (REPO / "scripts" / "live_only_103.sh").read_text()
    assert "RQ_LEGACY_103_LIVE_ONLY_ENABLED" in src
    assert "legacy rollback-only" in src
    assert "-m live.runner" in src
