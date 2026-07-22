"""Real-RunnerAdapter guards for --preflight (dry-run) mode.

Complements tests/test_runner_preflight_dry_run.py (which drives the runner
with fake pipeline/adapter modules). Here we exercise the ACTUAL RunnerAdapter
to prove, against isolated state paths, that:
  * preflight construction opens NO runs DB (the data/runs_*.db file is never
    created) — the single pre-commit persistence surface is neutralized; and
  * commit() (the single write chokepoint) refuses to write and flips the
    guard if it is ever entered during a dry-run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from live.runner import PreflightGuard

REPO = Path(__file__).resolve().parent.parent
STRATEGY = REPO / "backtesting" / "renquant_104"
PIPELINE_SRC = REPO / ".subrepo_runtime" / "repos" / "renquant-pipeline" / "src"
if PIPELINE_SRC.exists() and str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


def _cfg(tmp_path: Path) -> dict:
    return {
        "model_name": "renquant_104",
        "watchlist": ["AAA"],
        "benchmark": "SPY",
        "sector_map": {"AAA": "Tech"},
        "sector_etf_map": {},
        "ranking": {"panel_scoring": {"enabled": False}},
        "persistence": {
            "enabled": True,
            "db_path": str(tmp_path / "runs.db"),
            "sim_db_path": str(tmp_path / "sim_runs.db"),
        },
    }


def _broker():
    b = MagicMock()
    b.broker_name = None
    return b


@pytest.mark.integration
def test_preflight_adapter_opens_no_runs_db(tmp_path):
    from adapters.runner import RunnerAdapter

    adapter = RunnerAdapter(
        _cfg(tmp_path), models={}, broker=_broker(), strategy_dir=tmp_path,
        preflight=True, preflight_guard=PreflightGuard(),
    )
    # No DB connection, and NO runs*.db file created under the isolated path.
    assert adapter._db is None
    assert not (tmp_path / "runs.db").exists()
    assert list(tmp_path.glob("runs*.db")) == []


@pytest.mark.integration
def test_non_preflight_adapter_does_open_db(tmp_path):
    """Contrast: without --preflight the adapter DOES create the runs DB — this
    is exactly the write the probe must avoid."""
    from adapters.runner import RunnerAdapter

    adapter = RunnerAdapter(
        _cfg(tmp_path), models={}, broker=_broker(), strategy_dir=tmp_path,
        preflight=False,
    )
    try:
        assert adapter._db is not None
        assert (tmp_path / "runs.db").exists()
    finally:
        if adapter._db is not None:
            adapter._db.close()


@pytest.mark.integration
def test_preflight_commit_refuses_and_flips_guard(tmp_path):
    from adapters.runner import RunnerAdapter

    guard = PreflightGuard()
    adapter = RunnerAdapter(
        _cfg(tmp_path), models={}, broker=_broker(), strategy_dir=tmp_path,
        preflight=True, preflight_guard=guard,
    )
    # commit() is the single write chokepoint. Entering it during a dry-run
    # must write nothing and flip the guard so the attestation fails closed.
    adapter.commit(SimpleNamespace(_preflight=True, _preflight_guard=guard))

    assert guard.persisted is True
    assert guard.ordered is True
    assert guard.promoted is True
    assert guard.clean() is False
    # Nothing was persisted to the isolated path.
    assert list(tmp_path.glob("runs*.db")) == []
    assert list(tmp_path.glob("live_state*.json")) == []
