"""TDD step 3 — MetaLabelLoggingJob unit tests.

The Job wraps SnapshotHoldingsTask + should_skip when training mode is
off in config (default).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.meta_label.job_meta_label_log import MetaLabelLoggingJob  # noqa: E402
from kernel.meta_label.task_snapshot import SnapshotHoldingsTask     # noqa: E402


def _ctx_with_config(meta_cfg: dict | None = None, snapshot_logger=None):
    cfg = {}
    if meta_cfg is not None:
        cfg["meta_label_training"] = meta_cfg
    return SimpleNamespace(
        config=cfg,
        today=datetime.date(2025, 1, 15),
        holdings={},
        prices={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        hwm=100000.0,
        portfolio_value=95000.0,
        exits=[],
        candidates=[],
        snapshot_logger=snapshot_logger,
    )


class TestMetaLabelLoggingJobShape:
    def test_tasks_contains_snapshot_task(self):
        job = MetaLabelLoggingJob()
        tt = job.tasks
        types = [type(t).__name__ for t in tt]
        assert "SnapshotHoldingsTask" in types

    def test_should_skip_when_no_meta_label_training_block(self):
        job = MetaLabelLoggingJob()
        ctx = _ctx_with_config(meta_cfg=None)
        assert job.should_skip(ctx) is True

    def test_should_skip_when_explicitly_disabled(self):
        job = MetaLabelLoggingJob()
        ctx = _ctx_with_config(meta_cfg={"enabled": False})
        assert job.should_skip(ctx) is True

    def test_does_not_skip_when_enabled_and_logger_present(self):
        from kernel.meta_label.snapshot import SnapshotLogger
        job = MetaLabelLoggingJob()
        ctx = _ctx_with_config(
            meta_cfg={"enabled": True},
            snapshot_logger=SnapshotLogger(),
        )
        assert job.should_skip(ctx) is False

    def test_should_skip_when_enabled_but_no_logger(self):
        # Adapter never set the logger — gracefully skip rather than
        # silently no-op the wrong way. The task itself also has a
        # logger-None guard but Job-level skip is cleaner.
        job = MetaLabelLoggingJob()
        ctx = _ctx_with_config(
            meta_cfg={"enabled": True},
            snapshot_logger=None,
        )
        assert job.should_skip(ctx) is True
