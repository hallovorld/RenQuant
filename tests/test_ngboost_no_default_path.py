"""LoadNGBoostTask must NOT default to a hardcoded artifact filename.

§5.13.14 regression guard. Pre-fix: a sim config that enabled NGBoost
without supplying `artifact_path` silently loaded the production model
from `artifacts/prod/ngboost-head.alpha158_fund.json`, breaching
sim/prod isolation.

Post-fix: if `ngboost.enabled=true` and `artifact_path` is missing or
empty, LoadNGBoostTask logs an error and disables NGBoost for the run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _ctx(ngboost_cfg: dict):
    return SimpleNamespace(
        config={"ranking": {"panel_scoring": {"ngboost": ngboost_cfg}}},
    )


class TestNGBoostNoDefaultPath:

    def test_enabled_without_artifact_path_disables_quietly(self):
        """ngboost.enabled=true but no artifact_path → disabled, head=None."""
        from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask
        task = LoadNGBoostTask()
        ctx = _ctx({"enabled": True})  # no artifact_path
        task.run(ctx)
        assert ctx._ngboost_head is None

    def test_enabled_with_empty_artifact_path_disables(self):
        from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask
        task = LoadNGBoostTask()
        ctx = _ctx({"enabled": True, "artifact_path": ""})
        task.run(ctx)
        assert ctx._ngboost_head is None

    def test_disabled_short_circuits(self):
        """ngboost.enabled=false → task returns early, head left untouched."""
        from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask
        task = LoadNGBoostTask()
        ctx = _ctx({"enabled": False})
        # Pre-set head to a sentinel — task must NOT clobber it.
        ctx._ngboost_head = "sentinel"
        task.run(ctx)
        assert ctx._ngboost_head == "sentinel"
