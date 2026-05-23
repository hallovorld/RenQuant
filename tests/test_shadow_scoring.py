"""Regression tests for ApplyShadowScoringTask using MLflow tracking.

Pin the shadow model pattern (records what alt models WOULD do without
affecting primary orders) via 3rd-party MLflow library.

Verifies:
  1. No-op when no shadow_models configured (safe default)
  2. Shadow Task is registered in PanelScoringJob
  3. MLflow setup creates experiment + tracking URI works
  4. Persist via MLflow log_metrics + log_table works
  5. Source-level invariants (no order-placement calls)
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


@pytest.fixture(scope="module")
def shadow_mod():
    from kernel.panel_pipeline import shadow_scoring
    return shadow_scoring


class TestSourceContracts:
    """Pin behavior strings so future refactors can't silently change semantics."""

    def test_apply_shadow_task_registered_in_job(self):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "job_panel_scoring.py").read_text()
        assert "from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask" in src
        assert "ApplyShadowScoringTask()" in src

    def test_shadow_does_not_submit_orders(self, shadow_mod):
        """Shadow Task must NOT contain order-placement code paths."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "place_order" not in src
        assert "submit_order" not in src
        assert "broker." not in src

    def test_uses_mlflow_third_party(self, shadow_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "import mlflow" in src
        assert "mlflow.start_run" in src
        assert "mlflow.log_metrics" in src
        assert "mlflow.log_table" in src

    def test_default_experiment_name(self, shadow_mod):
        assert shadow_mod._DEFAULT_EXPERIMENT == "renquant_104_shadow"

    def test_2026_05_18_marker(self, shadow_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "2026-05-18" in src

    def test_shadow_runtime_has_disable_and_cache_guards(self, shadow_mod):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "shadow_enabled" in src
        assert "shadow_log_mlflow" in src
        assert "_SCORER_CACHE" in src


class TestMLflowSetup:
    """Verify MLflow integration works on a temp tracking dir."""

    def test_setup_creates_experiment(self, tmp_path, shadow_mod):
        uri = f"file:{tmp_path}/mlruns"
        exp_id = shadow_mod._ensure_mlflow_setup(uri, "test_exp_shadow")
        assert isinstance(exp_id, str)
        # 2nd call: same experiment, same ID
        exp_id2 = shadow_mod._ensure_mlflow_setup(uri, "test_exp_shadow")
        assert exp_id == exp_id2


class TestLogShadowRun:
    """Verify _log_shadow_run writes correct metrics + table."""

    def test_log_run_basic(self, tmp_path, shadow_mod):
        import mlflow
        uri = f"file:{tmp_path}/mlruns"
        exp_id = shadow_mod._ensure_mlflow_setup(uri, "test_log_shadow")
        primary = {"A": 0.10, "B": 0.05, "C": 0.02, "D": -0.05, "E": -0.10}
        shadow = {"A": 0.08, "B": 0.06, "C": 0.04, "D": -0.04, "E": -0.08}
        sorted_p = sorted(primary.items(), key=lambda x: -x[1])
        p_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_p)}
        sorted_s = sorted(shadow.items(), key=lambda x: -x[1])
        s_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_s)}

        shadow_mod._log_shadow_run(exp_id, "2026-05-18", "test_shadow",
                                    "patchtst", "xgb",
                                    primary, shadow, p_ranks, s_ranks)

        # Verify run was logged
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        assert len(runs) >= 1
        latest = runs.iloc[0]
        assert latest["tags.shadow_name"] == "test_shadow"
        assert latest["tags.shadow_kind"] == "patchtst"
        assert "metrics.mean_diff" in latest.index
        assert "metrics.corr_primary_shadow" in latest.index


class TestNoOpWhenNoShadow:
    """When config has no shadow_models, Task is no-op (no MLflow setup)."""

    def test_task_runs_silently_with_empty_config(self, shadow_mod):
        from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask
        class MockCtx:
            def __init__(self):
                self.config = {"ranking": {"panel_scoring": {}}}
                self.candidates = []
                self.holdings = []
                self.today = None
        ctx = MockCtx()
        task = ApplyShadowScoringTask()
        result = task.run(ctx)
        assert result is None or result is False
