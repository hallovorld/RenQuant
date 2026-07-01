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


class TestComputeShadowSummaryTopPicks:
    """2026-07-01: top-N recommendation list with an HONEST, relative-only
    confidence indicator (rank / percentile / z-score within today's scored
    universe — NEVER a fabricated probability).

    ``_compute_shadow_summary`` is the exact function ``ApplyShadowScoringTask
    .run`` calls, extracted as a pure function precisely so these values can
    be hand-verified against a small fixture without mocking MLflow / the
    model registry / scorer loading.

    Fixture: 10 tickers, shadow scores chosen so mean/std/rank/percentile/
    z-score can be independently hand-computed (values pinned below were
    cross-checked with a standalone numpy computation using the SAME
    population-std convention as the implementation, i.e. np.std(ddof=0)).
    """

    SHADOW = {
        "A": 0.10, "B": 0.08, "C": 0.05, "D": 0.02, "E": -0.01,
        "F": -0.05, "G": -0.08, "H": -0.10, "I": -0.15, "J": -0.20,
    }
    # Primary panel scores over the SAME universe, deliberately ranked
    # differently from shadow so in_primary_topN produces a real mix of
    # True/False. Primary top-5 by score desc: C, E, J, A, F.
    PRIMARY = {
        "A": 0.05, "B": -0.02, "C": 0.20, "D": 0.01, "E": 0.15,
        "F": 0.03, "G": -0.10, "H": 0.00, "I": -0.05, "J": 0.10,
    }

    def _sorted_and_ranks(self, scores):
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_scores)}
        return sorted_scores, ranks

    def _call(self, shadow_mod, top_n_picks=5, primary=None, shadow=None):
        primary = primary if primary is not None else self.PRIMARY
        shadow = shadow if shadow is not None else self.SHADOW
        sorted_primary, primary_ranks = self._sorted_and_ranks(primary)
        sorted_shadow, shadow_ranks = self._sorted_and_ranks(shadow)
        return shadow_mod._compute_shadow_summary(
            "patchtst_v1", "patchtst",
            primary, sorted_primary, primary_ranks,
            shadow, sorted_shadow, shadow_ranks,
            top_n_picks,
        )

    def test_top_picks_length_respects_top_n(self, shadow_mod):
        summary = self._call(shadow_mod, top_n_picks=5)
        assert summary["top_picks_n"] == 5
        assert len(summary["top_picks"]) == 5

    def test_top_picks_length_configurable(self, shadow_mod):
        summary = self._call(shadow_mod, top_n_picks=3)
        assert summary["top_picks_n"] == 3
        assert len(summary["top_picks"]) == 3
        assert [p["ticker"] for p in summary["top_picks"]] == ["A", "B", "C"]

    def test_rank_order_matches_score_order(self, shadow_mod):
        summary = self._call(shadow_mod)
        tickers = [p["ticker"] for p in summary["top_picks"]]
        assert tickers == ["A", "B", "C", "D", "E"]
        ranks = [p["shadow_rank"] for p in summary["top_picks"]]
        assert ranks == [1, 2, 3, 4, 5]

    def test_percentile_hand_verified(self, shadow_mod):
        """rank / n_universe * 100, n_universe=10 → clean round numbers."""
        summary = self._call(shadow_mod)
        expected_pct = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0}
        for p in summary["top_picks"]:
            assert p["shadow_percentile"] == expected_pct[p["ticker"]]

    def test_zscore_hand_verified(self, shadow_mod):
        """(score - mean) / std over the FULL 10-ticker shadow universe.
        Independently computed via numpy: mean=-0.034, std=0.0944669...
        """
        summary = self._call(shadow_mod)
        expected_z = {"A": 1.42, "B": 1.21, "C": 0.89, "D": 0.57, "E": 0.25}
        for p in summary["top_picks"]:
            assert p["shadow_zscore"] == pytest.approx(expected_z[p["ticker"]], abs=0.01)

    def test_zscore_matches_manual_formula(self, shadow_mod):
        """Belt-and-suspenders: recompute z from raw shadow_score/mean/std
        instead of trusting a second hardcoded table."""
        import numpy as np
        vals = np.array(list(self.SHADOW.values()), dtype=float)
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary = self._call(shadow_mod)
        for p in summary["top_picks"]:
            expected = round((p["shadow_score"] - mean) / std, 2)
            assert p["shadow_zscore"] == expected

    def test_in_primary_admitted_always_none_not_guessed(self, shadow_mod):
        """ApplyShadowScoringTask runs before SelectionJob populates
        ctx.orders — whether primary actually BOUGHT the ticker is not
        determinable at this point in the pipeline. Must be None, never a
        guessed True/False."""
        summary = self._call(shadow_mod)
        for p in summary["top_picks"]:
            assert p["in_primary_admitted"] is None

    def test_in_primary_topn_matches_actual_primary_ranking(self, shadow_mod):
        """Primary top-5 by score desc is C, E, J, A, F (hand-verified from
        PRIMARY fixture above). Shadow top-5 is A, B, C, D, E. Expect a real
        mix of True/False, not all-True or all-False."""
        summary = self._call(shadow_mod)
        by_ticker = {p["ticker"]: p["in_primary_topN"] for p in summary["top_picks"]}
        assert by_ticker == {
            "A": True,   # primary rank 4 (top5)
            "B": False,  # primary rank 8
            "C": True,   # primary rank 1 (top5)
            "D": False,  # primary rank 6
            "E": True,   # primary rank 2 (top5)
        }

    def test_existing_top10_overlap_and_spearman_unaffected(self, shadow_mod):
        """Refactor into a pure function must not change the pre-existing
        top10_overlap / spearman_vs_primary / top3 fields."""
        summary = self._call(shadow_mod)
        assert summary["top3"] == ["A", "B", "C"]
        assert summary["n_candidates"] == 10
        assert isinstance(summary["top10_overlap"], int)
        assert summary["spearman_vs_primary"] == summary["spearman_vs_primary"]  # not NaN (n=10>=5)

    def test_default_top_n_constant(self, shadow_mod):
        assert shadow_mod._DEFAULT_TOP_N_PICKS == 5

    def test_zscore_nan_when_shadow_scores_constant(self, shadow_mod):
        """Zero-variance universe must not raise ZeroDivisionError —
        z-score is undefined (NaN), not a fabricated value."""
        flat = {t: 0.05 for t in "ABCDE"}
        summary = self._call(shadow_mod, top_n_picks=3, primary=flat, shadow=flat)
        for p in summary["top_picks"]:
            z = p["shadow_zscore"]
            assert z != z, "expected NaN for zero-variance universe"  # NaN check


class TestApplyShadowScoringTaskUsesExtractedHelper:
    """Source-level contract: run() must call the extracted pure helper
    (reuse, not a second inline re-implementation) and must wire the
    configurable top-N constant through to it."""

    def test_run_calls_compute_shadow_summary(self):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "def _compute_shadow_summary(" in src
        assert "summary = _compute_shadow_summary(" in src

    def test_top_n_picks_is_configurable(self):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "shadow_top_n_picks" in src
        assert "_DEFAULT_TOP_N_PICKS" in src
