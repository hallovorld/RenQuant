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
import datetime as _dt
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

    # 2026-07-01 round 2 (Codex CHANGES_REQUESTED — admission gate): unless
    # a test explicitly overrides them, _call feeds a FRESH artifact + FULL
    # coverage so the pre-existing rank/percentile/z-score assertions below
    # (unrelated to the admission gate) keep exercising an actionable path.
    # TestComputeAdmission / TestComputeShadowSummaryAdmissionIntegration
    # below cover the NOT-actionable paths explicitly.
    DEFAULT_AS_OF_DATE = _dt.date(2026, 7, 1)
    DEFAULT_ARTIFACT_META = {
        "trained_date": "2026-06-30",
        "artifact_fingerprint": "sha256:testfp1234567890",
    }

    def _sorted_and_ranks(self, scores):
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_scores)}
        return sorted_scores, ranks

    def _call(self, shadow_mod, top_n_picks=5, primary=None, shadow=None,
              as_of_date=None, artifact_meta=None, n_expected_universe=None,
              min_coverage=None):
        primary = primary if primary is not None else self.PRIMARY
        shadow = shadow if shadow is not None else self.SHADOW
        sorted_primary, primary_ranks = self._sorted_and_ranks(primary)
        sorted_shadow, shadow_ranks = self._sorted_and_ranks(shadow)
        kwargs = dict(
            as_of_date=as_of_date if as_of_date is not None else self.DEFAULT_AS_OF_DATE,
            artifact_meta=(artifact_meta if artifact_meta is not None
                            else dict(self.DEFAULT_ARTIFACT_META)),
            n_expected_universe=(n_expected_universe if n_expected_universe is not None
                                   else len(shadow)),
        )
        if min_coverage is not None:
            kwargs["min_coverage"] = min_coverage
        return shadow_mod._compute_shadow_summary(
            "patchtst_v1", "patchtst",
            primary, sorted_primary, primary_ranks,
            shadow, sorted_shadow, shadow_ranks,
            top_n_picks,
            **kwargs,
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
        """(n_universe - rank + 1) / n_universe * 100, n_universe=10 → clean
        round numbers. FIXED 2026-07-01 round 2 (Codex CHANGES_REQUESTED):
        higher percentile = better rank, the conventional reading — was
        inverted (rank / n * 100, best name near the 1st percentile)."""
        summary = self._call(shadow_mod)
        expected_pct = {"A": 100.0, "B": 90.0, "C": 80.0, "D": 70.0, "E": 60.0}
        for p in summary["top_picks"]:
            assert p["shadow_percentile"] == expected_pct[p["ticker"]]

    def test_percentile_direction_best_rank_is_highest_percentile(self, shadow_mod):
        """Explicit direction check (not just round-number values): rank 1
        (best) must have a STRICTLY HIGHER percentile than rank 5 (worse)."""
        summary = self._call(shadow_mod)
        by_rank = {p["shadow_rank"]: p["shadow_percentile"] for p in summary["top_picks"]}
        assert by_rank[1] > by_rank[2] > by_rank[3] > by_rank[4] > by_rank[5]
        # Best-ranked name in a 10-name universe should read as the 100th
        # percentile, not the 10th/1st.
        assert by_rank[1] == 100.0

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


class TestFreshnessTier:
    """2026-07-01 round 2 (Codex CHANGES_REQUESTED): bucket artifact age into
    healthy/warn/escalate/breach/unknown — mirrors the tier VOCABULARY of
    renquant-orchestrator's model_freshness_monitor.py (separate repo, not
    imported) so an operator reading both alert streams sees the same words.
    """

    def test_healthy_below_warn_threshold(self, shadow_mod):
        assert shadow_mod._freshness_tier(0) == "healthy"
        assert shadow_mod._freshness_tier(27) == "healthy"

    def test_warn_band(self, shadow_mod):
        assert shadow_mod._freshness_tier(28) == "warn"
        assert shadow_mod._freshness_tier(32) == "warn"

    def test_escalate_band(self, shadow_mod):
        assert shadow_mod._freshness_tier(33) == "escalate"
        assert shadow_mod._freshness_tier(34) == "escalate"

    def test_breach_band(self, shadow_mod):
        assert shadow_mod._freshness_tier(35) == "breach"
        # Real known example from the review: PatchTST confirmed ~140 days
        # stale in this codebase — must land squarely in breach.
        assert shadow_mod._freshness_tier(140) == "breach"

    def test_unknown_when_age_missing_or_nan(self, shadow_mod):
        assert shadow_mod._freshness_tier(None) == "unknown"
        assert shadow_mod._freshness_tier(float("nan")) == "unknown"

    def test_documented_threshold_constants(self, shadow_mod):
        assert shadow_mod._FRESHNESS_WARN_DAYS == 28
        assert shadow_mod._FRESHNESS_ESCALATE_DAYS == 33
        assert shadow_mod._FRESHNESS_BREACH_DAYS == 35


class TestComputeAdmission:
    """2026-07-01 round 2 (Codex CHANGES_REQUESTED on umbrella PR #426):
    picks must be bound to an admission verdict (artifact freshness +
    scored-universe coverage) before they can be actionable. Covers the two
    REAL known failure modes cited in the review: a ~140d-stale PatchTST
    artifact, and an 83/292 censored-subset universe.
    """

    AS_OF = _dt.date(2026, 7, 1)

    def test_healthy_full_coverage_is_actionable(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": "2026-06-30"},
            n_scored=83, n_expected=83, min_coverage=0.80,
        )
        assert admission["verdict"] == "healthy"
        assert admission["actionable"] is True
        assert admission["coverage"] == 1.0
        assert admission["reasons"] == []

    def test_known_incident_140d_stale_is_breach_not_actionable(self, shadow_mod):
        trained = (self.AS_OF - _dt.timedelta(days=140)).isoformat()
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": trained},
            n_scored=83, n_expected=83, min_coverage=0.80,
        )
        assert admission["verdict"] == "breach"
        assert admission["actionable"] is False
        assert any("stale" in r for r in admission["reasons"])

    def test_escalate_tier_is_not_actionable(self, shadow_mod):
        trained = (self.AS_OF - _dt.timedelta(days=33)).isoformat()
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF, artifact_meta={"trained_date": trained},
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["verdict"] == "escalate"
        assert admission["actionable"] is False

    def test_missing_trained_date_is_unknown_not_actionable(self, shadow_mod):
        """Fail-closed: no provenance is never silently treated as fresh."""
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={}, n_scored=83, n_expected=83, min_coverage=0.80,
        )
        assert admission["verdict"] == "unknown"
        assert admission["actionable"] is False
        assert "trained_date" in admission["reasons"][0]

    def test_low_coverage_blocks_even_when_artifact_is_fresh(self, shadow_mod):
        """Real known example: an 83/292 (~28%) censored subset is not a
        comparable "rank 1" even when the artifact itself is fresh."""
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": "2026-06-30"},
            n_scored=83, n_expected=292, min_coverage=0.80,
        )
        assert admission["verdict"] == "healthy"
        assert admission["actionable"] is False
        assert admission["coverage"] == pytest.approx(83 / 292, abs=1e-4)
        assert any("coverage" in r for r in admission["reasons"])

    def test_coverage_fraction_correctly_computed_and_surfaced(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": "2026-06-30"},
            n_scored=150, n_expected=300, min_coverage=0.80,
        )
        assert admission["n_scored"] == 150
        assert admission["n_expected"] == 300
        assert admission["coverage"] == 0.5

    def test_zero_expected_universe_degrades_gracefully_not_a_false_fail(self, shadow_mod):
        """n_expected<=0 means the watchlist wasn't configured/available —
        coverage is UNKNOWN (None), not fabricated as 100% or force-failed."""
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": "2026-06-30"},
            n_scored=10, n_expected=0, min_coverage=0.80,
        )
        assert admission["coverage"] is None
        assert admission["n_expected"] is None
        assert admission["actionable"] is True

    def test_run_id_binds_date_name_and_fingerprint(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={
                "trained_date": "2026-06-30",
                "artifact_fingerprint": "sha256:abcdef0123456789",
            },
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["run_id"] == "2026-07-01:patchtst_v1:sha256:abcde"

    def test_default_min_coverage_constant(self, shadow_mod):
        assert shadow_mod._DEFAULT_MIN_COVERAGE == 0.80


class TestComputeAdmissionBindingDataCutoff:
    """2026-07-01 ROUND 3 (Codex #426 review point 1 named "trained cutoff"
    AND "feature-data cutoff" as separate provenance to bind): the age used
    for the verdict must PREFER a binding DATA cutoff field over
    ``trained_date`` whenever one is present — mirrors orchestrator's
    model_freshness_monitor.py DATA_CUTOFF_FIELDS priority. Real motivating
    case: hf_patchtst_scorer.py already stamps
    ``effective_train_cutoff_date`` into ``scorer.metadata`` at load time;
    round 2 computed age from ``trained_date`` alone and left that field
    unused, reintroducing the "fresh trained_date over stale data" risk
    this codebase already hit once (2026-06-15 model-stale-by-split-recipe).
    """

    AS_OF = _dt.date(2026, 7, 1)

    def test_binding_cutoff_preferred_over_trained_date(self, shadow_mod):
        """A recent trained_date with an OLD effective_train_cutoff_date
        (the real PatchTST shape: retrained recently but on stale data)
        must be judged on the cutoff, not the retrain run time."""
        admission = shadow_mod._compute_admission(
            name="patchtst_v1", as_of_date=self.AS_OF,
            artifact_meta={
                "trained_date": "2026-06-30",  # 1d old -> would read healthy
                "effective_train_cutoff_date": "2024-11-13",  # ~596d stale
            },
            n_scored=83, n_expected=83, min_coverage=0.80,
        )
        assert admission["verdict"] == "breach"
        assert admission["actionable"] is False
        assert admission["binding_cutoff"] == "2024-11-13"
        assert admission["binding_cutoff_field"] == "effective_train_cutoff_date"
        assert admission["age_days"] > 500

    def test_data_cutoff_field_priority_matches_orchestrator_order(self, shadow_mod):
        """label_observation_cutoff outranks effective_train_cutoff_date
        when both are present — same field priority as the orchestrator's
        DATA_CUTOFF_FIELDS."""
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={
                "label_observation_cutoff": "2026-06-29",
                "effective_train_cutoff_date": "2020-01-01",
            },
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["binding_cutoff_field"] == "label_observation_cutoff"
        assert admission["binding_cutoff"] == "2026-06-29"
        assert admission["verdict"] == "healthy"

    def test_trained_date_fallback_only_when_no_binding_cutoff_present(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={"trained_date": "2026-06-30"},
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["binding_cutoff"] is None
        assert admission["binding_cutoff_field"] is None
        assert admission["age_days"] == 1.0
        assert admission["verdict"] == "healthy"

    def test_unparseable_binding_cutoff_field_falls_back_to_trained_date(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={
                "trained_date": "2026-06-30",
                "effective_train_cutoff_date": "not-a-date",
            },
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["binding_cutoff"] is None
        assert admission["age_days"] == 1.0

    def test_lookahead_cutoff_fails_closed_to_breach(self, shadow_mod):
        """A cutoff LATER than as_of_date (look-ahead) must never read as a
        negative age / healthy."""
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF,
            artifact_meta={"effective_train_cutoff_date": "2026-08-15"},
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["age_days"] < 0
        assert admission["verdict"] == "breach"
        assert admission["actionable"] is False
        assert any("look-ahead" in r for r in admission["reasons"])

    def test_binding_cutoff_none_and_no_trained_date_is_unknown(self, shadow_mod):
        admission = shadow_mod._compute_admission(
            name="x", as_of_date=self.AS_OF, artifact_meta={},
            n_scored=10, n_expected=10, min_coverage=0.80,
        )
        assert admission["verdict"] == "unknown"
        assert admission["actionable"] is False
        assert admission["binding_cutoff"] is None
        assert admission["age_days"] is None


class TestComputeShadowSummaryAdmissionIntegration:
    """``_compute_shadow_summary`` must surface the admission verdict inline
    (``admission``/``actionable``/``run_id`` top-level keys) so callers
    (``ApplyShadowScoringTask.run`` logging, ``live/runner.py`` ntfy
    rendering) never need to recompute it."""

    SHADOW = TestComputeShadowSummaryTopPicks.SHADOW
    PRIMARY = TestComputeShadowSummaryTopPicks.PRIMARY

    def _sorted_and_ranks(self, scores):
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_scores)}
        return sorted_scores, ranks

    def test_actionable_true_when_fresh_and_full_coverage(self, shadow_mod):
        sorted_primary, primary_ranks = self._sorted_and_ranks(self.PRIMARY)
        sorted_shadow, shadow_ranks = self._sorted_and_ranks(self.SHADOW)
        summary = shadow_mod._compute_shadow_summary(
            "patchtst_v1", "patchtst",
            self.PRIMARY, sorted_primary, primary_ranks,
            self.SHADOW, sorted_shadow, shadow_ranks, 5,
            as_of_date=_dt.date(2026, 7, 1),
            artifact_meta={"trained_date": "2026-06-30"},
            n_expected_universe=10,
        )
        assert summary["actionable"] is True
        assert summary["admission"]["verdict"] == "healthy"
        assert summary["run_id"] == summary["admission"]["run_id"]
        # top_picks are still fully computed even when actionable — the
        # NOT-ACTIONABLE gate is applied by the ntfy RENDERER
        # (live/runner.py), not by hiding data in the audit trail here.
        assert len(summary["top_picks"]) == 5

    def test_actionable_false_when_stale_but_top_picks_still_computed(self, shadow_mod):
        """The audit trail (ctx._shadow_summary / MLflow) keeps the raw
        ranks even when not actionable — only the ntfy BODY suppresses
        them, so the diagnostic data isn't lost, just not presented as a
        recommendation."""
        sorted_primary, primary_ranks = self._sorted_and_ranks(self.PRIMARY)
        sorted_shadow, shadow_ranks = self._sorted_and_ranks(self.SHADOW)
        trained = (_dt.date(2026, 7, 1) - _dt.timedelta(days=140)).isoformat()
        summary = shadow_mod._compute_shadow_summary(
            "patchtst_v1", "patchtst",
            self.PRIMARY, sorted_primary, primary_ranks,
            self.SHADOW, sorted_shadow, shadow_ranks, 5,
            as_of_date=_dt.date(2026, 7, 1),
            artifact_meta={"trained_date": trained},
            n_expected_universe=10,
        )
        assert summary["actionable"] is False
        assert summary["admission"]["verdict"] == "breach"
        assert len(summary["top_picks"]) == 5

    def test_omitted_admission_kwargs_default_to_not_actionable(self, shadow_mod):
        """Fail-closed: a caller that omits as_of_date/artifact_meta (e.g.
        an old positional-only call site) gets UNKNOWN/not-actionable, never
        a silently-assumed-fresh pass."""
        sorted_primary, primary_ranks = self._sorted_and_ranks(self.PRIMARY)
        sorted_shadow, shadow_ranks = self._sorted_and_ranks(self.SHADOW)
        summary = shadow_mod._compute_shadow_summary(
            "patchtst_v1", "patchtst",
            self.PRIMARY, sorted_primary, primary_ranks,
            self.SHADOW, sorted_shadow, shadow_ranks, 5,
        )
        assert summary["actionable"] is False
        assert summary["admission"]["verdict"] == "unknown"


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

    def test_admission_gate_is_wired(self):
        """2026-07-01 round 2: run() must feed the admission gate (artifact
        metadata + expected-universe size), not just the top-N size."""
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "shadow_scoring.py").read_text()
        assert "def _compute_admission(" in src
        assert "admission = _compute_admission(" in src
        assert "artifact_meta=artifact_meta" in src
        assert "n_expected_universe=n_expected_universe" in src
        assert "shadow_min_coverage" in src
        assert "_DEFAULT_MIN_COVERAGE" in src
