"""Regression tests for VetoWeakBuysTask 2026-05-03 P0 fix.

Production incident summary
---------------------------
Commit 410758b (2026-04-29 12:21 PT) added ``buy_floor: 0.30`` to
production config with the rationale "new buys should pass the same
panel-score floor as rotation targets". Rotation evaluates the
*calibrated* rank_score (range [0, 1]); the buy_floor was applied to
``cand.panel_score``, which the chain ordering set to the *raw* XGBoost
rank:pairwise margin (range ~[0, 0.05]). 0.30 was unreachable on raw.

Result: production cron silently dropped 55/55 candidates daily for
~5 days (2026-04-29 → 2026-05-03). No fresh entries opened during this
window — every "buy" in the trades log was a TopUp on an existing
holding via ``TopUpHeldTask``. The portfolio froze in place.

Fix
---
* VetoWeakBuysTask reads ``cand.rank_score`` (calibrated) not
  ``cand.panel_score`` (raw).
* Task is reordered in PanelScoringJob.tasks to run AFTER
  ApplyGlobalCalibrationTask.

Invariant
---------
The buy_floor compares against the same scale that downstream tier
thresholds (rotation, QualityFloor) use — calibrated rank_score in
[0, 1].
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


def _make_cand(ticker: str, panel_score, rank_score):
    """Build a CandidateResult with explicit raw vs calibrated values."""
    from kernel.selection import CandidateResult
    return CandidateResult(
        ticker=ticker, raw_score=0.0,
        rank_score=rank_score, rs_score=0.0, detail="",
        expected_return=0.0, panel_score=panel_score,
    )


def _make_ctx(buy_floor=None):
    from types import SimpleNamespace
    return SimpleNamespace(
        config={"ranking": {"panel_scoring": {"buy_floor": buy_floor}}},
        candidates=[],
        holdings={},
        counters={},
    )


class TestProductionIncidentReproduction(unittest.TestCase):
    """The exact production scenario must now keep the candidate, not drop it."""

    def test_raw_below_floor_calibrated_above_keeps(self):
        """Raw 0.025 (XGB margin) calibrated 0.40 (probability), floor 0.30."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.30)
        ctx.candidates = [
            _make_cand("PRODBUG", panel_score=0.025, rank_score=0.40),
        ]
        VetoWeakBuysTask().run(ctx)
        # PRE-FIX: would compare 0.025 < 0.30 → DROP (the production bug).
        # POST-FIX: compares 0.40 ≥ 0.30 → KEEP.
        self.assertEqual([c.ticker for c in ctx.candidates], ["PRODBUG"])

    def test_raw_above_floor_calibrated_below_drops(self):
        """The mirror case — raw above (impossible in practice) but calibrated low."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.30)
        ctx.candidates = [
            _make_cand("X", panel_score=0.50, rank_score=0.10),  # calibrated 0.10
        ]
        VetoWeakBuysTask().run(ctx)
        # rank_score 0.10 < floor 0.30 → DROP.
        self.assertEqual([c.ticker for c in ctx.candidates], [])

    def test_typical_production_distribution_drops_bottom(self):
        """7 candidates spanning 0.05 → 0.45 calibrated, floor 0.30 keeps top 3."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.30)
        ctx.candidates = [
            _make_cand(f"T{i}", panel_score=0.025, rank_score=score)
            for i, score in enumerate([0.05, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45])
        ]
        VetoWeakBuysTask().run(ctx)
        # Keep T3 (0.30, ≥ floor), T4, T5, T6.
        self.assertEqual({c.ticker for c in ctx.candidates},
                         {"T3", "T4", "T5", "T6"})


class TestRankScoreSemantics(unittest.TestCase):
    def test_drops_by_rank_score_not_panel_score(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.5)
        # Both same panel_score, different rank_score:
        ctx.candidates = [
            _make_cand("LOW",  panel_score=0.9, rank_score=0.1),
            _make_cand("HIGH", panel_score=0.9, rank_score=0.9),
        ]
        VetoWeakBuysTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["HIGH"])

    def test_no_op_when_floor_unset(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=None)
        ctx.candidates = [_make_cand("A", panel_score=0.0, rank_score=0.0)]
        VetoWeakBuysTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 1)
        self.assertNotIn("panel_vetoed", ctx.counters)

    def test_missing_rank_score_keeps_candidate(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.3)
        ctx.candidates = [_make_cand("MISS", panel_score=0.1, rank_score=None)]
        VetoWeakBuysTask().run(ctx)
        # rank_score=None → keep (rs_score still ranks)
        self.assertEqual([c.ticker for c in ctx.candidates], ["MISS"])

    def test_nan_rank_score_drops(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        import math
        ctx = _make_ctx(buy_floor=0.3)
        ctx.candidates = [_make_cand("NAN", panel_score=0.5,
                                     rank_score=float("nan"))]
        VetoWeakBuysTask().run(ctx)
        # NaN → drop (would be live model crash signal)
        self.assertEqual([c.ticker for c in ctx.candidates], [])

    def test_counter_increments_on_drop(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.5)
        ctx.candidates = [
            _make_cand("A", panel_score=0.0, rank_score=0.1),
            _make_cand("B", panel_score=0.0, rank_score=0.6),
            _make_cand("C", panel_score=0.0, rank_score=0.2),
        ]
        VetoWeakBuysTask().run(ctx)
        self.assertEqual(ctx.counters["panel_vetoed"], 2)


# 2026-05-04: per-bar adaptive floor min(mean+std, cap), per user spec
# "暂时改成，取min[mean+std, 0.3]". The truly scientific number isn't a
# preset — it's tied to today's distribution AND bounded by the
# legacy 0.30 ceiling so we never get LESS strict than pre-fix.

class TestAdaptiveBuyFloor(unittest.TestCase):
    def _make_ctx_adaptive(self, cap=0.30):
        from types import SimpleNamespace
        return SimpleNamespace(
            config={"ranking": {"panel_scoring": {
                "buy_floor": "adaptive_mean_std_cap",
                "buy_floor_adaptive_cap": cap,
            }}},
            candidates=[],
            holdings={},
            counters={},
        )

    def test_adaptive_uses_mean_plus_std_when_below_cap(self):
        """Distribution: 0.10, 0.15, 0.20, 0.25, 0.30 (mean 0.20, std ≈ 0.079).
        mean+std ≈ 0.279 < cap 0.30 → floor = 0.279 → drops bottom up to 0.279."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = self._make_ctx_adaptive(cap=0.30)
        ctx.candidates = [
            _make_cand(f"T{i}", panel_score=0.0, rank_score=s)
            for i, s in enumerate([0.10, 0.15, 0.20, 0.25, 0.30])
        ]
        VetoWeakBuysTask().run(ctx)
        # Expected floor ≈ 0.20 + 0.079 = 0.279. Candidates ≥ 0.279 kept.
        kept = {c.ticker for c in ctx.candidates}
        # T3 (0.25) → drop  (0.25 < 0.279)
        # T4 (0.30) → keep  (0.30 ≥ 0.279)
        self.assertEqual(kept, {"T4"})

    def test_adaptive_capped_when_distribution_wide(self):
        """Distribution: 0.10, 0.30, 0.50, 0.70, 0.90 (mean 0.50, std ≈ 0.316).
        mean+std ≈ 0.816 > cap 0.30 → floor = 0.30 → drops only those <0.30."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = self._make_ctx_adaptive(cap=0.30)
        ctx.candidates = [
            _make_cand(f"T{i}", panel_score=0.0, rank_score=s)
            for i, s in enumerate([0.10, 0.30, 0.50, 0.70, 0.90])
        ]
        VetoWeakBuysTask().run(ctx)
        kept = {c.ticker for c in ctx.candidates}
        # T0 (0.10) drops; T1..T4 ≥ 0.30 keep
        self.assertEqual(kept, {"T1", "T2", "T3", "T4"})

    def test_adaptive_falls_back_to_cap_with_under_2_cands(self):
        """One cand → no std defined → use cap directly."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = self._make_ctx_adaptive(cap=0.30)
        ctx.candidates = [_make_cand("ONLY", panel_score=0.0, rank_score=0.50)]
        VetoWeakBuysTask().run(ctx)
        # 0.50 ≥ 0.30 cap → keep
        self.assertEqual([c.ticker for c in ctx.candidates], ["ONLY"])

    def test_adaptive_per_bar_recomputes_each_call(self):
        """Same Task instance, two different bars → different floors."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        task = VetoWeakBuysTask()
        # Bar 1 — tight cluster around 0.30
        # mean=0.30, sample-stdev≈0.0158, mean+std≈0.316 > cap 0.30
        # → floor = min(0.316, 0.30) = 0.30 → keeps cands ≥ 0.30
        ctx1 = self._make_ctx_adaptive(cap=0.30)
        ctx1.candidates = [
            _make_cand(f"X{i}", panel_score=0.0, rank_score=s)
            for i, s in enumerate([0.28, 0.29, 0.30, 0.31, 0.32])
        ]
        task.run(ctx1)
        kept1 = {c.ticker for c in ctx1.candidates}
        # X0=0.28, X1=0.29 drop; X2=0.30, X3=0.31, X4=0.32 keep
        self.assertEqual(kept1, {"X2", "X3", "X4"})
        # Bar 2 — wide spread, mean=0.40, sample-stdev≈0.279,
        # mean+std≈0.679 > cap → floor = 0.30 → keeps cands ≥ 0.30
        ctx2 = self._make_ctx_adaptive(cap=0.30)
        ctx2.candidates = [
            _make_cand(f"Y{i}", panel_score=0.0, rank_score=s)
            for i, s in enumerate([0.10, 0.20, 0.40, 0.60, 0.80])
        ]
        task.run(ctx2)
        kept2 = {c.ticker for c in ctx2.candidates}
        self.assertEqual(kept2, {"Y2", "Y3", "Y4"})


# 2026-05-04 user mandate: "rank_score need to be collected properly
# for future fine tune". Snapshot the full pre-veto candidate list so
# the persistence layer captures BOTH kept and vetoed rows.

class TestPreVetoCandidateSnapshot(unittest.TestCase):
    def test_snapshot_captures_all_candidates_before_veto(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.5)
        cands = [
            _make_cand("A", panel_score=0.0, rank_score=0.1),  # vetoed
            _make_cand("B", panel_score=0.0, rank_score=0.6),  # kept
            _make_cand("C", panel_score=0.0, rank_score=0.2),  # vetoed
        ]
        ctx.candidates = list(cands)
        VetoWeakBuysTask().run(ctx)
        snap = getattr(ctx, "_full_candidate_snapshot", None)
        self.assertIsNotNone(snap)
        self.assertEqual([c.ticker for c in snap], ["A", "B", "C"])
        # Survivors only in ctx.candidates
        self.assertEqual([c.ticker for c in ctx.candidates], ["B"])

    def test_blocked_by_ticker_records_veto_reason(self):
        """Vetoed candidates must have a per-ticker reason in
        ctx._blocked_by_ticker so record_candidate_scores can persist
        it as the SQL `blocked_by` column."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.5)
        ctx.candidates = [
            _make_cand("BELOW", panel_score=0.0, rank_score=0.1),
            _make_cand("OK",    panel_score=0.0, rank_score=0.7),
        ]
        VetoWeakBuysTask().run(ctx)
        blocked = ctx._blocked_by_ticker
        self.assertEqual(blocked.get("BELOW"), "veto:rank_score_below_floor")
        # The kept one is NOT in blocked map
        self.assertNotIn("OK", blocked)

    def test_nan_rank_score_tagged_distinctly(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=0.3)
        ctx.candidates = [_make_cand("NAN", panel_score=0.0,
                                      rank_score=float("nan"))]
        VetoWeakBuysTask().run(ctx)
        self.assertEqual(ctx._blocked_by_ticker.get("NAN"),
                          "veto:rank_score_nan")

    def test_snapshot_taken_even_when_floor_disabled(self):
        """Even with buy_floor=None (no veto), snapshot must be set so
        the adapter persists the full bar's distribution."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        ctx = _make_ctx(buy_floor=None)
        ctx.candidates = [
            _make_cand("X", panel_score=0.0, rank_score=0.5),
            _make_cand("Y", panel_score=0.0, rank_score=0.1),
        ]
        VetoWeakBuysTask().run(ctx)
        snap = getattr(ctx, "_full_candidate_snapshot", None)
        self.assertIsNotNone(snap)
        self.assertEqual([c.ticker for c in snap], ["X", "Y"])


class TestAdapterParityForFullSnapshot(unittest.TestCase):
    """Both sim AND runner adapters must persist the pre-veto snapshot
    so SQL queries on candidate_scores see the full distribution, not
    just survivors. lean.py doesn't write to the DB so it's exempt.

    Source-level pin (the adapter wiring is straightforward and easy
    to silently regress in a future refactor)."""

    def _check_adapter(self, adapter_filename: str, label: str):
        """The adapter must reference _full_candidate_snapshot BEFORE
        the record_candidate_scores() call (so the snapshot is the
        list passed in). Search a window AROUND the call site, not
        just forward, since the lookup is the few lines preceding it.
        """
        path = REPO / "backtesting" / "renquant_104" / "adapters" / adapter_filename
        src = path.read_text()
        idx = src.find("record_candidate_scores(")
        assert idx >= 0, f"{label} missing record_candidate_scores"
        # Slice 800 chars BEFORE the call + 200 after, so the
        # `cand_pool = getattr(ctx, "_full_candidate_snapshot", ...)`
        # binding is captured.
        start = max(0, idx - 800)
        block = src[start:idx + 200]
        self.assertIn("_full_candidate_snapshot", block,
                      f"{label} must read ctx._full_candidate_snapshot")
        # And the snapshot lookup must come BEFORE the call.
        snap_idx = block.find("_full_candidate_snapshot")
        call_idx = block.find("record_candidate_scores(")
        self.assertLess(snap_idx, call_idx,
                         f"{label}: _full_candidate_snapshot must be "
                         f"resolved BEFORE the record_candidate_scores call")

    def test_sim_adapter_reads_full_snapshot(self):
        self._check_adapter("sim.py", "sim.py")

    def test_runner_adapter_reads_full_snapshot(self):
        self._check_adapter("runner.py", "runner.py (live-side parity with sim)")


class TestPanelScoringJobOrdering(unittest.TestCase):
    """Veto must run AFTER ApplyGlobalCalibrationTask in the chain."""

    def test_veto_is_after_calibration(self):
        from kernel.panel_pipeline.job_panel_scoring import (
            PanelScoringJob, VetoWeakBuysTask, ApplyGlobalCalibrationTask,
            ApplyScoresTask,
        )
        tasks = PanelScoringJob().tasks
        names = [type(t).__name__ for t in tasks]
        self.assertIn("ApplyGlobalCalibrationTask", names)
        self.assertIn("VetoWeakBuysTask", names)
        self.assertIn("ApplyScoresTask", names)
        # Veto AFTER calibration:
        self.assertGreater(names.index("VetoWeakBuysTask"),
                           names.index("ApplyGlobalCalibrationTask"))
        # Veto AFTER ApplyScores (always — that's where rank_score is set):
        self.assertGreater(names.index("VetoWeakBuysTask"),
                           names.index("ApplyScoresTask"))

    def test_veto_is_before_kelly(self):
        """Kelly sizing should size only the surviving candidates."""
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        tasks = PanelScoringJob().tasks
        names = [type(t).__name__ for t in tasks]
        if "ApplyKellySizingTask" in names:
            self.assertLess(names.index("VetoWeakBuysTask"),
                            names.index("ApplyKellySizingTask"))


if __name__ == "__main__":
    unittest.main()
