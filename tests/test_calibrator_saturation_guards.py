"""Regression guards for the 2026-05-15 calibrator-saturation P0.

User-observed incident (2026-05-12 → 2026-05-15):
  * Almost every candidate calibrated to rank_score ~ 1.0000
  * Kelly sizing logged `mu_none=33` for every candidate (no μ on objects)
  * QP fell back to uniform 10.17% per-position target (no mu-weighting)

Root causes:
  1. prod calibrator artifact had expected_return.y in [-29.9%, +100.0%]
     and probability.y reaching 1.0 — CLAUDE.md §5.13.12 violation
     (output not clipped at train site).
  2. recent-12mo calibrator had y in [0.0, 1.0] for probability with
     ~16% of training samples landing in the saturated >=0.95 tail.

These tests pin:
  A. GlobalPanelCalibration.load() clips out-of-range y values and warns.
  B. ApplyGlobalCalibrationTask logs CALIBRATOR-SATURATED when post-
     calibrate rank_score IQR < 0.05 or upper-tail saturation >= 50%.
  C. ApplyGlobalCalibrationTask logs CALIBRATOR-ER-OUT-OF-RANGE when
     any |expected_return| > 0.20.

Phase 1 of P0 triage (2026-05-15) — detection-only, no behavior change.
Phase 2 wires mu/sigma; Phase 3 retrains calibrator with clips.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _write_artifact(tmp: Path, *, prob_y, er_y, prob_x=None, er_x=None):
    """Helper: write a calibrator artifact with given y arrays."""
    if prob_x is None:
        prob_x = np.linspace(-1.0, 1.0, len(prob_y)).tolist()
    if er_x is None:
        er_x = np.linspace(-1.0, 1.0, len(er_y)).tolist()
    payload = {
        "version": 1,
        "kind": "global_panel_calibration",
        "trained_date": "2026-05-15",
        "probability":     {"x": list(prob_x), "y": list(prob_y)},
        "expected_return": {"x": list(er_x),   "y": list(er_y)},
        "metadata": {},
    }
    p = tmp / "cal.json"
    p.write_text(json.dumps(payload))
    return p


class TestCalibratorLoadGuards:
    """Guard A — load() clips out-of-range y values."""

    def test_load_clips_er_out_of_bounds(self, caplog):
        """expected_return.y with |y|>0.20 should be clipped to ±0.20."""
        from training_panel.global_calibrator import GlobalPanelCalibration

        with tempfile.TemporaryDirectory() as tmp:
            artifact = _write_artifact(
                Path(tmp),
                prob_y=[0.1, 0.3, 0.5, 0.7, 0.9],
                er_y=[-0.5, -0.1, 0.0, 0.1, +1.0],  # ±1.0 obviously broken
            )
            with caplog.at_level(logging.WARNING):
                cal = GlobalPanelCalibration.load(artifact)

            assert float(cal.er_y.max()) <= 0.20 + 1e-9, (
                f"er_y not clipped: max={cal.er_y.max()}"
            )
            assert float(cal.er_y.min()) >= -0.20 - 1e-9, (
                f"er_y not clipped: min={cal.er_y.min()}"
            )
            assert any("expected_return.y" in r.message and "0.20" in r.message
                       for r in caplog.records), \
                f"expected ER warning log missing; got: {[r.message for r in caplog.records]}"

    def test_load_clips_probability_out_of_bounds(self, caplog):
        """probability.y outside [0,1] should be clipped + warn."""
        from training_panel.global_calibrator import GlobalPanelCalibration

        with tempfile.TemporaryDirectory() as tmp:
            artifact = _write_artifact(
                Path(tmp),
                prob_y=[-0.1, 0.0, 0.5, 1.0, 1.1],  # leaks beyond [0,1]
                er_y=[-0.05, 0.0, 0.0, 0.05, 0.1],
            )
            with caplog.at_level(logging.WARNING):
                cal = GlobalPanelCalibration.load(artifact)

            assert float(cal.prob_y.min()) >= 0.0 - 1e-9
            assert float(cal.prob_y.max()) <= 1.0 + 1e-9
            assert any("probability.y" in r.message
                       for r in caplog.records)

    def test_load_clean_artifact_no_warning(self, caplog):
        """An in-bounds artifact should load silently — no clip warnings."""
        from training_panel.global_calibrator import GlobalPanelCalibration

        with tempfile.TemporaryDirectory() as tmp:
            artifact = _write_artifact(
                Path(tmp),
                prob_y=[0.10, 0.30, 0.50, 0.70, 0.90],
                er_y=[-0.05, -0.02, 0.00, 0.02, 0.08],
            )
            with caplog.at_level(logging.WARNING):
                GlobalPanelCalibration.load(artifact)
            assert not any("clipping" in r.message.lower()
                            or "out of [0,1]" in r.message
                            for r in caplog.records), \
                f"clean artifact should not warn; got: {[r.message for r in caplog.records]}"

    def test_train_site_clips_er_to_20pct(self):
        """Phase 4 — fit_global_calibrator must clip ER to ±0.20 at train
        site. Pre-fix the clip was ±1.0 which let +100% returns flow into
        er_y and saturate the calibrator's upper tail."""
        from training_panel.global_calibrator import fit_global_calibrator
        import pandas as pd

        # Synthesize panel with deliberate tail-event returns (+200%, -150%).
        n = 2000
        rng = np.random.default_rng(seed=42)
        scores = pd.Series(
            rng.normal(0, 1, n),
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        # Returns: most ~ ±5%, but a few tail events at ±200%
        returns = rng.normal(0, 0.05, n)
        returns[0] = +2.0   # +200% — should be clipped to +0.20
        returns[1] = -1.5   # -150% — should be clipped to -0.20
        fwd = pd.Series(returns, index=scores.index)

        cal = fit_global_calibrator(
            {"T01": scores}, {"T01": fwd},
            lookahead_days=10, min_rows=100,
        )
        assert float(cal.er_y.max()) <= 0.20 + 1e-9, (
            f"train-site clip not enforced: er_y.max()={cal.er_y.max()}"
        )
        assert float(cal.er_y.min()) >= -0.20 - 1e-9, (
            f"train-site clip not enforced: er_y.min()={cal.er_y.min()}"
        )

    def test_preclip_snapshot_triggers_guard(self, caplog):
        """The pre-clip snapshot (preserved as `.pre-2026-05-15-clip.json`)
        is the live evidence of the bug class this guard catches. Verifies
        the load-time guard fires + clips er_y to ±0.20 on the actual
        broken artifact, AND that the CURRENT prod artifact (post-refit)
        loads silently. Together these pin both branches of the invariant."""
        from training_panel.global_calibrator import GlobalPanelCalibration

        snap = REPO_ROOT / ("backtesting/renquant_104/artifacts/prod/"
                             "panel-rank-calibration.pre-2026-05-15-clip.json")
        prod = REPO_ROOT / ("backtesting/renquant_104/artifacts/prod/"
                             "panel-rank-calibration.json")
        if not snap.exists() or not prod.exists():
            pytest.skip(f"calibrator artifacts not on disk")

        # Broken snapshot trips the guard
        with caplog.at_level(logging.WARNING):
            cal_bad = GlobalPanelCalibration.load(snap)
        assert float(cal_bad.er_y.max()) <= 0.20 + 1e-9
        assert any("expected_return" in r.message and "0.20" in r.message
                   for r in caplog.records), \
            "snapshot has |er_y|>0.20 but guard didn't fire"

        # Clean current prod loads silently — no clip warning
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            cal_good = GlobalPanelCalibration.load(prod)
        assert float(cal_good.er_y.max()) <= 0.20 + 1e-9
        assert not any("clipping" in r.message.lower()
                       for r in caplog.records), \
            "clean prod artifact triggered clip warning unexpectedly"


class TestApplyGlobalCalibrationSaturationGuard:
    """Guard B — ApplyGlobalCalibrationTask warns on saturated output."""

    def _make_ctx(self, candidate_rank_scores, candidate_ers=None):
        """Build a minimal ctx that satisfies ApplyGlobalCalibrationTask's
        access pattern.

        The mock calibrator passes panel_score → rank_score identity-style
        so we control the post-calibrate rank_score by setting panel_score
        to the desired output. Similarly for expected_return.
        """
        if candidate_ers is None:
            candidate_ers = [0.0] * len(candidate_rank_scores)
        # candidate's panel_score == target rank_score (mock identity map)
        ps_to_rank = dict(enumerate(candidate_rank_scores))
        ps_to_er   = dict(enumerate(candidate_ers))
        cands = []
        for i, _ in enumerate(candidate_rank_scores):
            c = SimpleNamespace(
                ticker=f"T{i:02d}",
                # encode index in panel_score so mock can dispatch
                panel_score=float(i),
                rank_score=None,
                expected_return=None,
                mu=None, sigma=None,
                kelly_target_pct=0.0,
            )
            cands.append(c)
        # Mock calibrator: panel_score (= index) → desired rank_score / er
        cal = SimpleNamespace(
            calibrate_probability=lambda s: ps_to_rank[int(s)],
            expected_return     =lambda s: ps_to_er[int(s)],
        )
        ctx = SimpleNamespace(
            config={"ranking": {"panel_scoring": {
                "global_calibration": {"enabled": True}}}},
            candidates=cands,
            holdings={},
            regime="BULL_CALM",
            _global_calibrator=cal,
            _regime_calibrators=None,
        )
        return ctx

    def test_saturated_upper_tail_warns(self, caplog):
        """50%+ of candidates with rank_score >= 0.95 → guard fires."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        # 8/10 candidates saturated at the top
        ctx = self._make_ctx(
            [0.97, 0.98, 0.99, 0.96, 0.97, 0.98, 0.99, 0.95, 0.40, 0.50],
        )
        with caplog.at_level(logging.WARNING):
            ApplyGlobalCalibrationTask().run(ctx)
        assert any("CALIBRATOR-SATURATED" in r.message
                   for r in caplog.records), \
            f"saturation guard didn't fire; logs: {[r.message for r in caplog.records]}"

    def test_low_iqr_warns(self, caplog):
        """IQR < 0.05 → guard fires (collapsed ranking)."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        # All scores in [0.50, 0.51] — IQR ~ 0.005
        ctx = self._make_ctx(
            [0.500, 0.502, 0.504, 0.506, 0.508, 0.510],
        )
        with caplog.at_level(logging.WARNING):
            ApplyGlobalCalibrationTask().run(ctx)
        assert any("CALIBRATOR-SATURATED" in r.message
                   for r in caplog.records)

    def test_healthy_distribution_no_warning(self, caplog):
        """Spread-out rank_scores → no saturation warning."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ctx = self._make_ctx(
            [0.10, 0.25, 0.40, 0.50, 0.60, 0.72, 0.81, 0.88],
        )
        with caplog.at_level(logging.WARNING):
            ApplyGlobalCalibrationTask().run(ctx)
        assert not any("CALIBRATOR-SATURATED" in r.message
                        for r in caplog.records)

    def test_extreme_expected_return_warns(self, caplog):
        """|expected_return| > 0.20 → ER-out-of-range guard fires."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ctx = self._make_ctx(
            [0.40, 0.50, 0.60],
            candidate_ers=[0.05, +0.85, -0.30],  # 85% predicted return — broken
        )
        with caplog.at_level(logging.WARNING):
            ApplyGlobalCalibrationTask().run(ctx)
        assert any("CALIBRATOR-ER-OUT-OF-RANGE" in r.message
                   for r in caplog.records)

    def test_in_range_expected_return_no_warn(self, caplog):
        """|expected_return| ≤ 0.20 → no ER warning."""
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
        )
        ctx = self._make_ctx(
            [0.40, 0.50, 0.60],
            candidate_ers=[0.05, +0.18, -0.12],
        )
        with caplog.at_level(logging.WARNING):
            ApplyGlobalCalibrationTask().run(ctx)
        assert not any("CALIBRATOR-ER-OUT-OF-RANGE" in r.message
                        for r in caplog.records)
