"""Calibrator load + LEAN guard must NOT default to a hardcoded prod path.

§5.13.14 regression guards. Both code paths used to fall back to
artifacts/prod/... silently when sim/research configs omitted
artifact_path. Calibrator side was a load-only misleading-results risk
(no corruption). LEAN guard side could read the WRONG trained_date and
falsely pass or fail the leakage check for a sim backtest.

Post-fix: both refuse to default, log a clear message, and skip the
operation rather than silently use prod.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── Calibrator ──────────────────────────────────────────────────────────


def _ctx_for_calibrator(global_cal_cfg: dict):
    return SimpleNamespace(
        config={
            "_strategy_dir": str(REPO_ROOT / "backtesting" / "renquant_104"),
            "ranking": {
                "panel_scoring": {
                    "global_calibration": global_cal_cfg,
                },
            },
        },
    )


class TestCalibratorNoDefaultPath:

    def test_enabled_without_artifact_path_disables(self):
        """enabled=true but no artifact_path → calibrator stays None."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        task = LoadGlobalCalibrationTask()
        ctx = _ctx_for_calibrator({"enabled": True})
        task.run(ctx)
        assert ctx._global_calibrator is None
        assert ctx._global_calibrator_missing_reason == "calibrator_missing_path"

    def test_enabled_with_empty_artifact_path_disables(self):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        task = LoadGlobalCalibrationTask()
        ctx = _ctx_for_calibrator({"enabled": True, "artifact_path": ""})
        task.run(ctx)
        assert ctx._global_calibrator is None
        assert ctx._global_calibrator_missing_reason == "calibrator_missing_path"

    def test_apply_missing_calibrator_fails_closed(self):
        """Enabled calibrator with no loaded artifact must not leave raw scores tradable."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        from kernel.selection import CandidateResult

        ctx = _ctx_for_calibrator({"enabled": True, "artifact_path": "missing.json"})
        ctx.candidates = [
            CandidateResult(ticker="AAA", raw_score=0.2, rank_score=0.2,
                            rs_score=0.0, detail="", panel_score=0.2),
        ]
        ctx.holdings = {}
        ctx.buy_blocked = False
        ctx.skip_buys = False
        ctx._global_calibrator_missing_reason = "calibrator_missing_path"

        result = ApplyGlobalCalibrationTask().run(ctx)

        assert result is False
        assert ctx.candidates == []
        assert ctx.buy_blocked is True
        assert ctx.skip_buys is True
        assert ctx._calibrator_contract_failed is True
        assert ctx._blocked_by_ticker["AAA"] == "calibrator_missing_path"

    def test_disabled_short_circuits(self):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        task = LoadGlobalCalibrationTask()
        ctx = _ctx_for_calibrator({"enabled": False})
        ctx._global_calibrator = "sentinel"
        task.run(ctx)
        assert ctx._global_calibrator == "sentinel"


# ── LEAN leakage guard ──────────────────────────────────────────────────


class TestLeanGuardNoDefaultPath:

    def test_no_artifact_path_skips_guard_without_raising(self, tmp_path):
        """panel_scoring.enabled=true but no artifact_path → guard skips
        (logs warning) rather than reading prod artifact."""
        from kernel.walk_forward.lean_guard import assert_lean_panel_no_leakage
        config = {
            "ranking": {"panel_scoring": {"enabled": True}},
            "backtest_end": "2024-01-01",
        }
        # Should not raise (guard skipped). No assertion on side-effect —
        # absence of exception IS the pass condition.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False
        )

    def test_disabled_panel_scoring_returns_early(self, tmp_path):
        """panel_scoring.enabled=false → guard returns without reading."""
        from kernel.walk_forward.lean_guard import assert_lean_panel_no_leakage
        config = {
            "ranking": {"panel_scoring": {"enabled": False}},
            "backtest_end": "2024-01-01",
        }
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False
        )

    def test_live_mode_returns_early(self, tmp_path):
        """is_live_mode=True → guard always returns early."""
        from kernel.walk_forward.lean_guard import assert_lean_panel_no_leakage
        config = {
            "ranking": {"panel_scoring": {"enabled": True}},
            "backtest_end": "2024-01-01",
        }
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=True
        )
