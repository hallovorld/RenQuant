"""Regression test: calibrator expected_return.y MUST be in [-1, +1].

Bug found 2026-05-09 audit hunt: 66/208 expected_return.y knots in
panel-rank-calibration.json exceeded ±100% (max +401%). Root cause:
fit_global_calibrator's isotonic ER head was fit on raw fwd_60d returns
without clipping. Individual ticker rows with > 100% 60-day return
(small-cap runaways, IPO pops) propagated into er_y. The QP solver
then read these as μ_i and inflated position weights / Kelly sizing
on top-scored tickers.

Fix: training_panel/global_calibrator.py now clips fwd_all to [-1, +1]
BEFORE isotonic fit, AND clips emitted er_y as defense-in-depth.

This test pins the invariant on every shipped calibrator artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


class TestCalibratorERRange:

    def test_production_calibrator_y_in_range(self):
        """Production panel-rank-calibration.json's expected_return.y MUST
        all be in [-1, +1]. If not, the QP solver's μ vector is poisoned."""
        path = REPO / "backtesting" / "renquant_104" / "artifacts" / "panel-rank-calibration.json"
        if not path.exists():
            pytest.skip("Production calibrator not present")
        m = json.loads(path.read_text())
        y = m.get("expected_return", {}).get("y", [])
        assert y, "calibrator has no expected_return.y values"

        out_of_range = [(i, v) for i, v in enumerate(y) if v > 1.0 or v < -1.0]
        assert not out_of_range, \
            f"AUDIT REGRESSION: {len(out_of_range)}/{len(y)} expected_return.y " \
            f"knots outside [-1, +1]. The QP solver uses these as μ_i — values " \
            f"like +4.0 mean 'expected +400% return in 60d', which inflates " \
            f"position weights wildly. First 5 violations: {out_of_range[:5]}"


class TestFitClampsExtremes:
    """Audit guard: fit_global_calibrator must clip extreme fwd returns
    BEFORE feeding them to IsotonicRegression."""

    def test_global_calibrator_clips_before_isotonic_fit(self):
        src = (REPO / "backtesting" / "renquant_104" / "training_panel"
               / "global_calibrator.py").read_text()
        # The clip must happen before iso_er.fit
        clip_idx = src.find("np.clip(fwd_all, -1.0, 1.0)")
        fit_idx = src.find("iso_er = IsotonicRegression")
        assert clip_idx > 0, \
            "AUDIT REGRESSION: fit_global_calibrator no longer clips fwd_all to " \
            "[-1, +1] before isotonic ER fit. Extreme outliers (e.g. fwd_60d " \
            "= +400%) will propagate into er_y again."
        assert clip_idx < fit_idx, \
            "AUDIT REGRESSION: clip occurs AFTER iso_er.fit — the fit consumed " \
            "raw values; the clip is now no-op."

    def test_global_calibrator_clips_emitted_er_y(self):
        src = (REPO / "backtesting" / "renquant_104" / "training_panel"
               / "global_calibrator.py").read_text()
        # Defense-in-depth: even after fwd_for_er clip, emit er_y clipped.
        assert "er_y = np.clip(er_y, -1.0, 1.0)" in src, \
            "AUDIT REGRESSION: emitted er_y is no longer defense-clipped to " \
            "[-1, +1]. sklearn's isotonic CAN extrapolate knots outside fit " \
            "range in degenerate cases."
