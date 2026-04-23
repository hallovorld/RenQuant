"""Tests for training_panel/global_calibrator.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.global_calibrator import (  # noqa: E402
    GlobalPanelCalibration,
    fit_global_calibrator,
)


def _synthetic_panel(n_tickers: int = 10, n_bars: int = 200, seed: int = 0,
                      signal: float = 0.5, noise: float = 0.05):
    """Synthesize panel scores + forward returns with known correlation.

    signal=0.5 means forward_return ≈ 0.5 × panel_score + noise.
    """
    rng = np.random.default_rng(seed)
    panel_scores: dict[str, pd.Series] = {}
    future_returns: dict[str, pd.Series] = {}
    idx = pd.bdate_range("2024-01-02", periods=n_bars)
    for t in range(n_tickers):
        tk = f"T{t}"
        raw = rng.normal(0, 0.5, n_bars)
        fwd = signal * raw + rng.normal(0, noise, n_bars)
        panel_scores[tk]   = pd.Series(raw, index=idx)
        future_returns[tk] = pd.Series(fwd, index=idx)
    return panel_scores, future_returns


class TestFit:
    def test_fit_monotonic_output(self):
        ps, fr = _synthetic_panel(signal=0.5, noise=0.03)
        cal = fit_global_calibrator(ps, fr, lookahead_days=5, threshold=0.02)
        # Monotonic in raw score → probability
        test_xs = np.linspace(-1.5, 1.5, 50)
        probs = cal.calibrate_probability_vec(test_xs)
        diffs = np.diff(probs)
        # isotonic guarantees non-decreasing
        assert (diffs >= -1e-9).all(), "probability head must be monotone non-decreasing"

    def test_fit_recovers_positive_signal(self):
        ps, fr = _synthetic_panel(signal=0.8, noise=0.02)
        cal = fit_global_calibrator(ps, fr, lookahead_days=5, threshold=0.0)
        # High raw score → high calibrated P
        assert cal.calibrate_probability(1.5) > cal.calibrate_probability(-1.5)
        # ER head: high raw → high ER
        assert cal.expected_return(1.5) > cal.expected_return(-1.5)

    def test_metadata_populated(self):
        ps, fr = _synthetic_panel(seed=42)
        cal = fit_global_calibrator(ps, fr)
        assert cal.metadata["n_tickers"] == 10
        assert cal.metadata["n_rows"] > 1000
        assert "pool_ic" in cal.metadata
        assert "prob_base_rate" in cal.metadata

    def test_raises_on_insufficient_data(self):
        # Too few rows across all tickers
        idx = pd.bdate_range("2024-01-02", periods=30)
        ps = {"T0": pd.Series([0.5] * 30, index=idx)}
        fr = {"T0": pd.Series([0.01] * 30, index=idx)}
        with pytest.raises(ValueError, match="min_rows"):
            fit_global_calibrator(ps, fr, min_rows=1000)

    def test_ignores_tickers_missing_from_future_returns(self):
        ps = {"T0": pd.Series([0.5] * 200,
                              index=pd.bdate_range("2024-01-02", periods=200)),
              "T1": pd.Series([0.3] * 200,
                              index=pd.bdate_range("2024-01-02", periods=200))}
        fr = {"T0": pd.Series([0.02] * 200,
                              index=pd.bdate_range("2024-01-02", periods=200))}
        cal = fit_global_calibrator(ps, fr, min_rows=100)
        # Only T0 contributed
        assert cal.metadata["n_rows"] == 200
        assert cal.metadata["n_tickers"] == 1   # only one ticker had forward returns


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        ps, fr = _synthetic_panel(seed=7)
        cal = fit_global_calibrator(ps, fr)
        path = tmp_path / "panel-cal.json"
        cal.save(path, metadata={"training_notes": "unit-test"})
        assert path.exists()

        cal2 = GlobalPanelCalibration.load(path)
        # Point predictions must match
        for x in [-1.0, -0.3, 0.0, 0.5, 1.2]:
            assert cal2.calibrate_probability(x) == pytest.approx(
                cal.calibrate_probability(x), rel=1e-6,
            )
            assert cal2.expected_return(x) == pytest.approx(
                cal.expected_return(x), rel=1e-6,
            )
        assert cal2.metadata.get("training_notes") == "unit-test"

    def test_load_rejects_wrong_kind(self, tmp_path):
        import json as _j
        path = tmp_path / "wrong.json"
        path.write_text(_j.dumps({"kind": "something_else"}))
        with pytest.raises(ValueError, match="global_panel_calibration"):
            GlobalPanelCalibration.load(path)


class TestApplyInPipeline:
    """ApplyGlobalCalibrationTask writes calibrated rank_score/expected_return."""

    def _ctx_with_candidates(self, tmp_path, calib):
        import datetime as dt
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult

        cfg = {
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "global_calibration": {
                        "enabled": True,
                        "artifact_path": str(tmp_path / "cal.json"),
                    },
                },
            },
        }
        ctx = InferenceContext(config=cfg, today=dt.date(2026, 4, 22))
        ctx.candidates = [
            CandidateResult(ticker="HIGH", raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=1.5),
            CandidateResult(ticker="LOW",  raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=-1.5),
        ]
        ctx.holdings = {}
        ctx._global_calibrator = calib  # noqa: SLF001
        return ctx

    def test_rank_score_reflects_calibration(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ps, fr = _synthetic_panel(signal=0.8, noise=0.02)
        cal = fit_global_calibrator(ps, fr, threshold=0.02)
        ctx = self._ctx_with_candidates(tmp_path, cal)

        ApplyGlobalCalibrationTask().run(ctx)

        high = next(c for c in ctx.candidates if c.ticker == "HIGH")
        low  = next(c for c in ctx.candidates if c.ticker == "LOW")
        # Higher panel_score → higher calibrated rank_score
        assert high.rank_score > low.rank_score
        # rank_score should land in [0, 1]
        assert 0.0 <= high.rank_score <= 1.0
        assert 0.0 <= low.rank_score  <= 1.0
        # expected_return written too
        assert high.expected_return > low.expected_return

    def test_noop_when_ngboost_enabled(self, tmp_path):
        """NGBoost's μ−λσ already gives a calibrated score — global calibrator defers."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ps, fr = _synthetic_panel()
        cal = fit_global_calibrator(ps, fr)
        ctx = self._ctx_with_candidates(tmp_path, cal)
        # Enable NGBoost → global calibration should short-circuit
        ctx.config["ranking"]["panel_scoring"]["ngboost"] = {"enabled": True}

        ApplyGlobalCalibrationTask().run(ctx)

        # rank_score should be UNCHANGED (still 0.5 from construction)
        for c in ctx.candidates:
            assert c.rank_score == 0.5
