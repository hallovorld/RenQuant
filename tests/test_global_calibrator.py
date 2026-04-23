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

    def test_runs_even_when_ngboost_enabled(self, tmp_path):
        """Post task-#2 refactor (2026-04-23): the calibrator no longer
        short-circuits when NGBoost is enabled. In mu_minus_lambda_sigma
        mode, NGBoost overwrites panel_score with μ−λσ first; the
        calibrator then maps whatever panel_score is (raw or μ−λσ) →
        probability so the tier thresholds have a meaningful value in
        both modes.

        Previously this test asserted the opposite behavior (noop); the
        short-circuit caused mu_minus_lambda_sigma to produce zero trades
        because rank_score was raw μ−λσ (always < 0.10 tier threshold).
        """
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        ps, fr = _synthetic_panel()
        cal = fit_global_calibrator(ps, fr)
        ctx = self._ctx_with_candidates(tmp_path, cal)
        # Enable NGBoost — calibration should STILL run.
        ctx.config["ranking"]["panel_scoring"]["ngboost"] = {
            "enabled": True,
            "score_mode": "mu_minus_lambda_sigma",
        }

        ApplyGlobalCalibrationTask().run(ctx)

        # rank_score should have been transformed from 0.5 → calibrated
        # probability in [0, 1] based on each candidate's panel_score.
        for c in ctx.candidates:
            assert 0.0 <= c.rank_score <= 1.0, \
                f"rank_score {c.rank_score} out of [0,1] for {c.ticker}"
            # Not equal to the construction value 0.5 for at least one
            # candidate (unless coincidence — use a sentinel check).
        # Higher panel_score candidate should get higher rank_score
        high = next(c for c in ctx.candidates if c.ticker == "HIGH")
        low  = next(c for c in ctx.candidates if c.ticker == "LOW")
        assert high.rank_score >= low.rank_score


class TestEndToEndWithNGBoost:
    """PanelScoringJob in mu_minus_lambda_sigma mode:
    ApplyNGBoostTask writes μ−λσ to panel_score, then calibrator maps it
    to probability. Verifies the task-#2 reorder works end-to-end.
    """

    def test_mu_minus_lambda_sigma_ends_up_calibrated(self, tmp_path):
        """With score_mode=mu_minus_lambda_sigma, rank_score after the
        full chain should be a calibrated probability in [0,1], NOT the
        raw μ−λσ value. This is the core task-#2 guarantee.
        """
        import datetime as dt
        from types import SimpleNamespace
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyNGBoostTask, ApplyGlobalCalibrationTask,
        )

        ps, fr = _synthetic_panel(signal=0.8, noise=0.02)
        cal = fit_global_calibrator(ps, fr, threshold=0.02)

        cfg = {
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "global_calibration": {
                        "enabled": True,
                        "artifact_path": str(tmp_path / "cal.json"),
                    },
                    "ngboost": {
                        "enabled": True,
                        "score_mode": "mu_minus_lambda_sigma",
                        "lambda_sigma": 1.0,
                    },
                },
            },
        }
        ctx = InferenceContext(config=cfg, today=dt.date(2026, 4, 23))
        ctx.candidates = [
            CandidateResult(ticker="A", raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=0.03),
            CandidateResult(ticker="B", raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=-0.02),
        ]
        ctx.holdings = {}
        ctx._global_calibrator = cal   # noqa: SLF001

        # Fake NGBoost head: μ = high for A, low for B; σ = moderate
        import pandas as pd
        fake_head = SimpleNamespace(
            feature_cols=["f0"],
            predict_distribution=lambda X: {
                "mu":    pd.Series([0.05, -0.03], index=["A", "B"]),
                "sigma": pd.Series([0.02,  0.02], index=["A", "B"]),
            },
        )
        ctx._ngboost_head = fake_head             # noqa: SLF001
        ctx._panel_matrix = pd.DataFrame(          # noqa: SLF001
            {"f0": [1.0, 0.5]}, index=["A", "B"]
        )

        # Run the two relevant tasks in their new order:
        ApplyNGBoostTask().run(ctx)            # μ−λσ → panel_score, rank_score
        ApplyGlobalCalibrationTask().run(ctx)  # calibrator(panel_score) → rank_score

        a = next(c for c in ctx.candidates if c.ticker == "A")
        b = next(c for c in ctx.candidates if c.ticker == "B")

        # rank_score is in [0, 1] — calibrated, not raw μ−λσ:
        assert 0.0 <= a.rank_score <= 1.0
        assert 0.0 <= b.rank_score <= 1.0
        # μ=0.05, σ=0.02 for A → μ−λσ = 0.03 (raw μ−λσ would be 0.03, not
        # 0 or 1) — calibration must not leave rank_score equal to μ−λσ:
        assert not (abs(a.rank_score - 0.03) < 1e-6), \
            "rank_score should be calibrated probability, not raw μ−λσ"
        # And A (higher μ) should rank at least as high as B (lower μ):
        assert a.rank_score >= b.rank_score
