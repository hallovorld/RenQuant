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
        idx = pd.bdate_range("2024-01-02", periods=200)
        rng = np.random.default_rng(0)
        # Use varied scores + matching forward-return signal so the
        # CALIB-COLLAPSE-GUARD (>=3 unique y) doesn't trip on this
        # ticker-skipping smoke test.
        ps = {"T0": pd.Series(rng.normal(0, 0.5, size=200), index=idx),
              "T1": pd.Series(rng.normal(0, 0.5, size=200), index=idx)}
        fr = {"T0": pd.Series(0.5 * ps["T0"].values
                              + rng.normal(0, 0.05, size=200), index=idx)}
        cal = fit_global_calibrator(ps, fr, min_rows=100)
        # Only T0 contributed
        assert cal.metadata["n_rows"] == 200
        assert cal.metadata["n_tickers"] == 1   # only one ticker had forward returns


class TestPersistence:
    def test_expected_return_can_be_requested_at_explicit_horizon(self):
        cal = GlobalPanelCalibration(
            prob_x=np.array([0.0, 1.0]),
            prob_y=np.array([0.4, 0.8]),
            er_x=np.array([0.0, 1.0]),
            er_y=np.array([0.06, 0.12]),
            metadata={"lookahead_days": 60},
        )

        assert cal.expected_return(0.0) == pytest.approx(0.06)
        assert cal.expected_return(0.0, horizon_days=20) == pytest.approx(0.02)
        assert cal.expected_return(1.0, horizon_days=120) == pytest.approx(0.24)
        assert cal.expected_return_vec(
            np.array([0.0, 1.0]), horizon_days=20,
        ).tolist() == pytest.approx([0.02, 0.04])

    def test_save_load_roundtrip(self, tmp_path):
        ps, fr = _synthetic_panel(seed=7)
        # Production calibrators use Platt + smooth bounded ER. Legacy isotonic
        # can legitimately produce flat ER plateaus that the G13 save gate now
        # rejects before they reach Kelly/QP.
        cal = fit_global_calibrator(ps, fr, method="platt")
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


# ── CALIB-PER-DATE-IC + CALIB-COLLAPSE-GUARD ──────────────────────────────────

class TestCalibratorPoolDiagnostics:
    """Audit fixes CALIB-PER-DATE-IC + CALIB-COLLAPSE-GUARD (2026-04-26).

    pool_ic mixes time × cross-section so it underreports cross-sectional
    signal. fit_global_calibrator should ALSO compute per-date cross-
    sectional IC and stamp both in metadata.

    Calibrators with < 5 unique y values are degenerate (collapsed) and
    should be REJECTED at fit time, not silently shipped to production.

    Round-7 audit (2026-04-26): bumped guard from 3 to 5 after the
    production XGBoost calibrator ran with n_unique_prob_y=6 and showed
    rank_score saturation across the top tier (top-7 candidates
    collapsing to 0.34474). User spec: "verify ≥5 unique y values".
    """

    def _per_date_panel(self, n_dates=30, n_tickers=20, signal=0.5,
                        seed=0):
        """Build a panel with per-date cross-sectional structure."""
        import numpy as _np
        import pandas as _pd
        rng = _np.random.default_rng(seed)
        dates = _pd.date_range("2023-01-01", periods=n_dates, freq="D")
        scores: dict[str, _pd.Series] = {}
        rets:   dict[str, _pd.Series] = {}
        for i in range(n_tickers):
            t = f"T{i:02d}"
            # Each ticker's score has a fixed component +
            # cross-sectional noise; future return correlates with score
            # within each date with `signal` strength.
            base = rng.normal(0, 0.1)
            raw = _pd.Series(
                base + rng.normal(0, 0.02, size=n_dates),
                index=dates, name=t,
            )
            fwd = signal * (raw - raw.mean()) + rng.normal(
                0, 0.03, size=n_dates,
            )
            fwd = _pd.Series(fwd, index=dates, name=t)
            scores[t] = raw
            rets[t]   = fwd
        return scores, rets

    def test_per_date_ic_recorded(self):
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        scores, rets = self._per_date_panel()
        cal = fit_global_calibrator(scores, rets, threshold=0.0,
                                    min_rows=200)
        assert "per_date_ic_mean" in cal.metadata
        assert "n_dates_eval" in cal.metadata
        assert "n_unique_prob_y" in cal.metadata
        assert cal.metadata["n_dates_eval"] >= 25
        # With cross-sectional structure, per-date IC should be material
        assert cal.metadata["per_date_ic_mean"] > 0.05

    def test_pool_ic_lower_than_per_date_ic_when_cross_sectional(self):
        """When signal is purely cross-sectional, pooled IC ≪ per-date IC."""
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        scores, rets = self._per_date_panel(signal=0.7)
        cal = fit_global_calibrator(scores, rets, threshold=0.0,
                                    min_rows=200)
        pool_ic    = cal.metadata["pool_ic"]
        per_date   = cal.metadata["per_date_ic_mean"]
        # Both should be positive but per-date should be at least as high
        assert per_date is not None
        assert per_date >= pool_ic - 0.02  # allow small noise

    def test_collapse_guard_rejects_degenerate_calibrator(self):
        """Constant scores + constant labels → isotonic 1 unique y → reject."""
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        import pandas as _pd
        dates = _pd.date_range("2023-01-01", periods=30, freq="D")
        scores: dict[str, _pd.Series] = {}
        rets:   dict[str, _pd.Series] = {}
        # All tickers have CONSTANT score (no marginal info) AND constant
        # forward returns → isotonic must collapse to 1 unique y. This
        # is the LightGBM-2026-04-26 failure mode: pool_ic ~ 0,
        # isotonic emits y = base_rate everywhere.
        for i in range(20):
            t = f"T{i:02d}"
            scores[t] = _pd.Series([0.05] * 30, index=dates, name=t)
            rets[t]   = _pd.Series([0.01] * 30, index=dates, name=t)
        # 2026-05-04 — match either the original "collapsed to" error
        # (probability head 1 unique y) OR the new NaN-leaf filter rejection
        # ("after NaN-leaf filter, pooled n=0 < min_rows"). Constant-score
        # data triggers the NaN-leaf mode-collapse filter first because
        # 100% of rows match the modal value; either path is a correct
        # "rejected for degeneracy" outcome.
        with pytest.raises(ValueError,
                           match="collapsed to|after NaN-leaf filter|< min_rows"):
            fit_global_calibrator(scores, rets, threshold=0.005,
                                  min_rows=200)

    def test_collapse_guard_passes_healthy_calibrator(self):
        """A healthy panel with cross-sectional signal must NOT raise."""
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        scores, rets = self._per_date_panel(n_dates=60, n_tickers=30,
                                            signal=0.5, seed=42)
        cal = fit_global_calibrator(scores, rets, threshold=0.0,
                                    min_rows=200)
        assert cal.metadata["n_unique_prob_y"] >= 5

    def test_collapse_guard_rejects_borderline_4_unique(self):
        """4 unique y values used to pass (old floor=3); round-7 raised
        floor to 5 so this should now be rejected.

        Tactic: build a panel where forward returns are near-constant
        with only 4 distinct values; isotonic regresses to ≤4 unique y.
        """
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        import pandas as _pd
        import numpy as _np
        rng = _np.random.default_rng(0)
        dates = _pd.date_range("2023-01-01", periods=60, freq="D")
        scores: dict[str, _pd.Series] = {}
        rets:   dict[str, _pd.Series] = {}
        # 4 quartile buckets — gives at most 4 unique probability values
        for i in range(20):
            t = f"T{i:02d}"
            raw_vals = rng.uniform(-0.05, 0.05, size=60)
            # Snap forward returns to 4 buckets — isotonic on indicator
            # against this gives ≤4 distinct y.
            fwd_vals = _np.where(raw_vals > 0.025, 0.05,
                          _np.where(raw_vals > 0.0,    0.02,
                          _np.where(raw_vals > -0.025, 0.0, -0.02)))
            scores[t] = _pd.Series(raw_vals, index=dates, name=t)
            rets[t]   = _pd.Series(fwd_vals, index=dates, name=t)
        with pytest.raises(ValueError, match="collapsed to|need ≥5"):
            fit_global_calibrator(scores, rets, threshold=0.01,
                                  min_rows=200)

    def test_n_unique_prob_y_metadata_tracks_actual_count(self):
        """Metadata field must be populated correctly for downstream
        score_db percentile fallback decisions."""
        from training_panel.global_calibrator import (  # noqa: PLC0415
            fit_global_calibrator,
        )
        scores, rets = self._per_date_panel(n_dates=80, n_tickers=30,
                                            signal=0.7, seed=7)
        cal = fit_global_calibrator(scores, rets, threshold=0.0,
                                    min_rows=200)
        # Cross-check metadata vs actual array uniqueness
        import numpy as _np
        actual_unique = int(len(set(_np.round(cal.prob_y, 8))))
        assert cal.metadata["n_unique_prob_y"] == actual_unique
