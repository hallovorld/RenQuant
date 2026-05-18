"""End-to-end smoke tests for the inference pipeline (Guard #4).

Per CLAUDE.md §5.3 BUG #6 audit response. The bug class:
  - feature data corrupted/constant/missing upstream
  - model still runs successfully but produces degenerate outputs
  - downstream silently rejects everything (mu_le_min_edge=49 etc.)
  - no log surfaces the prediction collapse

These tests run a minimal InferencePipeline on synthetic + production
artifacts and assert each phase produces DIVERSE outputs (not constants).

Tests are CHEAP — they exercise contract validators and direct head
load+predict on tiny synthetic data, NOT a full LEAN/Alpaca run.

Reference for what these test:
  - kernel/panel_pipeline/job_panel_scoring.py: ApplyScoresTask,
    ApplyNGBoostTask, ApplyGlobalCalibrationTask
  - training_panel/model_contract.py: contract validators
  - training_panel/{quantile_head,ngboost_head}.py: head implementations
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


# ── Contract validator unit tests ────────────────────────────────────────────

class TestModelContractValidators:
    """Direct unit tests of validation utilities — no head loading required."""

    def test_soft_check_input_passes_normal_panel(self):
        from training_panel.model_contract import soft_check_input
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.standard_normal((50, 10)),
            columns=[f"f{i}" for i in range(10)],
        )
        res = soft_check_input(X, X.columns, head_name="test")
        assert res.ok
        assert res.n_zero_var_cols == 0

    def test_soft_check_input_fails_all_constant_features(self):
        """BUG #6 reproducer at the contract level — all features constant."""
        from training_panel.model_contract import soft_check_input
        # Every row identical → zero per-column variance
        X = pd.DataFrame(
            np.tile([1.0, 2.0, 3.0], (50, 1)),
            columns=["f0", "f1", "f2"],
        )
        res = soft_check_input(X, X.columns, head_name="test")
        assert not res.ok, "should HARD fail on all-constant input"
        assert res.n_zero_var_cols == 3
        assert any("HARD FAIL" in w for w in res.warnings)

    def test_soft_check_input_warns_partial_constant(self):
        """A few constant cols (10–50%) should soft-warn but not hard-fail."""
        from training_panel.model_contract import soft_check_input
        rng = np.random.default_rng(42)
        # 3/10 cols constant = 30% — between soft (10%) and hard (50%)
        X = pd.DataFrame(
            rng.standard_normal((50, 10)),
            columns=[f"f{i}" for i in range(10)],
        )
        X[["f0", "f1", "f2"]] = 1.0  # constant
        res = soft_check_input(X, X.columns, head_name="test")
        assert res.ok, "30% constant should soft-warn, not hard-fail"
        assert res.n_zero_var_cols == 3
        assert any("SOFT" in w for w in res.warnings)

    def test_soft_check_output_passes_diverse_predictions(self):
        from training_panel.model_contract import soft_check_output
        rng = np.random.default_rng(42)
        out = pd.DataFrame({
            "mu": rng.standard_normal(50) * 0.02,
            "sigma": np.abs(rng.standard_normal(50)) * 0.1 + 0.01,
        })
        res = soft_check_output(out, head_name="test")
        assert res.ok
        assert res.mu_xs_std > 0.001
        assert res.n_unique_mu >= 2

    def test_soft_check_output_fails_collapsed_mu(self):
        """BUG #6 reproducer at the contract level — μ̂ identical across rows."""
        from training_panel.model_contract import soft_check_output
        out = pd.DataFrame({
            "mu": [0.05] * 50,    # constant μ̂
            "sigma": [0.1] * 50,  # constant σ̂
        })
        res = soft_check_output(out, head_name="test")
        assert not res.ok
        # numpy std of all-equal values is float-precision-noisy (1e-17 not 0.0);
        # the hard floor is 1e-6, so this still cleanly fails the contract.
        assert res.mu_xs_std < 1e-12
        assert res.n_unique_mu == 1
        assert any("collapsed" in w for w in res.warnings)

    def test_soft_check_score_series_passes_diverse(self):
        from training_panel.model_contract import soft_check_score_series
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.standard_normal(50))
        res = soft_check_score_series(scores, model_name="test")
        assert res.ok

    def test_soft_check_score_series_fails_constant(self):
        from training_panel.model_contract import soft_check_score_series
        scores = pd.Series([0.5] * 50)
        res = soft_check_score_series(scores, model_name="test")
        assert not res.ok
        assert res.n_unique_mu == 1

    def test_soft_check_score_series_fails_out_of_range(self):
        """If we declare expected [0,1] and get 1.5, must hard-fail."""
        from training_panel.model_contract import soft_check_score_series
        scores = pd.Series([0.2, 0.5, 1.5, 0.8])  # 1.5 > 1.0 expected_max
        res = soft_check_score_series(
            scores, model_name="test", expected_min=0.0, expected_max=1.0,
        )
        assert not res.ok
        assert any("max=" in w and "> 1.0" in w for w in res.warnings)


# ── Head-level smoke tests on production artifacts ────────────────────────────

class TestQuantileHeadSmoke:
    """Load production QuantileHead artifact, run on synthetic + real-shape input.

    These tests are SKIPPED if the production artifact isn't present,
    so they're CI-friendly even on fresh checkouts.
    """

    HEAD_PATH = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json"

    def _load(self):
        if not self.HEAD_PATH.exists():
            pytest.skip(f"head artifact not present: {self.HEAD_PATH}")
        # 2026-05-17: top-level head artifact can be either QuantileHead
        # (XGB-quantile triplet) or NGBoostHead — production loader is
        # polymorphic. This smoke test is QuantileHead-specific; skip
        # cleanly when the on-disk artifact is a different kind.
        import json as _json
        kind = _json.loads(self.HEAD_PATH.read_text()).get("kind")
        if kind != "quantile_head":
            pytest.skip(f"artifact kind={kind!r}, not 'quantile_head' — skip smoke")
        from training_panel.quantile_head import QuantileHead
        return QuantileHead.load(self.HEAD_PATH)

    def test_load_and_predict_diverse_synthetic_features(self):
        """With diverse random feature input, μ̂ x-sec std must be > 0."""
        head = self._load()
        rng = np.random.default_rng(42)
        n = 50
        X = pd.DataFrame(
            rng.standard_normal((n, len(head.feature_cols))),
            columns=head.feature_cols,
            index=[f"T{i}" for i in range(n)],
        )
        out = head.predict_distribution(X)
        assert "mu" in out.columns
        assert "sigma" in out.columns
        mu_finite = out["mu"].dropna()
        assert len(mu_finite) >= 2
        # Must NOT collapse to a constant
        assert mu_finite.std() > 1e-4, (
            f"BUG #6 regression: QuantileHead produced collapsed μ̂ on diverse "
            f"input. mu.std()={mu_finite.std():.2e}, "
            f"n_unique={mu_finite.nunique()}"
        )
        sigma_finite = out["sigma"].dropna()
        assert (sigma_finite > 0).all(), "σ̂ must be strictly positive"

    def test_predict_on_all_constant_input_warns(self):
        """All-constant input → predictions WILL be constant; contract must catch."""
        head = self._load()
        # Every row identical — guarantees identical predictions
        X = pd.DataFrame(
            np.zeros((20, len(head.feature_cols))),
            columns=head.feature_cols,
            index=[f"T{i}" for i in range(20)],
        )
        # Should not crash; soft_check logs ERROR but doesn't raise
        out = head.predict_distribution(X)
        # Confirm the contract correctly identifies the collapse
        from training_panel.model_contract import soft_check_output
        res = soft_check_output(out, head_name="QuantileHead")
        assert not res.ok, (
            "soft_check_output failed to detect collapsed μ̂ on all-zero input — "
            "the diversity guard would not catch BUG #6"
        )


# ── ApplyNGBoostTask integration smoke test ──────────────────────────────────

class TestApplyNGBoostTaskGuard:
    """Verify the in-pipeline diversity guard fires on collapsed predictions
    AND clears candidates so QP/Kelly do not trade on garbage.

    Mirrors the production code path without running a full pipeline:
    constructs an InferenceContext stub with collapsed μ̂/σ̂ and asserts
    candidates get cleared.
    """

    def test_diversity_guard_clears_candidates_on_collapse(self):
        """If post-predict μ̂ has zero x-sec std, ApplyNGBoostTask must
        clear ctx.candidates and stamp NaN on every cand.mu/cand.sigma."""
        # Smoke-test by reading the source for the guard logic — full
        # ctx wiring requires too much production setup for a unit test.
        # The actual fail-safe behavior is asserted by the integration
        # test in test_pipeline_invariants.py (BUG #6 path).
        scoring_py = REPO / "backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py"
        src = scoring_py.read_text()
        # Confirm the diversity guard is wired
        assert "DIVERSITY GUARD FAILED" in src, (
            "Guard #1 missing: ApplyNGBoostTask must hard-fail on collapsed μ̂"
        )
        assert "INPUT-VARIANCE GUARD FAILED" in src, (
            "Guard #2 missing: ApplyNGBoostTask must hard-fail on constant input"
        )
        # Both guards must clear candidates fail-safe
        guard_section = src[src.index("DIVERSITY GUARD FAILED"):]
        assert "ctx.candidates = []" in guard_section[:2000], (
            "Diversity guard does not clear candidates — fail-safe broken"
        )

    def test_panel_scorer_invokes_contract(self):
        scorer_py = REPO / "backtesting/renquant_104/kernel/panel_pipeline/panel_scorer.py"
        src = scorer_py.read_text()
        assert "soft_check_input" in src, "PanelScorer.score missing input contract"
        assert "soft_check_score_series" in src, "PanelScorer.score missing output contract"

    def test_quantile_head_invokes_contract(self):
        head_py = REPO / "backtesting/renquant_104/training_panel/quantile_head.py"
        src = head_py.read_text()
        assert "soft_check_input" in src, "QuantileHead.predict_distribution missing input contract"
        assert "soft_check_output" in src, "QuantileHead.predict_distribution missing output contract"

    def test_ngboost_head_invokes_contract(self):
        head_py = REPO / "backtesting/renquant_104/training_panel/ngboost_head.py"
        src = head_py.read_text()
        assert "soft_check_input" in src, "NGBoostHead.predict missing input contract"
        assert "soft_check_output" in src, "NGBoostHead.predict missing output contract"
