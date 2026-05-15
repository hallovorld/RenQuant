"""Panel-scoring alignment tests for renquant_104.

Verifies that the three runtime entry points handle the `ranking.panel_scoring.enabled`
flag consistently:

  1. `prepare_inference_panel_frames` is called identically by the LEAN
     adapter and the live RunnerAdapter when the flag is on.
  2. `PanelScoringJob.should_skip` honours the same flag inside the
     `InferencePipeline`.
  3. When the flag is off, no panel frames are produced and the Job skips.

These tests guard against divergence between the LEAN/live/sim paths as
panel scoring is wired in (task #37).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _tiny_ohlcv(days: int = 30, base: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=days)
    rng = np.random.default_rng(1)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
    return pd.DataFrame(
        {
            "open":   close,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": np.ones(days) * 1e6,
        },
        index=idx,
    )


def _panel_enabled_config() -> dict:
    return {
        "watchlist": ["AAA", "BBB"],
        "sector_map": {"AAA": "tech", "BBB": "tech"},
        "sector_etf_map": {"tech": "XLK"},
        "benchmark": "SPY",
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            },
        },
    }


def _panel_disabled_config() -> dict:
    cfg = _panel_enabled_config()
    cfg["ranking"]["panel_scoring"]["enabled"] = False
    return cfg


def _make_ctx(config: dict):
    from kernel.pipeline.context import InferenceContext
    ctx = InferenceContext(config=config, today=datetime.date(2026, 4, 20))
    ctx.candidates = []  # the Job should_skip=True when no candidates
    return ctx


# ── PanelScoringJob flag parity ──────────────────────────────────────────────

class TestPanelJobFlag:
    """PanelScoringJob.should_skip must gate on the same flag the adapters read."""

    def test_should_skip_when_disabled(self):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        from kernel.selection import CandidateResult
        ctx = _make_ctx(_panel_disabled_config())
        ctx.candidates = [
            CandidateResult(ticker="AAA", raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0)
        ]
        assert PanelScoringJob().should_skip(ctx) is True

    def test_not_skipped_when_enabled_with_candidates(self):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        from kernel.selection import CandidateResult
        ctx = _make_ctx(_panel_enabled_config())
        ctx.candidates = [
            CandidateResult(ticker="AAA", raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0)
        ]
        assert PanelScoringJob().should_skip(ctx) is False


# ── Adapter-level parity: LEAN vs Runner both call prepare_inference_panel_frames ──

class TestAdapterPanelPrep:
    """When flag=True, both LeanAdapter and RunnerAdapter must populate frames."""

    def test_runner_adapter_prepares_frames_when_enabled(self):
        from adapters.runner import RunnerAdapter

        config = _panel_enabled_config()
        broker = MagicMock()
        broker.get_account_value.return_value = 100_000.0
        broker.get_cash.return_value = 100_000.0
        broker.get_all_positions.return_value = []

        adapter = RunnerAdapter(config, models={}, broker=broker,
                                strategy_dir=_STRATEGY_DIR, sell_only=False)

        sentinel_ff = {"AAA": pd.DataFrame({"f": [1]})}
        sentinel_fac = {"AAA": pd.DataFrame({"g": [1]})}

        with patch("kernel.data.fetch_ohlcv", return_value=_tiny_ohlcv()), \
             patch("training_panel.pipeline.prepare_inference_panel_frames",
                   # T2-2 (2026-04-27): function now returns 4-tuple
                   # (ff, fac, macro, emb). 4th value None when embeddings disabled.
                   return_value=(sentinel_ff, sentinel_fac, None, None)) as prep_mock:
            ctx = adapter.make_context()

        assert prep_mock.called
        assert getattr(ctx, "_panel_feature_frames", None) is sentinel_ff
        assert getattr(ctx, "_panel_factor_frames", None) is sentinel_fac
        # Bug #25: macro frame is None in this test (no macros) but the
        # _panel_macro_frame attribute should be populated.
        assert getattr(ctx, "_panel_macro_frame", "MISSING") is None

    def test_runner_adapter_skips_prep_when_disabled(self):
        from adapters.runner import RunnerAdapter

        config = _panel_disabled_config()
        broker = MagicMock()
        broker.get_account_value.return_value = 100_000.0
        broker.get_cash.return_value = 100_000.0
        broker.get_all_positions.return_value = []

        adapter = RunnerAdapter(config, models={}, broker=broker,
                                strategy_dir=_STRATEGY_DIR, sell_only=False)

        with patch("kernel.data.fetch_ohlcv", return_value=_tiny_ohlcv()), \
             patch("training_panel.pipeline.prepare_inference_panel_frames") as prep_mock:
            ctx = adapter.make_context()

        assert not prep_mock.called
        assert getattr(ctx, "_panel_feature_frames", None) is None
        assert getattr(ctx, "_panel_factor_frames", None) is None

    def test_runner_adapter_skips_prep_when_sell_only(self):
        """Sell-only intraday runs don't need panel ranking — skip the heavy prep."""
        from adapters.runner import RunnerAdapter

        config = _panel_enabled_config()
        broker = MagicMock()
        broker.get_account_value.return_value = 100_000.0
        broker.get_cash.return_value = 100_000.0
        broker.get_all_positions.return_value = []

        adapter = RunnerAdapter(config, models={}, broker=broker,
                                strategy_dir=_STRATEGY_DIR, sell_only=True)

        with patch("kernel.data.fetch_ohlcv", return_value=_tiny_ohlcv()), \
             patch("training_panel.pipeline.prepare_inference_panel_frames") as prep_mock:
            adapter.make_context()

        assert not prep_mock.called


# ── Pipeline ordering: PanelScoringJob runs BEFORE RankingJob ────────────────

class TestPipelineOrdering:
    """Panel scores must overwrite rank_score BEFORE RankingJob composes final ranks."""

    def test_panel_job_imported_lazily_inside_run(self):
        """Guards against reintroducing a circular import (see pp_inference.py docstring)."""
        import kernel.pipeline.pp_inference as mod
        # If the import were at module scope, it'd be an attribute of the module.
        assert "PanelScoringJob" not in dir(mod)

    def test_run_calls_panel_job_before_ranking_job(self):
        """Verify InferencePipeline.run() invokes PanelScoringJob before RankingJob."""
        import kernel.pipeline.pp_inference as pp
        from kernel.pipeline.context import InferenceContext

        called: list[str] = []

        class FakeJob:
            def __init__(self, name: str):
                self._name = name
            def run(self, ctx):
                called.append(self._name)

        ctx = InferenceContext(config=_panel_enabled_config(),
                               today=datetime.date(2026, 4, 20))
        ctx.holdings = {}
        ctx.buy_blocked = True  # skip buy-scan so test stays small
        ctx.bear_only = False

        with patch.object(pp, "RegimeJob",   lambda: FakeJob("regime")), \
             patch.object(pp, "DrawdownJob", lambda: FakeJob("drawdown")), \
             patch.object(pp, "BuyGatesJob", lambda: FakeJob("gates")), \
             patch.object(pp, "RankingJob",  lambda: FakeJob("ranking")), \
             patch.object(pp, "RotationJob", lambda: FakeJob("rotation")), \
             patch.object(pp, "SelectionJob", lambda: FakeJob("selection")), \
             patch("kernel.panel_pipeline.job_panel_scoring.PanelScoringJob",
                   lambda: FakeJob("panel")):
            pp.InferencePipeline().run(ctx)

        # Order must be: regime → drawdown → gates → (no sells/buys) → panel → ranking → rotation → selection
        assert "panel" in called
        assert "ranking" in called
        assert called.index("panel") < called.index("ranking")


# ── Panel veto on weak buys ──────────────────────────────────────────────────

class TestPanelVetoWeakBuys:
    """VetoWeakBuysTask drops candidates below panel_scoring.buy_floor."""

    def _make_cands(self, scores):
        from kernel.selection import CandidateResult
        return [
            CandidateResult(ticker=f"T{i}", raw_score=0, rank_score=s,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=s)
            for i, s in enumerate(scores)
        ]

    def test_drops_candidates_below_floor(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        cfg = _panel_enabled_config()
        cfg["ranking"]["panel_scoring"]["buy_floor"] = 0.5
        ctx = _make_ctx(cfg)
        ctx.candidates = self._make_cands([0.2, 0.6, 0.4, 0.9])

        VetoWeakBuysTask().run(ctx)

        assert {c.ticker for c in ctx.candidates} == {"T1", "T3"}
        assert ctx.counters["panel_vetoed"] == 2

    def test_no_op_when_floor_unset(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        cfg = _panel_enabled_config()  # no buy_floor key
        ctx = _make_ctx(cfg)
        ctx.candidates = self._make_cands([0.0, 0.1, 0.9])

        VetoWeakBuysTask().run(ctx)

        assert len(ctx.candidates) == 3
        assert "panel_vetoed" not in ctx.counters

    def test_keeps_candidates_with_missing_rank_score(self):
        """Candidates without a rank_score are kept — rs_score still ranks them.

        2026-05-03 fix: VetoWeakBuysTask now reads ``cand.rank_score``
        (calibrated, post-ApplyGlobalCalibration) instead of
        ``cand.panel_score`` (raw XGB margin). The "missing-score keep"
        path therefore checks rank_score=None, not panel_score=None.
        """
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        from kernel.selection import CandidateResult
        cfg = _panel_enabled_config()
        cfg["ranking"]["panel_scoring"]["buy_floor"] = 0.5
        ctx = _make_ctx(cfg)
        ctx.candidates = [
            CandidateResult(ticker="A", raw_score=0, rank_score=None,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=0.1),
            CandidateResult(ticker="B", raw_score=0, rank_score=0.2,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=0.2),
        ]
        VetoWeakBuysTask().run(ctx)
        tickers = {c.ticker for c in ctx.candidates}
        assert "A" in tickers     # kept — rank_score missing
        assert "B" not in tickers  # dropped — below floor (rank_score 0.2 < 0.5)


# ── Conviction-scaled sizing ──────────────────────────────────────────────────

class TestPanelConvictionSizing:
    """conviction_multiplier maps panel_score → [min_mult, 1.0]."""

    def test_disabled_returns_one(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": False, "floor": 0.0, "ceiling": 1.0, "min_mult": 0.5}
        assert conviction_multiplier(0.3, cfg) == 1.0

    def test_none_panel_score_returns_one(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": True, "floor": 0.0, "ceiling": 1.0, "min_mult": 0.5}
        assert conviction_multiplier(None, cfg) == 1.0

    def test_below_floor_returns_min_mult(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": True, "floor": 0.2, "ceiling": 0.8, "min_mult": 0.5}
        assert conviction_multiplier(0.1, cfg) == 0.5

    def test_above_ceiling_returns_one(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": True, "floor": 0.2, "ceiling": 0.8, "min_mult": 0.5}
        assert conviction_multiplier(0.9, cfg) == 1.0

    def test_midpoint_scales_halfway(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": True, "floor": 0.0, "ceiling": 1.0, "min_mult": 0.5}
        # panel_score=0.5 → frac=0.5 → 0.5 + 0.5 * 0.5 = 0.75
        assert conviction_multiplier(0.5, cfg) == pytest.approx(0.75)

    def test_malformed_config_returns_one(self):
        from kernel.sizing import conviction_multiplier
        cfg = {"enabled": True, "floor": 0.8, "ceiling": 0.2, "min_mult": 0.5}  # inverted
        assert conviction_multiplier(0.5, cfg) == 1.0


# ── Panel-score rotation advantage ──────────────────────────────────────────

class TestPanelRotationAdvantage:
    """BuildPairsTask rejects rotation pairs where cand panel_score doesn't beat held's."""

    def _run_rotation(self, cfg_overrides: dict, cand_ps: float, held_ps: float):
        """Drive BuildPairsTask with ER-good rotation candidates and panel scores."""
        import datetime as dt
        from kernel.pipeline.context import InferenceContext
        from kernel.panel_pipeline.job_panel_scoring import BuildFeatureMatrixTask  # noqa: F401
        from kernel.pipeline.task_rotation import BuildPairsTask
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState

        cfg = _panel_enabled_config()
        cfg["rotation"] = {
            "enabled": True,
            "min_expected_advantage_pct": 0.01,
            "target_horizon_days": 20,
            "transaction_cost_pct": 0.0,
            "min_rotation_hold_days": 10,
            "lt_protection_days": 0,
            "max_rotations_per_bar": 2,
        }
        cfg["tax"] = {"short_term_rate": 0.0, "long_term_rate": 0.0,
                      "long_term_threshold_days": 365}
        cfg["ranking"]["panel_scoring"].update(cfg_overrides)

        ctx = InferenceContext(config=cfg, today=dt.date(2026, 4, 20))
        ctx.holdings = {
            "OLD": HoldingState(
                entry_price=100.0,
                entry_date=dt.date(2026, 1, 1),
                high_watermark=100.0,
                rank_score=0.2,
                expected_return=0.01,
                panel_score=held_ps,
            )
        }
        ctx.prices = {"OLD": 100.0, "NEW": 100.0}
        ctx.ranked = [
            CandidateResult(ticker="NEW", raw_score=0, rank_score=0.8,
                            rs_score=0, detail="", expected_return=0.10,
                            panel_score=cand_ps)
        ]
        ctx.regime = "BULL_CALM"
        ctx.bear_only = False

        BuildPairsTask().run(ctx)
        return ctx.rotations

    def test_pair_emitted_when_advantage_disabled(self):
        pairs = self._run_rotation({"rotation_advantage": 0.0},
                                   cand_ps=0.3, held_ps=0.7)
        assert len(pairs) == 1
        assert pairs[0].buy_ticker == "NEW"

    def test_pair_rejected_when_panel_advantage_insufficient(self):
        pairs = self._run_rotation({"rotation_advantage": 0.20},
                                   cand_ps=0.5, held_ps=0.45)  # only +0.05
        assert pairs == []

    def test_pair_emitted_when_panel_advantage_met(self):
        pairs = self._run_rotation({"rotation_advantage": 0.10},
                                   cand_ps=0.8, held_ps=0.4)   # +0.40
        assert len(pairs) == 1

    def test_skips_gate_when_panel_score_missing(self):
        """Fall back to ER-only rule when either side lacks a panel_score."""
        pairs = self._run_rotation({"rotation_advantage": 0.50},
                                   cand_ps=None, held_ps=0.4)
        assert len(pairs) == 1


# ── NGBoost wiring ────────────────────────────────────────────────────────────

def _ngboost_enabled_config(artifact_path: str | None = None) -> dict:
    cfg = _panel_enabled_config()
    ngb = {
        "enabled": True,
        "score_mode": "mu_minus_lambda_sigma",
        "lambda_sigma": 1.0,
    }
    if artifact_path is not None:
        ngb["artifact_path"] = artifact_path
    cfg["ranking"]["panel_scoring"]["ngboost"] = ngb
    return cfg


def _make_fake_head(feature_cols, mu_map, sigma_map):
    """Stand-in NGBoostHead whose predict_distribution is deterministic."""
    class _FakeHead:
        def __init__(self):
            self.feature_cols = list(feature_cols)
        def predict_distribution(self, X):
            idx = X.index
            mu    = pd.Series([mu_map.get(t, 0.0)    for t in idx], index=idx)
            sigma = pd.Series([sigma_map.get(t, 0.1) for t in idx], index=idx)
            return pd.DataFrame({"mu": mu, "sigma": sigma})
    return _FakeHead()


class TestNGBoostFlagParity:
    """NGBoost sub-flag is read by PanelScoringJob tasks, not the adapters."""

    def test_panel_job_still_skips_when_outer_flag_off(self):
        from kernel.panel_pipeline.job_panel_scoring import PanelScoringJob
        from kernel.selection import CandidateResult
        cfg = _panel_disabled_config()
        # Turn ngboost on while outer flag is off — whole job must still skip.
        cfg["ranking"]["panel_scoring"]["ngboost"] = {"enabled": True}
        ctx = _make_ctx(cfg)
        ctx.candidates = [CandidateResult(ticker="AAA", raw_score=0,
                                          rank_score=0.5, rs_score=0,
                                          detail="", expected_return=0)]
        assert PanelScoringJob().should_skip(ctx) is True

    def test_load_ngboost_noop_when_ngboost_disabled(self):
        from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask
        ctx = _make_ctx(_panel_enabled_config())  # ngboost omitted → disabled
        out = LoadNGBoostTask().run(ctx)
        assert out is None
        assert getattr(ctx, "_ngboost_head", None) is None

    def test_load_ngboost_records_failure_without_raising(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask
        cfg = _ngboost_enabled_config(
            artifact_path=str(tmp_path / "missing.json"),
        )
        cfg["_strategy_dir"] = str(tmp_path)
        ctx = _make_ctx(cfg)
        LoadNGBoostTask().run(ctx)  # must not raise
        assert ctx._ngboost_head is None  # noqa: SLF001


class TestApplyNGBoostScoring:
    """ApplyNGBoostTask writes μ/σ and optionally overrides rank_score."""

    def _ctx_with_matrix(self, feature_cols, tickers):
        cfg = _ngboost_enabled_config()
        ctx = _make_ctx(cfg)
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState
        ctx.candidates = [
            CandidateResult(ticker=t, raw_score=0, rank_score=0.5,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=0.5)
            for t in tickers
        ]
        ctx.holdings = {}
        # 2026-05-15 fixture fix: panel-pipeline INPUT-VARIANCE GUARD
        # (job_panel_scoring.py:990) clears candidates when >20% of feature
        # columns are constant. Pre-fix fixture used [0.0]*n → all-constant
        # → guard fired in tests. Use per-row varied values so the guard
        # passes (per-row σ > 0). Test math still works because the fake
        # head ignores input features (returns hard-coded mu/sigma per row).
        X = pd.DataFrame(
            {c: [float(j) + ci * 0.01 for j in range(len(tickers))]
             for ci, c in enumerate(feature_cols)},
            index=tickers,
        )
        ctx._panel_matrix = X  # noqa: SLF001
        return ctx

    def test_applies_mu_sigma_and_overrides_rank_score_by_default(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        feats = ["f1", "f2"]
        ctx = self._ctx_with_matrix(feats, ["A", "B"])
        # A: μ=0.08 σ=0.02 → combined 0.06;  B: μ=0.02 σ=0.20 → combined -0.18
        head = _make_fake_head(feats,
                               mu_map={"A": 0.08, "B": 0.02},
                               sigma_map={"A": 0.02, "B": 0.20})
        ctx._ngboost_head = head  # noqa: SLF001

        ApplyNGBoostTask().run(ctx)

        ca = next(c for c in ctx.candidates if c.ticker == "A")
        cb = next(c for c in ctx.candidates if c.ticker == "B")
        assert ca.mu == pytest.approx(0.08)
        assert ca.sigma == pytest.approx(0.02)
        assert ca.rank_score  == pytest.approx(0.06)
        assert ca.panel_score == pytest.approx(0.06)
        assert cb.rank_score  == pytest.approx(-0.18)

    def test_additive_mode_keeps_rank_score(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        feats = ["f1"]
        ctx = self._ctx_with_matrix(feats, ["A"])
        ctx.config["ranking"]["panel_scoring"]["ngboost"]["score_mode"] = "additive"
        head = _make_fake_head(feats, mu_map={"A": 0.05}, sigma_map={"A": 0.1})
        ctx._ngboost_head = head  # noqa: SLF001

        ApplyNGBoostTask().run(ctx)
        c = ctx.candidates[0]
        assert c.mu    == pytest.approx(0.05)
        assert c.sigma == pytest.approx(0.1)
        assert c.rank_score  == 0.5   # unchanged
        assert c.panel_score == 0.5   # unchanged

    def test_lambda_sigma_scales_penalty(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        feats = ["f1"]
        ctx = self._ctx_with_matrix(feats, ["A"])
        ctx.config["ranking"]["panel_scoring"]["ngboost"]["lambda_sigma"] = 3.0
        head = _make_fake_head(feats, mu_map={"A": 0.1}, sigma_map={"A": 0.2})
        ctx._ngboost_head = head  # noqa: SLF001

        ApplyNGBoostTask().run(ctx)
        # 0.1 - 3.0 * 0.2 = -0.5
        assert ctx.candidates[0].rank_score == pytest.approx(-0.5)

    def test_fills_missing_columns_with_zero_and_runs(self):
        """Audit N-25 (2026-04-25): pre-fix this no-op'd the entire NGBoost
        prediction whenever a single feature column was missing — silently
        leaving the candidate with no μ/σ. Post-fix, missing columns get
        filled with 0.0 (z-scored neutral) and predictions still run, so
        Kelly sizing and σ-sizing can proceed on a partial feature set.

        2026-04-28 self-audit (CRIT-1): the new drift detector hard-fails
        when >5% of cols missing (default), to prevent silent macro-residual
        regressions. This test exercises the zero-fill path (intended
        behaviour for SMALL drift), so set the threshold to 1.0 for the
        single-feature 100%-missing case to keep the soft path active.
        """
        from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask
        feats = ["x_expected"]
        ctx = self._ctx_with_matrix(["x_different"], ["A"])
        # CRIT-1: opt-out of the hard-fail for this fixture (single-col
        # panel where 1/1 missing would otherwise trigger drift fail-safe).
        ctx.config["ranking"]["panel_scoring"]["ngboost"]["max_feature_drift_pct"] = 1.0
        head = _make_fake_head(feats, mu_map={"A": 0.1}, sigma_map={"A": 0.05})
        ctx._ngboost_head = head  # noqa: SLF001

        ApplyNGBoostTask().run(ctx)
        # μ/σ now populated — pre-fix they would still be None.
        assert ctx.candidates[0].mu == pytest.approx(0.1)
        assert ctx.candidates[0].sigma == pytest.approx(0.05)


class TestSigmaSizing:
    """σ-sizing multiplier is plumbed through SizeAndEmitTask."""

    def test_sigma_sizing_reduces_max_pct_for_high_sigma(self):
        from kernel.sizing import sigma_multiplier
        cfg = {"enabled": True, "floor": 0.3, "ceiling": 1.0}
        # σ_median = 0.1, candidate σ = 0.2 → 0.5 multiplier
        assert sigma_multiplier(0.2, 0.1, cfg) == pytest.approx(0.5)

    def test_sigma_sizing_respects_floor(self):
        from kernel.sizing import sigma_multiplier
        cfg = {"enabled": True, "floor": 0.4, "ceiling": 1.0}
        # σ_median = 0.1, candidate σ = 10.0 → ratio 0.01 → clipped to floor 0.4
        assert sigma_multiplier(10.0, 0.1, cfg) == 0.4

    def test_sigma_sizing_respects_ceiling(self):
        from kernel.sizing import sigma_multiplier
        cfg = {"enabled": True, "floor": 0.3, "ceiling": 1.0}
        # σ_median = 0.1, candidate σ = 0.01 → ratio 10.0 → clipped to ceiling 1.0
        assert sigma_multiplier(0.01, 0.1, cfg) == 1.0

    def test_sigma_sizing_disabled_returns_one(self):
        from kernel.sizing import sigma_multiplier
        cfg = {"enabled": False, "floor": 0.3, "ceiling": 1.0}
        assert sigma_multiplier(0.5, 0.1, cfg) == 1.0

    def test_sigma_sizing_none_returns_one(self):
        from kernel.sizing import sigma_multiplier
        cfg = {"enabled": True, "floor": 0.3, "ceiling": 1.0}
        assert sigma_multiplier(None, 0.1, cfg) == 1.0
        assert sigma_multiplier(0.2, None, cfg) == 1.0

    def test_universe_sigma_median(self):
        from kernel.sizing import universe_sigma_median
        assert universe_sigma_median([0.1, 0.3, 0.2]) == pytest.approx(0.2)
        assert universe_sigma_median([0.2]) == 0.2
        assert universe_sigma_median([None, 0.0, -1.0]) is None
        assert universe_sigma_median([]) is None

    def test_size_and_emit_applies_sigma_multiplier(self, monkeypatch):
        """End-to-end: SizeAndEmitTask reads σ from candidates and scales max_pct."""
        import datetime as dt
        from kernel.pipeline.task_selection import SizeAndEmitTask
        from kernel.pipeline.context import InferenceContext
        from kernel.selection import CandidateResult

        cfg = _panel_enabled_config()
        cfg["regime_params"] = {
            "BULL_CALM": {"max_position_pct": 0.20, "cash_reserve_pct": 0.0},
        }
        cfg["ranking"]["panel_scoring"]["sigma_sizing"] = {
            "enabled": True, "floor": 0.3, "ceiling": 1.0,
        }

        ctx = InferenceContext(config=cfg, today=dt.date(2026, 4, 20))
        ctx.regime = "BULL_CALM"
        ctx.confidence = 1.0
        ctx.portfolio_value = 100_000.0
        ctx.cash = 100_000.0
        low_sigma  = CandidateResult(ticker="LO", raw_score=0, rank_score=0.9,
                                     rs_score=0, detail="",
                                     panel_score=0.9, sigma=0.05)
        high_sigma = CandidateResult(ticker="HI", raw_score=0, rank_score=0.9,
                                     rs_score=0, detail="",
                                     panel_score=0.9, sigma=0.20)
        ctx.ranked = [low_sigma, high_sigma]
        ctx._selected = ["LO", "HI"]  # noqa: SLF001
        ctx.prices = {"LO": 100.0, "HI": 100.0}

        SizeAndEmitTask().run(ctx)

        orders = {o["ticker"]: o for o in ctx.orders}
        assert "LO" in orders and "HI" in orders
        # σ_median between 0.05 and 0.20 is 0.125.
        # LO: 0.125 / 0.05 = 2.5, clipped to ceiling 1.0. Max_pct = 0.20 * 1.0 = 0.20.
        # HI: 0.125 / 0.20 = 0.625. Max_pct = 0.20 * 0.625 = 0.125.
        assert orders["HI"]["sigma_mult"] == pytest.approx(0.625)
        assert orders["LO"]["sigma_mult"] == pytest.approx(1.0)
        # LO should end up larger than HI (same price, same portfolio_value).
        assert orders["LO"]["shares"] > orders["HI"]["shares"]


# ── CACHE-DIR-SNAPSHOT audit fix ──────────────────────────────────────────────

class TestCacheDirSnapshotFallback:
    """Audit fix CACHE-DIR-SNAPSHOT (2026-04-26): when sim A/B uses
    snapshot_artifacts, _strategy_dir points to a tmpdir whose
    parent.parent doesn't have data/ — must fall back to cwd."""

    def test_snapshot_path_falls_back_to_cwd(self, tmp_path):
        """strategy_dir under tmpdir → derived path missing → use cwd."""
        from training_panel.pp_panel_training import _resolve_cache_dir
        # Simulate snapshot: tmpdir as strategy_dir
        fake_strategy_dir = tmp_path / "fake_snapshot" / "renquant_104"
        fake_strategy_dir.mkdir(parents=True)
        # Don't create data/ under tmp parent — simulates real bug
        result = _resolve_cache_dir(
            "data/fundamentals",
            {"_strategy_dir": str(fake_strategy_dir)},
        )
        # Result should be either the (non-existent) snapshot-derived
        # path OR the cwd path. Either way, no exception.
        assert isinstance(result, type(fake_strategy_dir))

    def test_absolute_path_unchanged(self):
        from training_panel.pp_panel_training import _resolve_cache_dir
        from pathlib import Path
        abs_path = "/tmp/some_abs"
        result = _resolve_cache_dir(abs_path, {"_strategy_dir": "/foo"})
        assert str(result) == abs_path

    def test_existing_strategy_relative_takes_precedence(self, tmp_path):
        from training_panel.pp_panel_training import _resolve_cache_dir
        # Build a fake repo_root / data / fundamentals
        fake_repo = tmp_path / "fake_repo"
        fake_strategy_dir = fake_repo / "backtesting" / "renquant_104"
        fake_strategy_dir.mkdir(parents=True)
        fake_data = fake_repo / "data" / "fundamentals"
        fake_data.mkdir(parents=True)
        result = _resolve_cache_dir(
            "data/fundamentals",
            {"_strategy_dir": str(fake_strategy_dir)},
        )
        # Should resolve to fake_repo / data / fundamentals (which exists)
        assert result == fake_data
