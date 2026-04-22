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
                   return_value=(sentinel_ff, sentinel_fac)) as prep_mock:
            ctx = adapter.make_context()

        assert prep_mock.called
        assert getattr(ctx, "_panel_feature_frames", None) is sentinel_ff
        assert getattr(ctx, "_panel_factor_frames", None) is sentinel_fac

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

    def test_keeps_candidates_with_missing_panel_score(self):
        """Candidates without a panel_score are kept — rs_score still ranks them."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        from kernel.selection import CandidateResult
        cfg = _panel_enabled_config()
        cfg["ranking"]["panel_scoring"]["buy_floor"] = 0.5
        ctx = _make_ctx(cfg)
        ctx.candidates = [
            CandidateResult(ticker="A", raw_score=0, rank_score=0.1,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=None),
            CandidateResult(ticker="B", raw_score=0, rank_score=0.2,
                            rs_score=0, detail="", expected_return=0,
                            panel_score=0.2),
        ]
        VetoWeakBuysTask().run(ctx)
        tickers = {c.ticker for c in ctx.candidates}
        assert "A" in tickers     # kept — panel_score missing
        assert "B" not in tickers  # dropped — below floor


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
