"""Regression tests for RegimeRouterScorer (Phase 3, 2026-05-19).

Pin the routing logic that materializes Phase 0 finding (XGB / HF fail
in different regimes — ensemble per regime).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


@pytest.fixture
def fake_scorers():
    """Two fake scorers — XGB-like (no history) + HF-like (history)."""
    class FakeScorer:
        def __init__(self, name, history, feat_cols, seq_len, bias):
            self.name = name
            self.requires_history = history
            self.feature_cols = list(feat_cols)
            self.seq_len = seq_len
            self.bias = bias  # adds bias to make scorers distinguishable

        def score(self, X):
            # Returns bias + sum of features (deterministic)
            return pd.Series(self.bias + X.sum(axis=1).values,
                              index=X.index, name="panel_score")

        def score_with_history(self, panel_hist, target_tickers):
            # Pick latest row per ticker, sum + bias
            latest = (panel_hist.sort_values("date")
                      .groupby("ticker").tail(1)
                      .set_index("ticker"))
            target_present = [t for t in target_tickers if t in latest.index]
            scores = self.bias + latest.loc[target_present][self.feature_cols].sum(axis=1)
            return pd.Series(scores.values, index=target_present,
                              name="panel_score")

    return {
        "xgb": FakeScorer("xgb", False, ["f1", "f2"], 1, bias=100.0),
        "hf_patchtst": FakeScorer("hf_patchtst", True, ["f1", "f2"], 8, bias=0.0),
    }


class TestRouting:
    def test_bear_routes_to_hf(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        scorer, key = r._pick_scorer("BEAR")
        assert key == "hf_patchtst"

    def test_choppy_routes_to_hf(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        scorer, key = r._pick_scorer("CHOPPY")
        assert key == "hf_patchtst"

    def test_bull_calm_routes_to_xgb(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        scorer, key = r._pick_scorer("BULL_CALM")
        assert key == "xgb"

    def test_bull_volatile_routes_to_xgb(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        scorer, key = r._pick_scorer("BULL_VOLATILE")
        assert key == "xgb"

    def test_unknown_regime_falls_back_to_default(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        scorer, key = r._pick_scorer("PHANTOM_REGIME")
        assert key == "xgb"  # default

    def test_custom_routing_overrides_default(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        custom = {"BEAR": "xgb", "BULL_CALM": "hf_patchtst"}  # inverse
        r = RegimeRouterScorer(scorers=fake_scorers, routing=custom,
                                default_scorer_key="xgb")
        assert r._pick_scorer("BEAR")[1] == "xgb"
        assert r._pick_scorer("BULL_CALM")[1] == "hf_patchtst"


class TestEndToEndScoring:
    def test_bear_uses_hf_score_with_history(self, fake_scorers):
        """In BEAR regime, score_with_history routes to HF (history)
        scorer, returns its scores (bias=0)."""
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        panel = pd.DataFrame({
            "ticker": ["A", "B"] * 5,
            "date": list(pd.date_range("2024-01-01", periods=5)) * 2,
            "f1": list(range(10)),
            "f2": list(range(10, 20)),
        })
        scores = r.score_with_history(panel, ["A", "B"],
                                        current_regime="BEAR")
        # HF score = bias (0) + f1 + f2 — should be > 0
        assert (scores > 0).all()
        # HF's bias is 0, not 100 (would be XGB)
        assert scores.max() < 100

    def test_bull_volatile_uses_xgb_score(self, fake_scorers):
        """In BULL_VOLATILE regime, routes to XGB (no history)."""
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        panel = pd.DataFrame({
            "ticker": ["A", "B"] * 5,
            "date": list(pd.date_range("2024-01-01", periods=5)) * 2,
            "f1": list(range(10)),
            "f2": list(range(10, 20)),
        })
        scores = r.score_with_history(panel, ["A", "B"],
                                        current_regime="BULL_VOLATILE")
        # XGB score = bias (100) + features → > 100
        assert (scores > 100).all()


class TestSourceContracts:
    def test_default_routing_matches_phase0_finding(self):
        from kernel.panel_pipeline.regime_router_scorer import DEFAULT_ROUTING
        # Phase 0: BEAR/CHOPPY → HF (HF stable in crash), BULL_* → XGB
        assert DEFAULT_ROUTING["BEAR"] == "hf_patchtst"
        assert DEFAULT_ROUTING["CHOPPY"] == "hf_patchtst"
        assert DEFAULT_ROUTING["BULL_CALM"] == "xgb"
        assert DEFAULT_ROUTING["BULL_VOLATILE"] == "xgb"

    def test_feature_cols_is_union(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        # Both scorers have ["f1", "f2"]
        assert set(r.feature_cols) == {"f1", "f2"}

    def test_seq_len_is_max(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        # XGB seq=1, HF seq=8 — max is 8
        assert r.seq_len == 8

    def test_requires_history_true_if_any_scorer_does(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        # HF requires history → router does too
        assert r.requires_history is True


class TestErrorPaths:
    def test_empty_scorers_raises(self):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        with pytest.raises(ValueError, match="needs ≥1 scorer"):
            RegimeRouterScorer(scorers={}, default_scorer_key="xgb")

    def test_default_scorer_key_must_be_in_scorers(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        with pytest.raises(ValueError, match="not in"):
            RegimeRouterScorer(scorers=fake_scorers,
                                default_scorer_key="missing_key")

    def test_missing_routed_scorer_raises_at_construction(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        routing = {"BEAR": "hf_patchtst", "BULL_CALM": "ghost_model"}

        with pytest.raises(ValueError, match="routing references missing scorer"):
            RegimeRouterScorer(
                scorers=fake_scorers,
                routing=routing,
                default_scorer_key="xgb",
            )

    def test_missing_feature_columns_raise_not_zero_fill(self, fake_scorers):
        from kernel.panel_pipeline.regime_router_scorer import RegimeRouterScorer
        r = RegimeRouterScorer(scorers=fake_scorers, default_scorer_key="xgb")
        panel = pd.DataFrame({
            "ticker": ["A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "f1": [1.0, 2.0],
            # f2 intentionally absent.
        })

        with pytest.raises(RuntimeError, match="missing feature columns"):
            r.score_with_history(panel, ["A", "B"], current_regime="BEAR")


class TestModelRegistryIntegration:
    def test_regime_router_kind_registered(self):
        from kernel.panel_pipeline.model_registry import registry
        h = registry.get("regime_router")
        assert h.requires_history is True
        assert callable(h.scorer_loader)

    def test_train_cmd_raises(self):
        """regime_router is inference-only composition."""
        from kernel.panel_pipeline.model_registry import registry
        h = registry.get("regime_router")
        with pytest.raises(NotImplementedError):
            h.train_cmd(None)
