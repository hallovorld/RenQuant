"""Tests for TransformerPanelScorer + PanelScorer.load dispatch on .pt / kind.

Scope (step 2b): ensure a saved PanelTransformerModel artifact loads via
PanelScorer.load for either the .pt path or the .json sidecar, scores a
feature matrix shaped like inference expects, and never shadows the
default XGBoost dispatch when the artifact isn't a transformer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

torch = pytest.importorskip("torch")

from training_panel.transformer_model import PanelTransformerModel  # noqa: E402
from kernel.panel_pipeline.panel_scorer import PanelScorer           # noqa: E402
from kernel.panel_pipeline.transformer_scorer import TransformerPanelScorer  # noqa: E402


def _train_tiny_transformer(tmp_path: Path) -> tuple[Path, list[str], PanelTransformerModel]:
    """Train a 2-epoch transformer on a synthetic 6-ticker × 10-date panel."""
    rng = np.random.default_rng(0)
    feature_cols = ["f0", "f1", "f2"]
    rows = []
    gs = []
    for d in range(10):
        n = 6
        X = rng.normal(size=(n, 3)).astype(np.float32)
        y = X.sum(axis=1) + rng.normal(size=n) * 0.2
        for t in range(n):
            rows.append({
                "date": d, "ticker": f"T{t}",
                **{c: float(X[t, i]) for i, c in enumerate(feature_cols)},
                "label": float(y[t]),
            })
        gs.append(n)
    panel = pd.DataFrame(rows)
    m = PanelTransformerModel(params={
        "max_epochs": 2, "d_model": 16, "n_heads": 2, "n_layers": 1,
        "batch_size": 4, "device": "cpu", "seed": 1,
    })
    m.train(panel, np.array(gs, dtype=int), feature_cols, num_boost_round=2)
    art = tmp_path / "panel-transformer.pt"
    m.save(art)
    return art, feature_cols, m


# ── Scorer basics ─────────────────────────────────────────────────────────────

class TestTransformerScorerRoundtrip:
    def test_score_returns_one_value_per_ticker(self, tmp_path: Path):
        art, cols, _ = _train_tiny_transformer(tmp_path)
        scorer = TransformerPanelScorer.load(art)
        # Build a single-date feature matrix (6 tickers × 3 features)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(6, 3)).astype(np.float32)
        matrix = pd.DataFrame(X, columns=cols, index=[f"T{i}" for i in range(6)])
        scores = scorer.score(matrix)
        assert len(scores) == 6
        assert list(scores.index) == list(matrix.index), (
            "scorer must preserve caller's index order"
        )
        assert scores.dtype.kind == "f"
        assert not scores.isna().any()

    def test_missing_features_raise_keyerror(self, tmp_path: Path):
        art, cols, _ = _train_tiny_transformer(tmp_path)
        scorer = TransformerPanelScorer.load(art)
        matrix = pd.DataFrame({cols[0]: [0.0, 0.0]}, index=["A", "B"])
        with pytest.raises(KeyError, match="missing columns"):
            scorer.score(matrix)


# ── PanelScorer.load dispatch ────────────────────────────────────────────────

class TestPanelScorerDispatch:
    def test_pt_path_returns_transformer_scorer(self, tmp_path: Path):
        art, _, _ = _train_tiny_transformer(tmp_path)
        scorer = PanelScorer.load(art)
        assert isinstance(scorer, TransformerPanelScorer), (
            "PanelScorer.load on a .pt must return TransformerPanelScorer"
        )

    def test_json_sidecar_path_returns_transformer_scorer(self, tmp_path: Path):
        art, _, _ = _train_tiny_transformer(tmp_path)
        sidecar = art.with_suffix(".json")
        assert sidecar.exists()
        scorer = PanelScorer.load(sidecar)
        assert isinstance(scorer, TransformerPanelScorer), (
            "PanelScorer.load on the .json sidecar must also dispatch to transformer"
        )

    def test_xgboost_artifact_dispatch_routing(self, tmp_path: Path, monkeypatch):
        """A legacy XGBoost artifact (no `kind`, no `.pt`) must NOT be
        mistakenly routed to the transformer loader.

        We assert routing by stubbing the three loader constructors and
        verifying which one gets called — avoids actually running a real
        XGBoost load, which is irrelevant to this test and has proved
        flaky in CI.
        """
        payload = {
            "version": 1,
            "feature_cols": ["x"],
            "booster_raw_json": "{}",   # placeholder; we never call through
        }
        art = tmp_path / "panel-ltr.json"
        art.write_text(json.dumps(payload))

        calls: list[str] = []

        class _FakeBooster:
            def load_model(self, *a, **k): calls.append("xgb_load_model")

        import kernel.panel_pipeline.panel_scorer as ps_mod
        monkeypatch.setattr(ps_mod.xgb, "Booster", _FakeBooster)

        import kernel.panel_pipeline.transformer_scorer as tps_mod
        monkeypatch.setattr(tps_mod.TransformerPanelScorer, "load",
                            classmethod(lambda cls, p: calls.append("transformer_load")))

        PanelScorer.load(art)
        assert "xgb_load_model" in calls, (
            "legacy XGBoost artifact must route to xgboost loader"
        )
        assert "transformer_load" not in calls, (
            "XGBoost artifact was misrouted to transformer loader"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
