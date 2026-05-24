"""Tests for kernel/panel_pipeline/panel_scorer.py and gate helpers."""
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


def _trained_artifact(tmp_path: Path):
    """Train a tiny PanelLTRModel and save an artifact. Returns (path, model, panel)."""
    from training_panel.ltr_model import PanelLTRModel

    rng = np.random.default_rng(0)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=20)
    for d in dates:
        for i in range(6):
            x1 = rng.normal()
            x2 = rng.normal()
            label = x1 + 0.2 * rng.normal()
            rows.append({
                "date": d, "ticker": f"T{i}", "x1": x1, "x2": x2,
                "label": label, "weight": 1.0,
            })
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    grp = panel.groupby("date", sort=True).size().values.astype(np.int32)
    m = PanelLTRModel()
    m.train(panel, grp, feature_cols=["x1", "x2"],
            num_boost_round=30, early_stopping_rounds=None)
    path = tmp_path / "panel_model.json"
    m.save(path, metadata={"training_notes": "unit test"})
    return path, m, panel


class TestPanelScorerLoad:
    def test_load_roundtrip(self, tmp_path):
        from kernel.panel_pipeline import PanelScorer
        path, _, _ = _trained_artifact(tmp_path)
        scorer = PanelScorer.load(path)
        assert scorer.booster is not None
        assert scorer.feature_cols == ["x1", "x2"]

    def test_metadata_populated_from_artifact(self, tmp_path):
        from kernel.panel_pipeline import PanelScorer
        path, _, _ = _trained_artifact(tmp_path)
        scorer = PanelScorer.load(path)
        assert scorer.metadata["training_notes"] == "unit test"
        assert "version" in scorer.metadata

    def test_nested_artifact_metadata_promoted_for_runtime_contracts(self, tmp_path):
        from kernel.panel_pipeline import PanelScorer
        path, _, _ = _trained_artifact(tmp_path)
        payload = json.loads(path.read_text())
        payload["metadata"] = {
            "wf_gate_metadata": {
                "passed": False,
                "sanity_regime_ic": {
                    "regimes": {
                        "BULL_CALM": {
                            "eligible": True,
                            "passed": False,
                            "mean_ic": 0.01,
                        },
                    },
                },
            },
            "score_sample_range": [-0.1, 0.2],
        }
        path.write_text(json.dumps(payload))

        scorer = PanelScorer.load(path)

        assert scorer.metadata["wf_gate_metadata"]["passed"] is False
        assert scorer.metadata["score_sample_range"] == [-0.1, 0.2]
        assert scorer.metadata["metadata"]["wf_gate_metadata"]["passed"] is False


class TestPanelScorerScore:
    def test_matches_training_predictions(self, tmp_path):
        """Saving and re-loading must yield identical predictions on the same
        rows — this is the inference-side guarantee of artifact fidelity."""
        from kernel.panel_pipeline import PanelScorer
        path, model, panel = _trained_artifact(tmp_path)
        # Pick "today" = the last date
        last_date = panel["date"].max()
        today_rows = panel[panel["date"] == last_date].set_index("ticker")

        scorer = PanelScorer.load(path)
        scores = scorer.score(today_rows[["x1", "x2"]])

        # Compare against the freshly trained model's prediction on the same rows
        model_preds = model.predict(today_rows[["x1", "x2"]])
        assert np.allclose(scores.values, model_preds.values, atol=1e-9)

    def test_score_preserves_index(self, tmp_path):
        from kernel.panel_pipeline import PanelScorer
        path, _, panel = _trained_artifact(tmp_path)
        last_date = panel["date"].max()
        today_rows = panel[panel["date"] == last_date].set_index("ticker")

        scorer = PanelScorer.load(path)
        scores = scorer.score(today_rows[["x1", "x2"]])
        assert list(scores.index) == list(today_rows.index)

    def test_missing_feature_column_raises(self, tmp_path):
        from kernel.panel_pipeline import PanelScorer
        path, _, _ = _trained_artifact(tmp_path)
        scorer = PanelScorer.load(path)

        # feature matrix missing x2
        bad = pd.DataFrame({"x1": [0.1, 0.2, 0.3]}, index=["A", "B", "C"])
        with pytest.raises(KeyError):
            scorer.score(bad)


class TestComputePanelScoresHelper:
    def test_one_shot_returns_series(self, tmp_path):
        from kernel.panel_pipeline import compute_panel_scores
        path, _, panel = _trained_artifact(tmp_path)
        last_date = panel["date"].max()
        today_rows = panel[panel["date"] == last_date].set_index("ticker")

        out = compute_panel_scores(path, today_rows[["x1", "x2"]])
        assert isinstance(out, pd.Series)
        assert len(out) == len(today_rows)


class TestTopNByScore:
    def test_top_n_returns_highest(self):
        from kernel.panel_pipeline import top_n_by_score
        scores = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.9})
        top = top_n_by_score(scores, n=2)
        assert top == ["D", "B"]

    def test_zero_n_returns_empty(self):
        from kernel.panel_pipeline import top_n_by_score
        scores = pd.Series({"A": 1.0, "B": 2.0})
        assert top_n_by_score(scores, n=0) == []

    def test_nan_scores_excluded(self):
        from kernel.panel_pipeline import top_n_by_score
        scores = pd.Series({"A": 0.1, "B": np.nan, "C": 0.3, "D": 0.2})
        assert top_n_by_score(scores, n=5) == ["C", "D", "A"]

    def test_n_larger_than_scores_returns_all(self):
        from kernel.panel_pipeline import top_n_by_score
        scores = pd.Series({"A": 0.1, "B": 0.2})
        assert set(top_n_by_score(scores, n=10)) == {"A", "B"}


class TestProbabilityGate:
    def test_keeps_above_threshold(self):
        from kernel.panel_pipeline import probability_gate
        scores = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.9})
        out = probability_gate(scores, threshold=0.3)
        assert out == ["D", "B", "C"]

    def test_empty_when_none_pass(self):
        from kernel.panel_pipeline import probability_gate
        scores = pd.Series({"A": 0.1, "B": 0.2})
        assert probability_gate(scores, threshold=1.0) == []

    def test_nan_excluded(self):
        from kernel.panel_pipeline import probability_gate
        scores = pd.Series({"A": 0.5, "B": np.nan, "C": 0.7})
        assert probability_gate(scores, threshold=0.0) == ["C", "A"]
