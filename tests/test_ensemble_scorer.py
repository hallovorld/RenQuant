"""Unit tests for kernel/panel_pipeline/ensemble_scorer.py.

Scope: rank-averaging semantics (scale-invariant), weight application,
union of feature_cols, and tie-handling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.panel_pipeline.ensemble_scorer import EnsemblePanelScorer  # noqa: E402


class _StubScorer:
    """Duck-typed scorer matching PanelScorer / TransformerPanelScorer surface."""

    def __init__(self, cols: list[str], out: pd.Series):
        self.feature_cols = list(cols)
        self._out = out

    def score(self, matrix: pd.DataFrame) -> pd.Series:
        # Assert the caller gave us the columns we asked for.
        assert all(c in matrix.columns for c in self.feature_cols)
        return self._out.loc[matrix.index]


class TestEnsembleRankAveraging:
    def test_scale_invariant(self):
        """Scorer A's raw values are tiny; scorer B's are huge. Rank
        averaging should give them equal weight."""
        idx = ["T0", "T1", "T2", "T3"]
        matrix = pd.DataFrame({"f": [0, 0, 0, 0]}, index=idx)
        a = _StubScorer(["f"], pd.Series([0.001, 0.002, 0.003, 0.004], index=idx))
        b = _StubScorer(["f"], pd.Series([100, 200, 300, 400],         index=idx))
        ens = EnsemblePanelScorer([a, b])
        out = ens.score(matrix)
        # Both backends agree on ranking → output = rank-normalized.
        assert out["T0"] == pytest.approx(0.0)
        assert out["T1"] == pytest.approx(1/3)
        assert out["T2"] == pytest.approx(2/3)
        assert out["T3"] == pytest.approx(1.0)
        assert out.idxmax() == "T3"

    def test_weights_are_normalized(self):
        idx = ["A", "B", "C"]
        matrix = pd.DataFrame({"f": [0.0]*3}, index=idx)
        # Scorer "x" ranks A > B > C (rank A=1, B=0.5, C=0)
        x = _StubScorer(["f"], pd.Series([3.0, 2.0, 1.0], index=idx))
        # Scorer "y" reverses: C > B > A (rank A=0, B=0.5, C=1.0)
        y = _StubScorer(["f"], pd.Series([1.0, 2.0, 3.0], index=idx))
        # 100% weight on x.
        ens = EnsemblePanelScorer([x, y], weights=[1.0, 0.0])
        out = ens.score(matrix)
        assert out["A"] == 1.0 and out["C"] == 0.0
        # 100% weight on y — should invert.
        ens2 = EnsemblePanelScorer([x, y], weights=[0.0, 1.0])
        out2 = ens2.score(matrix)
        assert out2["C"] == 1.0 and out2["A"] == 0.0
        # 50/50 — everything equal (3 tickers, disagreement cancels).
        ens3 = EnsemblePanelScorer([x, y])
        out3 = ens3.score(matrix)
        assert out3["A"] == pytest.approx(0.5)
        assert out3["B"] == pytest.approx(0.5)
        assert out3["C"] == pytest.approx(0.5)

    def test_feature_cols_union(self):
        a = _StubScorer(["f0", "f1"], pd.Series([1.0], index=["X"]))
        b = _StubScorer(["f1", "f2"], pd.Series([1.0], index=["X"]))
        ens = EnsemblePanelScorer([a, b])
        assert ens.feature_cols == ["f0", "f1", "f2"], (
            "union must preserve order and deduplicate"
        )

    def test_ties_get_average_rank(self):
        idx = ["A", "B", "C", "D"]
        matrix = pd.DataFrame({"f": [0.0]*4}, index=idx)
        # B and C tie on the middle rank.
        s = _StubScorer(["f"], pd.Series([1.0, 2.0, 2.0, 3.0], index=idx))
        ens = EnsemblePanelScorer([s])
        out = ens.score(matrix)
        # Sorted-descending positions: D(0), B(1)=C(2), A(3).
        # After tie averaging: D=0 → norm 1.0; B=C=1.5 → norm 1-1.5/3=0.5; A=3 → norm 0.
        assert out["D"] == pytest.approx(1.0)
        assert out["A"] == pytest.approx(0.0)
        assert out["B"] == pytest.approx(0.5)
        assert out["C"] == pytest.approx(0.5)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            EnsemblePanelScorer([])

    def test_rejects_weight_length_mismatch(self):
        s = _StubScorer(["f"], pd.Series([1.0], index=["A"]))
        with pytest.raises(ValueError):
            EnsemblePanelScorer([s], weights=[0.5, 0.5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
