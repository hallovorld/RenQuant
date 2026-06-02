"""Tests for RegimeEnsemblePanelScorer (Track C, 2026-06-02).

Pin the dispatch + back-compat contract for the regime-specialist
ensemble. The scorer is the production-side counterpart to
``scripts/train_per_regime_panel.py`` — it loads up to 4 specialist
artifacts and dispatches at score time based on the regime detector's
final regime + confidence + posterior emitted by ``RegimeFinalizeTask``.

Test matrix (per Track C plan + ensemble-scorer contract):

  1. ``specialists`` field absent → ``load_from_config`` returns the
     legacy global scorer untouched (back-compat).
  2. All 4 specialists present, high confidence → uses
     ``specialists[final_regime].score(...)`` hard.
  3. 3 of 4 specialists present, high confidence, final_regime is the
     missing one → falls back to the global scorer.
  4. All 4 specialists, low confidence, top-2 posterior — blends top-2
     specialists by their normalized posteriors.
  5. Specialist artifact with mismatched feature recipe (introduces a
     column the global doesn't have) → raises
     ``StaleSpecialistArtifact`` at load time.

References:
  - doc/research/2026-06-02-bull-calm-signal-recovery-plan.md (Track C)
  - kernel/panel_pipeline/regime_ensemble_scorer.py
  - CLAUDE.md §1 PRIME DIRECTIVE, §7.1 paired tests, §7.6 fingerprint guards
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


# ── Stub PanelScorer.load to avoid real XGBoost dependency ─────────────────
# We patch the module-level ``PanelScorer.load`` to read a minimal JSON file
# and return a ``_FakeScorer`` whose .score() is a deterministic function of
# its (configurable) bias. That's enough to verify routing.

class _FakeScorer:
    """Stand-in for ``PanelScorer`` — bias-shifted column sum."""

    def __init__(self, name: str, bias: float, feature_cols: list[str],
                 metadata: dict | None = None):
        self.name = name
        self.bias = float(bias)
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}

    def score(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"FakeScorer.score: missing {missing}")
        vals = self.bias + X[self.feature_cols].sum(axis=1).values
        return pd.Series(vals, index=X.index, name="panel_score")


@pytest.fixture
def fake_artifacts(tmp_path):
    """Write small JSON sidecars + register a fake PanelScorer.load patch.

    Returns a dict: {label → (path, FakeScorer)}.
    """
    artifacts: dict[str, tuple[Path, _FakeScorer]] = {}
    feat_cols = ["f1", "f2"]
    for label, bias in [
        ("global", 0.0),
        ("BULL_CALM", 100.0),
        ("BEAR", 200.0),
        ("BULL_VOLATILE", 300.0),
        ("CHOPPY", 400.0),
    ]:
        p = tmp_path / f"panel-ltr.{label}.json"
        p.write_text(json.dumps({
            "label": label,
            "feature_cols": feat_cols,
            "kind": "panel_ltr_xgboost",
        }))
        artifacts[label] = (p, _FakeScorer(label, bias, feat_cols,
                                            metadata={"artifact_path": str(p)}))
    return artifacts


@pytest.fixture
def patched_scorer_loader(monkeypatch, fake_artifacts):
    """Make PanelScorer.load(path) return the fake scorer for that path."""
    from kernel.panel_pipeline import regime_ensemble_scorer as res
    by_path = {str(p): sc for (p, sc) in fake_artifacts.values()}

    def fake_load(path):
        p = str(path)
        if p not in by_path:
            raise FileNotFoundError(p)
        return by_path[p]

    monkeypatch.setattr(res.PanelScorer, "load", staticmethod(fake_load))
    return by_path


@pytest.fixture
def feature_matrix():
    """Common 3-ticker feature matrix indexed by ticker."""
    return pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]},
        index=["AAA", "BBB", "CCC"],
    )


# ── Test 1: back-compat ────────────────────────────────────────────────────

class TestBackCompat:
    """No specialists configured → return the legacy global scorer."""

    def test_no_specialists_returns_global(
        self, patched_scorer_loader, fake_artifacts, feature_matrix,
    ):
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            load_panel_scorer_with_ensemble,
            RegimeEnsemblePanelScorer,
        )
        global_path, global_scorer = fake_artifacts["global"]
        cfg = {"artifact_path": str(global_path)}
        loaded = load_panel_scorer_with_ensemble(cfg)
        # Legacy contract: returns the raw PanelScorer (NOT wrapped)
        assert not isinstance(loaded, RegimeEnsemblePanelScorer)
        assert loaded is global_scorer
        # Scoring path is unchanged
        out = loaded.score(feature_matrix)
        assert out.tolist() == [5.0, 7.0, 9.0]


# ── Test 2: hard-pick when confidence ≥ threshold ──────────────────────────

class TestHardPick:
    """High-confidence regime — bypass the global, use the specialist."""

    def _build(self, fake_artifacts, *, regimes: list[str]):
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            load_panel_scorer_with_ensemble,
        )
        global_path, _ = fake_artifacts["global"]
        cfg = {
            "artifact_path": str(global_path),
            "specialists": {r: str(fake_artifacts[r][0]) for r in regimes},
            "specialist_confidence_threshold": 0.8,
        }
        return load_panel_scorer_with_ensemble(cfg)

    def test_all_specialists_high_conf_uses_specialist(
        self, patched_scorer_loader, fake_artifacts, feature_matrix,
    ):
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            RegimeEnsemblePanelScorer,
        )
        ens = self._build(fake_artifacts,
                          regimes=["BULL_CALM", "BEAR", "BULL_VOLATILE", "CHOPPY"])
        assert isinstance(ens, RegimeEnsemblePanelScorer)
        ctx = types.SimpleNamespace(
            final_regime="BULL_CALM",
            regime_confidence=0.9,
            regime_posterior={"BULL_CALM": 0.9, "BEAR": 0.1},
        )
        out = ens.score(ctx, feature_matrix)
        # BULL_CALM bias = 100; col sums = [5, 7, 9]
        assert out.tolist() == [105.0, 107.0, 109.0]

    def test_three_of_four_high_conf_missing_specialist_falls_back(
        self, patched_scorer_loader, fake_artifacts, feature_matrix,
    ):
        # Build with BULL_CALM missing; final_regime = the missing one.
        ens = self._build(fake_artifacts,
                          regimes=["BEAR", "BULL_VOLATILE", "CHOPPY"])
        ctx = types.SimpleNamespace(
            final_regime="BULL_CALM",
            regime_confidence=0.95,
            regime_posterior={"BULL_CALM": 0.95, "BEAR": 0.05},
        )
        out = ens.score(ctx, feature_matrix)
        # Global bias = 0; col sums = [5, 7, 9]
        assert out.tolist() == [5.0, 7.0, 9.0]


# ── Test 3: blend on low confidence ────────────────────────────────────────

class TestBlend:
    """Confidence below threshold → posterior-weighted blend of top-k."""

    def test_blend_top_two_by_posterior(
        self, patched_scorer_loader, fake_artifacts, feature_matrix,
    ):
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            load_panel_scorer_with_ensemble,
        )
        global_path, _ = fake_artifacts["global"]
        cfg = {
            "artifact_path": str(global_path),
            "specialists": {
                r: str(fake_artifacts[r][0])
                for r in ["BULL_CALM", "BEAR", "BULL_VOLATILE", "CHOPPY"]
            },
            "specialist_confidence_threshold": 0.8,
            "specialist_blend_top_k": 2,
        }
        ens = load_panel_scorer_with_ensemble(cfg)
        # confidence below 0.8 → blend top-2 of posterior
        # posterior: BULL_CALM 0.6, BEAR 0.3, CHOPPY 0.1
        # top-2 normalized: BULL_CALM = 0.6/0.9 ≈ 0.6667
        #                    BEAR      = 0.3/0.9 ≈ 0.3333
        # scores per ticker (sum f1+f2 = 5/7/9):
        #   BULL_CALM specialist → 100 + sum
        #   BEAR      specialist → 200 + sum
        # blended → 0.6667 * (100 + sum) + 0.3333 * (200 + sum)
        #         = sum + (66.67 + 66.67)
        #         = sum + 133.333…
        ctx = types.SimpleNamespace(
            final_regime="BULL_CALM",
            regime_confidence=0.4,
            regime_posterior={"BULL_CALM": 0.6, "BEAR": 0.3, "CHOPPY": 0.1},
        )
        out = ens.score(ctx, feature_matrix)
        expected = [
            (100.0 + 5.0) * (0.6 / 0.9) + (200.0 + 5.0) * (0.3 / 0.9),
            (100.0 + 7.0) * (0.6 / 0.9) + (200.0 + 7.0) * (0.3 / 0.9),
            (100.0 + 9.0) * (0.6 / 0.9) + (200.0 + 9.0) * (0.3 / 0.9),
        ]
        np.testing.assert_allclose(out.values, expected, rtol=1e-10)


# ── Test 4: stale-fingerprint specialist ───────────────────────────────────

class TestStaleSpecialist:
    """A specialist that introduces a new feature column raises clearly."""

    def test_mismatched_recipe_raises(
        self, monkeypatch, fake_artifacts, tmp_path,
    ):
        from kernel.panel_pipeline import regime_ensemble_scorer as res
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            load_panel_scorer_with_ensemble,
            StaleSpecialistArtifact,
        )
        # Build a "rogue" specialist whose feature_cols introduces f3.
        global_path, global_scorer = fake_artifacts["global"]
        rogue_path = tmp_path / "panel-ltr.rogue.json"
        rogue_path.write_text(json.dumps({"label": "rogue"}))
        rogue = _FakeScorer("rogue", 999.0, ["f1", "f2", "f3"],
                             metadata={"artifact_path": str(rogue_path)})
        by_path = {
            str(global_path): global_scorer,
            str(rogue_path):  rogue,
        }

        def fake_load(path):
            p = str(path)
            if p not in by_path:
                raise FileNotFoundError(p)
            return by_path[p]

        monkeypatch.setattr(res.PanelScorer, "load", staticmethod(fake_load))
        cfg = {
            "artifact_path": str(global_path),
            "specialists": {"BULL_CALM": str(rogue_path)},
        }
        with pytest.raises(StaleSpecialistArtifact, match=r"specialist\[BULL_CALM\]"):
            load_panel_scorer_with_ensemble(cfg)


# ── Test 5 (bonus): ctx=None falls back gracefully ─────────────────────────

class TestCtxNoneFallback:
    """When ctx is None or fields are missing, fall back to global."""

    def test_ctx_none_uses_global(
        self, patched_scorer_loader, fake_artifacts, feature_matrix,
    ):
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            load_panel_scorer_with_ensemble,
        )
        global_path, _ = fake_artifacts["global"]
        cfg = {
            "artifact_path": str(global_path),
            "specialists": {
                "BULL_CALM": str(fake_artifacts["BULL_CALM"][0]),
            },
        }
        ens = load_panel_scorer_with_ensemble(cfg)
        out = ens.score(None, feature_matrix)
        # Global bias = 0
        assert out.tolist() == [5.0, 7.0, 9.0]
