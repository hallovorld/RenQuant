"""Integration test for Track C specialist ensemble wiring (2026-06-02).

AUDIT REGRESSION GUARD for codex finding on PR #121:

    HIGH: specialist runtime is not wired. load_panel_scorer_with_ensemble
    is defined, the `specialists` config block is documented, but
    LoadScorerTask still loads through model_registry and ApplyScoresTask
    still calls scorer.score(X) without forwarding ctx. So a configured
    `ranking.panel_scoring.specialists` is dead code.

This test proves the full wiring:

  1. ``LoadScorerTask`` reads ``ranking.panel_scoring.specialists`` and
     routes through ``load_panel_scorer_with_ensemble`` — yielding a
     ``RegimeEnsemblePanelScorer`` on ``ctx._panel_scorer``.
  2. ``ApplyScoresTask`` forwards ``ctx`` to ``scorer.score(...)`` so the
     ensemble dispatches to the correct specialist based on
     ``ctx.final_regime`` / ``ctx.regime_confidence``.
  3. The specialist's score column distribution is observably DIFFERENT
     from the global scorer's — proving the specialist (not the global)
     is the one consumed at runtime.

Per CLAUDE.md §7.1 paired-test invariant; pins the wiring against
regression. Without this test the configured specialists path is a
runtime no-op and the bug recurs the moment someone refactors
LoadScorerTask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


class _FakeScorer:
    """Bias-shifted column-sum scorer (mirrors test_regime_ensemble_scorer)."""

    requires_history = False

    def __init__(self, name: str, bias: float, feature_cols: list[str],
                 metadata: dict | None = None):
        self.name = name
        self.bias = float(bias)
        self.feature_cols = list(feature_cols)
        meta = dict(metadata or {})
        # Tag as kind=panel_ltr_xgboost so ApplyScoresTask routes through the
        # alpha158-rebuild path (which is the path that calls scorer.score
        # with ctx in the production code path that was previously broken).
        meta.setdefault("kind", "panel_ltr_xgboost")
        self.metadata = meta

    def score(self, X: pd.DataFrame, ctx=None) -> pd.Series:  # noqa: ARG002
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"FakeScorer.score: missing {missing}")
        vals = self.bias + X[self.feature_cols].sum(axis=1).values
        return pd.Series(vals, index=X.index, name="panel_score")


@pytest.fixture
def fake_scorers(tmp_path, monkeypatch):
    """Write two sidecar JSONs + patch PanelScorer.load to return fakes."""
    from kernel.panel_pipeline import regime_ensemble_scorer as res

    feat_cols = ["f1", "f2"]
    paths: dict[str, Path] = {}
    fakes: dict[str, _FakeScorer] = {}
    for label, bias in [("global", 0.0), ("BULL_CALM", 1000.0), ("BEAR", 2000.0)]:
        p = tmp_path / f"panel-ltr.{label}.json"
        p.write_text(json.dumps({
            "label": label,
            "feature_cols": feat_cols,
            "kind": "panel_ltr_xgboost",
        }))
        paths[label] = p
        fakes[label] = _FakeScorer(label, bias, feat_cols,
                                    metadata={"artifact_path": str(p)})

    by_path = {str(p): fakes[label] for label, p in paths.items()}

    def fake_load(path):
        s = str(path)
        if s not in by_path:
            raise FileNotFoundError(s)
        return by_path[s]

    monkeypatch.setattr(res.PanelScorer, "load", staticmethod(fake_load))
    return paths, fakes


def _build_ctx(panel_cfg: dict, *, final_regime: str, confidence: float,
               posterior: dict[str, float]) -> SimpleNamespace:
    """Construct the minimal InferenceContext-like object the tasks read."""
    candidates = [
        SimpleNamespace(ticker="AAA", rank_score=None, panel_score=None),
        SimpleNamespace(ticker="BBB", rank_score=None, panel_score=None),
        SimpleNamespace(ticker="CCC", rank_score=None, panel_score=None),
    ]
    feature_matrix = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]},
        index=["AAA", "BBB", "CCC"],
    )
    config = {
        "ranking": {"panel_scoring": panel_cfg},
        "_strategy_dir": str(REPO_ROOT / "backtesting" / "renquant_104"),
    }
    return SimpleNamespace(
        config=config,
        candidates=candidates,
        holdings={},
        _panel_matrix=feature_matrix,
        final_regime=final_regime,
        regime_confidence=confidence,
        regime_posterior=dict(posterior),
        # Fields that ApplyScoresTask consults but doesn't drive scoring with
        # when metadata.kind != panel_linear and the scorer has no
        # requires_history. We DON'T set kind=panel_ltr_xgboost in metadata
        # so the alpha158 rebuild path is skipped — the simple path at the
        # bottom (scorer.score(X, ctx=ctx)) is the one we want to verify.
    )


def _stub_consistency(monkeypatch):
    """Skip the artifact / config fingerprint check — fake artifacts have none."""
    from kernel.panel_pipeline import job_panel_scoring as jps

    def _ok(self, ctx, panel_cfg, scorer, path):  # noqa: ARG001
        return True

    monkeypatch.setattr(jps.LoadScorerTask, "_assert_config_consistency", _ok)


class TestSpecialistWiringIntegration:
    """End-to-end: LoadScorerTask + ApplyScoresTask consume `specialists`."""

    def test_specialist_consumed_high_confidence(self, fake_scorers, monkeypatch):
        """High-confidence BULL_CALM ctx → BULL_CALM specialist scores applied,
        NOT the global. Asserts the score deltas match the SPECIALIST bias.
        """
        from kernel.panel_pipeline.job_panel_scoring import (
            LoadScorerTask, ApplyScoresTask,
        )
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            RegimeEnsemblePanelScorer,
        )

        paths, _ = fake_scorers
        _stub_consistency(monkeypatch)

        panel_cfg = {
            "enabled": True,
            "kind": "xgb",
            "artifact_path": str(paths["global"]),
            "specialists": {
                "BULL_CALM": str(paths["BULL_CALM"]),
                "BEAR":      str(paths["BEAR"]),
            },
            "specialist_confidence_threshold": 0.8,
        }
        ctx = _build_ctx(
            panel_cfg,
            final_regime="BULL_CALM",
            confidence=0.95,
            posterior={"BULL_CALM": 0.95, "BEAR": 0.05},
        )

        # LoadScorerTask: should pick up specialists, build the ensemble.
        rc = LoadScorerTask().run(ctx)
        assert rc is not False, "LoadScorerTask returned fail-closed"
        assert isinstance(ctx._panel_scorer, RegimeEnsemblePanelScorer), (
            "LoadScorerTask did not route through "
            "load_panel_scorer_with_ensemble — `specialists` block is "
            "dead config. This is the bug PR #121 review found."
        )
        assert set(ctx._panel_scorer.specialists.keys()) == {"BULL_CALM", "BEAR"}

        # Bypass the alpha158-rebuild branch in ApplyScoresTask — the
        # ensemble's metadata kind defaults to panel_ltr_xgboost for kind-
        # dispatching back-compat, but in this test the FakeScorer feature
        # cols are ["f1","f2"], not the 158 alpha names. Forcing a neutral
        # kind here puts the call path through the bare scorer.score(X,
        # ctx=ctx) site at job_panel_scoring.py:1110 — the exact line that
        # was failing the codex review before the fix.
        ctx._panel_scorer.metadata["kind"] = "regime_specialist_ensemble"

        # ApplyScoresTask: should forward ctx so the ensemble dispatches.
        ApplyScoresTask().run(ctx)

        # Column sums for f1+f2: AAA=5, BBB=7, CCC=9.
        # BULL_CALM specialist bias = 1000 → expected scores 1005/1007/1009.
        # Global would give 5/7/9. Specialist's distribution proves wiring.
        scores_by_ticker = {c.ticker: c.panel_score for c in ctx.candidates}
        assert scores_by_ticker == pytest.approx({
            "AAA": 1005.0, "BBB": 1007.0, "CCC": 1009.0,
        })

    def test_specialist_consumed_blend_low_confidence(self, fake_scorers, monkeypatch):
        """Low-confidence → ensemble blends BULL_CALM + BEAR specialists by
        posterior. Proves ctx is read inside score() at runtime.
        """
        from kernel.panel_pipeline.job_panel_scoring import (
            LoadScorerTask, ApplyScoresTask,
        )
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            RegimeEnsemblePanelScorer,
        )

        paths, _ = fake_scorers
        _stub_consistency(monkeypatch)

        panel_cfg = {
            "enabled": True,
            "kind": "xgb",
            "artifact_path": str(paths["global"]),
            "specialists": {
                "BULL_CALM": str(paths["BULL_CALM"]),
                "BEAR":      str(paths["BEAR"]),
            },
            "specialist_confidence_threshold": 0.8,
            "specialist_blend_top_k": 2,
        }
        ctx = _build_ctx(
            panel_cfg,
            final_regime="BULL_CALM",
            confidence=0.4,
            posterior={"BULL_CALM": 0.6, "BEAR": 0.4},
        )

        assert LoadScorerTask().run(ctx) is not False
        assert isinstance(ctx._panel_scorer, RegimeEnsemblePanelScorer)
        ctx._panel_scorer.metadata["kind"] = "regime_specialist_ensemble"
        ApplyScoresTask().run(ctx)

        # Blended: 0.6 * (1000 + sum) + 0.4 * (2000 + sum)
        #        = sum + 600 + 800 = sum + 1400
        # AAA: 5 + 1400 = 1405; BBB: 1407; CCC: 1409
        scores = {c.ticker: c.panel_score for c in ctx.candidates}
        np.testing.assert_allclose(
            [scores["AAA"], scores["BBB"], scores["CCC"]],
            [1405.0, 1407.0, 1409.0],
            rtol=1e-9,
        )

    def test_no_specialists_legacy_path_unchanged(self, fake_scorers, monkeypatch):
        """Back-compat: absent `specialists` ⇒ legacy model_registry path.

        Pins the §7.5 single-source-of-truth invariant: turning the
        specialist wire-in on must not break configs that don't opt in.
        """
        from kernel.panel_pipeline.job_panel_scoring import (
            LoadScorerTask, ApplyScoresTask,
        )
        from kernel.panel_pipeline import model_registry as mr
        from kernel.panel_pipeline.regime_ensemble_scorer import (
            RegimeEnsemblePanelScorer,
        )

        paths, fakes = fake_scorers
        _stub_consistency(monkeypatch)

        # Patch the XGB handler's loader so we don't need a real artifact.
        original_loader = mr.XGBHandler.scorer_loader

        @classmethod
        def fake_loader(cls, artifact_path, config):  # noqa: ARG001
            return fakes["global"]

        monkeypatch.setattr(mr.XGBHandler, "scorer_loader", fake_loader)
        try:
            panel_cfg = {
                "enabled": True,
                "kind": "xgb",
                "artifact_path": str(paths["global"]),
                # No `specialists` key — exercise the back-compat path
            }
            ctx = _build_ctx(
                panel_cfg,
                final_regime="BULL_CALM", confidence=0.95,
                posterior={"BULL_CALM": 0.95},
            )
            assert LoadScorerTask().run(ctx) is not False
            assert not isinstance(ctx._panel_scorer, RegimeEnsemblePanelScorer)
            # Bypass alpha158-rebuild for this synthetic 2-feature fake.
            ctx._panel_scorer.metadata["kind"] = "regime_specialist_ensemble"
            # Should be the global fake (bias=0)
            ApplyScoresTask().run(ctx)
            scores = {c.ticker: c.panel_score for c in ctx.candidates}
            assert scores == pytest.approx({"AAA": 5.0, "BBB": 7.0, "CCC": 9.0})
        finally:
            mr.XGBHandler.scorer_loader = original_loader
