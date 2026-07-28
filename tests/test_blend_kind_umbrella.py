"""Umbrella-kernel `kind="blend"` acceptance (pipeline#218 umbrella mirror).

2026-07-28 rehearsal-caught fork divergence: the shadow_blend full-lane
rehearsal fail-closed with ``panel_scorer_invalid_kind`` because pipeline#218
only patched the renquant-pipeline kernel copies while ``live.runner``
executes the UMBRELLA-LOCAL fork under ``backtesting/renquant_104/kernel/``.

Pins here:
  1. The umbrella registry dispatches kind "blend" by DELEGATING to the
     pinned renquant-pipeline ``load_blend_scorer`` — the REAL pinned
     shadow_blend profile (``.subrepo_runtime`` configs, read-only) loads
     against the real artifacts (read-only) with BOTH component pins
     verified, and scores a small synthetic raw union matrix.
  2. An unregistered kind still fail-closes as ``panel_scorer_invalid_kind``
     (the exact failure the rehearsal surfaced must stay armed for real
     bogus kinds).
  3. The kind-branch sites the mirror touched behave: DriftGuardTask skips
     the structural check for blend (union features are rebuilt in
     ApplyScoresTask), and LoadScorerTask anchors its strict-consistency
     path on component 0 when no top-level ``artifact_path`` is configured
     (both the fresh-load and preloaded branches).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"


def _pinned_configs_dir() -> Path | None:
    """The pinned strategy configs dir (read-only). Resolution mirrors the
    runtime: RENQUANT_SUBREPO_ROOT first, then the repo-local runtime."""
    candidates = []
    subrepo_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if subrepo_root:
        candidates.append(Path(subrepo_root) / "renquant-strategy-104" / "configs")
    candidates.append(REPO_ROOT / ".subrepo_runtime" / "repos"
                      / "renquant-strategy-104" / "configs")
    for c in candidates:
        if (c / "strategy_config.shadow_blend.json").is_file():
            return c
    return None


_CONFIGS = _pinned_configs_dir()

needs_pinned_profile = pytest.mark.skipif(
    _CONFIGS is None,
    reason="pinned shadow_blend profile not present "
           "(set RENQUANT_SUBREPO_ROOT or run on the ops machine)",
)


def _blend_config() -> dict:
    raw = json.loads(
        (_CONFIGS / "strategy_config.shadow_blend.json").read_text())
    panel_cfg = raw["ranking"]["panel_scoring"]
    return {
        "ranking": {"panel_scoring": panel_cfg},
        "_strategy_dir": str(STRATEGY_DIR),
    }


class TestBlendKindRegistry:
    def test_blend_kind_registered(self):
        from kernel.panel_pipeline.model_registry import registry

        handler = registry.get("blend")
        assert handler.requires_history is False
        assert "blend" in registry.list()

    def test_bogus_kind_still_rejected(self):
        from kernel.panel_pipeline.model_registry import registry

        with pytest.raises(ValueError, match="not registered"):
            registry.get("definitely_not_a_kind")

    @needs_pinned_profile
    def test_real_pinned_profile_loads_and_scores(self):
        """The umbrella load+score path accepts the REAL pinned shadow_blend
        profile: both pins verified fail-closed, union features exposed, and
        a small synthetic candidate frame produces finite blend z-sums."""
        from kernel.panel_pipeline.model_registry import registry

        cfg = _blend_config()
        panel_cfg = cfg["ranking"]["panel_scoring"]
        handler = registry.get("blend")
        scorer = handler.scorer_loader(None, cfg)  # artifact_path ignored

        meta = scorer.metadata
        assert meta["kind"] == "blend"
        comps = meta["components"]
        assert len(comps) == 2
        # Loader verified the pins; re-check the observed identities against
        # the config pins here so the test fails loudly on a silent skip.
        for comp_meta, comp_cfg in zip(comps, panel_cfg["components"]):
            pinned = comp_cfg["expected_content_sha256"].split(":", 1)[-1].lower()
            observed = comp_meta["content_sha256"].split(":", 1)[-1].lower()
            assert observed.startswith(pinned), (
                f"content pin not honoured: {pinned} vs {observed}")
        assert meta["config_fingerprint"].startswith("sha256:")
        assert scorer.requires_history is False
        assert len(scorer.feature_cols) > 100  # union of both alpha158+fund legs

        rng = np.random.default_rng(7)
        tickers = ["SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E", "SYN_F"]
        X = pd.DataFrame(
            rng.normal(0.0, 1.0, size=(len(tickers), len(scorer.feature_cols))),
            index=tickers,
            columns=scorer.feature_cols,
        )
        scores = scorer.score(X, ctx=None)
        assert isinstance(scores, pd.Series)
        assert list(scores.index) == tickers
        finite = scores[np.isfinite(scores)]
        assert len(finite) >= 2, f"expected >=2 finite blend scores, got {scores}"
        # Both legs healthy on a 6-name random cross-section.
        assert meta["degraded_reason"] is None

    @needs_pinned_profile
    def test_missing_pin_fails_closed(self):
        """Dropping a required pin must raise (never silently load)."""
        from kernel.panel_pipeline.model_registry import registry

        cfg = _blend_config()
        panel_cfg = cfg["ranking"]["panel_scoring"]
        broken = [dict(c) for c in panel_cfg["components"]]
        broken[1].pop("expected_config_fingerprint")
        cfg["ranking"]["panel_scoring"] = dict(panel_cfg, components=broken)
        with pytest.raises(Exception):
            registry.get("blend").scorer_loader(None, cfg)


class TestLoadScorerTaskBlendDispatch:
    def _ctx(self, panel_cfg: dict) -> SimpleNamespace:
        return SimpleNamespace(
            config={
                "ranking": {"panel_scoring": panel_cfg},
                "_strategy_dir": str(STRATEGY_DIR),
            },
            candidates=[SimpleNamespace(ticker="AAPL")],
            holdings={},
            counters={},
        )

    def test_invalid_kind_fail_close_stays_armed(self, monkeypatch):
        """The exact rehearsal failure reason must stay armed for kinds that
        really are unregistered."""
        import kernel.panel_pipeline.job_panel_scoring as jps

        recorded = {}

        def _record(ctx, **kw):
            recorded.update(kw)

        monkeypatch.setattr(jps, "_submit_gate_verdict", _record)
        ctx = self._ctx({
            "enabled": True,
            "kind": "bogus_kind",
            "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
        })
        out = jps.LoadScorerTask().run(ctx)
        assert out is False
        assert ctx.skip_buys is True
        assert recorded.get("reason") == "panel_scorer_invalid_kind"

    def test_component0_anchor_without_top_level_path(self):
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask

        panel_cfg = {
            "enabled": True,
            "kind": "blend",
            "components": [
                {"artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json"},
                {"artifact_path": "artifacts/shadow/panel-clf.top-decile.fwd60.json"},
            ],
        }
        ctx = self._ctx(panel_cfg)
        p = LoadScorerTask._blend_component0_path(ctx, panel_cfg)
        assert p is not None
        assert p.name == "panel-ltr.alpha158_fund.json"
        assert LoadScorerTask._blend_component0_path(ctx, {"components": []}) is None

    def test_preloaded_blend_anchors_consistency_on_component0(self, monkeypatch):
        """Preloaded branch (adapter/LEAN): no top-level artifact_path and a
        stat-less composite metadata must still anchor the strict gate on
        component 0 — the #218 preloaded-branch fix, mirrored."""
        from kernel.panel_pipeline.job_panel_scoring import LoadScorerTask

        seen = {}

        def _capture(ctx, panel_cfg, scorer, path):
            seen["path"] = path
            return True

        monkeypatch.setattr(
            LoadScorerTask, "_assert_config_consistency",
            staticmethod(_capture),
        )
        panel_cfg = {
            "enabled": True,
            "kind": "blend",
            "components": [
                {"artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json"},
                {"artifact_path": "artifacts/shadow/panel-clf.top-decile.fwd60.json"},
            ],
        }
        ctx = self._ctx(panel_cfg)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"kind": "blend"}, feature_cols=["a"],
        )
        LoadScorerTask().run(ctx)
        assert seen["path"] is not None
        assert seen["path"].name == "panel-ltr.alpha158_fund.json"


class TestBlendKindBranchSites:
    def test_drift_guard_skips_blend(self):
        from kernel.panel_pipeline.tasks_feature_matrix import DriftGuardTask

        ctx = SimpleNamespace(
            _panel_matrix=pd.DataFrame({"x": [1.0]}, index=["AAPL"]),
            _panel_scorer=SimpleNamespace(
                metadata={"kind": "blend"},
                feature_cols=["not_in_matrix"],  # would trip the check if run
                requires_history=False,
            ),
        )
        assert DriftGuardTask().run(ctx) is None

    def test_resolve_frames_target_only_matrix_for_blend(self):
        from kernel.panel_pipeline.tasks_feature_matrix import (
            ResolveInferenceFramesTask,
        )

        ctx = SimpleNamespace(
            candidates=[SimpleNamespace(ticker="AAPL")],
            holdings={},
            _panel_scorer=SimpleNamespace(
                metadata={"kind": "blend"},
                requires_history=False,
            ),
            _panel_feature_frames=None,
        )
        out = ResolveInferenceFramesTask().run(ctx)
        assert out is False  # alpha158-rebuild short-circuit
        assert "__alpha158_target__" in ctx._panel_matrix.columns
        assert list(ctx._panel_matrix.index) == ["AAPL"]
