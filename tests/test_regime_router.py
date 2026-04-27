"""Tests for kernel/panel_pipeline/regime_router.py — T2-3 Phase A.

Pin the contracts so future training/wiring work can't silently break the
config gate, default-fallback semantics, or the operator-visible WARNING
that fires when ensemble is enabled but per-regime artifacts are missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

from kernel.panel_pipeline.regime_router import (   # noqa: E402
    KNOWN_REGIMES,
    RegimeRouter,
)


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    """Bare strategy dir with an empty `artifacts/` subfolder."""
    (tmp_path / "artifacts").mkdir()
    return tmp_path


def _write_stub_artifact(strategy_dir: Path, name: str, payload: dict | None = None) -> Path:
    payload = payload if payload is not None else {"kind": "panel_ltr_xgboost", "stub": True}
    p = strategy_dir / "artifacts" / name
    p.write_text(json.dumps(payload))
    return p


# ── KNOWN_REGIMES contract ──────────────────────────────────────────────────────

class TestKnownRegimes:
    def test_known_regimes_set(self):
        """The 4 macro regimes must match kernel/regime.py output names so the
        dispatch on `ctx.regime` is direct."""
        assert KNOWN_REGIMES == ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR")


# ── has_regime_ensemble ────────────────────────────────────────────────────────

class TestHasRegimeEnsemble:
    def test_false_when_config_disabled(self, strategy_dir):
        cfg = {"panel_ltr": {"regime_ensemble": {"enabled": False}}}
        router = RegimeRouter(strategy_dir, cfg)
        assert router.has_regime_ensemble() is False

    def test_false_when_no_regime_section(self, strategy_dir):
        """Missing config section defaults to disabled."""
        router = RegimeRouter(strategy_dir, {"panel_ltr": {}})
        assert router.has_regime_ensemble() is False

    def test_false_when_enabled_but_no_artifacts(self, strategy_dir):
        cfg = {"panel_ltr": {"regime_ensemble": {"enabled": True}}}
        router = RegimeRouter(strategy_dir, cfg)
        # No regime artifacts in strategy_dir/artifacts
        assert router.has_regime_ensemble() is False

    def test_true_when_enabled_and_one_artifact_present(self, strategy_dir):
        _write_stub_artifact(strategy_dir, "panel-ltr.regime-bull_calm.json")
        cfg = {"panel_ltr": {"regime_ensemble": {"enabled": True}}}
        router = RegimeRouter(strategy_dir, cfg)
        assert router.has_regime_ensemble() is True

    def test_true_when_all_four_artifacts_present(self, strategy_dir):
        for r in KNOWN_REGIMES:
            _write_stub_artifact(strategy_dir, f"panel-ltr.regime-{r.lower()}.json")
        cfg = {"panel_ltr": {"regime_ensemble": {"enabled": True}}}
        router = RegimeRouter(strategy_dir, cfg)
        assert router.has_regime_ensemble() is True


# ── pick_artifact ──────────────────────────────────────────────────────────────

class TestPickArtifact:
    def test_returns_regime_specific_path_when_exists(self, strategy_dir):
        p = _write_stub_artifact(strategy_dir, "panel-ltr.regime-bear.json")
        router = RegimeRouter(strategy_dir, {})
        picked = router.pick_artifact("BEAR")
        assert picked == p

    def test_falls_back_to_default_when_regime_missing(self, strategy_dir, caplog):
        """Regime ensemble enabled but a specific regime artifact is missing
        → return default panel-ltr.json so the system degrades gracefully
        rather than crashing."""
        import logging
        caplog.set_level(logging.WARNING, logger="kernel.panel_pipeline.regime_router")
        router = RegimeRouter(strategy_dir, {})
        picked = router.pick_artifact("CHOPPY")
        assert picked == strategy_dir / "artifacts" / "panel-ltr.json"

    def test_unknown_regime_warns_and_returns_default(self, strategy_dir, caplog):
        """Unknown regime label (e.g. typo / future regime not in KNOWN_REGIMES)
        must NOT silently load a wrong artifact — it falls back to default
        with a WARN."""
        import logging
        caplog.set_level(logging.WARNING, logger="kernel.panel_pipeline.regime_router")
        router = RegimeRouter(strategy_dir, {})
        picked = router.pick_artifact("MARS_RECESSION")
        assert picked == strategy_dir / "artifacts" / "panel-ltr.json"
        text = " ".join(rec.message for rec in caplog.records)
        assert "unknown regime" in text or "MARS_RECESSION" in text

    def test_audit_2nd_round_6_fallback_logs_warning(self, strategy_dir, caplog):
        """Audit 2nd-round #6 fix (2026-04-27): when ensemble is enabled but
        no per-regime artifact exists, the fallback to default panel-ltr.json
        must surface as WARNING (not silent INFO) so operators see the
        degradation. Pre-fix: silent INFO log buried this fact."""
        import logging
        caplog.set_level(logging.WARNING, logger="kernel.panel_pipeline.regime_router")
        # No artifacts written. Ensemble enabled.
        cfg = {"panel_ltr": {"regime_ensemble": {"enabled": True}}}
        router = RegimeRouter(strategy_dir, cfg)
        router.pick_artifact("BULL_CALM")
        warning_messages = [
            rec.message for rec in caplog.records
            if rec.levelno >= logging.WARNING
        ]
        joined = " ".join(warning_messages)
        # The WARN must mention regime + fallback so operator can act
        assert "BULL_CALM" in joined
        assert "fallback" in joined.lower() or "default" in joined.lower()


# ── inventory ──────────────────────────────────────────────────────────────────

class TestInventory:
    def test_inventory_reports_all_four_regimes(self, strategy_dir):
        router = RegimeRouter(strategy_dir, {})
        inv = router.inventory()
        assert set(inv.keys()) == set(KNOWN_REGIMES)
        for regime in KNOWN_REGIMES:
            entry = inv[regime]
            # exists must be False (no artifacts written)
            assert entry["exists"] is False
            assert entry["size"] is None
            assert entry["mtime"] is None
            # path key must point at regime-specific artifact
            assert f"regime-{regime.lower()}.json" in entry["path"]

    def test_inventory_picks_up_existing_artifact_metadata(self, strategy_dir):
        p = _write_stub_artifact(strategy_dir, "panel-ltr.regime-choppy.json",
                                  {"kind": "panel_ltr_xgboost", "stub": True})
        router = RegimeRouter(strategy_dir, {})
        inv = router.inventory()
        assert inv["CHOPPY"]["exists"] is True
        assert inv["CHOPPY"]["size"] is not None
        assert inv["CHOPPY"]["size"] == p.stat().st_size
        assert inv["CHOPPY"]["mtime"] is not None
        # And the others stay False
        for r in KNOWN_REGIMES:
            if r != "CHOPPY":
                assert inv[r]["exists"] is False


# ── Defensive contract checks (would catch refactor regressions) ────────────────

class TestDefensiveContracts:
    def test_artifact_naming_convention(self, strategy_dir):
        """Per-regime artifact naming MUST be `panel-ltr.regime-<lowercase>.json`
        so training/finalization scripts can write them deterministically.
        Pinning the convention here prevents drift between trainer + router."""
        router = RegimeRouter(strategy_dir, {})
        # Inspect the path produced by pick_artifact (using existence check
        # to force the regime-specific branch)
        for regime in KNOWN_REGIMES:
            _write_stub_artifact(strategy_dir, f"panel-ltr.regime-{regime.lower()}.json")
        for regime in KNOWN_REGIMES:
            picked = router.pick_artifact(regime)
            assert picked.name == f"panel-ltr.regime-{regime.lower()}.json"

    def test_default_path_is_panel_ltr_json(self, strategy_dir):
        """Default fallback artifact MUST be artifacts/panel-ltr.json — that's
        the PROD path the rest of the pipeline reads."""
        router = RegimeRouter(strategy_dir, {})
        picked = router.pick_artifact("BEAR")  # missing → default
        assert picked == strategy_dir / "artifacts" / "panel-ltr.json"
