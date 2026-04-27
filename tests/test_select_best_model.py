"""Tests for scripts/select_best_model.py — Phase 3 backend tournament.

Pin the discovery / scoring / ranking semantics so a future change to
the script's heuristics doesn't silently flip which artifact wins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import select_best_model as sbm   # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_panel_artifact(path: Path, *, ic: float, rows: int = 70000,
                          features: int = 28, sim_smoke: dict | None = None,
                          trained_date: str = "2026-04-26") -> Path:
    md = {
        "oos_mean_ic":  ic,
        "oos_std_ic":   0.02,
        "panel_shape":  {"rows": rows, "tickers": 99, "dates": 753},
        "trained_date": trained_date,
    }
    if sim_smoke is not None:
        md["sim_smoke"] = sim_smoke
    data = {
        "kind":         "panel_ltr_xgboost",
        "feature_cols": [f"f{i}" for i in range(features)],
        "metadata":     md,
    }
    path.write_text(json.dumps(data))
    return path


# ── Discovery ─────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_finds_current_and_baks(self, tmp_path):
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.json", ic=0.05)
        _write_panel_artifact(adir / "panel-ltr.xgboost.bak.json", ic=0.05)
        _write_panel_artifact(adir / "panel-ltr.lightgbm.bak.json", ic=0.04)
        _write_panel_artifact(adir / "panel-ltr.macro-enabled.bak.json", ic=0.03)
        cands = sbm.discover_candidates(adir)
        names = {c.name for c in cands}
        assert "current" in names
        assert "xgboost" in names
        assert "lightgbm" in names
        assert "macro-enabled" in names

    def test_skips_staging_previous_pretrain(self, tmp_path):
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.json", ic=0.05)
        _write_panel_artifact(adir / "panel-ltr.staging.json", ic=0.04)
        _write_panel_artifact(adir / "panel-ltr.previous.json", ic=0.045)
        _write_panel_artifact(adir / "panel-ltr.pre-train.json", ic=0.045)
        cands = sbm.discover_candidates(adir)
        names = {c.name for c in cands}
        assert names == {"current"}   # staging/previous/pre-train all skipped

    def test_handles_corrupt_artifact(self, tmp_path):
        """A malformed JSON should be skipped with a warning, not crash."""
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.json", ic=0.05)
        (adir / "panel-ltr.broken.bak.json").write_text("{not json")
        cands = sbm.discover_candidates(adir)
        names = {c.name for c in cands}
        assert "current" in names
        assert "broken" not in names

    def test_extracts_sim_smoke_metrics(self, tmp_path):
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.xgboost.bak.json", ic=0.05,
                              sim_smoke={"apy": 0.18, "sharpe": 1.5,
                                         "calmar": 2.1, "turnover_ratio": 2.5})
        cands = sbm.discover_candidates(adir)
        c = cands[0]
        assert c.sim_apy == 0.18
        assert c.sim_sharpe == 1.5
        assert c.sim_calmar == 2.1
        assert c.sim_turnover == 2.5


# ── Composite scoring ─────────────────────────────────────────────────────────

class TestComposite:
    def test_higher_ic_ranks_first_when_only_ic_present(self, tmp_path):
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.bad.bak.json",  ic=0.02)
        _write_panel_artifact(adir / "panel-ltr.good.bak.json", ic=0.06)
        _write_panel_artifact(adir / "panel-ltr.mid.bak.json",  ic=0.04)
        cands = sbm.discover_candidates(adir)
        ranked = sbm.score_candidates(cands, sbm.parse_weights(None))
        names = [c.name for c in ranked]
        assert names == ["good", "mid", "bad"]

    def test_sharpe_dominates_when_weight_overridden(self, tmp_path):
        """ic-weight=0, sharpe-weight=1 → ranking purely by Sharpe."""
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.a.bak.json", ic=0.06,
                              sim_smoke={"apy": 0.10, "sharpe": 0.5, "calmar": 0.5})
        _write_panel_artifact(adir / "panel-ltr.b.bak.json", ic=0.02,
                              sim_smoke={"apy": 0.20, "sharpe": 2.0, "calmar": 2.0})
        cands = sbm.discover_candidates(adir)
        weights = sbm.parse_weights("ic=0,sharpe=1,calmar=0")
        ranked = sbm.score_candidates(cands, weights)
        # Higher sharpe (b) wins despite worse IC
        assert ranked[0].name == "b"

    def test_missing_metrics_get_neutral_zscore(self, tmp_path):
        """A candidate without sim_smoke shouldn't get unfair credit /
        penalty — its z=0 (neutral) keeps it above-fail-below-pass."""
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.full.bak.json", ic=0.04,
                              sim_smoke={"apy": 0.10, "sharpe": 1.0, "calmar": 1.5})
        _write_panel_artifact(adir / "panel-ltr.icnly.bak.json", ic=0.04)
        cands = sbm.discover_candidates(adir)
        ranked = sbm.score_candidates(cands, sbm.parse_weights(None))
        # IC ties → composite reduces to sim component — full beats icnly
        assert ranked[0].name == "full"

    def test_constant_metric_yields_zero_zscore(self, tmp_path):
        """All-equal IC → all z=0 → composite=0 across the board (no
        artificial winner)."""
        adir = tmp_path / "artifacts"
        adir.mkdir()
        _write_panel_artifact(adir / "panel-ltr.a.bak.json", ic=0.04)
        _write_panel_artifact(adir / "panel-ltr.b.bak.json", ic=0.04)
        cands = sbm.discover_candidates(adir)
        ranked = sbm.score_candidates(cands, sbm.parse_weights(None))
        assert all(c.composite == 0.0 for c in ranked)


# ── Promote winner ────────────────────────────────────────────────────────────

class TestPromoteWinner:
    def test_promote_swaps_files(self, tmp_path):
        # Build a minimal strategy dir layout
        strategy_dir = tmp_path / "renquant_X"
        adir = strategy_dir / "artifacts"
        adir.mkdir(parents=True)
        # Wire kernel/model_acceptance.promote — needs the strategy_dir
        # to be on sys.path. Easier: insert the real renquant_104 kernel.
        sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))
        from kernel.model_acceptance import promote   # noqa

        active   = adir / "panel-ltr.json"
        new      = adir / "panel-ltr.winner.bak.json"
        _write_panel_artifact(active, ic=0.04)
        _write_panel_artifact(new,    ic=0.06)

        cands = sbm.discover_candidates(adir)
        rc = sbm.promote_winner(strategy_dir, "winner", cands)
        assert rc == 0
        # active now contains the winner artifact
        active_data = json.loads(active.read_text())
        assert active_data["metadata"]["oos_mean_ic"] == 0.06
        # prior preserved
        prev = adir / "panel-ltr.previous.json"
        assert prev.exists()

    def test_promote_current_is_noop(self, tmp_path):
        strategy_dir = tmp_path / "rq"
        adir = strategy_dir / "artifacts"
        adir.mkdir(parents=True)
        _write_panel_artifact(adir / "panel-ltr.json", ic=0.05)
        cands = sbm.discover_candidates(adir)
        rc = sbm.promote_winner(strategy_dir, "current", cands)
        assert rc == 0
        # No .previous.json created on no-op
        assert not (adir / "panel-ltr.previous.json").exists()

    def test_promote_unknown_name_fails(self, tmp_path):
        strategy_dir = tmp_path / "rq"
        adir = strategy_dir / "artifacts"
        adir.mkdir(parents=True)
        _write_panel_artifact(adir / "panel-ltr.json", ic=0.05)
        cands = sbm.discover_candidates(adir)
        rc = sbm.promote_winner(strategy_dir, "ghost", cands)
        assert rc == 1


# ── Weight parsing ────────────────────────────────────────────────────────────

class TestWeightParsing:
    def test_default(self):
        w = sbm.parse_weights(None)
        assert w == {"ic": 0.5, "sharpe": 0.3, "calmar": 0.2}

    def test_full_override(self):
        w = sbm.parse_weights("ic=0.7,sharpe=0.2,calmar=0.1")
        assert w["ic"] == 0.7

    def test_ignores_blanks(self):
        w = sbm.parse_weights("ic=0.5,  ,sharpe=0.5")
        assert w["sharpe"] == 0.5
