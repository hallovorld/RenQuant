"""Tests for kernel.preflight — pre-flight smoke test for cron startup.

Catches the class of bugs where artifact / config / state file drifts.
Mandatory at every cron startup; HARD failures raise PreflightFailed
which the runner converts to ntfy + abort.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightFailed,
    run_preflight,
    _check_model_artifact,
    _check_best_iter,
    _check_config_fingerprint,
    _check_watchlist_size,
    _check_feature_coverage,
    _check_state_file,
    _check_broker_connect,
)
from kernel.config_consistency import (  # noqa: E402
    fingerprint_config, _model_relevant_fields,
)


# ── Fixture: synthetic strategy dir + healthy artifact ─────────────────────

@pytest.fixture
def healthy_setup(tmp_path):
    """Build a config + artifact that should pass every check."""
    cfg = {
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "panel_ltr": {
            "lookahead_days": 10,
            "xgb_params": {"objective": "rank:pairwise"},
            "asset_embeddings": {"enabled": False},
            "min_best_iter": 20,
            "artifact_path": "artifacts/panel-ltr.json",
        },
    }
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    panel = art_dir / "panel-ltr.json"
    panel.write_text(json.dumps({
        "best_iter": 50,
        "oos_mean_ic": 0.045,
        "feature_cols": ["rsi", "macd", "bbp"],
        "config_fingerprint": fingerprint_config(cfg),
        "config_fingerprint_fields": _model_relevant_fields(cfg),
    }))
    return cfg, tmp_path


# ── P-MODEL-ARTIFACT ───────────────────────────────────────────────────────

class TestCheckModelArtifact:
    def test_pass_when_artifact_exists(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_model_artifact(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_artifact_missing(self, tmp_path):
        cfg = {"panel_ltr": {"artifact_path": "artifacts/missing.json"}}
        r = _check_model_artifact(cfg, tmp_path)
        assert not r.ok and r.severity == "hard"
        assert "missing" in r.message.lower()

    def test_fail_on_unreadable_json(self, tmp_path):
        cfg = {"panel_ltr": {"artifact_path": "artifacts/x.json"}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/x.json").write_text("{not json")
        r = _check_model_artifact(cfg, tmp_path)
        assert not r.ok


# ── P-BEST-ITER (BUG-CV-2 invariant) ───────────────────────────────────────

class TestCheckBestIter:
    def test_pass_when_above_threshold(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_best_iter(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_below_threshold(self, tmp_path):
        """Production was discovered with best_iter=4 today.
        This check refuses to trade on an undertrained model."""
        cfg = {
            "panel_ltr": {
                "min_best_iter": 20,
                "artifact_path": "artifacts/panel-ltr.json",
            },
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
            "oos_mean_ic": 0.04,
        }))
        r = _check_best_iter(cfg, tmp_path)
        assert not r.ok and r.severity == "hard"
        assert "best_iter=4" in r.message
        assert "Retrain required" in r.message

    def test_soft_pass_when_best_iter_missing(self, tmp_path):
        """Legacy artifacts (e.g. transformer) may not stamp best_iter."""
        cfg = {"panel_ltr": {"artifact_path": "artifacts/panel-ltr.json"}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "oos_mean_ic": 0.04,
        }))
        r = _check_best_iter(cfg, tmp_path)
        assert r.ok and r.severity == "soft"


# ── P-CONFIG-FP (catches the 24h watchlist-mismatch incident class) ────────

class TestCheckConfigFingerprint:
    def test_pass_when_fingerprints_match(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_config_fingerprint(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_watchlist_drifted(self, healthy_setup):
        cfg, sd = healthy_setup
        # Mutate config so watchlist differs from stored fingerprint
        cfg["watchlist"] = ["AAPL", "MSFT", "NVDA", "TSLA"]   # extra ticker
        r = _check_config_fingerprint(cfg, sd)
        assert not r.ok and r.severity == "hard"
        assert "watchlist" in r.message

    def test_fail_when_objective_changed(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["panel_ltr"]["xgb_params"]["objective"] = "rank:ndcg"
        r = _check_config_fingerprint(cfg, sd)
        assert not r.ok
        assert "objective" in r.message

    def test_soft_pass_when_no_fingerprint_in_artifact(self, tmp_path):
        """Pre-2026-04-28 artifacts lack fingerprint — soft pass."""
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {"artifact_path": "artifacts/panel-ltr.json"},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 50,
        }))
        r = _check_config_fingerprint(cfg, tmp_path)
        assert r.ok and r.severity == "soft"


# ── P-WATCHLIST ────────────────────────────────────────────────────────────

class TestCheckWatchlist:
    def test_pass_when_watchlist_matches(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_watchlist_size(cfg, sd)
        assert r.ok

    def test_fail_when_watchlist_differs(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["watchlist"] = ["AAPL", "MSFT"]   # missing NVDA
        r = _check_watchlist_size(cfg, sd)
        assert not r.ok
        assert "in_trained_not_live" in r.message


# ── P-FEATURE-COVER ────────────────────────────────────────────────────────

class TestCheckFeatureCoverage:
    def test_skip_when_ngboost_disabled(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_feature_coverage(cfg, sd)
        assert r.ok and r.severity == "soft"

    def test_pass_when_features_overlap(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["ranking"] = {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}
        # Stamp NGBoost head with same features as panel
        (sd / "artifacts/ngboost-head.json").write_text(json.dumps({
            "feature_cols": ["rsi", "macd", "bbp"],
        }))
        r = _check_feature_coverage(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_too_many_features_missing(self, healthy_setup):
        """Today's macro-drift bug class: NGBoost head has 184 macro
        cols but panel only has 27 → 84.8% missing."""
        cfg, sd = healthy_setup
        cfg["ranking"] = {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}
        # NGBoost head has 5 features but panel only has 3 → 40% missing
        (sd / "artifacts/ngboost-head.json").write_text(json.dumps({
            "feature_cols": ["rsi", "macd", "bbp", "vxx_z", "hyg_z"],
        }))
        r = _check_feature_coverage(cfg, sd)
        assert not r.ok and r.severity == "hard"
        assert "vxx_z" in r.message or "hyg_z" in r.message


# ── P-STATE-FILE ───────────────────────────────────────────────────────────

class TestCheckStateFile:
    def test_skip_when_no_broker_name(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_state_file(cfg, sd, None)
        assert r.ok and r.severity == "soft"

    def test_pass_when_state_file_absent(self, healthy_setup):
        """First run: state file doesn't exist yet — soft pass."""
        cfg, sd = healthy_setup
        r = _check_state_file(cfg, sd, "paper")
        assert r.ok and r.severity == "soft"

    def test_pass_when_state_file_valid(self, healthy_setup):
        cfg, sd = healthy_setup
        (sd / "live_state.paper.json").write_text(json.dumps({"x": 1}))
        r = _check_state_file(cfg, sd, "paper")
        assert r.ok and r.severity == "hard"

    def test_fail_on_corrupt_state_file(self, healthy_setup):
        cfg, sd = healthy_setup
        (sd / "live_state.paper.json").write_text("{not json")
        r = _check_state_file(cfg, sd, "paper")
        assert not r.ok and r.severity == "hard"


# ── P-BROKER-CONNECT ───────────────────────────────────────────────────────

class TestCheckBrokerConnect:
    def test_skip_when_no_broker(self):
        r = _check_broker_connect(None)
        assert r.ok and r.severity == "soft"

    def test_pass_when_broker_connects(self):
        broker = MagicMock()
        broker.connect = MagicMock(return_value=None)
        broker.get_account_value = MagicMock(return_value=10000.0)
        r = _check_broker_connect(broker)
        assert r.ok and r.severity == "hard"

    def test_fail_when_broker_raises(self):
        broker = MagicMock()
        broker.connect = MagicMock(side_effect=RuntimeError("API down"))
        r = _check_broker_connect(broker)
        assert not r.ok and r.severity == "hard"


# ── Orchestrator ───────────────────────────────────────────────────────────

class TestRunPreflight:
    def test_strict_mode_raises_on_hard_failure(self, tmp_path):
        """Production undertrained model (best_iter=4) must abort cron."""
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {
                "lookahead_days": 10,
                "xgb_params": {"objective": "rank:pairwise"},
                "asset_embeddings": {"enabled": False},
                "min_best_iter": 20,
                "artifact_path": "artifacts/panel-ltr.json",
            },
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
            "config_fingerprint": fingerprint_config(cfg),
            "config_fingerprint_fields": _model_relevant_fields(cfg),
            "feature_cols": [],
        }))
        with pytest.raises(PreflightFailed, match="best_iter=4"):
            run_preflight(cfg, broker=None, strategy_dir=tmp_path)

    def test_strict_false_returns_results_without_raise(self, tmp_path):
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {"min_best_iter": 20,
                           "artifact_path": "artifacts/panel-ltr.json"},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
        }))
        results = run_preflight(cfg, broker=None, strategy_dir=tmp_path, strict=False)
        assert any(not r.ok and r.severity == "hard" for r in results)

    def test_pass_on_healthy_setup(self, healthy_setup):
        cfg, sd = healthy_setup
        results = run_preflight(cfg, broker=None, strategy_dir=sd)
        # All hard checks pass; some soft (NGBoost disabled, state file absent)
        hards = [r for r in results if r.severity == "hard"]
        assert all(r.ok for r in hards), \
            f"hard failures: {[r.message for r in hards if not r.ok]}"


# ── PreflightFailed exception ──────────────────────────────────────────────

class TestPreflightFailed:
    def test_message_lists_each_failure(self):
        failures = [
            PreflightCheck("P-X", "hard", False, "reason X"),
            PreflightCheck("P-Y", "hard", False, "reason Y"),
        ]
        e = PreflightFailed(failures)
        assert "P-X" in str(e) and "reason X" in str(e)
        assert "P-Y" in str(e) and "reason Y" in str(e)
        assert "No orders placed" in str(e)
