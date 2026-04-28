"""Tests for kernel/config_consistency.py.

Guards against the 3-in-24h class of bugs where strategy_config.json
drifts out of sync with the trained panel-ltr.json artifact. See
doc/archives/audits/2026-04-28-deep-audit.md and
doc/archives/audits/2026-04-28-nvts-buy-postmortem.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.config_consistency import (   # noqa: E402
    ConfigModelMismatch,
    assert_consistent,
    fingerprint_config,
    _model_relevant_fields,
)


def _cfg(watchlist=None, lookahead=10, objective="rank:pairwise", emb=False):
    return {
        "watchlist": list(watchlist or ["AAPL", "MSFT"]),
        "panel_ltr": {
            "lookahead_days": lookahead,
            "xgb_params":     {"objective": objective},
            "asset_embeddings": {"enabled": emb},
        },
    }


# ── _model_relevant_fields ──────────────────────────────────────────────────

class TestRelevantFields:
    def test_watchlist_sorted_in_projection(self):
        out = _model_relevant_fields(_cfg(watchlist=["MSFT", "AAPL"]))
        # Ticker order in config doesn't matter for hash
        assert out["watchlist"] == ["AAPL", "MSFT"]

    def test_extracts_lookahead(self):
        out = _model_relevant_fields(_cfg(lookahead=20))
        assert out["lookahead_days"] == 20

    def test_extracts_objective(self):
        out = _model_relevant_fields(_cfg(objective="rank:ndcg"))
        assert out["objective"] == "rank:ndcg"

    def test_extracts_embedding_flag(self):
        out = _model_relevant_fields(_cfg(emb=True))
        assert out["asset_embeddings"] is True


# ── fingerprint_config ──────────────────────────────────────────────────────

class TestFingerprint:
    def test_fingerprint_is_deterministic(self):
        c = _cfg()
        assert fingerprint_config(c) == fingerprint_config(c)

    def test_fingerprint_starts_with_sha256_prefix(self):
        fp = fingerprint_config(_cfg())
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 16

    def test_watchlist_change_changes_fingerprint(self):
        a = fingerprint_config(_cfg(watchlist=["AAPL", "MSFT"]))
        b = fingerprint_config(_cfg(watchlist=["AAPL", "MSFT", "NVDA"]))
        assert a != b

    def test_watchlist_order_does_NOT_change_fingerprint(self):
        # Sorted internally → "MSFT, AAPL" hashes same as "AAPL, MSFT"
        a = fingerprint_config(_cfg(watchlist=["AAPL", "MSFT", "NVDA"]))
        b = fingerprint_config(_cfg(watchlist=["NVDA", "MSFT", "AAPL"]))
        assert a == b

    def test_lookahead_change_changes_fingerprint(self):
        # 10d vs 60d (the M1 ensemble case)
        a = fingerprint_config(_cfg(lookahead=10))
        b = fingerprint_config(_cfg(lookahead=60))
        assert a != b

    def test_objective_change_changes_fingerprint(self):
        # rank:ndcg vs rank:pairwise — the 4-27 incident
        a = fingerprint_config(_cfg(objective="rank:pairwise"))
        b = fingerprint_config(_cfg(objective="rank:ndcg"))
        assert a != b

    def test_embedding_flag_change_changes_fingerprint(self):
        # T2-2 embedding on/off
        a = fingerprint_config(_cfg(emb=False))
        b = fingerprint_config(_cfg(emb=True))
        assert a != b

    def test_irrelevant_field_does_NOT_change_fingerprint(self):
        # Adding unrelated fields shouldn't change model fingerprint
        a = _cfg()
        b = _cfg()
        b["unrelated_setting"] = {"foo": "bar"}
        b["panel_ltr"]["unrelated"] = 42
        assert fingerprint_config(a) == fingerprint_config(b)


# ── assert_consistent ──────────────────────────────────────────────────────

class TestAssertConsistent:
    def test_match_passes_silently(self, caplog):
        cfg = _cfg()
        artifact = {
            "config_fingerprint": fingerprint_config(cfg),
            "config_fingerprint_fields": _model_relevant_fields(cfg),
        }
        # No raise
        assert_consistent(cfg, artifact, strict=True)

    def test_mismatch_raises_in_strict(self):
        cfg = _cfg(watchlist=["AAPL"])
        # Simulate artifact trained on different watchlist
        other = _cfg(watchlist=["AAPL", "MSFT", "NVDA"])
        artifact = {
            "config_fingerprint": fingerprint_config(other),
            "config_fingerprint_fields": _model_relevant_fields(other),
        }
        with pytest.raises(ConfigModelMismatch, match="watchlist"):
            assert_consistent(cfg, artifact, strict=True)

    def test_mismatch_logs_in_non_strict(self, caplog):
        import logging
        cfg = _cfg(watchlist=["AAPL"])
        other = _cfg(watchlist=["AAPL", "MSFT"])
        artifact = {
            "config_fingerprint": fingerprint_config(other),
            "config_fingerprint_fields": _model_relevant_fields(other),
        }
        with caplog.at_level(logging.ERROR):
            assert_consistent(cfg, artifact, strict=False)
        assert any("MISMATCH" in r.message for r in caplog.records)

    def test_unstamped_artifact_passes_with_warning(self, caplog):
        import logging
        cfg = _cfg()
        artifact = {}  # no fingerprint stored
        with caplog.at_level(logging.WARNING):
            assert_consistent(cfg, artifact, strict=True)  # should NOT raise
        assert any("no fingerprint" in r.message for r in caplog.records)

    def test_mismatch_reports_specific_field(self):
        # Simulates "config flipped to rank:ndcg but model trained pairwise"
        cfg = _cfg(objective="rank:ndcg")
        other = _cfg(objective="rank:pairwise")
        artifact = {
            "config_fingerprint": fingerprint_config(other),
            "config_fingerprint_fields": _model_relevant_fields(other),
        }
        with pytest.raises(ConfigModelMismatch, match="objective"):
            assert_consistent(cfg, artifact, strict=True)

    def test_mismatch_includes_remediation(self):
        cfg = _cfg(lookahead=10)
        other = _cfg(lookahead=60)
        artifact = {
            "config_fingerprint": fingerprint_config(other),
            "config_fingerprint_fields": _model_relevant_fields(other),
        }
        with pytest.raises(ConfigModelMismatch) as exc_info:
            assert_consistent(cfg, artifact, strict=True)
        msg = str(exc_info.value)
        assert "Retrain" in msg or "retrain" in msg
        assert "checkpoint" in msg.lower()


# ── Real-world incident regressions ─────────────────────────────────────────

class TestIncidentRegressions:
    def test_24h_incident_1_macro_feature_drift(self):
        """2026-04-27a: config disabled macro but ngboost-head was trained
        with 184 macro features. Caught here as embedding mismatch via
        watchlist proxy."""
        # config: short watchlist (post-macro-disable)
        cfg = _cfg(watchlist=["AAPL", "MSFT"], emb=False)
        # artifact trained when emb was True
        old = _cfg(watchlist=["AAPL", "MSFT"], emb=True)
        artifact = {
            "config_fingerprint": fingerprint_config(old),
            "config_fingerprint_fields": _model_relevant_fields(old),
        }
        with pytest.raises(ConfigModelMismatch, match="asset_embeddings"):
            assert_consistent(cfg, artifact, strict=True)

    def test_24h_incident_2_ndcg_config_flip(self):
        """2026-04-27b: config said rank:ndcg but model trained pairwise."""
        cfg = _cfg(objective="rank:ndcg")
        old = _cfg(objective="rank:pairwise")
        artifact = {
            "config_fingerprint": fingerprint_config(old),
            "config_fingerprint_fields": _model_relevant_fields(old),
        }
        with pytest.raises(ConfigModelMismatch, match="objective"):
            assert_consistent(cfg, artifact, strict=True)

    def test_24h_incident_3_watchlist_227_mismatch(self):
        """2026-04-28a: config locked watchlist=227 but auto-revert
        only restored 103-trained model."""
        big = ["TICKER_%03d" % i for i in range(227)]
        small = big[:103]
        cfg = _cfg(watchlist=big)
        old = _cfg(watchlist=small)
        artifact = {
            "config_fingerprint": fingerprint_config(old),
            "config_fingerprint_fields": _model_relevant_fields(old),
        }
        with pytest.raises(ConfigModelMismatch, match="watchlist"):
            assert_consistent(cfg, artifact, strict=True)
