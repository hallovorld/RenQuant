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

    # ── 2026-05-04 invariant: training resolution must be in fingerprint
    # so daily↔hourly artifact-vs-config drift fails LOUD instead of
    # silently fail-safing every bar via DriftGuardTask. See incident:
    # panel-ltr.json (trained 2026-05-03 with hourly+minute features) +
    # 2026-05-04 daily-only mandate → strategy made 0 trades for 252
    # days because DriftGuardTask saw 13/27 features missing.

    def test_extracts_training_resolution(self):
        out = _model_relevant_fields(_cfg())
        assert out["training_resolution"] == "daily"

    def test_extracts_hourly_enabled_default_false(self):
        out = _model_relevant_fields(_cfg())
        assert out["hourly_enabled"] is False

    def test_extracts_minute_enabled_default_false(self):
        out = _model_relevant_fields(_cfg())
        assert out["minute_enabled"] is False

    def test_extracts_hourly_enabled_when_set_true(self):
        cfg = _cfg()
        cfg["panel_ltr"]["hourly"] = {"enabled": True}
        out = _model_relevant_fields(cfg)
        assert out["hourly_enabled"] is True

    def test_extracts_minute_enabled_when_set_true(self):
        cfg = _cfg()
        cfg["panel_ltr"]["minute"] = {"enabled": True}
        out = _model_relevant_fields(cfg)
        assert out["minute_enabled"] is True

    def test_extracts_watchlist_sector_map(self):
        cfg = _cfg(watchlist=["MSFT", "AAPL"])
        cfg["sector_map"] = {"AAPL": "giant_tech", "MSFT": "software", "TSLA": "auto"}
        cfg["sector_etf_map"] = {"giant_tech": "XLK", "software": "XLK", "auto": "XLY"}
        out = _model_relevant_fields(cfg)
        assert out["sector_map"] == {
            "AAPL": "giant_tech",
            "MSFT": "software",
        }
        assert out["sector_etf_map"] == {
            "giant_tech": "XLK",
            "software": "XLK",
        }

    def test_missing_watchlist_sector_is_visible_in_projection(self):
        cfg = _cfg(watchlist=["AAPL", "BAC"])
        cfg["sector_map"] = {"AAPL": "giant_tech"}
        cfg["sector_etf_map"] = {"giant_tech": "XLK"}
        out = _model_relevant_fields(cfg)
        assert out["sector_map"]["BAC"] is None


class TestFingerprintResolutionInvariant:
    """Daily-mode vs hourly-mode produce DISTINCT fingerprints — the
    smoke test that prevents the 2026-05-03/04 stale-artifact incident
    from silently recurring on the next Sunday retrain."""

    def test_hourly_flag_changes_fingerprint(self):
        cfg_daily = _cfg()
        cfg_hourly = _cfg()
        cfg_hourly["panel_ltr"]["hourly"] = {"enabled": True}
        assert fingerprint_config(cfg_daily) != fingerprint_config(cfg_hourly)

    def test_minute_flag_changes_fingerprint(self):
        cfg_daily = _cfg()
        cfg_minute = _cfg()
        cfg_minute["panel_ltr"]["minute"] = {"enabled": True}
        assert fingerprint_config(cfg_daily) != fingerprint_config(cfg_minute)

    def test_training_resolution_change_changes_fingerprint(self):
        cfg_daily = _cfg()
        cfg_hourly_res = _cfg()
        cfg_hourly_res["panel_ltr"]["training_resolution"] = "hourly"
        assert fingerprint_config(cfg_daily) != fingerprint_config(cfg_hourly_res)

    def test_sector_map_change_changes_fingerprint(self):
        cfg_a = _cfg()
        cfg_b = _cfg()
        cfg_a["sector_map"] = {"AAPL": "giant_tech", "MSFT": "giant_tech"}
        cfg_b["sector_map"] = {"AAPL": "giant_tech", "MSFT": "software"}
        cfg_a["sector_etf_map"] = {"giant_tech": "XLK", "software": "XLK"}
        cfg_b["sector_etf_map"] = {"giant_tech": "XLK", "software": "XLK"}
        assert fingerprint_config(cfg_a) != fingerprint_config(cfg_b)

    def test_sector_etf_map_change_changes_fingerprint(self):
        cfg_a = _cfg()
        cfg_b = _cfg()
        sector_map = {"AAPL": "giant_tech", "MSFT": "software"}
        cfg_a["sector_map"] = sector_map
        cfg_b["sector_map"] = sector_map
        cfg_a["sector_etf_map"] = {"giant_tech": "XLK", "software": "XLK"}
        cfg_b["sector_etf_map"] = {"giant_tech": "XLK", "software": "IGV"}
        assert fingerprint_config(cfg_a) != fingerprint_config(cfg_b)


class TestNGBoostSaveTaskStampsFingerprint:
    """2026-05-04 source-level pin: NGBoostSaveTask must stamp the same
    config_fingerprint + config_fingerprint_fields into ngboost-head.json
    that SaveArtifactTask stamps into panel-ltr.json. Pre-fix only
    panel-ltr was stamped, leaving the NGBoost head's artifact lacking
    a way to detect resolution drift at adapter init.

    This is a source-level pin (not a behavioral test) because spinning
    up a full training context is expensive. The two operative lines
    must be present and use the same `_model_relevant_fields` helper."""

    def test_ngboost_save_stamps_fingerprint(self):
        path = (REPO_ROOT / "backtesting" / "renquant_104"
                / "training_panel" / "pp_panel_training.py")
        src = path.read_text()
        idx_class = src.find("class NGBoostSaveTask")
        # next class boundary
        idx_next = src.find("\nclass ", idx_class + 1)
        body = src[idx_class:idx_next] if idx_next > 0 else src[idx_class:]
        # Both fields stamped from the SAME helper as panel-LTR
        assert 'meta["config_fingerprint"]' in body, (
            "NGBoostSaveTask must stamp config_fingerprint (2026-05-04)"
        )
        assert 'meta["config_fingerprint_fields"]' in body
        assert "fingerprint_config(ctx.config)" in body
        assert "_model_relevant_fields(ctx.config)" in body
        # And the fail-soft try/except so a config_consistency import
        # failure doesn't kill training.
        assert "try:" in body
        assert "except Exception" in body


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
