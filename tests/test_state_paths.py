"""Tests for kernel/state_paths.py — broker-isolated state file paths.

Covers TEST-1 + VAL-1 from doc/archives/audits/2026-04-28-deep-audit.md.

The 2026-04-27 incident (paper smoke contaminating live_state.json) is
why this module exists; these tests verify the isolation invariants and
input validation hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.state_paths import (   # noqa: E402
    ALLOWED_BROKERS,
    _safe_broker,
    live_state_legacy_path,
    live_state_path,
    resolve_live_state_read,
    runs_db_legacy_path,
    runs_db_path,
)


# ── _safe_broker ────────────────────────────────────────────────────────────

class TestSafeBroker:
    def test_none_returns_unknown(self):
        assert _safe_broker(None) == "unknown"

    def test_empty_string_returns_unknown(self):
        assert _safe_broker("") == "unknown"

    def test_alpaca_passes_through(self):
        assert _safe_broker("alpaca") == "alpaca"

    def test_paper_passes_through(self):
        assert _safe_broker("paper") == "paper"

    def test_dash_replaced_with_underscore(self):
        # alpaca-paper is in ALLOWED_BROKERS; output is normalized to underscore
        assert _safe_broker("alpaca-paper") == "alpaca_paper"

    def test_underscore_form_also_allowed(self):
        assert _safe_broker("alpaca_paper") == "alpaca_paper"

    def test_ibkr_allowed(self):
        assert _safe_broker("ibkr") == "ibkr"

    def test_unknown_broker_rejected(self):
        # VAL-1: defense against typo / malicious caller
        with pytest.raises(ValueError, match="Unknown broker_name"):
            _safe_broker("unknownbroker")

    def test_path_traversal_rejected(self):
        # VAL-1: defense against path-traversal via crafted broker_name
        with pytest.raises(ValueError, match="Unknown broker_name"):
            _safe_broker("../../etc/passwd")

    def test_allowed_brokers_set_complete(self):
        # Sanity: all known brokers mentioned in live/runner.py choices are
        # accepted. If new brokers are added there, ALLOWED_BROKERS must
        # be updated in lockstep.
        for b in ["paper", "alpaca", "alpaca-paper", "ibkr"]:
            assert b in ALLOWED_BROKERS or b.replace("-", "_") in ALLOWED_BROKERS


# ── live_state_path ─────────────────────────────────────────────────────────

class TestLiveStatePath:
    def test_includes_broker_tag(self, tmp_path):
        p = live_state_path(tmp_path, "alpaca")
        assert p.name == "live_state.alpaca.json"
        assert p.parent == tmp_path

    def test_dash_normalised_to_underscore(self, tmp_path):
        p = live_state_path(tmp_path, "alpaca-paper")
        assert p.name == "live_state.alpaca_paper.json"

    def test_none_falls_back_to_unknown(self, tmp_path):
        p = live_state_path(tmp_path, None)
        assert p.name == "live_state.unknown.json"


# ── live_state_legacy_path ──────────────────────────────────────────────────

class TestLiveStateLegacyPath:
    def test_returns_bare_legacy(self, tmp_path):
        p = live_state_legacy_path(tmp_path)
        assert p.name == "live_state.json"
        assert p.parent == tmp_path


# ── resolve_live_state_read ─────────────────────────────────────────────────

class TestResolveLiveStateRead:
    def test_uses_primary_when_exists(self, tmp_path):
        primary = tmp_path / "live_state.alpaca.json"
        primary.write_text("{}")
        path, used_legacy = resolve_live_state_read(tmp_path, "alpaca")
        assert path == primary
        assert used_legacy is False

    def test_falls_back_to_legacy(self, tmp_path):
        legacy = tmp_path / "live_state.json"
        legacy.write_text("{}")
        path, used_legacy = resolve_live_state_read(tmp_path, "alpaca")
        assert path == legacy
        assert used_legacy is True

    def test_returns_primary_when_neither_exists(self, tmp_path):
        # No file written. Returns the primary path (caller handles missing).
        path, used_legacy = resolve_live_state_read(tmp_path, "alpaca")
        assert path == tmp_path / "live_state.alpaca.json"
        assert used_legacy is False

    def test_primary_takes_priority_over_legacy(self, tmp_path):
        # Both exist (during migration window) → primary wins
        primary = tmp_path / "live_state.alpaca.json"
        primary.write_text('{"new":1}')
        legacy = tmp_path / "live_state.json"
        legacy.write_text('{"old":1}')
        path, used_legacy = resolve_live_state_read(tmp_path, "alpaca")
        assert path == primary
        assert used_legacy is False


# ── runs_db_path ────────────────────────────────────────────────────────────

class TestRunsDbPath:
    def test_inserts_broker_tag_before_db(self):
        p = runs_db_path("data/runs.db", "alpaca")
        assert p == Path("data/runs.alpaca.db")

    def test_dash_normalised(self):
        p = runs_db_path("data/runs.db", "alpaca-paper")
        assert p == Path("data/runs.alpaca_paper.db")

    def test_none_falls_back_to_unknown(self):
        p = runs_db_path("data/runs.db", None)
        assert p == Path("data/runs.unknown.db")

    def test_idempotent_when_already_tagged(self):
        # TEST-1 idempotence (regression for the double-tagging bug):
        # passing an already-broker-tagged path back through with the
        # SAME broker should NOT double-tag it.
        already = "data/runs.alpaca.db"
        p = runs_db_path(already, "alpaca")
        assert p == Path("data/runs.alpaca.db")
        # Different broker still applies (reasonable: different broker
        # → different file); current behaviour double-tags. Documenting:
        p2 = runs_db_path(already, "paper")
        # Acceptable behaviour — it's caller error to pass another broker's
        # tagged path; result is "data/runs.alpaca.paper.db". Caller should
        # pass the bare base.
        assert p2 == Path("data/runs.alpaca.paper.db")

    def test_unknown_broker_rejected(self):
        # VAL-1: rejection at this layer too
        with pytest.raises(ValueError):
            runs_db_path("data/runs.db", "../../etc/passwd")


# ── runs_db_legacy_path ─────────────────────────────────────────────────────

class TestRunsDbLegacyPath:
    def test_returns_bare_path(self):
        p = runs_db_legacy_path("data/runs.db")
        assert p == Path("data/runs.db")
