"""Tests for the experiment_configs DB table + inflate/diff helpers.

Per CLAUDE.md follow-up (Task #38): file-system side configs are noisy
(60+ stale entries from past experiments). This commit ships a SQLite
table to keep them, with helpers:

  diff_overrides(base, side)         → flat dict of dotted-key changes
  apply_overrides(base, overrides)   → reconstruct full config
  inflate_experiment_config(label)   → round-trip from DB

Per CLAUDE.md §2: every feature ships with a test. These pin the
diff/apply round-trip + DB roundtrip semantics so future refactors
can't silently corrupt experimental configs.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestDiffOverrides:
    """Pin diff_overrides semantics: only different leaves end up in output."""

    def test_unchanged_keys_omitted(self):
        from migrate_experiment_configs_to_db import diff_overrides
        base = {"a": 1, "b": 2}
        side = {"a": 1, "b": 2}
        assert diff_overrides(base, side) == {}

    def test_changed_leaf_emitted_as_dotted(self):
        from migrate_experiment_configs_to_db import diff_overrides
        base = {"x": {"y": {"z": 1}}}
        side = {"x": {"y": {"z": 99}}}
        assert diff_overrides(base, side) == {"x.y.z": 99}

    def test_new_key_in_side_emitted(self):
        from migrate_experiment_configs_to_db import diff_overrides
        base = {"a": 1}
        side = {"a": 1, "b": 2}
        assert diff_overrides(base, side) == {"b": 2}

    def test_list_compared_by_value_equality(self):
        from migrate_experiment_configs_to_db import diff_overrides
        base = {"watchlist": ["AAPL", "MSFT"]}
        side = {"watchlist": ["AAPL", "MSFT", "NVDA"]}
        # Lists compared by != → emitted whole when different
        out = diff_overrides(base, side)
        assert out == {"watchlist": ["AAPL", "MSFT", "NVDA"]}


class TestApplyOverrides:
    def test_round_trip(self):
        from migrate_experiment_configs_to_db import apply_overrides, diff_overrides
        base = {"a": 1, "nested": {"b": 2, "c": 3}}
        side = {"a": 1, "nested": {"b": 99, "c": 3, "d": "new"}}
        ov = diff_overrides(base, side)
        # diff_overrides only emits the *changed* leaves. apply_overrides
        # rebuilds side by overlaying those onto base.
        rebuilt = apply_overrides(base, ov)
        assert rebuilt == side

    def test_apply_creates_missing_parents(self):
        from migrate_experiment_configs_to_db import apply_overrides
        base = {}
        ov = {"deep.nested.key": "value"}
        out = apply_overrides(base, ov)
        assert out == {"deep": {"nested": {"key": "value"}}}

    def test_apply_does_not_mutate_base(self):
        from migrate_experiment_configs_to_db import apply_overrides
        base = {"a": [1, 2, 3]}
        ov = {"b": "added"}
        out = apply_overrides(base, ov)
        assert "b" not in base   # base unchanged
        assert base["a"] == [1, 2, 3]
        assert out == {"a": [1, 2, 3], "b": "added"}


class TestStoreInflateRoundTrip:
    """End-to-end: store → inflate returns the original config."""

    def _setup(self):
        tmp = Path(tempfile.mkdtemp())
        db = tmp / "runs.db"
        strat = tmp / "strat"
        strat.mkdir()
        base = {
            "model_name": "test",
            "watchlist": ["AAPL", "MSFT"],
            "rotation": {"joint_actions": {"qp_min_invested_pct": 0.5}},
        }
        (strat / "strategy_config.json").write_text(json.dumps(base))
        return tmp, db, strat, base

    def test_round_trip_preserves_overrides(self):
        from migrate_experiment_configs_to_db import (
            store_experiment_config, inflate_experiment_config,
            diff_overrides,
        )
        _tmp, db, strat, base = self._setup()
        side = {
            "model_name": "test",
            "watchlist": ["AAPL", "MSFT"],
            "rotation": {"joint_actions": {"qp_min_invested_pct": 0.7}},
        }
        ov = diff_overrides(base, side)
        store_experiment_config(
            label="exp1", base_config_name="strategy_config.json",
            overrides=ov, db_path=db,
        )
        inflated = inflate_experiment_config(
            "exp1", db_path=db, strategy_dir=strat,
        )
        assert inflated == side, (
            f"DB round-trip lost overrides. Original side:\n{side}\n"
            f"Inflated:\n{inflated}\n"
            f"Stored overrides:\n{ov}"
        )

    def test_unknown_label_raises_keyerror(self):
        from migrate_experiment_configs_to_db import (
            inflate_experiment_config, store_experiment_config,
        )
        _tmp, db, strat, _base = self._setup()
        # Ensure table exists by storing & deleting
        store_experiment_config(
            label="seed", base_config_name="strategy_config.json",
            overrides={}, db_path=db,
        )
        with pytest.raises(KeyError, match="missing_label"):
            inflate_experiment_config(
                "missing_label", db_path=db, strategy_dir=strat,
            )

    def test_audit_label_indexed(self):
        from migrate_experiment_configs_to_db import store_experiment_config
        _tmp, db, _strat, _base = self._setup()
        store_experiment_config(
            label="exp_alpha158", base_config_name="strategy_config.json",
            overrides={"x": 1}, audit_label="ALPHA158_LINEAR", db_path=db,
        )
        conn = sqlite3.connect(str(db))
        try:
            indexes = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()]
            assert "idx_experiment_configs_audit_label" in indexes
            row = conn.execute(
                "SELECT audit_label FROM experiment_configs WHERE label=?",
                ("exp_alpha158",),
            ).fetchone()
            assert row[0] == "ALPHA158_LINEAR"
        finally:
            conn.close()

    def test_upsert_updates_overrides(self):
        from migrate_experiment_configs_to_db import (
            store_experiment_config, inflate_experiment_config,
        )
        _tmp, db, strat, _base = self._setup()
        store_experiment_config(
            label="up1", base_config_name="strategy_config.json",
            overrides={"rotation.joint_actions.qp_min_invested_pct": 0.5},
            db_path=db,
        )
        store_experiment_config(
            label="up1", base_config_name="strategy_config.json",
            overrides={"rotation.joint_actions.qp_min_invested_pct": 0.9},
            db_path=db,
        )
        inflated = inflate_experiment_config("up1", db_path=db, strategy_dir=strat)
        assert inflated["rotation"]["joint_actions"]["qp_min_invested_pct"] == 0.9
