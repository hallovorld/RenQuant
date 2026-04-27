"""Phase 4 tests: challenger / shadow infrastructure.

User spec 2026-04-26: "phase 1,2,3,4全做". Phase 4 ships PLATFORM:
config block, ChallengerEvaluator API, runs.db schema for
challenger_decisions, log_decision() / compare_window() helpers.

Live-runner wiring is intentionally NOT done today (Phase 4b) —
operator decides when to flip the bit. These tests pin the contract
so Phase 4b can integrate without re-discovering API shape.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.challenger import (   # noqa: E402
    ChallengerConfig,
    ChallengerEvaluator,
    log_decision,
    compare_window,
)
from kernel.persistence import ensure_schema   # noqa: E402
import sqlite3 as _sqlite


# ── ChallengerConfig ──────────────────────────────────────────────────────────

class TestChallengerConfig:
    def test_disabled_by_default(self):
        cc = ChallengerConfig.from_strategy_config({})
        assert cc.enabled is False
        assert cc.artifact_path is None
        assert cc.shadow_period_days == 0

    def test_loads_from_acceptance_block(self):
        config = {
            "acceptance": {
                "challenger": {
                    "enabled":            True,
                    "artifact_path":      "artifacts/panel-ltr.macro-enabled.bak.json",
                    "name":               "macro-enabled",
                    "shadow_period_days": 14,
                },
            },
        }
        cc = ChallengerConfig.from_strategy_config(config)
        assert cc.enabled is True
        assert cc.name == "macro-enabled"
        assert cc.shadow_period_days == 14

    def test_strategy_config_block_present(self):
        cfg_path = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
        cfg = json.loads(cfg_path.read_text())
        ch = cfg["acceptance"]["challenger"]
        assert ch["enabled"] is False    # default OFF
        assert "shadow_period_days" in ch
        assert ch["shadow_period_days"] == 0


# ── ChallengerEvaluator ───────────────────────────────────────────────────────

class TestChallengerEvaluator:
    def test_disabled_returns_none(self, tmp_path):
        ev = ChallengerEvaluator.maybe_load({}, tmp_path)
        assert ev is None

    def test_enabled_but_artifact_missing_returns_none(self, tmp_path):
        config = {
            "acceptance": {
                "challenger": {
                    "enabled":       True,
                    "artifact_path": "artifacts/does_not_exist.json",
                    "name":          "ghost",
                },
            },
        }
        ev = ChallengerEvaluator.maybe_load(config, tmp_path)
        assert ev is None

    def test_score_with_no_scorer_returns_empty_series(self):
        cc = ChallengerConfig(enabled=False, artifact_path=None,
                              name=None, shadow_period_days=0)
        ev = ChallengerEvaluator(cc, scorer=None)
        s = ev.score(pd.DataFrame({"a": [1.0, 2.0]}))
        assert isinstance(s, pd.Series)
        assert s.empty

    def test_score_with_empty_input_returns_empty_series(self):
        cc = ChallengerConfig(enabled=False, artifact_path=None,
                              name=None, shadow_period_days=0)
        # Even with a fake scorer, empty X → empty Series (no crash)
        class _S:
            def score(self, X): return pd.Series(dtype=float)
        ev = ChallengerEvaluator(cc, scorer=_S())
        s = ev.score(pd.DataFrame())
        assert s.empty


# ── DB persistence ────────────────────────────────────────────────────────────

class TestChallengerDB:
    def _open_db(self, tmp_path):
        db_path = tmp_path / "runs.db"
        conn = _sqlite.connect(str(db_path), isolation_level=None)
        ensure_schema(conn)
        return conn

    def test_schema_present_after_init_db(self, tmp_path):
        conn = self._open_db(tmp_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='challenger_decisions'"
        )
        assert cur.fetchone() is not None
        # Indexes
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_challenger_%'"
        )
        idx = {r[0] for r in cur.fetchall()}
        assert "idx_challenger_run" in idx
        assert "idx_challenger_window" in idx

    def test_log_decision_inserts_row(self, tmp_path):
        conn = self._open_db(tmp_path)
        log_decision(conn,
                     run_id="run_abc",
                     decision_date=pd.Timestamp("2026-04-26"),
                     ticker="AAPL",
                     challenger_name="macro-enabled",
                     challenger_score=0.42,
                     challenger_rank_score=0.71,
                     challenger_action="BUY",
                     actual_score=0.38,
                     actual_action="BUY")
        conn.commit()
        cur = conn.execute("SELECT COUNT(*) FROM challenger_decisions")
        assert cur.fetchone()[0] == 1

    def test_log_decision_handles_nones(self, tmp_path):
        """Some decisions don't have all values (e.g. challenger no-decision)."""
        conn = self._open_db(tmp_path)
        log_decision(conn,
                     run_id="r",
                     decision_date=pd.Timestamp("2026-04-26"),
                     ticker="MSFT",
                     challenger_name="macro-enabled",
                     challenger_score=None,
                     challenger_rank_score=None,
                     challenger_action="HOLD",
                     actual_score=0.50,
                     actual_action="BUY")
        conn.commit()
        cur = conn.execute("SELECT challenger_score, actual_score FROM challenger_decisions")
        row = cur.fetchone()
        assert row[0] is None
        assert row[1] == 0.50

    def test_compare_window_empty_returns_zeros(self, tmp_path):
        conn = self._open_db(tmp_path)
        v = compare_window(conn,
                           challenger_name="x",
                           start_date=pd.Timestamp("2026-04-01"),
                           end_date=pd.Timestamp("2026-04-30"))
        assert v["n_decisions"] == 0
        assert v["agreement_rate"] == 0.0

    def test_compare_window_aggregates_correctly(self, tmp_path):
        conn = self._open_db(tmp_path)
        # Insert: 3 agreements, 2 disagreements over a window
        rows = [
            # decision_date,         ticker, ch_action, actual_action, ch_score, actual_score
            (pd.Timestamp("2026-04-20"), "A", "BUY",  "BUY",  0.7, 0.6),
            (pd.Timestamp("2026-04-21"), "B", "HOLD", "HOLD", 0.1, 0.2),
            (pd.Timestamp("2026-04-22"), "C", "BUY",  "BUY",  0.8, 0.7),
            (pd.Timestamp("2026-04-23"), "D", "BUY",  "HOLD", 0.6, 0.3),  # ch-only buy
            (pd.Timestamp("2026-04-24"), "E", "HOLD", "BUY",  0.2, 0.5),  # live-only buy
        ]
        for d, t, ca, aa, cs, asc in rows:
            log_decision(conn, run_id="r", decision_date=d, ticker=t,
                         challenger_name="x",
                         challenger_score=cs, challenger_rank_score=cs,
                         challenger_action=ca,
                         actual_score=asc, actual_action=aa)
        conn.commit()

        v = compare_window(conn, challenger_name="x",
                           start_date=pd.Timestamp("2026-04-20"),
                           end_date=pd.Timestamp("2026-04-30"))
        assert v["n_decisions"] == 5
        assert v["agreement_rate"] == 0.6   # 3/5
        assert v["challenger_only_buy"] == 1
        assert v["live_only_buy"] == 1
        # Score corr should be positive (both rise/fall together-ish)
        assert v["score_corr"] is not None

    def test_compare_window_filters_by_name(self, tmp_path):
        conn = self._open_db(tmp_path)
        # Two challengers — compare_window should only see the named one
        log_decision(conn, run_id="r", decision_date=pd.Timestamp("2026-04-20"),
                     ticker="A", challenger_name="alpha",
                     challenger_score=0.5, challenger_rank_score=0.5,
                     challenger_action="BUY",
                     actual_score=0.5, actual_action="BUY")
        log_decision(conn, run_id="r", decision_date=pd.Timestamp("2026-04-20"),
                     ticker="A", challenger_name="beta",
                     challenger_score=0.5, challenger_rank_score=0.5,
                     challenger_action="HOLD",
                     actual_score=0.5, actual_action="BUY")
        conn.commit()
        v_alpha = compare_window(conn, challenger_name="alpha",
                                 start_date=pd.Timestamp("2026-04-01"),
                                 end_date=pd.Timestamp("2026-04-30"))
        v_beta  = compare_window(conn, challenger_name="beta",
                                 start_date=pd.Timestamp("2026-04-01"),
                                 end_date=pd.Timestamp("2026-04-30"))
        assert v_alpha["n_decisions"] == 1 and v_alpha["agreement_rate"] == 1.0
        assert v_beta["n_decisions"]  == 1 and v_beta["agreement_rate"]  == 0.0
