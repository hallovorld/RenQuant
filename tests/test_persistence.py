"""Tests for kernel/persistence.py — SQLite decision-trace."""
from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.persistence import (  # noqa: E402
    ensure_schema,
    get_connection,
    record_pipeline_run,
    record_candidate_scores,
    record_trades,
    record_rotations,
    record_training_run,
    record_ticker_daily_state,
    decision_trace_integrity_report,
    validate_decision_trace_integrity,
)


def _cfg(tmp_path: Path, enabled: bool = True) -> dict:
    # Both db_path (live role) and sim_db_path (sim role) set to distinct
    # tmp files — SimAdapter uses role="sim" and writes to sim_db_path,
    # RunnerAdapter uses role="live" and writes to db_path.
    return {
        "persistence": {
            "enabled":     enabled,
            "db_path":     str(tmp_path / "runs.db"),
            "sim_db_path": str(tmp_path / "sim_runs.db"),
        },
        "model_name": "renquant-104-test",
    }


class TestConnectionLifecycle:
    def test_disabled_returns_none(self, tmp_path):
        conn = get_connection(_cfg(tmp_path, enabled=False))
        assert conn is None

    def test_enabled_creates_db_file(self, tmp_path):
        conn = get_connection(_cfg(tmp_path, enabled=True))
        assert conn is not None
        assert (tmp_path / "runs.db").exists()
        conn.close()

    def test_schema_tables_exist(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"pipeline_runs", "candidate_scores", "trades", "rotations",
                "training_runs"}.issubset(tables)
        conn.close()

    def test_legacy_trades_table_migrates_before_trade_date_index(self, tmp_path):
        """Regression: live DBs created before trade_date existed failed
        ensure_schema at CREATE INDEX idx_trades_date before ALTER TABLE ran."""
        db = tmp_path / "runs.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE trades (
                run_id TEXT,
                ticker TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                invest REAL,
                target_pct REAL,
                exit_reason TEXT,
                pnl_pct REAL,
                hold_days INTEGER,
                tax REAL,
                rank_score REAL,
                conviction REAL,
                sigma_mult REAL,
                mu REAL,
                sigma REAL
            );
        """)
        ensure_schema(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(trades)")}
        assert "trade_date" in cols
        assert "tax_cash_debited" in cols
        assert "tax_cash_debit_mode" in cols
        assert "tax_lot_method" in cols
        assert "expected_return_horizon_days" in cols
        assert "mu_horizon_days" in cols
        assert "idx_trades_date" in indexes
        conn.close()

    def test_legacy_ticker_daily_state_rebuild_keeps_horizon_columns(self, tmp_path):
        """Legacy date-keyed TDS rebuild must preserve post-migration columns."""
        db = tmp_path / "runs.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE ticker_daily_state (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                regime TEXT,
                confidence REAL,
                in_watchlist INTEGER,
                in_universe INTEGER,
                pending_at_broker INTEGER,
                has_position INTEGER,
                position_qty REAL,
                position_pct REAL,
                model_type TEXT,
                model_action TEXT,
                sell_streak INTEGER,
                panel_score REAL,
                rank_score REAL,
                expected_return REAL,
                kelly_target_pct REAL,
                mu REAL,
                sigma REAL,
                in_candidates INTEGER,
                selected INTEGER,
                blocked_by TEXT,
                sector TEXT,
                PRIMARY KEY (date, ticker)
            );
            INSERT INTO ticker_daily_state
              (date, ticker, expected_return, mu, selected)
              VALUES ('2026-05-24', 'AAA', 0.01, 0.02, 0);
        """)

        ensure_schema(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(ticker_daily_state)")}
        assert "expected_return_horizon_days" in cols
        assert "mu_horizon_days" in cols
        record_ticker_daily_state(
            conn,
            run_id="run-new",
            run_date=datetime.date(2026, 5, 25),
            rows=[{
                "ticker": "AAA",
                "expected_return": 0.03,
                "expected_return_horizon_days": 60,
                "mu": 0.04,
                "mu_horizon_days": 60,
                "selected": 1,
            }],
        )
        row = conn.execute(
            """SELECT expected_return_horizon_days, mu_horizon_days
               FROM ticker_daily_state WHERE run_id='run-new' AND ticker='AAA'"""
        ).fetchone()
        assert row == (60, 60)
        conn.close()


class TestPipelineRun:
    def test_insert_and_read_back(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn,
            run_type="sim",
            run_date=datetime.date(2026, 4, 22),
            strategy="renquant_104",
            regime="BULL_CALM",
            confidence=0.82,
            portfolio_value=123_456.0,
            cash=10_000.0,
            n_candidates=15,
            n_exits=2,
            n_rotations=1,
            n_buys=3,
            buy_blocked=True,
            skip_buys=False,
            bear_only=False,
            counters={"qp_delta_below_min_dw": 7},
            run_bundle={"artifact_hashes": {"panel": "sha256:test"}},
        )
        assert rid is not None
        row = conn.execute(
            """SELECT buy_blocked, skip_buys, bear_only, counters_json, run_bundle_json
                 FROM pipeline_runs WHERE run_id = ?""",
            (rid,),
        ).fetchone()
        assert row[:3] == (1, 0, 0)
        assert '"qp_delta_below_min_dw": 7' in row[3]
        assert '"panel": "sha256:test"' in row[4]
        conn.close()

    def test_noop_when_disabled(self, tmp_path):
        """All record_* calls must be safe no-ops when the connection is None."""
        result = record_pipeline_run(
            None, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        assert result is None


class TestRotations:
    def test_record_rotations_classifies_accepted_and_blocked_pairs(self, tmp_path):
        from kernel.rotation import RotationPair

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn,
            run_type="sim",
            run_date=datetime.date(2026, 5, 25),
            strategy="test",
        )
        accepted = RotationPair(
            sell_ticker="AAA",
            buy_ticker="BBB",
            sell_score=0.30,
            buy_score=0.60,
            sell_er=0.01,
            buy_er=0.05,
            horizon_days=20,
            raw_advantage=0.04,
            tax_drag=0.01,
            transaction_cost=0.001,
            net_advantage=0.029,
            threshold=0.02,
            margin_realized=0.009,
        )
        blocked = RotationPair(
            sell_ticker="CCC",
            buy_ticker="DDD",
            sell_score=0.40,
            buy_score=0.55,
            sell_er=0.02,
            buy_er=0.03,
            horizon_days=20,
            raw_advantage=0.01,
            tax_drag=0.00,
            transaction_cost=0.001,
            net_advantage=0.009,
            threshold=0.02,
            margin_realized=-0.011,
        )
        ctx = SimpleNamespace(
            rotations=[accepted, blocked],
            rotations_blocked=[{
                "sell": "CCC",
                "buy": "DDD",
                "reason": "insufficient_cash",
            }],
            orders=[{
                "ticker": "BBB",
                "order_type": "ROTATION",
                "decision_inputs": {"sell_ticker": "AAA", "buy_ticker": "BBB"},
            }],
        )

        record_rotations(conn, rid, ctx)

        rows = conn.execute(
            """SELECT cand_ticker, held_ticker, decision, cand_er, held_er,
                      raw_adv, net_adv, tax_drag, threshold
                 FROM rotations ORDER BY cand_ticker""",
        ).fetchall()
        assert rows == [
            ("BBB", "AAA", "accepted", 0.05, 0.01, 0.04, 0.029, 0.01, 0.02),
            ("DDD", "CCC", "blocked:insufficient_cash", 0.03, 0.02, 0.01, 0.009, 0.0, 0.02),
        ]
        conn.close()


class TestCandidateScores:
    def test_records_candidates_and_holdings(self, tmp_path):
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
            strategy="test",
        )
        c1 = CandidateResult(ticker="AAA", raw_score=0.5, rank_score=0.6,
                             rs_score=0.1, detail="", expected_return=0.02,
                             panel_score=0.7, mu=0.01, sigma=0.03)
        c2 = CandidateResult(ticker="BBB", raw_score=0.2, rank_score=0.3,
                             rs_score=0.0, detail="", expected_return=0.0,
                             panel_score=0.2, mu=-0.01, sigma=0.05)
        hs_held = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 3, 1),
            high_watermark=105.0,
            rank_score=0.4, panel_score=0.5, mu=0.0, sigma=0.04,
        )
        holdings = {"ZZZ": hs_held}

        record_candidate_scores(conn, rid, [c1, c2], holdings, selected_tickers={"AAA"})

        rows = conn.execute(
            "SELECT ticker, role, selected FROM candidate_scores WHERE run_id = ?", (rid,),
        ).fetchall()
        by_key = {(r[0], r[1]): r for r in rows}
        assert ("AAA", "candidate") in by_key
        assert ("BBB", "candidate") in by_key
        assert ("ZZZ", "holding")   in by_key
        assert by_key[("AAA", "candidate")][2] == 1   # selected
        assert by_key[("BBB", "candidate")][2] == 0   # not selected
        conn.close()

    def test_excluded_holding_tickers_are_not_candidate_score_rows(self, tmp_path):
        from kernel.exits import HoldingState

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
            strategy="test",
        )
        spy = HoldingState(
            entry_price=500.0,
            entry_date=datetime.date(2026, 3, 1),
            high_watermark=510.0,
            rank_score=None,
        )
        alpha = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 3, 1),
            high_watermark=105.0,
            rank_score=0.6,
        )

        record_candidate_scores(
            conn,
            rid,
            [],
            {"SPY": spy, "AAA": alpha},
            selected_tickers=set(),
            excluded_holding_tickers={"SPY"},
        )

        rows = conn.execute(
            "SELECT ticker, role FROM candidate_scores WHERE run_id = ? ORDER BY ticker",
            (rid,),
        ).fetchall()
        assert rows == [("AAA", "holding")]
        conn.close()

    def test_candidate_trace_pool_includes_short_candidates_with_distinct_role(self, tmp_path):
        from types import SimpleNamespace
        from kernel.decision_trace import candidate_trace_pool
        from kernel.selection import CandidateResult

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
            strategy="test",
        )
        long = CandidateResult(
            ticker="AAA", raw_score=0.5, rank_score=0.6, rs_score=0.1,
            panel_score=0.7, mu=0.01, sigma=0.03,
        )
        short = CandidateResult(
            ticker="AAA", raw_score=-1.5, rank_score=0.2, rs_score=0.0,
            panel_score=-1.5, mu=-0.02, sigma=0.04,
        )
        short.trace_role = "short_candidate"
        ctx = SimpleNamespace(candidates=[long], short_candidates=[short])

        record_candidate_scores(
            conn,
            rid,
            candidate_trace_pool(ctx),
            {},
            selected_tickers=set(),
        )

        rows = conn.execute(
            "SELECT ticker, role, panel_score FROM candidate_scores "
            "WHERE run_id = ? ORDER BY role",
            (rid,),
        ).fetchall()
        assert rows == [
            ("AAA", "candidate", 0.7),
            ("AAA", "short_candidate", -1.5),
        ]
        conn.close()

    def test_blocked_map_recorded(self, tmp_path):
        from kernel.selection import CandidateResult
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        c1 = CandidateResult(ticker="AAA", raw_score=0, rank_score=0.9,
                             rs_score=0, detail="", expected_return=0)
        record_candidate_scores(
            conn, rid, [c1], {}, selected_tickers=set(),
            blocked_map={"AAA": "sector_cap"},
        )
        row = conn.execute(
            "SELECT blocked_by FROM candidate_scores WHERE run_id = ? AND ticker = ?",
            (rid, "AAA"),
        ).fetchone()
        assert row[0] == "sector_cap"
        conn.close()

    def test_sector_map_lookup_is_case_stable(self, tmp_path):
        """Selected rows must keep sector metadata even if ticker case drifts."""
        from kernel.selection import CandidateResult

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        cand = CandidateResult(
            ticker="bac", raw_score=0.5, rank_score=0.7,
            rs_score=0.0, detail="", expected_return=0.02,
        )

        record_candidate_scores(
            conn,
            rid,
            [cand],
            {},
            selected_tickers={"bac"},
            sector_map={"BAC": "finance"},
        )

        row = conn.execute(
            """SELECT sector FROM candidate_scores
               WHERE run_id = ? AND ticker = 'bac'""",
            (rid,),
        ).fetchone()
        assert row[0] == "finance"
        conn.close()

    def test_qp_delta_recorded_for_candidates_and_holdings(self, tmp_path):
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        cand = CandidateResult(ticker="AAA", raw_score=0, rank_score=0.8,
                               rs_score=0, detail="", expected_return=0.01)
        holding = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 3, 1),
            high_watermark=105.0,
        )

        record_candidate_scores(
            conn, rid, [cand], {"BBB": holding},
            selected_tickers={"AAA"},
            qp_delta_by_ticker={"AAA": 0.035, "BBB": -0.020},
            qp_target_by_ticker={"AAA": 0.085, "BBB": 0.050},
            qp_status="optimal",
        )

        rows = {
            (r[0], r[1]): r[2:]
            for r in conn.execute(
                "SELECT ticker, role, qp_delta_w, qp_target_w, qp_status "
                "FROM candidate_scores WHERE run_id = ?",
                (rid,),
            )
        }
        assert rows[("AAA", "candidate")][0] == pytest.approx(0.035)
        assert rows[("AAA", "candidate")][1] == pytest.approx(0.085)
        assert rows[("AAA", "candidate")][2] == "optimal"
        assert rows[("BBB", "holding")][0] == pytest.approx(-0.020)
        assert rows[("BBB", "holding")][1] == pytest.approx(0.050)
        assert rows[("BBB", "holding")][2] == "optimal"
        conn.close()

    def test_score_horizon_fields_recorded_for_candidates_and_holdings(self, tmp_path):
        from kernel.selection import CandidateResult
        from kernel.exits import HoldingState

        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        cand = CandidateResult(
            ticker="AAA",
            raw_score=0,
            rank_score=0.8,
            rs_score=0,
            detail="",
            expected_return=0.04,
            expected_return_horizon_days=60,
            mu=0.04,
            mu_horizon_days=60,
        )
        holding = HoldingState(
            entry_price=100.0,
            entry_date=datetime.date(2026, 3, 1),
            high_watermark=105.0,
            expected_return=0.03,
            expected_return_horizon_days=60,
            mu=0.03,
            mu_horizon_days=60,
        )

        record_candidate_scores(conn, rid, [cand], {"BBB": holding}, selected_tickers=set())

        rows = {
            (r[0], r[1]): r[2:]
            for r in conn.execute(
                """SELECT ticker, role, expected_return_horizon_days, mu_horizon_days
                     FROM candidate_scores WHERE run_id = ?""",
                (rid,),
            )
        }
        assert rows[("AAA", "candidate")] == (60, 60)
        assert rows[("BBB", "holding")] == (60, 60)
        conn.close()

    def test_non_selected_candidate_gets_explicit_default_reason(self, tmp_path):
        from kernel.selection import CandidateResult
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        cand = CandidateResult(ticker="AAA", raw_score=0, rank_score=0.8,
                               rs_score=0, detail="", expected_return=0.01)

        record_candidate_scores(
            conn, rid, [cand], {}, selected_tickers=set(),
        )

        row = conn.execute(
            "SELECT blocked_by FROM candidate_scores WHERE run_id = ? AND ticker = ?",
            (rid, "AAA"),
        ).fetchone()
        assert row[0] == "candidate_not_selected"
        conn.close()

    def test_missing_candidate_scores_persist_as_null_not_zero(self, tmp_path):
        from kernel.selection import CandidateResult
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        cand = CandidateResult(
            ticker="AAA",
            raw_score=None,
            rank_score=None,
            rs_score=None,
            detail="",
            expected_return=None,
        )

        record_candidate_scores(
            conn, rid, [cand], {}, selected_tickers=set(),
        )

        row = conn.execute(
            """SELECT raw_score, rank_score, rs_score
                 FROM candidate_scores
                WHERE run_id = ? AND ticker = ?""",
            (rid, "AAA"),
        ).fetchone()
        assert row == (None, None, None)
        conn.close()

    def test_selected_candidate_clears_stale_block_reason(self, tmp_path):
        """AUDIT REGRESSION GUARD: selected rows are outcomes, not blocks.

        Kelly/QP diagnostics can stamp zero reasons before a later portfolio
        layer emits a buy. Persisting both selected=1 and blocked_by corrupts
        decision-factor attribution.
        """
        from kernel.selection import CandidateResult
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        c1 = CandidateResult(ticker="AAA", raw_score=0, rank_score=0.9,
                             rs_score=0, detail="", expected_return=0)
        c2 = CandidateResult(ticker="BBB", raw_score=0, rank_score=0.4,
                             rs_score=0, detail="", expected_return=0)
        record_candidate_scores(
            conn, rid, [c1, c2], {}, selected_tickers={"AAA"},
            blocked_map={"AAA": "kelly_zero:mu_none", "BBB": "tier"},
        )
        rows = dict(conn.execute(
            "SELECT ticker, blocked_by FROM candidate_scores WHERE run_id = ?",
            (rid,),
        ).fetchall())
        assert rows == {"AAA": None, "BBB": "tier"}
        conn.close()

    def test_ticker_daily_state_selected_clears_stale_block_reason(self, tmp_path):
        """AUDIT REGRESSION GUARD: ticker_daily_state uses the same
        selected=>not-blocked invariant as candidate_scores."""
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 4, 22),
            rows=[
                {"ticker": "AAA", "selected": 1, "blocked_by": "kelly_zero:mu_none"},
                {"ticker": "BBB", "selected": 0, "blocked_by": "tier"},
            ],
        )
        rows = dict(conn.execute(
            "SELECT ticker, blocked_by FROM ticker_daily_state WHERE run_id = ?",
            (rid,),
        ).fetchall())
        assert rows == {"AAA": None, "BBB": "tier"}
        conn.close()


class TestTrades:
    def test_records_buys_and_sells(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 4, 22),
        )
        record_trades(conn, rid, [
            {"ticker": "AAA", "action": "buy", "shares": 10, "price": 100.0,
             "invest": 1000.0, "rank_score": 0.6, "mu": 0.02, "sigma": 0.04},
            {"ticker": "BBB", "action": "sell", "price": 95.0,
             "exit_reason": "trailing_stop", "pnl_pct": -0.05, "hold_days": 42,
             "tax": 20.0},
        ])
        rows = conn.execute(
            "SELECT ticker, action, price, exit_reason FROM trades WHERE run_id = ?",
            (rid,),
        ).fetchall()
        assert len(rows) == 2
        kinds = {(r[0], r[1]) for r in rows}
        assert ("AAA", "buy") in kinds and ("BBB", "sell") in kinds
        # spot-check the sell row
        sell = next(r for r in rows if r[1] == "sell")
        assert sell[3] == "trailing_stop"
        conn.close()

    def test_records_trade_decision_tree_payloads(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "date": datetime.date(2026, 5, 22),
            "shares": 4,
            "price": 125.0,
            "invest": 500.0,
            "gross_pnl": 40.0,
            "proceeds_basis": 500.0,
            "net_pnl_after_tax": 32.0,
            "tax_cash_debited": 0.0,
            "tax_cash_debit_mode": "reporting_only",
            "tax_lot_method": "hifo",
            "panel_score": 0.58,
            "expected_return": 0.021,
            "expected_return_horizon_days": 60,
            "kelly_target_pct": 0.08,
            "mu_horizon_days": 60,
            "regime": "BULL_CALM",
            "confidence": 0.72,
            "order_type": "QP_BUY",
            "source": "JointPortfolioQPJob.JointPortfolioQPTask",
            "source_job": "JointPortfolioQPJob",
            "source_task": "JointPortfolioQPTask",
            "order_source": "JointPortfolioQPJob.JointPortfolioQPTask",
            "attribution_version": "order_attribution_v1",
            "score_snapshot": {
                "rank_score": 0.61,
                "panel_score": 0.58,
                "mu": 0.014,
                "sigma": 0.032,
                "kelly_target_pct": 0.08,
                "confidence": 0.72,
                "regime": "BULL_CALM",
            },
            "decision_inputs": {
                "acceptance_reason": "qp_target_weight_increase",
                "target_w": 0.08,
                "current_w": 0.00,
                "delta_w": 0.08,
            },
        }])
        row = conn.execute(
            """SELECT trade_date, order_type, source_job, source_task,
                      order_source, attribution_version,
                      score_snapshot_json, decision_inputs_json,
                      gross_pnl, proceeds_basis, net_pnl_after_tax,
                      tax_cash_debited, tax_cash_debit_mode,
                      tax_lot_method, panel_score, expected_return,
                      expected_return_horizon_days, kelly_target_pct,
                      mu_horizon_days, regime, confidence
                 FROM trades WHERE run_id = ? AND ticker = 'AAA'""",
            (rid,),
        ).fetchone()
        assert row[0] == "2026-05-22"
        assert row[1] == "QP_BUY"
        assert row[2] == "JointPortfolioQPJob"
        assert row[3] == "JointPortfolioQPTask"
        assert row[4] == "JointPortfolioQPJob.JointPortfolioQPTask"
        assert row[5] == "order_attribution_v1"
        score_snapshot = json.loads(row[6])
        decision_inputs = json.loads(row[7])
        assert score_snapshot["rank_score"] == pytest.approx(0.61)
        assert score_snapshot["regime"] == "BULL_CALM"
        assert decision_inputs["acceptance_reason"] == "qp_target_weight_increase"
        assert decision_inputs["delta_w"] == pytest.approx(0.08)
        assert row[8] == pytest.approx(40.0)
        assert row[9] == pytest.approx(500.0)
        assert row[10] == pytest.approx(32.0)
        assert row[11] == pytest.approx(0.0)
        assert row[12] == "reporting_only"
        assert row[13] == "hifo"
        assert row[14] == pytest.approx(0.58)
        assert row[15] == pytest.approx(0.021)
        assert row[16] == 60
        assert row[17] == pytest.approx(0.08)
        assert row[18] == 60
        assert row[19] == "BULL_CALM"
        assert row[20] == pytest.approx(0.72)
        conn.close()

    def test_record_trades_fills_minimal_decision_payload(self, tmp_path):
        """Executed-trade rows must stay replayable even if a caller omits
        rich attribution. The fallback is explicit, not silently NULL."""
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
        }])
        row = conn.execute(
            """SELECT score_snapshot_json, decision_inputs_json
                 FROM trades WHERE run_id = ? AND ticker = 'AAA'""",
            (rid,),
        ).fetchone()
        score_snapshot = json.loads(row[0])
        decision_inputs = json.loads(row[1])
        assert score_snapshot["attribution_missing"] is True
        assert decision_inputs["acceptance_reason"] == "recorded_trade"
        assert decision_inputs["ticker"] == "AAA"
        conn.close()

    def test_decision_trace_integrity_report_pins_run_contract(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[
                {"ticker": "AAA", "selected": 1, "blocked_by": "tier"},
                {"ticker": "BBB", "selected": 0, "blocked_by": "tier"},
            ],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA", "BBB", "CCC"],
        )

        assert report["ok"] is False
        assert report["missing_watchlist_tickers"] == ["CCC"]
        assert report["selected_blocked_rows"] == 0
        assert report["candidate_selected_blocked_rows"] == 0
        assert report["candidate_reason_gaps"] == 0
        assert report["decision_reason_gaps"] == 0
        assert report["trade_payload_gaps"] == 0
        assert report["fallback_trade_attribution_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_allows_extra_trade_attempt_ticker(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="live", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[
                {"ticker": "AAA", "selected": 0, "blocked_by": "no_model_signal"},
                {"ticker": "BBB", "selected": 0, "blocked_by": "broker_rejected"},
            ],
        )
        record_trades(conn, rid, [{
            "ticker": "BBB",
            "action": "buy_rejected",
            "shares": 1,
            "price": 100.0,
            "blocked_by": "broker_rejected",
            "score_snapshot": {"attempt_status": "buy_rejected"},
            "decision_inputs": {"attempt_status": "buy_rejected"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is True
        assert report["extra_tickers"] == ["BBB"]
        assert report["unexplained_extra_tickers"] == []
        conn.close()

    def test_decision_trace_integrity_report_flags_candidate_reason_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        conn.execute(
            """INSERT INTO candidate_scores
               (run_id, ticker, role, selected, blocked_by)
               VALUES (?, 'AAA', 'candidate', 0, NULL)""",
            (rid,),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{"ticker": "AAA", "selected": 0, "blocked_by": "not_selected"}],
        )

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["candidate_reason_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_missing_block_reason(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[
                {"ticker": "AAA", "selected": 0, "blocked_by": None},
                {"ticker": "BBB", "selected": 1, "blocked_by": "stale"},
            ],
        )

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA", "BBB"],
        )

        assert report["ok"] is False
        assert report["decision_reason_gaps"] == 1
        assert report["selected_blocked_rows"] == 0
        conn.close()

    def test_decision_trace_integrity_report_flags_sell_share_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 1,
                "model_type": "xgb",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "sell",
            "price": 100.0,
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"exit_reason": "qp_sell"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["sell_share_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_sell_economic_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 1,
                "model_type": "xgb",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "sell",
            "shares": 1,
            "price": 100.0,
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"exit_reason": "stop_loss"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["sell_economic_gaps"] == 1
        conn.close()

    @pytest.mark.parametrize(
        ("gross_pnl", "tax", "net_pnl_after_tax"),
        [
            (100.0, 10.0, 999.0),   # net must equal gross - tax
            (100.0, -1.0, 101.0),   # tax cannot be negative
            (100.0, 120.0, -20.0),  # tax cannot exceed positive gross
            (-10.0, 1.0, -11.0),    # losing sell must not carry positive tax
            (float("nan"), 0.0, 0.0),
        ],
    )
    def test_decision_trace_integrity_report_flags_corrupt_sell_economics(
        self,
        tmp_path,
        gross_pnl,
        tax,
        net_pnl_after_tax,
    ):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 1,
                "model_type": "xgb",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "sell",
            "shares": 1,
            "price": 100.0,
            "gross_pnl": gross_pnl,
            "tax": tax,
            "net_pnl_after_tax": net_pnl_after_tax,
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"exit_reason": "stop_loss"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["sell_economic_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_accepts_valid_sell_economics(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 0,
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "sell",
            "shares": 1,
            "price": 100.0,
            "gross_pnl": 100.0,
            "tax": 25.0,
            "net_pnl_after_tax": 75.0,
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"exit_reason": "stop_loss"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is True
        assert report["sell_economic_gaps"] == 0
        conn.close()

    def test_decision_trace_integrity_report_checks_short_cover_economics(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 0,
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "short_cover",
            "shares": 1,
            "price": 100.0,
            "gross_pnl": 100.0,
            "tax": 25.0,
            "net_pnl_after_tax": 999.0,
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"exit_reason": "short_cover"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["sell_economic_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_qp_attribution_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "held_no_new_buy",
                "in_universe": 1,
                "model_type": "xgb",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "sell",
            "shares": 1,
            "price": 100.0,
            "source_job": "JointPortfolioQPJob",
            "score_snapshot": {"rank_score": 0.40},
            "decision_inputs": {"source_job": "JointPortfolioQPJob"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["qp_trade_attribution_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_qp_buy_horizon_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 1,
                "blocked_by": None,
                "in_universe": 1,
                "model_type": "xgb",
                "sector": "tech",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
            "source_job": "JointPortfolioQPJob",
            "score_snapshot": {
                "rank_score": 0.61,
                "mu": 0.02,
                "sigma": 0.18,
            },
            "decision_inputs": {
                "source_job": "JointPortfolioQPJob",
                "delta_w": 0.05,
                "target_w": 0.05,
                "solver_status": "optimal",
            },
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["qp_buy_horizon_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_score_horizon_gaps(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        conn.execute(
            """INSERT INTO candidate_scores
               (run_id, ticker, role, selected, blocked_by, expected_return)
               VALUES (?, 'AAA', 'candidate', 0, 'not_selected', 0.03)""",
            (rid,),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "not_selected",
                "in_universe": 1,
                "model_type": "xgb",
                "expected_return": 0.03,
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
            "expected_return": 0.03,
            "score_snapshot": {"rank_score": 0.61},
            "decision_inputs": {"acceptance_reason": "unit"},
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["candidate_horizon_gaps"] == 1
        assert report["decision_horizon_gaps"] == 1
        assert report["trade_horizon_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_accepts_qp_buy_horizon(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 1,
                "blocked_by": None,
                "in_universe": 1,
                "model_type": "xgb",
                "sector": "tech",
            }],
        )
        record_trades(conn, rid, [{
            "ticker": "AAA",
            "action": "buy",
            "shares": 1,
            "price": 100.0,
            "source_job": "JointPortfolioQPJob",
            "score_snapshot": {
                "rank_score": 0.61,
                "mu": 0.02,
                "mu_horizon_days": 60,
                "sigma": 0.18,
                "expected_return": 0.02,
                "expected_return_horizon_days": 60,
            },
            "decision_inputs": {
                "source_job": "JointPortfolioQPJob",
                "delta_w": 0.05,
                "target_w": 0.05,
                "solver_status": "optimal",
                "expected_return_horizon_days": 60,
                "mu_horizon_days": 60,
            },
        }])

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is True
        assert report["qp_buy_horizon_gaps"] == 0
        conn.close()

    def test_decision_trace_integrity_report_flags_model_type_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "not_selected",
                "in_universe": 1,
                "model_type": None,
            }],
        )

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["AAA"],
        )

        assert report["ok"] is False
        assert report["model_type_gaps"] == 1
        conn.close()

    def test_decision_trace_integrity_report_flags_selected_sector_gap(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "BAC",
                "selected": 1,
                "blocked_by": None,
                "in_universe": 1,
                "model_type": "xgb",
                "sector": None,
            }],
        )
        conn.execute(
            """INSERT INTO candidate_scores
               (run_id, ticker, role, selected, blocked_by, sector)
               VALUES (?, 'BAC', 'candidate', 1, NULL, NULL)""",
            (rid,),
        )

        report = decision_trace_integrity_report(
            conn, rid, expected_watchlist=["BAC"],
        )

        assert report["ok"] is False
        assert report["selected_sector_gaps"] == 1
        assert report["candidate_selected_sector_gaps"] == 1
        conn.close()

    def test_validate_decision_trace_integrity_raises_when_strict(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.update({
            "watchlist": ["AAA"],
            "persistence": {
                **cfg["persistence"],
                "strict_decision_trace_integrity": True,
            },
        })
        conn = get_connection(cfg)
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )

        with pytest.raises(RuntimeError, match="decision trace integrity failed"):
            validate_decision_trace_integrity(
                conn, rid, cfg, context="unit-test",
            )
        conn.close()

    def test_validate_decision_trace_integrity_passes_complete_trace(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.update({
            "watchlist": ["AAA"],
            "persistence": {
                **cfg["persistence"],
                "strict_decision_trace_integrity": True,
            },
        })
        conn = get_connection(cfg)
        rid = record_pipeline_run(
            conn, run_type="sim", run_date=datetime.date(2026, 5, 22),
        )
        record_ticker_daily_state(
            conn,
            run_id=rid,
            run_date=datetime.date(2026, 5, 22),
            rows=[{
                "ticker": "AAA",
                "selected": 0,
                "blocked_by": "no_model_signal",
                "in_universe": 0,
            }],
        )

        report = validate_decision_trace_integrity(
            conn, rid, cfg, context="unit-test",
        )

        assert report["ok"] is True
        conn.close()


class TestTrainingRun:
    def test_insert(self, tmp_path):
        conn = get_connection(_cfg(tmp_path))
        tid = record_training_run(
            conn,
            strategy="renquant_104",
            artifact_type="panel-ltr",
            config_snapshot={"num_boost_round": 300},
            oos_mean_ic=0.04,
            train_ic=0.33,
            n_rows=80627,
            feature_cols=["beta_60d_z", "hurst_proxy"],
            artifact_path="artifacts/panel-ltr.json",
        )
        assert tid is not None
        row = conn.execute(
            "SELECT artifact_type, oos_mean_ic FROM training_runs WHERE run_id = ?", (tid,),
        ).fetchone()
        assert row[0] == "panel-ltr"
        assert row[1] == pytest.approx(0.04)
        conn.close()


class TestSimAdapterIntegration:
    """SimAdapter.commit() writes to the DB when persistence.enabled is on."""

    def test_sim_adapter_writes_run_when_enabled(self, tmp_path, monkeypatch):
        from adapters.sim import SimAdapter
        import pandas as pd
        import numpy as np

        idx = pd.bdate_range("2024-01-02", periods=60)
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
        spy_df = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.ones(60) * 1e6,
        }, index=idx)

        cfg = _cfg(tmp_path, enabled=True)
        cfg.update({
            "watchlist": ["AAPL"],
            "sector_map": {"AAPL": "giant_tech"},
            "sector_etf_map": {"giant_tech": "XLK"},
            "tax": {},
            "regime": {},
        })
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv={"SPY": spy_df, "AAPL": spy_df}, spy_df=spy_df, sector_etf_map={},
            initial_cash=100_000,
        )
        today = idx[30]
        ctx = adapter.make_context(today)
        # Fake a minimal "pipeline output" so commit writes something:
        ctx.regime     = "BULL_CALM"
        ctx.confidence = 0.7
        ctx.candidates = []
        ctx.exits      = []
        ctx.rotations  = []
        ctx.orders     = []
        adapter.commit(ctx)

        # SimAdapter writes to sim_runs.db (role="sim") per 2026-04-24
        # DB separation — NOT to the live runs.db.
        import sqlite3
        conn = sqlite3.connect(tmp_path / "sim_runs.db")
        n = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        assert n == 1
        row = conn.execute(
            "SELECT run_type, regime FROM pipeline_runs"
        ).fetchone()
        assert row == ("sim", "BULL_CALM")
        tds = conn.execute(
            "SELECT ticker, in_watchlist FROM ticker_daily_state"
        ).fetchone()
        assert tds == ("AAPL", 1)
        conn.close()
        # Live DB should NOT have been touched.
        assert not (tmp_path / "runs.db").exists(), \
            "SimAdapter must not write to the live DB"

    def test_sim_adapter_noop_when_disabled(self, tmp_path):
        from adapters.sim import SimAdapter
        import pandas as pd
        import numpy as np

        idx = pd.bdate_range("2024-01-02", periods=30)
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))
        spy_df = pd.DataFrame({
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.ones(30) * 1e6,
        }, index=idx)

        cfg = _cfg(tmp_path, enabled=False)
        cfg.update({"watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {}})
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv={"SPY": spy_df}, spy_df=spy_df, sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._db is None  # noqa: SLF001
        # DB file should NOT exist
        assert not (tmp_path / "runs.db").exists()
