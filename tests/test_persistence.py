"""Tests for kernel/persistence.py — SQLite decision-trace."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
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
    record_training_run,
    record_ticker_daily_state,
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
                      score_snapshot_json, decision_inputs_json
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
        cfg.update({"watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {}})
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv={"SPY": spy_df}, spy_df=spy_df, sector_etf_map={},
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
