"""Tests for the WF cut loader — §8 Step 4e.

Pins the data adapter that materialises ``AllocatorReplayBar`` from
the sim decision trace. The replay harness math (PR #131) is tested
separately; this file only covers the loader's contract:

1. DB row → bar emission grouping (per-date).
2. Constraint snapshot defaults match the per-regime spec in §1
   PRIME DIRECTIVE (BULL_CALM tighter than other regimes).
3. fwd_horizon_days picks the right column from
   ticker_forward_returns; unsupported horizons fail loud.
4. Missing DB → friendly SystemExit (#128 pattern).
5. Empty date range → empty list.
6. Smoke: piping the loader output into ``replay_all`` produces
   non-empty per-allocator results.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.allocator_replay import (  # noqa: E402
    AllocatorReplayBar,
    replay_all,
)
from kernel.portfolio_qp.baseline_allocators import (  # noqa: E402
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402
from kernel.portfolio_qp.wf_replay_loader import (  # noqa: E402
    load_replay_bars_from_sim_db,
)


def _build_fixture_db(
    db_path: Path,
    *,
    dates: list[str],
    tickers_per_date: dict[str, list[str]],
    regime_per_date: dict[str, str],
    mu_per_pair: dict[tuple[str, str], float] | None = None,
    sigma_per_pair: dict[tuple[str, str], float] | None = None,
    fwd_60_per_pair: dict[tuple[str, str], float] | None = None,
) -> None:
    """Create a minimal sim_runs.db with the columns the loader reads."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE score_distribution (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            raw_panel REAL,
            rank_score REAL,
            mu REAL,
            sigma REAL,
            regime TEXT,
            is_holding INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE ticker_forward_returns (
            as_of_date TEXT,
            ticker TEXT,
            close_price REAL,
            fwd_1d REAL,
            fwd_5d REAL,
            fwd_10d REAL,
            fwd_20d REAL,
            fwd_60d REAL,
            updated_at TEXT
        )
    """)
    mu_per_pair = mu_per_pair or {}
    sigma_per_pair = sigma_per_pair or {}
    fwd_per_pair = fwd_60_per_pair or {}
    for date in dates:
        regime = regime_per_date.get(date)
        for ticker in tickers_per_date.get(date, []):
            mu = mu_per_pair.get((date, ticker), 0.02)
            sigma = sigma_per_pair.get((date, ticker), 0.10)
            fwd = fwd_per_pair.get((date, ticker), 0.005)
            cur.execute(
                "INSERT INTO score_distribution "
                "(run_id, date, ticker, mu, sigma, regime) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("run-test", date, ticker, mu, sigma, regime),
            )
            cur.execute(
                "INSERT INTO ticker_forward_returns "
                "(as_of_date, ticker, fwd_60d) VALUES (?, ?, ?)",
                (date, ticker, fwd),
            )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_db():
    """3-date / 3-ticker fixture DB. BULL_CALM on dates 1+2; CHOPPY on 3."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "sim_runs.db"
        _build_fixture_db(
            db,
            dates=["2026-01-01", "2026-01-02", "2026-01-03"],
            tickers_per_date={
                "2026-01-01": ["AAPL", "MSFT", "GOOG"],
                "2026-01-02": ["AAPL", "MSFT", "GOOG"],
                "2026-01-03": ["AAPL", "MSFT"],  # only 2 — still emits
            },
            regime_per_date={
                "2026-01-01": "BULL_CALM",
                "2026-01-02": "BULL_CALM",
                "2026-01-03": "CHOPPY",
            },
            mu_per_pair={
                ("2026-01-01", "AAPL"): 0.05,
                ("2026-01-01", "MSFT"): 0.04,
                ("2026-01-01", "GOOG"): 0.03,
                ("2026-01-02", "AAPL"): 0.04,
                ("2026-01-02", "MSFT"): 0.05,
                ("2026-01-02", "GOOG"): 0.03,
                ("2026-01-03", "AAPL"): 0.02,
                ("2026-01-03", "MSFT"): 0.01,
            },
            fwd_60_per_pair={
                ("2026-01-01", "AAPL"): 0.05,
                ("2026-01-01", "MSFT"): 0.04,
                ("2026-01-01", "GOOG"): 0.03,
                ("2026-01-02", "AAPL"): 0.04,
                ("2026-01-02", "MSFT"): 0.05,
                ("2026-01-02", "GOOG"): 0.03,
                ("2026-01-03", "AAPL"): 0.01,
                ("2026-01-03", "MSFT"): 0.02,
            },
        )
        yield db


class TestBarEmissionGrouping:
    def test_one_bar_per_date(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-03",
        )
        # 3 unique dates → 3 bars
        assert len(bars) == 3
        assert [b.bar_date for b in bars] == ["2026-01-01", "2026-01-02", "2026-01-03"]

    def test_tickers_sorted_within_bar(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-01",
        )
        bar = bars[0]
        assert list(bar.snap.tickers) == sorted(bar.snap.tickers)

    def test_mu_sigma_fwd_shapes_match(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-03",
        )
        for bar in bars:
            n = bar.snap.n
            assert bar.mu.shape == (n,)
            assert bar.sigma.shape == (n,)
            assert bar.fwd_return.shape == (n,)


class TestPerRegimeConstraintDefaults:
    def test_bull_calm_max_position_pct_is_0_15(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-01",
        )
        snap = bars[0].snap
        assert bars[0].regime == "BULL_CALM"
        assert np.allclose(snap.w_upper_hard, 0.15)
        assert np.allclose(snap.w_upper, 0.15)

    def test_non_bull_calm_max_position_pct_is_0_20(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-03", "2026-01-03",
        )
        snap = bars[0].snap
        assert bars[0].regime == "CHOPPY"
        assert np.allclose(snap.w_upper_hard, 0.20)

    def test_cash_reserve_default_5pct(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-01",
        )
        assert bars[0].snap.cash_reserve == 0.05


class TestFwdHorizonSelection:
    def test_unsupported_horizon_raises(self, fixture_db):
        with pytest.raises(ValueError, match="fwd_horizon_days=99"):
            load_replay_bars_from_sim_db(
                fixture_db, "2026-01-01", "2026-01-01",
                fwd_horizon_days=99,
            )

    def test_default_horizon_60_days(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-01",
        )
        # The fixture stamped fwd_60d only — confirm it's wired.
        np.testing.assert_array_equal(bars[0].fwd_return,
                                       np.array([0.05, 0.03, 0.04]))


class TestLoadFailureModes:
    def test_missing_db_raises_systemexit(self):
        with pytest.raises(SystemExit, match="DB not found"):
            load_replay_bars_from_sim_db(
                "/nonexistent/sim_runs.db",
                "2026-01-01", "2026-01-31",
            )

    def test_empty_date_range_returns_empty_list(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2030-01-01", "2030-01-31",
        )
        assert bars == []


class TestReplayHarnessIntegration:
    def test_bars_consumed_by_replay_all(self, fixture_db):
        bars = load_replay_bars_from_sim_db(
            fixture_db, "2026-01-01", "2026-01-03",
        )
        results = replay_all(
            {
                "equal_weight": equal_weight_top_k,
                "inverse_vol": inverse_vol_top_k,
                "fractional_kelly": fractional_kelly_top_k,
            },
            bars,
        )
        # Every allocator has 3 daily returns (3 bars)
        for name, res in results.items():
            assert res.bars == 3
            assert len(res.daily_returns_net) == 3


class TestStep4hSectorSnapshot:
    def test_build_sector_matrix_from_map(self):
        """#136/#154 Step-4h: today's sector_map -> (S, cap_vec, names)."""
        from kernel.portfolio_qp.wf_replay_loader import _build_sector_matrix
        tickers = ["AAPL", "MSFT", "JPM", "GS"]
        sector_map = {"AAPL": "tech", "MSFT": "tech", "JPM": "finance", "GS": "finance"}
        S, cap_vec, names = _build_sector_matrix(tickers, sector_map, max_per_sector=2, per_name_cap=0.15)
        assert names == ("finance", "tech")
        assert S.shape == (2, 4)
        # tech row: AAPL,MSFT = 1; finance row: JPM,GS = 1
        assert list(S[names.index("tech")]) == [1.0, 1.0, 0.0, 0.0]
        assert list(S[names.index("finance")]) == [0.0, 0.0, 1.0, 1.0]
        # cap = max_per_sector * per_name_cap = 2 * 0.15 = 0.30
        assert all(abs(c - 0.30) < 1e-9 for c in cap_vec)

    def test_build_sector_matrix_no_map_returns_none(self):
        from kernel.portfolio_qp.wf_replay_loader import _build_sector_matrix
        S, cap_vec, names = _build_sector_matrix(["AAPL"], {}, 2, 0.15)
        assert (S, cap_vec, names) == (None, None, None)

    def test_build_snapshot_with_sector_is_decision_grade(self):
        """Snapshot carrying sector caps -> constraint_fidelity decision_grade."""
        from kernel.portfolio_qp.wf_replay_loader import _build_snapshot
        from kernel.portfolio_qp.run_ab_replay import constraint_fidelity_block
        snap = _build_snapshot(
            ["AAPL", "MSFT", "JPM"], "BULL_CALM",
            sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "finance"},
            max_per_sector=2,
        )
        assert snap.sector_indicator is not None
        assert snap.sector_cap_vec is not None
        # one bar carrying sector caps -> decision_grade True
        from kernel.portfolio_qp.allocator_replay import AllocatorReplayBar
        import numpy as np
        bar = AllocatorReplayBar(
            bar_date="d-0", snap=snap,
            mu=np.array([0.02, 0.01, 0.015]),
            sigma=np.array([0.1, 0.1, 0.1]),
            fwd_return=np.array([0.001, -0.001, 0.0]),
            regime="BULL_CALM", cost_per_trade_bps=0.0,
        )
        cf = constraint_fidelity_block([bar])
        assert cf["decision_grade"] is True

    def test_build_snapshot_without_sector_not_decision_grade(self):
        from kernel.portfolio_qp.wf_replay_loader import _build_snapshot
        from kernel.portfolio_qp.run_ab_replay import constraint_fidelity_block
        from kernel.portfolio_qp.allocator_replay import AllocatorReplayBar
        import numpy as np
        snap = _build_snapshot(["AAPL", "MSFT"], "BULL_CALM")  # no sector args
        assert snap.sector_indicator is None
        bar = AllocatorReplayBar(
            bar_date="d-0", snap=snap,
            mu=np.array([0.02, 0.01]), sigma=np.array([0.1, 0.1]),
            fwd_return=np.array([0.001, -0.001]),
            regime="BULL_CALM", cost_per_trade_bps=0.0,
        )
        cf = constraint_fidelity_block([bar])
        assert cf["decision_grade"] is False
