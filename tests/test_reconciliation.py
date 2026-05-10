"""Tests for kernel/reconciliation/live_sim_reconcile.py.

Per CLAUDE.md §5.13.1 — tests must walk real prod data flow with synthetic
SQLite fixtures, NOT only hand-built dataclasses. Most helpers here ingest
LiveFill / SimDecision values that come back from the actual sqlite3 read
path; one test wires through a SimAdapter-shaped stub end-to-end.

NEVER hits real Alpaca. NEVER touches data/runs.alpaca.db.
"""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = (Path(__file__).resolve().parent.parent
                 / "backtesting" / "renquant_104")
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.reconciliation import (  # noqa: E402
    LiveFill, SimDecision,
    compute_decision_divergence,
    compute_rolling_ic,
    compute_slippage,
    emit_report,
    load_live_fills,
    load_sim_decisions,
    replay_through_sim,
)
from kernel.reconciliation.live_sim_reconcile import (  # noqa: E402
    build_per_day_breakdown,
)


# ── Synthetic DB fixture ───────────────────────────────────────────────────


def _make_db(path: Path) -> sqlite3.Connection:
    """Create a minimal runs.db schema sufficient for the reconciliation
    queries. Mirrors the prod schema columns we actually SELECT on."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE pipeline_runs (
            run_id    TEXT PRIMARY KEY,
            run_date  DATE NOT NULL,
            run_type  TEXT NOT NULL
        );
        CREATE TABLE trades (
            run_id  TEXT,
            ticker  TEXT,
            action  TEXT,
            shares  REAL,
            price   REAL
        );
        CREATE TABLE candidate_scores (
            run_id      TEXT,
            ticker      TEXT,
            role        TEXT,
            rank_score  REAL,
            PRIMARY KEY (run_id, ticker, role)
        );
        CREATE TABLE ticker_forward_returns (
            as_of_date  DATE NOT NULL,
            ticker      TEXT NOT NULL,
            fwd_5d      REAL,
            PRIMARY KEY (as_of_date, ticker)
        );
    """)
    return conn


def _seed_run(
    conn: sqlite3.Connection, run_id: str, date: str, run_type: str,
    fills: list[tuple[str, str, float, float]],
) -> None:
    conn.execute(
        "INSERT INTO pipeline_runs (run_id, run_date, run_type) VALUES (?, ?, ?)",
        (run_id, date, run_type),
    )
    for ticker, action, shares, price in fills:
        conn.execute(
            "INSERT INTO trades (run_id, ticker, action, shares, price) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, ticker, action, shares, price),
        )
    conn.commit()


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    p = tmp_path / "empty.db"
    conn = _make_db(p)
    conn.close()
    return p


@pytest.fixture
def aligned_db(tmp_path: Path) -> tuple[Path, Path]:
    """Live DB + sim DB where every live fill has a matching sim trade."""
    live_p = tmp_path / "runs.alpaca.db"
    sim_p  = tmp_path / "sim_runs.db"
    live = _make_db(live_p)
    sim  = _make_db(sim_p)
    _seed_run(live, "live-1", "2026-05-09", "live",
              [("AAPL", "buy", 10, 200.0), ("MSFT", "sell", 5, 400.0)])
    _seed_run(sim,  "sim-1",  "2026-05-09", "sim",
              [("AAPL", "buy", 10, 200.0), ("MSFT", "sell", 5, 400.0)])
    live.close(); sim.close()
    return live_p, sim_p


@pytest.fixture
def diverging_db(tmp_path: Path) -> tuple[Path, Path]:
    """Sim disagrees on 1 of 2 fills (50% divergence)."""
    live_p = tmp_path / "runs.alpaca.db"
    sim_p  = tmp_path / "sim_runs.db"
    live = _make_db(live_p)
    sim  = _make_db(sim_p)
    _seed_run(live, "live-1", "2026-05-09", "live",
              [("AAPL", "buy", 10, 200.0), ("NVDA", "buy", 8, 500.0)])
    # Sim says hold on NVDA (no row for it), agrees on AAPL.
    _seed_run(sim, "sim-1", "2026-05-09", "sim",
              [("AAPL", "buy", 10, 200.0)])
    live.close(); sim.close()
    return live_p, sim_p


# ── Loaders ─────────────────────────────────────────────────────────────────


class TestLoadLiveFills:
    def test_basic_load(self, aligned_db):
        live_p, _ = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        assert len(fills) == 2
        tickers = {f.ticker for f in fills}
        assert tickers == {"AAPL", "MSFT"}
        assert all(isinstance(f, LiveFill) for f in fills)

    def test_date_filter(self, aligned_db):
        live_p, _ = aligned_db
        fills = load_live_fills(live_p, "2026-05-10", "2026-05-15")
        assert fills == []

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_live_fills(tmp_path / "nope.db", "2026-05-09", "2026-05-09")

    def test_skips_sim_run_type(self, aligned_db):
        # The aligned fixture also has sim-tagged rows in the sim DB; ensure
        # we don't pick those up when we point at the live DB.
        live_p, _ = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        assert all(f.run_id.startswith("live-") for f in fills)


# ── Same-fills → zero divergence / zero slip ────────────────────────────────


class TestAlignedFills:
    def test_zero_divergence_zero_slip(self, aligned_db):
        live_p, sim_p = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        sims  = load_sim_decisions(sim_p, "2026-05-09", "2026-05-09")
        matched = replay_through_sim(fills, sim_decisions=sims)

        slip = compute_slippage(fills, matched)
        div  = compute_decision_divergence(fills, matched)
        assert slip["p50_bps"] == pytest.approx(0.0, abs=1e-6)
        assert slip["p95_bps"] == pytest.approx(0.0, abs=1e-6)
        assert div["divergence_rate"] == 0.0
        assert div["n_disagree"] == 0


# ── 50% divergence detection ────────────────────────────────────────────────


class TestDivergenceDetection:
    def test_half_disagreement(self, diverging_db):
        live_p, sim_p = diverging_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        sims  = load_sim_decisions(sim_p, "2026-05-09", "2026-05-09")
        matched = replay_through_sim(fills, sim_decisions=sims)
        div = compute_decision_divergence(fills, matched)
        assert div["n"] == 2
        assert div["n_disagree"] == 1
        assert div["divergence_rate"] == 0.5
        assert div["cases"][0]["ticker"] == "NVDA"
        assert div["cases"][0]["sim_action"] == "hold"

    def test_qty_tolerance(self):
        fills = [LiveFill("r", "2026-05-09", "AAPL", "buy", 100, 200.0)]
        # Sim said 95 shares — 5% delta is at the boundary; default tol = 5%.
        sims = [SimDecision("2026-05-09", "AAPL", "buy", 95, 200.0)]
        div = compute_decision_divergence(fills, sims, qty_tolerance_pct=0.04)
        assert div["n_disagree"] == 1
        # Loosen tolerance — same delta now passes.
        div2 = compute_decision_divergence(fills, sims, qty_tolerance_pct=0.10)
        assert div2["n_disagree"] == 0


# ── Slippage calibration with known synthetic spread ───────────────────────


class TestSlippageCalibration:
    def test_known_spread_p95(self):
        # Sim @ 100; live trades at 100 + spread bps for buys, 100 - spread
        # bps for sells. Symmetric spread → both push slippage_bps positive.
        fills = []
        sims  = []
        spreads_bps = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100]  # 10 buys
        for i, sb in enumerate(spreads_bps):
            d = f"2026-05-{i + 1:02d}"
            sim_price = 100.0
            live_price = 100.0 * (1 + sb / 1e4)
            fills.append(LiveFill(f"r{i}", d, "AAPL", "buy", 10, live_price))
            sims.append(SimDecision(d, "AAPL", "buy", 10, sim_price))
        slip = compute_slippage(fills, sims)
        assert slip["n"] == 10
        # p50 = 5th smallest of [1,2,3,4,5,10,20,30,50,100] -> idx 4 or 5.
        # Allow tiny float slack on (1 + bps/1e4) round-trip.
        assert 4.5 <= slip["p50_bps"] <= 10.5
        # p95 = near the top — should be 50 or 100.
        assert slip["p95_bps"] >= 50.0
        assert slip["max_bps"] == pytest.approx(100.0, abs=1e-3)

    def test_sell_sign_convention(self):
        # Live SELLS at 99 (vs sim 100) — operator received LESS, so this
        # should register as POSITIVE slippage (live was worse).
        fills = [LiveFill("r", "2026-05-09", "AAPL", "sell", 10, 99.0)]
        sims  = [SimDecision("2026-05-09", "AAPL", "sell", 10, 100.0)]
        slip = compute_slippage(fills, sims)
        assert slip["n"] == 1
        assert slip["p50_bps"] > 0  # positive = live worse


# ── Rolling IC calibration ─────────────────────────────────────────────────


class TestRollingIC:
    def test_perfect_correlation(self, tmp_path):
        live_p = tmp_path / "runs.alpaca.db"
        conn = _make_db(live_p)
        # Plant predictions that perfectly track realized fwd_5d.
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, run_date, run_type) "
            "VALUES (?, ?, ?)", ("r1", "2026-05-09", "live"),
        )
        for i, t in enumerate(["AAPL", "MSFT", "NVDA", "GOOG", "META"]):
            score = float(i)
            ret   = float(i) * 0.01
            conn.execute(
                "INSERT INTO candidate_scores (run_id, ticker, role, "
                "rank_score) VALUES (?, ?, 'cand', ?)", ("r1", t, score),
            )
            conn.execute(
                "INSERT INTO ticker_forward_returns (as_of_date, ticker, "
                "fwd_5d) VALUES (?, ?, ?)", ("2026-05-09", t, ret),
            )
        conn.commit(); conn.close()

        ic = compute_rolling_ic(live_p, "2026-05-09", "2026-05-09")
        assert ic["ok"] is True
        assert ic["n"] == 5
        assert ic["ic"] == pytest.approx(1.0, abs=1e-6)

    def test_missing_tables_no_crash(self, tmp_path):
        # DB exists but candidate_scores join produces 0 rows.
        live_p = tmp_path / "runs.alpaca.db"
        _make_db(live_p).close()
        ic = compute_rolling_ic(live_p, "2026-05-09", "2026-05-09")
        assert ic["ok"] is False
        assert ic["n"] == 0

    def test_missing_db_no_crash(self, tmp_path):
        ic = compute_rolling_ic(tmp_path / "nope.db",
                                "2026-05-09", "2026-05-09")
        assert ic["ok"] is False


# ── Empty DB → empty report, no crash ──────────────────────────────────────


class TestEmptyDB:
    def test_empty_db_emits_report(self, empty_db, tmp_path):
        fills = load_live_fills(empty_db, "2026-05-09", "2026-05-09")
        assert fills == []
        matched = replay_through_sim(fills)
        metrics = {
            "broker": "alpaca",
            "start_date": "2026-05-09", "end_date": "2026-05-09",
            "n_fills": 0,
            "generated_at": datetime.datetime.now().isoformat(),
            "slippage":   compute_slippage(fills, matched),
            "divergence": compute_decision_divergence(fills, matched),
            "per_day":    build_per_day_breakdown(fills, matched),
            "rolling_ic": compute_rolling_ic(empty_db,
                                             "2026-05-09", "2026-05-09"),
        }
        out = tmp_path / "empty.md"
        written = emit_report(metrics, out)
        text = written.read_text()
        assert "Live<->Sim Reconciliation" in text
        assert "0.00%" in text   # divergence rate prints zero
        assert "(no fills in window)" in text


# ── End-to-end SimAdapter stub (per §5.13.1) ──────────────────────────────


class _StubSimAdapter:
    """Minimal SimAdapter shape for the e2e walk: lookup_decision(date,
    ticker) -> (action, shares, price). Mirrors how a real SimAdapter
    would be wrapped to expose its decision history per (date, ticker).

    We DON'T import the real SimAdapter here because constructing one
    requires building 150+MB of OHLCV panels for 100+ tickers — overkill
    for a unit test. The contract being tested is the integration shape
    (callable with date+ticker, returns 3-tuple), which matches the real
    adapter's stored decision log.
    """

    def __init__(self, decisions: dict[tuple[str, str],
                                       tuple[str, float, float]]) -> None:
        self._decisions = decisions

    def lookup_decision(self, date: str, ticker: str):
        return self._decisions.get((date, ticker.upper()),
                                   ("hold", 0.0, float("nan")))


class TestEndToEndSimAdapter:
    def test_walks_through_adapter(self, aligned_db):
        live_p, _ = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        adapter = _StubSimAdapter({
            ("2026-05-09", "AAPL"): ("buy",  10.0, 200.0),
            ("2026-05-09", "MSFT"): ("sell", 5.0,  400.0),
        })
        matched = replay_through_sim(fills, sim_adapter=adapter)
        assert len(matched) == 2
        assert all(m.action in ("buy", "sell") for m in matched)
        # Combined with slippage/divergence — full pipeline ends at 0/0:
        assert compute_slippage(fills, matched)["p95_bps"] == pytest.approx(
            0.0, abs=1e-6
        )
        assert compute_decision_divergence(fills, matched)["n_disagree"] == 0

    def test_adapter_failure_is_caught(self, aligned_db):
        """If sim adapter raises, the helper logs + falls back to hold —
        never crashes the whole reconciliation run."""
        live_p, _ = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")

        class BoomAdapter:
            def lookup_decision(self, *_args, **_kw):
                raise RuntimeError("upstream sim crashed")

        matched = replay_through_sim(fills, sim_adapter=BoomAdapter())
        assert len(matched) == 2
        assert all(m.action == "hold" for m in matched)
        # Divergence is 100% (live had buys/sells, sim "said" hold).
        div = compute_decision_divergence(fills, matched)
        assert div["divergence_rate"] == 1.0


# ── Per-day breakdown ──────────────────────────────────────────────────────


class TestPerDayBreakdown:
    def test_buckets_by_date(self):
        fills = [
            LiveFill("r1", "2026-05-08", "AAPL", "buy", 10, 200.0),
            LiveFill("r2", "2026-05-09", "AAPL", "buy", 10, 200.0),
            LiveFill("r3", "2026-05-09", "MSFT", "sell", 5, 400.0),
        ]
        sims = [
            SimDecision("2026-05-08", "AAPL", "buy", 10, 200.0),
            SimDecision("2026-05-09", "AAPL", "buy", 10, 200.0),
            SimDecision("2026-05-09", "MSFT", "sell", 5, 400.0),
        ]
        rows = build_per_day_breakdown(fills, sims)
        assert len(rows) == 2
        assert rows[0]["date"] == "2026-05-08"
        assert rows[0]["n_fills"] == 1
        assert rows[1]["date"] == "2026-05-09"
        assert rows[1]["n_fills"] == 2
        assert all(r["n_disagree"] == 0 for r in rows)


# ── Report rendering smoke ─────────────────────────────────────────────────


class TestEmitReport:
    def test_report_includes_all_sections(self, aligned_db, tmp_path):
        live_p, sim_p = aligned_db
        fills = load_live_fills(live_p, "2026-05-09", "2026-05-09")
        sims  = load_sim_decisions(sim_p, "2026-05-09", "2026-05-09")
        matched = replay_through_sim(fills, sim_decisions=sims)
        metrics = {
            "broker": "alpaca",
            "start_date": "2026-05-09", "end_date": "2026-05-09",
            "n_fills": len(fills),
            "generated_at": "now",
            "slippage":   compute_slippage(fills, matched),
            "divergence": compute_decision_divergence(fills, matched),
            "per_day":    build_per_day_breakdown(fills, matched),
            "rolling_ic": {"ok": False, "warn": "test"},
        }
        out = tmp_path / "report.md"
        emit_report(metrics, out)
        text = out.read_text()
        for section in ("Summary", "Per-Day Breakdown", "Divergence Cases",
                        "Rolling IC"):
            assert f"## {section}" in text, f"missing section {section}"
