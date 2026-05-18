"""Tests for scripts/build_dashboard.py — daily metrics dashboard.

Four layers per CLAUDE.md §5.2 + user's directive on hand-rolled code:
  - Unit:        each section_* function with fixture data
  - Integration: build() composes sections + writes file
  - E2E:         output is valid Markdown (renderable on GitHub)
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Load the dashboard script as a module (it lives in scripts/, not on sys.path)
_DASH_PATH = REPO / "scripts" / "build_dashboard.py"
_spec = importlib.util.spec_from_file_location("build_dashboard", _DASH_PATH)
dashboard = importlib.util.module_from_spec(_spec)
sys.modules["build_dashboard"] = dashboard
_spec.loader.exec_module(dashboard)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_db(tmp_path):
    """Tiny in-memory db with portfolio_daily_metrics + trades tables.

    2026-05-17 fix: dates are computed relative to today so the
    section_recent_trades 7-day filter always sees the fixture trades.
    Pre-fix the hardcoded 2026-05-09 dates fell outside today's window
    once the calendar advanced, breaking the test silently.
    """
    today = datetime.date.today()
    def _d(days_back: int) -> str:
        return (today - datetime.timedelta(days=days_back)).isoformat()
    db_path = tmp_path / "fake_runs.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE portfolio_daily_metrics (
            as_of_date TEXT, run_type TEXT, strategy TEXT,
            portfolio_value REAL, daily_return REAL
        );
        CREATE TABLE trades (
            run_id TEXT, ticker TEXT, action TEXT, shares REAL,
            price REAL, exit_reason TEXT, pnl_pct REAL
        );
    """)
    cur.executemany(
        "INSERT INTO portfolio_daily_metrics VALUES (?,?,?,?,?)",
        [
            (_d(10), "live", "renquant-104", 10_000.0, 0.005),
            (_d(9),  "live", "renquant-104", 10_050.0, 0.005),
            (_d(8),  "live", "renquant-104", 10_100.0, 0.005),
            # Deployment-transition spike that should be filtered out
            (_d(22), "live", "renquant-104", 100_000.0, 8.87),
            (_d(21), "live", "renquant-104", 10_178.71, -0.898),
        ],
    )
    cur.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?)",
        [
            (f"{_d(2)}-live-x1", "AAPL", "buy",  10.0, 200.0,  None,      None),
            (f"{_d(2)}-live-x1", "MSFT", "buy",   5.0, 400.0,  None,      None),
            (f"{_d(1)}-live-x2", "NVDA", "sell",  3.0, 1200.0, "qp_sell", 0.05),
            (f"{_d(3)}-live-x3", "GOOG", "buy",   2.0, 180.0,  None,      None),
        ],
    )
    conn.commit()
    return conn


@pytest.fixture
def fake_state():
    return {
        "regime": "BULL_CALM",
        "regime_confidence": 0.72,
        "high_water_mark": 11_000.0,
        "holdings": {},
    }


# ── UNIT — section builders ──────────────────────────────────────────────────

class TestSectionHeader:

    def test_with_db_and_state(self, fake_db, fake_state):
        md = dashboard.section_header("alpaca", fake_db, fake_state)
        assert "RenQuant Dashboard" in md
        assert "alpaca" in md
        assert "$10,100.00" in md   # latest portfolio value
        assert "+0.50%" in md       # daily return
        assert "$11,000.00" in md   # HWM
        assert "BULL_CALM" in md
        assert "0.72" in md

    def test_no_db(self, fake_state):
        md = dashboard.section_header("paper", None, fake_state)
        assert "RenQuant Dashboard" in md
        assert "$11,000.00" in md   # HWM still rendered
        assert "—" in md            # missing fields show em-dash

    def test_no_state(self, fake_db):
        md = dashboard.section_header("alpaca", fake_db, {})
        # Header still renders without state
        assert "RenQuant Dashboard" in md
        assert "$10,100.00" in md


class TestSectionRecentTrades:

    def test_renders_with_trades(self, fake_db):
        md = dashboard.section_recent_trades(fake_db, n_days=7)
        assert "Recent trades" in md
        assert "AAPL" in md
        assert "MSFT" in md
        assert "NVDA" in md
        assert "qp_sell" in md
        assert "+5.00%" in md   # P/L on the NVDA sell

    def test_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE trades (run_id TEXT, ticker TEXT, action TEXT, "
            "shares REAL, price REAL, exit_reason TEXT, pnl_pct REAL)"
        )
        md = dashboard.section_recent_trades(conn)
        assert "No trades in window" in md

    def test_no_db(self):
        md = dashboard.section_recent_trades(None)
        assert "DB unavailable" in md


class TestSectionPnLSparkline:

    def test_filters_deployment_spikes(self, fake_db):
        md = dashboard.section_pnl_sparkline(fake_db)
        # Real daily returns shown
        assert "$10,000.00" in md
        assert "$10,050.00" in md
        assert "$10,100.00" in md
        # Spurious deployment rows filtered
        assert "$100,000.00" not in md
        assert "+887" not in md
        assert "-89" not in md

    def test_no_db_returns_empty(self):
        assert dashboard.section_pnl_sparkline(None) == ""


class TestSectionPriorities:

    def test_strips_existing_numbering(self, tmp_path, monkeypatch):
        # Create a fake roadmap with pre-numbered entries
        fake_rm = tmp_path / "roadmap.md"
        fake_rm.write_text(
            "## P0 — by ROI\n"
            "### 1. ⭐ Walk-forward gate\n"
            "Some text.\n"
            "### 2. Options-IV (Bali-Hovakimian 2009)\n"
            "More text.\n"
            "### ★ Execution-tactic block\n"
            "Even more.\n"
            "### 3. News sentiment FinBERT\n"
            "Yet more.\n"
            "## P1 — architecture\n"
            "### 11. Multi-horizon\n"
        )
        monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path.parent)
        # The function reads from REPO_ROOT/doc/roadmap.md — set up that path
        (tmp_path.parent / "doc").mkdir(parents=True, exist_ok=True)
        target = tmp_path.parent / "doc" / "roadmap.md"
        target.write_text(fake_rm.read_text())

        md = dashboard.section_priorities()
        # Each line numbered, no double-numbering
        assert "1. Walk-forward gate" in md
        assert "2. Options-IV" in md
        assert "3. Execution-tactic block" in md
        assert "1. 1." not in md   # numbering not duplicated
        assert "★" not in md       # ranking glyph stripped

        # Cleanup
        target.unlink()


# ── INTEGRATION — build() composes everything ────────────────────────────────

class TestBuildIntegration:

    def test_full_build_writes_file(self, tmp_path, monkeypatch, fake_db, fake_state):
        # Stage a fake repo layout under tmp_path that build() can read
        repo = tmp_path
        (repo / "data").mkdir()
        (repo / "backtesting" / "renquant_104").mkdir(parents=True)
        (repo / "doc").mkdir()

        # Move our fake_db to data/runs.alpaca.db
        fake_db_path = repo / "data" / "runs.alpaca.db"
        with sqlite3.connect(str(fake_db_path)) as new_db:
            for line in fake_db.iterdump():
                if line not in ("BEGIN TRANSACTION;", "COMMIT;"):
                    new_db.execute(line)
        # Stage live state
        state_path = repo / "backtesting" / "renquant_104" / "live_state.alpaca.json"
        state_path.write_text(json.dumps(fake_state))

        out_path = repo / "doc" / "dashboard.md"
        monkeypatch.setattr(dashboard, "REPO_ROOT", repo)

        md = dashboard.build(broker="alpaca", out_path=out_path)
        assert out_path.exists()
        assert len(md) > 100
        assert md == out_path.read_text()

        # Section headers all present
        for section in ["RenQuant Dashboard", "Recent trades",
                        "Portfolio P/L", "Model health"]:
            assert section in md

    def test_missing_db_still_produces_dashboard(self, tmp_path, monkeypatch):
        """Dashboard must render even if no broker DB exists yet
        (fresh install / never-traded case)."""
        repo = tmp_path
        (repo / "data").mkdir()
        (repo / "backtesting" / "renquant_104").mkdir(parents=True)
        (repo / "doc").mkdir()
        monkeypatch.setattr(dashboard, "REPO_ROOT", repo)

        out_path = repo / "doc" / "dashboard.md"
        md = dashboard.build(broker="alpaca", out_path=out_path)
        assert out_path.exists()
        assert "RenQuant Dashboard" in md
        # Should not raise even with all data missing


# ── E2E — output is valid Markdown ──────────────────────────────────────

class TestMarkdownValidity:

    def test_real_dashboard_is_well_formed(self, fake_db, fake_state, tmp_path,
                                           monkeypatch):
        """Run a full build and assert the output is well-formed:
          - balanced table pipes per row
          - no unclosed code fences
          - all H2 headers have content under them
        """
        repo = tmp_path
        (repo / "data").mkdir()
        (repo / "backtesting" / "renquant_104").mkdir(parents=True)
        (repo / "doc").mkdir()
        # Stage state
        state_path = repo / "backtesting" / "renquant_104" / "live_state.alpaca.json"
        state_path.write_text(json.dumps(fake_state))
        # Stage db
        db_path = repo / "data" / "runs.alpaca.db"
        with sqlite3.connect(str(db_path)) as new_db:
            for line in fake_db.iterdump():
                if line not in ("BEGIN TRANSACTION;", "COMMIT;"):
                    new_db.execute(line)
        monkeypatch.setattr(dashboard, "REPO_ROOT", repo)

        out_path = repo / "doc" / "dashboard.md"
        md = dashboard.build(broker="alpaca", out_path=out_path)

        # Markdown table pipe-balance: every table-data row has same pipe count
        for table_block in _extract_tables(md):
            counts = [line.count("|") for line in table_block.splitlines()
                      if line.strip().startswith("|")]
            assert len(set(counts)) == 1, \
                f"unbalanced table pipes: {counts}\n{table_block}"

        # No unclosed code fences
        assert md.count("```") % 2 == 0

        # Every H2 is followed by some content
        lines = md.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("## "):
                # Find next non-empty line within next 5 lines
                follow = [ln for ln in lines[i+1:i+6] if ln.strip()]
                assert follow, f"H2 with no content: {line!r}"


def _extract_tables(md: str) -> list[str]:
    """Yield each pipe-table block in *md*."""
    out: list[str] = []
    cur: list[str] = []
    for line in md.splitlines():
        if line.strip().startswith("|"):
            cur.append(line)
        else:
            if len(cur) >= 2:
                out.append("\n".join(cur))
            cur = []
    if len(cur) >= 2:
        out.append("\n".join(cur))
    return out
