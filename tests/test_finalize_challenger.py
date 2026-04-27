"""Tests for scripts/finalize_challenger.py — Phase 4b operator UX.

Pin the report-generation contract so future changes don't silently
break the markdown layout, recommendation heuristic, or fwd-return
join. The script is the operator-facing surface that turns a closed
shadow window into a decision artifact.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

import finalize_challenger as fc   # noqa: E402
from kernel.persistence import ensure_schema   # noqa: E402
from kernel.challenger import log_decision    # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _open_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "runs.db"), isolation_level=None)
    ensure_schema(conn)
    return conn


def _populate_decisions(conn, *, name="macro-enabled"):
    """Insert 10 synthetic decisions: 7 agree, 3 disagree."""
    rows = [
        # date,         ticker, ch_act, actual_act, ch_score, actual_score
        ("2026-04-12", "AAPL", "BUY",  "BUY",  0.72, 0.68),
        ("2026-04-13", "MSFT", "HOLD", "HOLD", 0.18, 0.21),
        ("2026-04-14", "GOOG", "BUY",  "BUY",  0.81, 0.79),
        ("2026-04-15", "NVDA", "BUY",  "HOLD", 0.65, 0.32),  # CH-only BUY
        ("2026-04-16", "AMZN", "HOLD", "HOLD", 0.15, 0.22),
        ("2026-04-17", "META", "BUY",  "BUY",  0.55, 0.51),
        ("2026-04-18", "TSLA", "HOLD", "BUY",  0.27, 0.66),  # live-only BUY
        ("2026-04-19", "ADBE", "BUY",  "BUY",  0.69, 0.72),
        ("2026-04-20", "AVGO", "BUY",  "HOLD", 0.71, 0.41),  # CH-only BUY
        ("2026-04-21", "ORCL", "HOLD", "HOLD", 0.10, 0.15),
    ]
    for d, t, ca, aa, cs, as_ in rows:
        log_decision(conn, run_id="r", decision_date=pd.Timestamp(d), ticker=t,
                     challenger_name=name,
                     challenger_score=cs, challenger_rank_score=cs,
                     challenger_action=ca,
                     actual_score=as_, actual_action=aa)
    conn.commit()


def _populate_forward_returns(conn):
    """Insert fwd returns matching the synthetic decisions."""
    fwd_rows = [
        ("2026-04-15", "NVDA", 100.0, 0.01,  0.062, 0.085),  # CH was right
        ("2026-04-18", "TSLA", 200.0, -0.02, -0.041, -0.06), # CH was right (held)
        ("2026-04-20", "AVGO", 150.0, 0.01,  0.025, 0.04),   # CH right
    ]
    for d, t, c, f1, f5, f20 in fwd_rows:
        conn.execute(
            "INSERT INTO ticker_forward_returns(as_of_date, ticker, close_price, "
            "fwd_1d, fwd_5d, fwd_10d, fwd_20d) VALUES (?,?,?,?,?,?,?)",
            (d, t, c, f1, f5, f5, f20),
        )
    conn.commit()


# ── Loading helpers ───────────────────────────────────────────────────────────

class TestArtifactLoadAuditFix6:
    def test_metadata_takes_precedence_with_warning_on_conflict(self, tmp_path, caplog):
        """Audit fix #6 (2026-04-26): when both metadata.X and top-level X
        exist with DIFFERENT values, prefer metadata (canonical) and warn.
        Pre-fix, the `or` fallback silently dropped the second value."""
        import logging as _logging
        path = tmp_path / "panel-ltr.json"
        path.write_text(json.dumps({
            "trained_date":  "2026-04-26",       # top-level
            "feature_cols":  ["a"],
            "oos_mean_ic":   0.05,                # top-level disagreeing
            "metadata": {
                "oos_mean_ic": 0.04,              # canonical
                "trained_date": "2026-04-26",
            },
        }))
        with caplog.at_level(_logging.WARNING, logger="finalize-challenger"):
            md = fc._load_artifact_metadata(path)
        assert md["oos_mean_ic"] == 0.04   # metadata wins
        assert any("present in BOTH metadata" in r.message for r in caplog.records)


class TestArtifactLoad:
    def test_missing_artifact_marked(self, tmp_path):
        md = fc._load_artifact_metadata(tmp_path / "absent.json")
        assert md["_missing"] is True

    def test_valid_artifact_extracts_metrics(self, tmp_path):
        path = tmp_path / "panel-ltr.json"
        path.write_text(json.dumps({
            "trained_date": "2026-04-26",
            "feature_cols": [f"f{i}" for i in range(28)],
            "panel_shape":  {"rows": 74000, "tickers": 99, "dates": 753},
            "oos_mean_ic":  0.0482,
            "best_iter":    9,
            "metadata":     {"sim_smoke": {"apy": 0.16, "sharpe": 1.4}},
        }))
        md = fc._load_artifact_metadata(path)
        assert md["feature_count"] == 28
        assert md["panel_rows"] == 74000
        assert md["oos_mean_ic"] == 0.0482
        assert md["sim_apy"] == 0.16


# ── Verdict computation ───────────────────────────────────────────────────────

class TestVerdict:
    def test_summary_stats_on_synthetic_window(self, tmp_path):
        conn = _open_db(tmp_path)
        _populate_decisions(conn)
        df = fc._read_decisions(conn, "macro-enabled",
                                pd.Timestamp("2026-04-01"),
                                pd.Timestamp("2026-04-30"))
        stats = fc._summary_stats(df)
        assert stats["n_decisions"] == 10
        assert stats["agreement_rate"] == 0.7   # 7/10
        assert stats["challenger_only_buy"] == 2   # NVDA + AVGO
        assert stats["live_only_buy"] == 1         # TSLA

    def test_top_disagreements_returns_only_disagree_rows(self, tmp_path):
        conn = _open_db(tmp_path)
        _populate_decisions(conn)
        df = fc._read_decisions(conn, "macro-enabled",
                                pd.Timestamp("2026-04-01"),
                                pd.Timestamp("2026-04-30"))
        top = fc._top_disagreements(df, n=10)
        # Only 3 disagreements (NVDA, TSLA, AVGO)
        assert len(top) == 3
        actions = set(zip(top["challenger_action"], top["actual_action"]))
        # No agreement pair should be in there
        assert ("BUY", "BUY") not in actions

    def test_forward_returns_join(self, tmp_path):
        conn = _open_db(tmp_path)
        _populate_decisions(conn)
        _populate_forward_returns(conn)
        df = fc._read_decisions(conn, "macro-enabled",
                                pd.Timestamp("2026-04-01"),
                                pd.Timestamp("2026-04-30"))
        df = fc._attach_forward_returns(conn, df)
        # NVDA / TSLA / AVGO have fwd data; rest are NaN
        nvda_row = df[df["ticker"] == "NVDA"].iloc[0]
        assert nvda_row["fwd_5d"] == pytest.approx(0.062)
        assert df[df["ticker"] == "AAPL"]["fwd_5d"].iloc[0] is None or pd.isna(
            df[df["ticker"] == "AAPL"]["fwd_5d"].iloc[0]
        )


# ── Recommendation heuristic ──────────────────────────────────────────────────

class TestRecommendation:
    def test_high_agreement_emits_safe_recommendation(self):
        rec = fc._heuristic_recommendation(
            {"n_decisions": 100, "agreement_rate": 0.92,
             "challenger_only_buy": 2, "live_only_buy": 1, "score_corr": 0.9},
            pd.DataFrame(),
        )
        assert "✅" in rec or "≥90%" in rec or "very high" in rec

    def test_low_agreement_flags(self):
        rec = fc._heuristic_recommendation(
            {"n_decisions": 100, "agreement_rate": 0.50,
             "challenger_only_buy": 30, "live_only_buy": 25, "score_corr": 0.3},
            pd.DataFrame(),
        )
        assert "🚩" in rec or "<75%" in rec or "low" in rec.lower()

    def test_zero_decisions_warns(self):
        rec = fc._heuristic_recommendation(
            {"n_decisions": 0, "agreement_rate": 0.0,
             "challenger_only_buy": 0, "live_only_buy": 0, "score_corr": None},
            pd.DataFrame(),
        )
        assert "🚫" in rec or "no decisions" in rec.lower()


# ── End-to-end render ─────────────────────────────────────────────────────────

class TestRender:
    def test_render_includes_all_sections(self, tmp_path):
        conn = _open_db(tmp_path)
        _populate_decisions(conn)
        _populate_forward_returns(conn)
        df = fc._attach_forward_returns(conn,
            fc._read_decisions(conn, "macro-enabled",
                               pd.Timestamp("2026-04-01"),
                               pd.Timestamp("2026-04-30")))
        stats = fc._summary_stats(df)
        top = fc._top_disagreements(df, n=10)
        live_md = {"trained_date": "2026-04-26", "feature_count": 28,
                   "oos_mean_ic": 0.0482, "_path": "/tmp/live.json"}
        ch_md = {"trained_date": "2026-04-26", "feature_count": 61,
                 "oos_mean_ic": 0.0393, "_path": "/tmp/ch.json"}
        rec = fc._heuristic_recommendation(stats, top)
        md = fc.render_report(
            strategy="renquant_104", challenger_name="macro-enabled",
            window_start=pd.Timestamp("2026-04-12"),
            window_end=pd.Timestamp("2026-04-26"),
            live_md=live_md, challenger_md=ch_md,
            stats=stats, top_dis=top, recommendation=rec,
        )
        # All required sections present
        assert "## 1. Model parameter comparison" in md
        assert "## 2. Decision statistics" in md
        assert "## 3. Top disagreements" in md
        assert "## 4. Operator recommendation" in md
        # Both models' params shown
        assert "0.0482" in md   # live IC
        assert "0.0393" in md   # challenger IC
        # Decision counts
        assert "10" in md       # n_decisions
        # Top disagreements include actual disagreement tickers
        assert "NVDA" in md or "TSLA" in md or "AVGO" in md
        # Recommendation present
        assert "macro-enabled" in md   # name interpolation


# ── ntfy push ─────────────────────────────────────────────────────────────────

class TestNtfy:
    def test_no_notify_env_skips(self, monkeypatch):
        monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
        # Should not raise even if URL is bogus
        fc.notify_ntfy("title", "body", "renquant-test-topic-DOES-NOT-EXIST")


# ── Main CLI integration ──────────────────────────────────────────────────────

class TestMainCLI:
    def test_returns_2_when_table_missing(self, tmp_path, monkeypatch, capsys):
        """No challenger_decisions table → exit 2 (not crash)."""
        # Build a minimal strategy_dir with a usable config
        sdir = tmp_path / "renquant_test"
        (sdir / "data").mkdir(parents=True)
        (sdir / "artifacts").mkdir(parents=True)
        (sdir / "strategy_config.json").write_text(json.dumps({
            "persistence": {"db_path": str(tmp_path / "runs.db")},
            "acceptance":  {"challenger": {"name": "test-ch"}},
        }))
        # Create empty DB without the table
        sqlite3.connect(str(tmp_path / "runs.db")).close()

        # Patch REPO_ROOT-derived strategy lookup
        monkeypatch.setattr(fc, "REPO_ROOT", tmp_path)
        # The script computes `strategy_dir = REPO_ROOT / "backtesting" / strategy`
        # Make that resolve into our test dir
        (tmp_path / "backtesting").mkdir(exist_ok=True)
        (tmp_path / "backtesting" / "renquant_test").symlink_to(sdir)

        rc = fc.main.__wrapped__() if hasattr(fc.main, "__wrapped__") else None
        # Easier: invoke main() with sys.argv override
        monkeypatch.setattr(sys, "argv", [
            "finalize_challenger.py",
            "--strategy", "renquant_test",
            "--challenger-name", "test-ch",
            "--no-ntfy",
        ])
        rc = fc.main()
        assert rc == 2
