"""DRPH tests — deterministic replay & parity harness substrate.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §IV + S2
item 5; prototype self-proofs: scripts/engineering/drph_core.py (PR #112).

Invariants pinned:
- identity verifies; sub-precision float wobble (≤1e-12) never diffs
- any real behavior change diffs AND is localized to its path
- corpus integrity check catches tampering before a verify is trusted
- capture→verify round-trip against a synthetic runs DB
- the committed golden case (2026-06-11 false-BEAR day) is intact
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.drph import PRECISION, ReplayCase, canonical_json, run_fingerprint  # noqa: E402

GOLDEN_CASE = REPO / "tests" / "drph_corpus" / "2026-06-11_false_bear"


def _decisions() -> dict:
    return {
        "run_fingerprint": run_fingerprint(
            config_sha="c1", panel_sha="p1", state_sha="s1",
            artifact_shas={"primary": "a1", "calibrator": "a2"},
            pin_digest="d1", env_sha="e1"),
        "book": {"regime": "CHOPPY", "buy_blocked": False,
                 "vol_5d": 0.26123456789},
        "tickers": [{"ticker": "MU", "selected": 1, "blocked_by": None,
                     "rank_score": 0.32513821903305495}],
    }


class TestCanonicalization:

    def test_identity(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={"state": {"h": 1}}, expected_decisions=_decisions())
        ok, diffs = case.verify(_decisions())
        assert ok and diffs == []

    def test_sub_precision_wobble_never_diffs(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={}, expected_decisions=_decisions())
        wobbled = json.loads(canonical_json(_decisions()))
        wobbled["book"]["vol_5d"] += 1e-12
        ok, _ = case.verify(wobbled)
        assert ok, f"wobble below 1e-{PRECISION} must not diff"

    def test_real_change_diffs_and_localizes(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={}, expected_decisions=_decisions())
        changed = json.loads(canonical_json(_decisions()))
        changed["tickers"][0]["selected"] = 0
        ok, diffs = case.verify(changed)
        assert not ok
        assert any("tickers[0].selected" in d for d in diffs), diffs

    def test_key_order_irrelevant(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={}, expected_decisions=_decisions())
        reordered = {k: _decisions()[k]
                     for k in reversed(list(_decisions()))}
        ok, _ = case.verify(reordered)
        assert ok

    def test_fingerprint_sorts_artifacts(self):
        a = run_fingerprint(config_sha="c", panel_sha="p", state_sha="s",
                            artifact_shas={"z": "1", "a": "2"},
                            pin_digest="d", env_sha="e")
        assert list(a["artifact_shas"]) == ["a", "z"]


class TestCorpusIntegrity:

    def test_tampered_expected_caught(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={"state": {"h": 1}}, expected_decisions=_decisions())
        f = tmp_path / "c" / "expected" / "decisions.json"
        f.write_text(f.read_text().replace("CHOPPY", "BEAR"))
        problems = case.check_integrity()
        assert problems and "expected" in problems[0]

    def test_tampered_input_caught(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={"state": {"h": 1}}, expected_decisions=_decisions())
        (tmp_path / "c" / "inputs" / "state.json").write_text('{"h":2}')
        problems = case.check_integrity()
        assert problems and "state" in problems[0]

    def test_clean_case_passes(self, tmp_path):
        case = ReplayCase(tmp_path / "c")
        case.write(inputs={"state": {"h": 1}}, expected_decisions=_decisions())
        assert case.check_integrity() == []


@pytest.fixture()
def synthetic_db(tmp_path):
    db = tmp_path / "runs.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE pipeline_runs (
        run_id TEXT PRIMARY KEY, run_date DATE, run_type TEXT, regime TEXT,
        confidence REAL, buy_blocked INTEGER, skip_buys INTEGER,
        bear_only INTEGER, n_candidates INTEGER, n_exits INTEGER,
        n_buys INTEGER, counters_json TEXT, run_bundle_json TEXT)""")
    conn.execute("""CREATE TABLE ticker_daily_state (
        run_id TEXT, date TEXT, ticker TEXT, regime TEXT, confidence REAL,
        in_watchlist INTEGER, in_universe INTEGER, has_position INTEGER,
        position_qty REAL, model_action TEXT, panel_score REAL,
        rank_score REAL, expected_return REAL, kelly_target_pct REAL,
        mu REAL, sigma REAL, in_candidates INTEGER, selected INTEGER,
        blocked_by TEXT, sector TEXT, qp_delta_w REAL, qp_target_w REAL,
        qp_status TEXT, model_admission_ok INTEGER,
        model_admission_reason TEXT, active_scorer TEXT)""")
    conn.execute("""CREATE TABLE live_state_snapshots (
        run_id TEXT PRIMARY KEY, state_json TEXT)""")
    conn.execute(
        "INSERT INTO pipeline_runs VALUES ('r1','2026-06-11','live','BEAR',"
        "0.9,1,0,1,0,2,0,'{\"panel_vetoed\": 71}','{\"config_hash\": \"x\"}')")
    conn.execute(
        "INSERT INTO ticker_daily_state VALUES ('r1','2026-06-11','MU','BEAR',"
        "0.9,1,1,1,3,'HOLD',0.5,0.3,0.01,0.02,0.1,0.2,1,0,'bear_only',"
        "'tech',0.0,0.0,'ok',1,'',\'patchtst\')")
    conn.execute(
        "INSERT INTO live_state_snapshots VALUES ('r1','{\"regime\": \"BEAR\"}')")
    conn.commit()
    conn.close()
    return db


class TestCaptureCli:

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "drph_capture.py"), *args],
            capture_output=True, text=True)

    def test_capture_then_verify_round_trip(self, synthetic_db, tmp_path):
        out = tmp_path / "case"
        r = self._run("capture", "--db", str(synthetic_db),
                      "--run-id", "r1", "--out", str(out))
        assert r.returncode == 0, r.stderr
        r = self._run("verify", "--db", str(synthetic_db),
                      "--run-id", "r1", "--case", str(out))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PARITY OK" in r.stdout

    def test_verify_fails_on_divergence(self, synthetic_db, tmp_path):
        out = tmp_path / "case"
        assert self._run("capture", "--db", str(synthetic_db),
                         "--run-id", "r1", "--out", str(out)).returncode == 0
        conn = sqlite3.connect(synthetic_db)
        conn.execute("UPDATE ticker_daily_state SET selected=1, "
                     "blocked_by=NULL WHERE ticker='MU'")
        conn.commit()
        conn.close()
        r = self._run("verify", "--db", str(synthetic_db),
                      "--run-id", "r1", "--case", str(out))
        assert r.returncode == 1
        assert "PARITY FAILED" in r.stdout
        assert "selected" in r.stdout

    def test_missing_run_id_fail_closed(self, synthetic_db, tmp_path):
        r = self._run("capture", "--db", str(synthetic_db),
                      "--run-id", "nope", "--out", str(tmp_path / "c"))
        assert r.returncode != 0

    def test_capture_refuses_without_provenance(self, synthetic_db, tmp_path):
        conn = sqlite3.connect(synthetic_db)
        conn.execute("UPDATE pipeline_runs SET run_bundle_json=NULL")
        conn.execute("DELETE FROM live_state_snapshots")
        conn.commit()
        conn.close()
        r = self._run("capture", "--db", str(synthetic_db),
                      "--run-id", "r1", "--out", str(tmp_path / "c"))
        assert r.returncode != 0
        assert "provenance" in (r.stdout + r.stderr)


class TestGoldenCorpus:
    """The committed 2026-06-11 false-BEAR case must stay intact."""

    def test_golden_case_integrity(self):
        assert GOLDEN_CASE.exists(), "golden corpus case missing"
        case = ReplayCase(GOLDEN_CASE)
        assert case.check_integrity() == []

    def test_golden_case_is_the_false_bear_day(self):
        expected = ReplayCase(GOLDEN_CASE).expected()
        assert expected["book"]["run_date"] == "2026-06-11"
        assert expected["book"]["regime"] == "BEAR"
        assert len(expected["tickers"]) == 142


class TestCorpusInventory:
    """Every committed case must be intact; sim cases must self-identify."""

    CORPUS = REPO / "tests" / "drph_corpus"

    def test_all_cases_intact(self):
        cases = [d for d in self.CORPUS.iterdir() if d.is_dir()]
        assert len(cases) >= 5
        for case_dir in cases:
            problems = ReplayCase(case_dir).check_integrity()
            assert problems == [], f"{case_dir.name}: {problems}"

    def test_regime_instability_trio_present(self):
        # The 2026-06-11 trio: three live runs, three different regimes.
        import json as _json
        regimes = {}
        for name in ("2026-06-11_false_bear", "2026-06-11_live_2f0ce396",
                     "2026-06-11_live_fbb8c140"):
            exp = ReplayCase(self.CORPUS / name).expected()
            regimes[name] = exp["book"]["regime"]
        assert sorted(regimes.values()) == ["BEAR", "BULL_CALM", "CHOPPY"]
