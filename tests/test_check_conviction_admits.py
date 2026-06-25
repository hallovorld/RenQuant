"""Tests for the reusable conviction-admits deploy guard."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cca", Path(__file__).resolve().parent.parent / "scripts" / "check_conviction_admits.py")
cca = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cca)


def test_count_admits_absolute_vs_demean():
    ers = [0.051, 0.034, 0.033, 0.0, -0.01, 0.006]  # mean ~0.019
    # absolute floor 0.03: 0.051/0.034/0.033 clear → 3
    assert cca.count_admits(ers, {"enabled": True, "mu_floor": 0.03})["admits"] == 3
    # demean (full mean): the footgun-fixed behaviour — only the top clears mean+0.03
    dem = cca.count_admits(ers, {"enabled": True, "mu_floor": 0.03, "demean_cross_sectional": True})
    assert dem["demean"] is True
    assert dem["admits"] == 1            # only 0.051 > mean(0.019)+0.03=0.049
    # disabled gate → everything "admits"
    assert cca.count_admits(ers, {"enabled": False})["admits"] == len(ers)


def _db(tmp_path, ers, rid="2026-06-24-live-aaa"):
    p = tmp_path / "runs.db"; con = sqlite3.connect(str(p))
    con.execute("create table candidate_scores (run_id text, ticker text, expected_return real)")
    con.executemany("insert into candidate_scores values (?,?,?)",
                    [(rid, f"T{i}", e) for i, e in enumerate(ers)])
    con.commit(); con.close(); return p


def _cfg(tmp_path, gate):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"ranking": {"panel_scoring": {"conviction_gate": gate}}}))
    return p


def test_evaluate_would_not_buy_triggers(tmp_path):
    db = _db(tmp_path, [0.034, 0.033, 0.02, 0.0])      # demean over mean → ~0 admits
    cfg = _cfg(tmp_path, {"enabled": True, "mu_floor": 0.03, "demean_cross_sectional": True})
    res = cca.evaluate(1, cfg, db)
    assert res["status"] == "WOULD_NOT_BUY"


def test_evaluate_ok_when_admits(tmp_path):
    db = _db(tmp_path, [0.10, 0.034, 0.02, 0.0, -0.01])  # one big winner clears mean+floor
    cfg = _cfg(tmp_path, {"enabled": True, "mu_floor": 0.03, "demean_cross_sectional": True})
    res = cca.evaluate(1, cfg, db)
    assert res["status"] == "OK" and res["admits"] >= 1


def test_cannot_evaluate_empty_db(tmp_path):
    db = _db(tmp_path, [])
    cfg = _cfg(tmp_path, {"enabled": True, "mu_floor": 0.03})
    assert cca.evaluate(1, cfg, db)["status"] == "CANNOT_EVALUATE"
