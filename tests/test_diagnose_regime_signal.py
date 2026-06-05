"""Unit tests for the per-regime signal diagnostic."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    path = REPO / "scripts" / "diagnose_regime_signal.py"
    spec = importlib.util.spec_from_file_location("diagnose_regime_signal", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _seed(db: Path):
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE ticker_forward_returns (as_of_date TEXT, ticker TEXT, fwd_20d REAL)")
    conn.execute("CREATE TABLE pipeline_runs (run_date TEXT, run_type TEXT, regime TEXT)")
    # two BULL_CALM dates, 5 tickers each with a clear dispersion
    for d in ("2024-01-01", "2024-01-31"):
        conn.execute("INSERT INTO pipeline_runs VALUES (?,?,?)", (d, "sim", "BULL_CALM"))
        for i, t in enumerate(["A", "B", "C", "D", "E"]):
            conn.execute("INSERT INTO ticker_forward_returns VALUES (?,?,?)",
                         (d, t, 0.01 * (i - 2)))  # -0.02..+0.02
    conn.commit(); conn.close()


def test_dispersion_and_horizon_validation(tmp_path):
    mod = _load()
    db = tmp_path / "sim.db"
    _seed(db)
    res = mod.run_diagnostic(db, 20)
    assert res["horizon_days"] == 20
    assert "BULL_CALM" in res["dispersion"]
    d = res["dispersion"]["BULL_CALM"]
    assert d["n_dates"] == 2
    assert d["dispersion_std"] > 0  # clear cross-sectional spread


def test_invalid_horizon_raises(tmp_path):
    mod = _load()
    import pytest
    with pytest.raises(ValueError, match="not in"):
        mod._fwd_col(7)


def test_missing_db_exits(tmp_path):
    mod = _load()
    import pytest
    with pytest.raises(SystemExit):
        mod.run_diagnostic(tmp_path / "nope.db", 20)


def test_text_render_has_both_tables(tmp_path):
    mod = _load()
    db = tmp_path / "sim.db"
    _seed(db)
    txt = mod.render_text(mod.run_diagnostic(db, 20))
    assert "disp(std)" in txt
    assert "momentum_IC" in txt
