"""promote() → MLflow champion alias wiring (eng plan S3-8 / errata B).

The WF gate is the only alias-mover; promote() is its sole exit, so the
alias move lives there. Registry failure must never roll back a
gate-passed promote (loud warning, file swap stands).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

import kernel.model_acceptance as ma  # noqa: E402

# Canonical passing fixture lives in test_promote_wf_gate; bare pytest
# does not put tests/ on sys.path (rootdir conftest absent) — import by
# path so this module is self-validating (the #322 lesson).
sys.path.insert(0, str(REPO / "tests"))
from test_promote_wf_gate import _wf_meta  # noqa: E402


def _staging(tmp_path) -> Path:
    p = tmp_path / "panel-ltr.alpha158_fund.staging.json"
    p.write_text(json.dumps({
        "kind": "xgb", "feature_cols": ["f1"],
        "metadata": {"wf_gate_metadata": _wf_meta(passed=True)},
    }))
    return p


class TestChampionWiring:

    def test_promote_registers_and_moves_champion(self, tmp_path, monkeypatch):
        calls = {}

        def _fake_register(active_path, data):
            calls["path"] = Path(active_path)
            calls["wf"] = (data.get("metadata") or {}).get("wf_gate_metadata")

        monkeypatch.setattr(ma, "_register_champion", _fake_register)
        active = tmp_path / "panel-ltr.alpha158_fund.json"
        ma.promote(_staging(tmp_path), active)
        assert calls["path"] == active
        assert calls["wf"]["passed"] is True
        assert calls["wf"]["trade_monotonicity"]["passed"] is True
        assert calls["wf"]["trade_monotonicity"]["regimes"][0]["eligible"] is True

    def test_registry_failure_does_not_roll_back_promote(self, tmp_path,
                                                         monkeypatch, caplog):
        import logging

        def _boom(active_path, data):
            raise RuntimeError("mlflow down")

        monkeypatch.setattr(ma, "_register_champion", _boom)
        active = tmp_path / "panel-ltr.alpha158_fund.json"
        with caplog.at_level(logging.WARNING):
            ma.promote(_staging(tmp_path), active)
        assert active.exists(), "file swap is the trading truth"
        assert any("champion registration failed" in r.message
                   for r in caplog.records)

    def test_register_champion_end_to_end(self, tmp_path, monkeypatch):
        # Real registry path against a tmp MLflow file backend.
        from kernel.registry import init_tracking, resolve_model_by_alias

        uri = f"file:{tmp_path / 'mlruns'}"
        init_tracking(uri)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        active = tmp_path / "panel-ltr.testmodel.json"
        active.write_text(json.dumps({"kind": "xgb", "feature_cols": ["f1"]}))
        ma._register_champion(active, json.loads(active.read_text()))
        got = resolve_model_by_alias("panel-ltr.testmodel", "champion")
        assert got["version"] == "1"
