"""Regression guards for the weekly WF gate CLI contract."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_wf_gate_accepts_weekly_strict_flag() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--strict"' in src, (
        "weekly_wf_promote.sh passes --strict; run_wf_gate.py must accept it"
    )


def test_wf_gate_defaults_to_walkforward_sim_config() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert 'default="strategy_config.sim_wl200.json"' in src


def test_wf_gate_sim_cuts_do_not_use_live_static_path_or_persistence() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--no-compare"' in src
    assert '"--no-persist"' in src
    assert '"--skip-preflight"' in src
    assert "returncode" in src and "sim cuts failed execution" in src


def test_wf_gate_sanity_reindexes_missing_optional_features() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "val.reindex(columns=feat_cols, fill_value=0).fillna(0)" in src


def test_run_sim_disables_live_freshness_by_default_for_historical_sims() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert 'data_freshness["enabled"] = False' in src
    assert "historical sim" in src
