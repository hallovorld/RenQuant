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


def test_wf_gate_supports_bounded_cut_parallelism() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--jobs"' in src
    assert "ThreadPoolExecutor" in src
    assert "wf_jobs" in src


def test_wf_gate_uses_current_python_environment_for_sim_cuts() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "PYTHON = sys.executable" in src
    assert '"/Users/renhao/miniconda3/envs/renquant/bin/python"' not in src


def test_wf_gate_stamps_benchmark_and_regime_context() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "cut_market_context" in src
    assert "spy_sharpe" in src
    assert "strategy_minus_spy_sharpe_mean" in src
    assert "n_cuts_beat_spy_sharpe" in src
    assert "hmm_regime_counts" in src
    assert "spy_grid_regime_counts" in src


def test_wf_gate_refuses_to_stamp_manifest_as_candidate_artifact() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "inspect_artifact_usage" in src
    assert "candidate_artifact_used" in src
    assert '"--allow-manifest-scope"' not in src
    assert 'artifact_scope_ok = bool(artifact_usage.get("candidate_artifact_used"))' in src
    assert "candidate artifact was not evaluated" in src


def test_model_acceptance_rejects_non_candidate_wf_metadata() -> None:
    src = (REPO / "backtesting/renquant_104/kernel/model_acceptance.py").read_text()
    assert 'wf.get("candidate_artifact_used") is False' in src
    assert "WF sim did not" in src
    assert "evaluate the candidate artifact" in src


def test_run_sim_disables_live_freshness_by_default_for_historical_sims() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert 'data_freshness["enabled"] = False' in src
    assert "historical sim" in src
