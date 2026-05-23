"""Regression guards for the weekly WF gate CLI contract."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_wf_gate_accepts_weekly_strict_flag() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--strict"' in src, (
        "weekly_wf_promote.sh passes --strict; run_wf_gate.py must accept it"
    )


def test_wf_gate_defaults_to_manifest_recipe_config() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert 'default="strategy_config.sim_wl200.json"' in src
    assert "Manifest configs validate the candidate" in src


def test_wf_gate_sim_cuts_do_not_use_live_static_path_or_persistence() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--no-compare"' in src
    assert '"--no-persist"' in src
    assert '"--skip-preflight"' in src
    assert "returncode" in src and "sim cuts failed execution" in src


def test_wf_gate_persists_trade_trace_by_default() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--trace-dir"' in src
    assert '"--no-trade-trace"' in src
    assert "wf_trade_traces" in src
    assert '"--trade-log-json"' in src
    assert '"--round-trips-csv"' in src
    assert "wf_trade_trace_dir" in src


def test_wf_gate_runs_qp_contract_and_trade_monotonicity_gates() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "validate_qp_contract_config" in src
    assert "run_trade_contract_gate" in src
    assert "trade_contract" in src
    assert "run_trade_monotonicity_gate" in src
    assert "trade_monotonicity" in src
    assert '"--skip-trade-gates"' in src


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


def test_wf_gate_bootstraps_repo_and_strategy_import_paths() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "for _p in (REPO, STRATEGY_DIR):" in src
    assert "sys.path.insert(0, _s)" in src


def test_wf_gate_stamps_benchmark_and_regime_context() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "cut_market_context" in src
    assert "spy_sharpe" in src
    assert "strategy_minus_spy_sharpe_mean" in src
    assert "n_cuts_beat_spy_sharpe" in src
    assert "hmm_regime_counts" in src
    assert "spy_grid_regime_counts" in src
    assert "trade_buy_regime_counts_total" in src
    assert "trade_sell_exit_reason_counts_total" in src


def test_wf_gate_trade_trace_summary_uses_production_decision_regimes(tmp_path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")
    trade_path = tmp_path / "trades.json"
    trade_path.write_text(json.dumps([
        {
            "action": "buy",
            "ticker": "AAPL",
            "regime": "BULL_CALM",
            "source_job": "JointPortfolioQPJob",
            "mu": 0.03,
            "sigma": 0.12,
        },
        {
            "action": "buy",
            "ticker": "MSFT",
            "regime": "CHOPPY",
            "source_job": "TopUpJob",
            "mu": None,
            "sigma": None,
        },
        {
            "action": "sell",
            "ticker": "AAPL",
            "regime": "BEAR",
            "source_job": "TickerSellJob",
            "exit_reason": "stop_loss",
        },
    ]))

    summary = mod._trade_trace_summary({"trade_json": str(trade_path)})

    assert summary["n_buys"] == 2
    assert summary["n_sells"] == 1
    assert summary["buy_regime_counts"] == {"BULL_CALM": 1, "CHOPPY": 1}
    assert summary["sell_regime_counts"] == {"BEAR": 1}
    assert summary["buy_source_counts"] == {"JointPortfolioQPJob": 1, "TopUpJob": 1}
    assert summary["sell_exit_reason_counts"] == {"stop_loss": 1}
    assert summary["buy_missing_mu"] == 1
    assert summary["buy_missing_sigma"] == 1


def test_wf_gate_rejects_positive_sharpe_when_all_cuts_lag_spy(monkeypatch) -> None:
    """Absolute Sharpe is not enough; WF acceptance must compare to SPY.

    Pre-fix, three cuts with Sharpe=+0.60 passed because mean Sharpe >= 0.40
    and all cuts were positive, even though each cut lost to SPY. That is the
    exact benchmark-blind failure mode the 2026-05-23 audit surfaced.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    fake_rows = [
        {
            "start": start,
            "end": end,
            "sharpe": 0.60,
            "apy": 0.08,
            "returncode": 0,
            "dominant_spy_grid_regime": "BULL_CALM",
            "market_context": {"spy_sharpe": 0.90, "spy_apy": 0.18},
            "trade_trace_summary": {"buy_regime_counts": {"BULL_CALM": 1}},
        }
        for start, end in mod.CUTS
    ]

    def fake_run_sim_cut(strategy_config, start, end, trace_dir=None):
        del strategy_config, trace_dir
        for row in fake_rows:
            if row["start"] == start and row["end"] == end:
                return dict(row)
        raise AssertionError((start, end))

    monkeypatch.setattr(mod, "run_sim_cut", fake_run_sim_cut)

    result = mod.run_walk_forward("unit_config.json", jobs=1)

    assert result["wf_3cut_sharpe_mean"] == 0.60
    assert result["n_positive_cuts"] == 3
    assert result["n_cuts_beat_spy_sharpe"] == 0
    assert result["passed"] is False
    assert result["benchmark_by_dominant_regime"]["BULL_CALM"]["n_cuts"] == 3
    assert result["regime_benchmark_failures"] == ["BULL_CALM"]
    assert "SPY" in result["reason"]


def test_wf_gate_refuses_to_stamp_manifest_as_candidate_artifact() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "inspect_artifact_usage" in src
    assert "candidate_artifact_used" in src
    assert '"--allow-manifest-scope"' not in src
    assert "recipe_validated" in src
    assert "manifest recipe mismatch" in src
    assert "no matching manifest recipe was validated" in src


def test_model_acceptance_rejects_non_candidate_wf_metadata() -> None:
    src = (REPO / "backtesting/renquant_104/kernel/model_acceptance.py").read_text()
    assert 'wf.get("candidate_artifact_used") is False and wf.get("recipe_validated") is not True' in src
    assert "not validate" in src
    assert "matching manifest recipe" in src


def test_wf_gate_has_recipe_fingerprint_contract() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "def _recipe_projection" in src
    assert "def _recipe_fingerprint" in src
    assert "candidate_recipe_fingerprint" in src
    assert "missing_features_vs_candidate" in src


def test_wf_gate_can_derive_prod_semantic_config() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"--derive-config-from-prod"' in src
    assert "build_wf_config_from_prod" in src
    assert "wf_eval_configs" in src


def test_run_sim_disables_live_freshness_by_default_for_historical_sims() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert 'data_freshness["enabled"] = False' in src
    assert "historical sim" in src


def test_run_sim_can_dump_trade_forensics() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert '"--trade-log-json"' in src
    assert '"--trade-log-csv"' in src
    assert '"--round-trips-csv"' in src
    assert '"--trade-report-md"' in src
    assert "write_trade_outputs" in src
    assert "end_prices" in src


def test_run_sim_enforces_qp_contract_before_simulation() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert "validate_qp_contract_config" in src
    assert '"--allow-raw-qp-mu"' in src
    assert "sys.exit(3)" in src


def test_sim_buy_trade_log_falls_back_to_context_regime() -> None:
    src = (REPO / "backtesting/renquant_104/adapters/sim.py").read_text()
    assert 'order.get("regime") or getattr(ctx, "regime", None)' in src
