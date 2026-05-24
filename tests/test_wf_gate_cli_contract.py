"""Regression guards for the weekly WF gate CLI contract."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_wf_gate_recipe_fingerprint_includes_feature_space_contract() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")
    base = {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["alpha", "fund"],
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {"objective": "rank:pairwise", "nthread": 14},
    }
    old = dict(base, feature_norm_kind=["global_z", "legacy_full_z"])
    new = dict(base, feature_norm_kind=["global_z", "robust_z"])

    assert mod._recipe_fingerprint(old) != mod._recipe_fingerprint(new)


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


def test_wf_gate_zero_trade_cuts_fail_without_traceback(monkeypatch, tmp_path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    def fake_run_sim_cut(_cfg, start, end, trace_dir):
        return {
            "start": start,
            "end": end,
            "sharpe": float("nan"),
            "apy": 0.0,
            "market_context": {"spy_sharpe": 1.0, "spy_apy": 0.10},
            "trade_trace_summary": {"n_buys": 0, "n_sells": 0},
            "trace_paths": {
                "round_trips_csv": str(tmp_path / f"{start}_{end}.csv"),
            },
            "returncode": 0,
        }

    monkeypatch.setattr(mod, "run_sim_cut", fake_run_sim_cut)

    result = mod.run_walk_forward("dummy_config.json", jobs=1, trace_dir=tmp_path)

    assert result["passed"] is False
    assert "zero trades across all WF cuts" in result["reason"]
    assert result["wf_3cut_apy_mean"] == 0.0
    assert result["n_cuts_beat_spy_sharpe"] == 0


def test_wf_trade_gates_handle_empty_round_trip_csv(tmp_path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")
    empty = tmp_path / "empty.round_trips.csv"
    empty.write_text("\n")
    wf_result = {"cuts": [{"trace_paths": {"round_trips_csv": str(empty)}}]}

    contract = mod.run_trade_contract_gate(wf_result, {})
    monotonicity = mod.run_trade_monotonicity_gate(wf_result)

    assert contract["passed"] is False
    assert "no round-trip" in contract["reason"]
    assert monotonicity["passed"] is False
    assert "no round-trip" in monotonicity["reason"]


def test_wf_gate_skip_flags_are_not_acceptance_passes() -> None:
    """Skipped required gates can be diagnostic-only, never promotable PASS."""
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")
    args = SimpleNamespace(
        skip_wf=True,
        skip_sanity=True,
        skip_trade_gates=True,
        skip_config_parity=True,
        no_trade_trace=True,
    )

    reasons = mod._required_validation_skip_reasons(args)
    overall = mod._compute_overall_pass(
        wf_result={"passed": True},
        sanity_result={"passed": True},
        trade_contract_result={"passed": True},
        trade_gate_result={"passed": True},
        validation_scope_ok=True,
        parity_result={"passed": True},
        skipped_required_gates=reasons,
    )

    assert set(reasons) == {
        "walk_forward_skipped",
        "sanity_skipped",
        "trade_gates_skipped",
        "config_parity_skipped",
        "trade_trace_disabled",
    }
    assert overall is False


def test_wf_gate_sanity_reindexes_missing_optional_features() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert "transform_feature_frame(" in src
    assert 'source_space="panel"' in src


def test_wf_gate_sanity_unavailable_fails_closed() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"passed": False' in src
    assert "panel missing — sanity unavailable" in src
    assert "sanity not implemented for this kind" in src
    assert "prediction failed: {exc}" in src


def test_wf_gate_sanity_records_method_and_shift_diagnostics() -> None:
    src = (REPO / "scripts/run_wf_gate.py").read_text()
    assert '"sanity_method": "existing_model_label_diagnostics"' in src
    assert '"sanity_method":       sanity_result.get("sanity_method")' in src
    assert "placebo_shift_diagnostics" in src
    assert "abs_ratio_to_aligned_real" in src
    assert "sanity_placebo_aligned_real_ic" in src


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
    assert '"benchmark_by_dominant_regime": wf_result.get("benchmark_by_dominant_regime")' in src
    assert '"regime_benchmark_failures": wf_result.get("regime_benchmark_failures")' in src
    assert '"performance_tax_basis_counts": wf_result.get("performance_tax_basis_counts")' in src
    assert '"sanity_regime_ic":    sanity_result.get("sanity_regime_ic")' in src
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


def test_wf_gate_counts_performance_tax_basis() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    rows = [
        {"performance_tax_basis": "annual_net"},
        {"performance_tax_basis": "annual_net"},
        {"performance_tax_basis": "event_level"},
        {"performance_tax_basis": None},
    ]

    assert mod._value_counts(rows, "performance_tax_basis") == {
        "annual_net": 2,
        "event_level": 1,
    }


def test_wf_gate_prefers_exact_annual_net_metrics_from_trace(monkeypatch, tmp_path) -> None:
    """AUDIT REGRESSION GUARD: WF metrics come from machine-readable trace.

    Pre-fix, run_wf_gate parsed rounded console strings like ``Sharpe=+0.10``
    and ``APY: 1.0%``. When the trace JSON carries annual-net metrics, the
    promotion gate must use those exact values and retain event-level fields
    for audit.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    class _Proc:
        returncode = 0
        stdout = "Risk: Sharpe=+0.10\nFinal value: $101,000  |  Return: 1.0%  |  APY: 1.0%\n"
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        del capture_output, text, timeout
        eq_path = Path(cmd[cmd.index("--equity-json") + 1])
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        eq_path.write_text(json.dumps({
            "apy": 0.0100,
            "sharpe": 0.1000,
            "annual_net_apy": 0.123456,
            "annual_net_sharpe": 0.654321,
            "event_level_apy": 0.0100,
            "event_level_sharpe": 0.1000,
            "event_level_tax_debited": 50.0,
            "annual_net_tax_estimate": 8.0,
            "tax_overstatement_vs_annual_net": 42.0,
            "equity": {"2024-01-02": 100000.0},
        }))
        return _Proc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "cut_market_context", lambda start, end: {
        "spy_sharpe": 0.20,
        "spy_apy": 0.02,
    })

    result = mod.run_sim_cut(
        "unit_config.json",
        "2024-01-02",
        "2024-12-31",
        tmp_path,
    )

    assert result["sharpe"] == 0.654321
    assert result["apy"] == 0.123456
    assert result["event_level_sharpe"] == 0.1000
    assert result["event_level_apy"] == 0.0100
    assert result["event_level_tax_debited"] == 50.0
    assert result["annual_net_tax_estimate"] == 8.0
    assert result["performance_tax_basis"] == "annual_net"
    assert result["tax_overstatement_vs_annual_net"] == 42.0


def test_wf_gate_trace_dir_repo_relative_path_is_not_double_prefixed() -> None:
    """Operators often pass repo-relative trace dirs from automation wrappers."""
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    trace_dir = mod._resolve_trace_dir_arg(
        "backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/unit"
    )

    assert trace_dir == (
        REPO / "backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/unit"
    )
    assert "backtesting/renquant_104/backtesting/renquant_104" not in str(trace_dir)


def test_wf_gate_trace_dir_artifacts_path_stays_strategy_relative() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    mod = importlib.import_module("run_wf_gate")

    trace_dir = mod._resolve_trace_dir_arg(
        "artifacts/diagnostics/wf_trade_traces/unit"
    )

    assert trace_dir == (
        REPO / "backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/unit"
    )


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
    helper = (
        REPO / "backtesting/renquant_104/kernel/trade_events.py"
    ).read_text()
    assert "build_buy_trade_event(" in src
    assert 'default_regime=getattr(ctx, "regime", None)' in src
    assert 'order.get("regime", default_regime)' in helper
