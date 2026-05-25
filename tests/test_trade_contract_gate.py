from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from trade_contracts import evaluate_trade_contract  # noqa: E402


class TestTradeContractRegressionGuard:
    """AUDIT REGRESSION GUARD: WF ledgers must be scientifically auditable."""

    def test_fails_when_qp_trade_is_missing_entry_mu_sigma(self) -> None:
        df = pd.DataFrame([
            {
                "status": "closed",
                "ticker": "AAPL",
                "entry_mu": None,
                "entry_sigma": None,
                "exit_regime": "BULL_CALM",
                "exit_stop_loss_pct": 0.15,
                "exit_max_single_day_loss_pct": 0.0,
                "exit_sdl_n_sigma": 3.0,
                "exit_trailing_stop_trigger_pct": 0.12,
                "exit_trailing_stop_trail_pct": 0.25,
                "exit_max_hold_days": 500,
            }
        ])

        report = evaluate_trade_contract(
            df, require_entry_mu=True, require_entry_sigma=True,
        )

        assert report.passed is False
        assert report.evidence["missing_entry_mu"] == 1
        assert report.evidence["missing_entry_sigma"] == 1

    def test_fails_when_qp_trade_is_missing_expected_return_horizon(self) -> None:
        df = pd.DataFrame([
            {
                "status": "closed",
                "ticker": "AAPL",
                "entry_mu": 0.02,
                "entry_sigma": 0.18,
                "entry_expected_return": None,
                "entry_expected_return_horizon_days": None,
                "entry_mu_horizon_days": None,
                "exit_regime": "BULL_CALM",
                "exit_stop_loss_pct": 0.15,
                "exit_max_single_day_loss_pct": 0.0,
                "exit_sdl_n_sigma": 3.0,
                "exit_trailing_stop_trigger_pct": 0.12,
                "exit_trailing_stop_trail_pct": 0.25,
                "exit_max_hold_days": 500,
            }
        ])

        report = evaluate_trade_contract(
            df,
            require_entry_mu=True,
            require_entry_sigma=True,
            require_entry_expected_return=True,
            require_entry_horizon=True,
        )

        assert report.passed is False
        assert report.evidence["missing_entry_expected_return"] == 1
        assert report.evidence["missing_entry_expected_return_horizon_days"] == 1
        assert report.evidence["missing_entry_mu_horizon_days"] == 1

    def test_passes_when_entry_model_and_exit_policy_fields_exist(self) -> None:
        df = pd.DataFrame([
            {
                "status": "closed",
                "ticker": "AAPL",
                "entry_mu": 0.02,
                "entry_mu_horizon_days": 60,
                "entry_sigma": 0.18,
                "entry_expected_return": 0.02,
                "entry_expected_return_horizon_days": 60,
                "exit_regime": "BULL_CALM",
                "exit_stop_loss_pct": 0.15,
                "exit_max_single_day_loss_pct": 0.0,
                "exit_sdl_n_sigma": 3.0,
                "exit_trailing_stop_trigger_pct": 0.12,
                "exit_trailing_stop_trail_pct": 0.25,
                "exit_max_hold_days": 500,
            }
        ])

        report = evaluate_trade_contract(
            df,
            require_entry_mu=True,
            require_entry_sigma=True,
            require_entry_expected_return=True,
            require_entry_horizon=True,
        )

        assert report.passed is True

    def test_benchmark_sleeve_entry_does_not_require_alpha_mu_sigma(self) -> None:
        df = pd.DataFrame([
            {
                "status": "closed",
                "ticker": "SPY",
                "entry_order_type": "BENCHMARK_SLEEVE_BUY",
                "entry_source_job": "BenchmarkSleeveJob",
                "entry_mu": None,
                "entry_sigma": None,
                "exit_regime": "BULL_CALM",
                "exit_stop_loss_pct": 0.15,
                "exit_max_single_day_loss_pct": 0.0,
                "exit_sdl_n_sigma": 3.0,
                "exit_trailing_stop_trigger_pct": 0.12,
                "exit_trailing_stop_trail_pct": 0.25,
                "exit_max_hold_days": 500,
            }
        ])

        report = evaluate_trade_contract(
            df, require_entry_mu=True, require_entry_sigma=True,
        )

        assert report.passed is True
        assert report.evidence["n_alpha_entry_auditable"] == 0
        assert report.evidence["n_benchmark_sleeve_auditable"] == 1
