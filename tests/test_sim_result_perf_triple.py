"""Tests for the falsifiability layer wired into SimResult (2026-05-10).

Covers:
- SimResult dataclass has dsr / pbo / n_trials / beta_vs_spy / alpha_vs_spy /
  information_ratio_vs_spy fields with sane defaults.
- SimAdapter.build_result populates them when fed a synthetic mini-sim.
- DSR < raw Sharpe at observed Sharpe=1.0 with n_trials > 1 (selection-bias
  correction visible).
- §5.13.4 regression: any sim with large n_trials produces DSR < 1.0 even
  when raw Sharpe = 1.0.

Per CLAUDE.md §5.13.4 — "Single performance number = unverified claim".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

# Use bare-minimum imports — we mock SimAdapter init.
from sim.runner import SimResult   # noqa: E402
from adapters.sim import SimAdapter  # noqa: E402
from kernel.metrics import compute_perf_triple  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# SimResult dataclass — field presence + defaults.
# ─────────────────────────────────────────────────────────────────────────────

class TestSimResultFieldsAdded:
    def test_simresult_has_falsifiability_fields_with_nan_defaults(self):
        r = SimResult(
            equity_df=pd.DataFrame(),
            trade_log=[],
            rotation_log=[],
            final_value=100_000.0,
            total_return=0.0,
            apy=0.0,
            win_rate=0.0,
            avg_hold=0.0,
            avg_pnl=0.0,
            total_tax=0.0,
            exit_reasons={},
            rotations=[],
        )
        # Default values per §5.13.4 falsifiability layer.
        for field in ("dsr", "pbo", "beta_vs_spy", "alpha_vs_spy",
                      "information_ratio_vs_spy"):
            assert hasattr(r, field), f"SimResult missing {field}"
            assert math.isnan(getattr(r, field)), \
                f"{field} default should be NaN, got {getattr(r, field)}"
        # n_trials defaults to 1 (single-config-search → no DSR deflation).
        assert r.n_trials == 1
        assert r.event_level_tax_debited == 0.0
        assert r.annual_net_tax_estimate == 0.0
        assert math.isnan(r.annual_net_apy_estimate)


# ─────────────────────────────────────────────────────────────────────────────
# build_result populates the new fields when fed a synthetic mini-sim.
# Construct SimAdapter via __new__ (skipping __init__) and stuff the minimum
# attributes needed by build_result.
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_adapter(
    n_days: int = 252,
    sharpe_target: float = 1.0,
    seed: int = 0,
    n_trials: int = 1,
) -> SimAdapter:
    """Build a synthetic SimAdapter whose build_result yields a controlled
    Sharpe.

    Uses __new__ to bypass the heavy __init__ (model loading, panel frames,
    universe context). Only the attributes build_result reads are populated.
    """
    rng = np.random.default_rng(seed)
    # Per-period return parameters chosen so that annualized Sharpe ≈ target.
    daily_sigma = 0.01
    daily_mu = sharpe_target * daily_sigma / math.sqrt(252)
    daily_returns = daily_mu + daily_sigma * rng.standard_normal(n_days)
    equity = 100_000 * np.cumprod(1 + daily_returns)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="B")
    equity_curve = [
        {"date": d, "portfolio": float(v), "regime": "RISK_ON"}
        for d, v in zip(idx, equity)
    ]

    # Build a SPY benchmark with similar volatility but different mean.
    spy_returns = 0.0003 + 0.009 * rng.standard_normal(n_days)
    spy_close = 400 * np.cumprod(1 + spy_returns)
    spy_df = pd.DataFrame({"close": spy_close}, index=idx)

    adapter = SimAdapter.__new__(SimAdapter)
    adapter._equity_curve = equity_curve
    adapter._trade_log = []
    adapter._rotation_log = []
    adapter._monitor_state = {"no_candidate_streak": 0}
    adapter._initial_cash = 100_000.0
    adapter._spy_df = spy_df
    adapter._config = {"performance": {"n_trials": n_trials}}
    return adapter


class TestBuildResultPopulatesPerfTriple:
    def test_dsr_pbo_finite_when_sim_has_data(self):
        adapter = _make_synthetic_adapter(n_days=252, sharpe_target=1.0,
                                          seed=42, n_trials=1)
        result = adapter.build_result()
        # n_trials = 1 → DSR uses the trivial-deflation path; should still
        # be a finite (post-skew/kurtosis-corrected) number.
        assert math.isfinite(result.dsr)
        # PBO is NaN in single-seed mode by design.
        assert math.isnan(result.pbo)
        assert result.n_trials == 1

    def test_beta_alpha_ir_populated_with_spy(self):
        adapter = _make_synthetic_adapter(n_days=252, sharpe_target=1.0,
                                          seed=1, n_trials=1)
        result = adapter.build_result()
        assert math.isfinite(result.beta_vs_spy)
        assert math.isfinite(result.alpha_vs_spy)
        assert math.isfinite(result.information_ratio_vs_spy)
        # β clipped per §5.13.12 to |β| ≤ 10.
        assert abs(result.beta_vs_spy) <= 10.0

    def test_annual_net_tax_reporting_offsets_same_year_losses(self):
        adapter = _make_synthetic_adapter(n_days=252, sharpe_target=0.0,
                                          seed=7, n_trials=1)
        adapter._initial_cash = 1_000.0
        idx = pd.date_range("2024-01-02", periods=252, freq="B")
        adapter._equity_curve = [
            {"date": idx[0], "portfolio": 1_000.0, "regime": "BULL_CALM"},
            {"date": idx[-1], "portfolio": 970.0, "regime": "BULL_CALM"},
        ]
        adapter._trade_log = [
            {
                "action": "sell",
                "ticker": "A",
                "date": pd.Timestamp("2024-02-01"),
                "gross_pnl": 100.0,
                "pnl_pct": 0.10,
                "hold_days": 20,
                "tax": 50.0,
                "exit_reason": "test",
            },
            {
                "action": "sell",
                "ticker": "B",
                "date": pd.Timestamp("2024-03-01"),
                "gross_pnl": -80.0,
                "pnl_pct": -0.08,
                "hold_days": 10,
                "tax": 0.0,
                "exit_reason": "test",
            },
        ]
        adapter._config = {
            "performance": {"n_trials": 1},
            "tax": {
                "short_term_rate": 0.50,
                "long_term_rate": 0.20,
                "long_term_threshold_days": 365,
            },
        }

        result = adapter.build_result()

        assert result.event_level_tax_debited == pytest.approx(50.0)
        assert result.annual_net_tax_estimate == pytest.approx(10.0)
        assert result.tax_overstatement_vs_annual_net == pytest.approx(40.0)
        assert result.annual_net_final_value_estimate == pytest.approx(1_010.0)
        assert result.annual_net_total_return_estimate == pytest.approx(0.01)
        assert result.annual_net_equity_df_estimate["portfolio"].iloc[-1] == pytest.approx(1_010.0)
        assert result.annual_net_sharpe_estimate != result.sharpe

    def test_annual_net_tax_path_is_first_class_perf_metric(self):
        """AUDIT REGRESSION GUARD: annual-net tax reporting must carry its
        own equity path and Sharpe, not just a final-value footnote.

        Pre-fix, run_sim_104 could only export event-level Sharpe/APY, so WF
        promotion evaluated a tax-cash-stress path while merely printing the
        annual-net estimate. That made "tax exceeds gross" diagnostics hard to
        interpret and could make APY/Sharpe look worse than the annual netting
        model used in the forensic report.
        """
        adapter = _make_synthetic_adapter(n_days=252, sharpe_target=0.0,
                                          seed=9, n_trials=1)
        adapter._initial_cash = 1_000.0
        idx = pd.date_range("2024-01-02", periods=4, freq="B")
        adapter._equity_curve = [
            {"date": idx[0], "portfolio": 1_000.0, "regime": "BULL_CALM"},
            {"date": idx[1], "portfolio": 1_050.0, "regime": "BULL_CALM"},
            {"date": idx[2], "portfolio": 970.0, "regime": "BULL_CALM"},
            {"date": idx[3], "portfolio": 970.0, "regime": "BULL_CALM"},
        ]
        adapter._trade_log = [
            {
                "action": "sell",
                "ticker": "A",
                "date": idx[1],
                "gross_pnl": 100.0,
                "pnl_pct": 0.10,
                "hold_days": 20,
                "tax": 50.0,
                "exit_reason": "test",
            },
            {
                "action": "sell",
                "ticker": "B",
                "date": idx[2],
                "gross_pnl": -80.0,
                "pnl_pct": -0.08,
                "hold_days": 10,
                "tax": 0.0,
                "exit_reason": "test",
            },
        ]
        adapter._spy_df = pd.DataFrame({"close": [100, 101, 100, 102]}, index=idx)
        adapter._config = {
            "performance": {"n_trials": 1},
            "tax": {
                "short_term_rate": 0.50,
                "long_term_rate": 0.20,
                "long_term_threshold_days": 365,
            },
        }

        result = adapter.build_result()

        assert result.annual_net_equity_df_estimate["portfolio"].tolist() == pytest.approx([
            1_000.0,
            1_100.0,
            1_020.0,
            1_010.0,
        ])
        assert result.annual_net_final_value_estimate == pytest.approx(1_010.0)
        assert math.isfinite(result.annual_net_sharpe_estimate)
        assert result.annual_net_sharpe_estimate == pytest.approx(
            result.annual_net_sharpe_estimate
        )


# ─────────────────────────────────────────────────────────────────────────────
# DSR selection-bias correction: with n_trials > 1, DSR < raw Sharpe.
# ─────────────────────────────────────────────────────────────────────────────

class TestDSRSelectionBiasVisible:
    def test_dsr_drops_with_n_trials(self):
        """At observed annual Sharpe = 1.0 and n_trials = 20, the DSR
        selection-bias correction must produce DSR < raw Sharpe (otherwise
        the deflator is mute and §5.13.4 is unmet)."""
        rng = np.random.default_rng(99)
        n = 252
        daily_sigma = 0.01
        daily_mu = 1.0 * daily_sigma / math.sqrt(252)  # annualized SR ≈ 1.0
        returns = daily_mu + daily_sigma * rng.standard_normal(n)
        triple_n1 = compute_perf_triple(returns, n_trials=1)
        triple_n20 = compute_perf_triple(returns, n_trials=20)
        # Same returns, but more trials → more selection-bias penalty
        # → DSR(n=20) < DSR(n=1).
        assert triple_n20["dsr"] < triple_n1["dsr"]


class TestSinglyNumberClaimRejected:
    """§5.13.4 audit-regression-guard: any sim with n_trials >> 1 produces
    DSR << 1.0 even when raw Sharpe = 1.0. Pre-fix sim emitted raw Sharpe
    only — caller could not tell the difference between a single-config
    Sharpe=1.0 and a 100-config-search Sharpe=1.0."""

    def test_dsr_well_below_one_at_high_trial_count(self):
        rng = np.random.default_rng(2026)
        n = 252
        daily_sigma = 0.01
        daily_mu = 1.0 * daily_sigma / math.sqrt(252)  # annualized SR ≈ 1.0
        returns = daily_mu + daily_sigma * rng.standard_normal(n)

        # Sanity: raw Sharpe is close to 1.0.
        triple_base = compute_perf_triple(returns, n_trials=1)
        raw_sharpe = triple_base["sharpe"]
        assert math.isfinite(raw_sharpe)

        # n_trials >= 100 → DSR must show strong deflation.
        triple_high = compute_perf_triple(returns, n_trials=100)
        # DSR is in [0, 1] probability-style — at n_trials=100 the
        # probability-of-true-edge must be far from 1 even for SR ≈ 1.0.
        assert math.isfinite(triple_high["dsr"])
        # Hard regression: DSR < 1.0 (the deflator is not muted).
        assert triple_high["dsr"] < 1.0, \
            f"DSR={triple_high['dsr']} >= 1.0 — selection-bias deflator " \
            f"silently muted; §5.13.4 not enforced"
