"""Tests for scripts/validate_buy_logic.py — the sim-runner script.

Self-audit fix (2026-04-26): validate_buy_logic.py shipped without
tests and crashed all 4 sims at the FINAL summary step
('result.buys()' — buys is a @property, not a method). 50 minutes
of compute lost. These tests pin the contract for _summarise +
_diff_table + _apply_overrides + _build_tag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_buy_logic import (  # noqa: E402
    _apply_overrides,
    _build_tag,
    _diff_table,
    _summarise,
)


class _FakeSimResult:
    """Stub matching the SimResult API surface that _summarise reads.

    SimResult fields: equity_df, trade_log, total_return, apy, win_rate,
    final_value, longest_no_trade_streak, plus @property buys/sells.
    """
    def __init__(self, *, equity_df=None, trade_log=None,
                 total_return=0.0, apy=0.0, win_rate=0.0,
                 final_value=100_000.0, longest_no_trade_streak=0):
        self.equity_df = equity_df
        self.trade_log = trade_log or []
        self.total_return = total_return
        self.apy = apy
        self.win_rate = win_rate
        self.final_value = final_value
        self.longest_no_trade_streak = longest_no_trade_streak
        self.rotation_log = []   # SimResult requirement
        self.exit_reasons = {}
        self.rotations = []

    @property
    def buys(self):
        return [t for t in self.trade_log if t.get("action") == "buy"]

    @property
    def sells(self):
        return [t for t in self.trade_log if t.get("action") == "sell"]


def _equity_df(n_days: int = 250, drift: float = 0.0008,
                vol: float = 0.012, seed: int = 0) -> pd.DataFrame:
    """Build an equity_df with `portfolio` and `regime` columns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n_days)
    eq = 100_000 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    return pd.DataFrame(
        {"portfolio": eq, "regime": "BULL_CALM"},
        index=idx,
    )


# ── _summarise — the function that crashed all sims ──────────────────────────

class TestSummarise:
    def test_buys_attr_not_called_as_method(self):
        """REGRESSION: VALIDATE-BUYS-CALL — result.buys is a property,
        not a method. Pre-fix, _summarise called it as buys() → crash."""
        result = _FakeSimResult(
            apy=0.30, total_return=0.50,
            equity_df=_equity_df(),
            trade_log=[
                {"action": "buy", "ticker": "A"},
                {"action": "sell", "ticker": "A"},
            ],
        )
        # MUST NOT crash:
        out = _summarise(result, "test")
        assert out["n_buys"] == 1
        assert out["n_sells"] == 1
        assert out["n_trades"] == 2

    def test_basic_metrics_computed(self):
        result = _FakeSimResult(apy=0.39, total_return=0.85,
                                 equity_df=_equity_df(),
                                 trade_log=[{"action": "buy"}] * 12)
        out = _summarise(result, "tag1")
        assert out["tag"] == "tag1"
        assert out["apy"] == pytest.approx(0.39)
        assert out["total_return"] == pytest.approx(0.85)
        assert out["n_trades"] == 12

    def test_sharpe_computed_from_equity_df(self):
        """Sharpe must be computed (not pre-existing on SimResult)."""
        result = _FakeSimResult(equity_df=_equity_df(drift=0.001, vol=0.01))
        out = _summarise(result, "tag")
        # Positive drift + low vol → positive Sharpe
        assert out["sharpe"] > 0

    def test_max_dd_computed(self):
        """max_dd computed from peak-to-trough on equity_df.portfolio."""
        # Build equity that drops from 100k → 80k then back
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        port = [100, 110, 120, 100, 80, 90, 100, 110, 120, 130]
        eq = pd.DataFrame({"portfolio": port, "regime": "BULL_CALM"}, index=idx)
        result = _FakeSimResult(equity_df=eq)
        out = _summarise(result, "tag")
        # Peak 120 → trough 80 → DD = (80-120)/120 = -0.333
        assert out["max_dd"] == pytest.approx(0.333, abs=0.001)

    def test_no_equity_df_safe(self):
        """When equity_df is None, sharpe + max_dd default to 0; no crash."""
        result = _FakeSimResult(equity_df=None, apy=0.1)
        out = _summarise(result, "tag")
        assert out["sharpe"] == 0.0
        assert out["max_dd"] == 0.0
        assert out["apy"] == 0.1

    def test_empty_equity_df_safe(self):
        """Empty equity_df doesn't crash."""
        eq = pd.DataFrame({"portfolio": [], "regime": []})
        result = _FakeSimResult(equity_df=eq)
        out = _summarise(result, "tag")
        assert out["sharpe"] == 0.0
        assert out["max_dd"] == 0.0

    def test_empty_trade_log_safe(self):
        result = _FakeSimResult(trade_log=[])
        out = _summarise(result, "tag")
        assert out["n_trades"] == 0
        assert out["n_buys"] == 0
        assert out["n_sells"] == 0

    def test_zero_volatility_no_sharpe_div_by_zero(self):
        """Constant equity → vol=0 → sharpe=0 (no nan/inf)."""
        idx = pd.date_range("2024-01-02", periods=20, freq="B")
        eq = pd.DataFrame({"portfolio": [100_000] * 20, "regime": "FLAT"},
                           index=idx)
        result = _FakeSimResult(equity_df=eq)
        out = _summarise(result, "tag")
        assert out["sharpe"] == 0.0
        assert np.isfinite(out["max_dd"])


# ── _diff_table ──────────────────────────────────────────────────────────────

class TestDiffTable:
    def test_returns_string(self):
        baseline = {"apy": 0.30, "total_return": 0.50, "sharpe": 1.0,
                     "max_dd": 0.10, "n_trades": 50, "n_buys": 25,
                     "n_sells": 25, "longest_no_trade_streak": 5}
        candidate = dict(baseline); candidate["apy"] = 0.35
        out = _diff_table(baseline, candidate)
        assert isinstance(out, str)
        assert "apy" in out
        assert "Verdict" in out

    def test_verdict_up_for_apy_improvement(self):
        baseline = {k: 0.0 for k in (
            "apy", "total_return", "sharpe", "max_dd",
            "n_trades", "n_buys", "n_sells", "longest_no_trade_streak"
        )}
        candidate = dict(baseline); candidate["apy"] = 0.05
        out = _diff_table(baseline, candidate)
        assert "✅" in out

    def test_verdict_down_for_apy_loss(self):
        baseline = {k: 0.0 for k in (
            "apy", "total_return", "sharpe", "max_dd",
            "n_trades", "n_buys", "n_sells", "longest_no_trade_streak"
        )}
        candidate = dict(baseline); candidate["apy"] = -0.10
        out = _diff_table(baseline, candidate)
        assert "❌" in out


# ── _apply_overrides ─────────────────────────────────────────────────────────

class TestApplyOverrides:
    def _base_cfg(self):
        return {"ranking": {}, "rotation": {}}

    def test_no_overrides_returns_copy(self):
        cfg = self._base_cfg()
        out = _apply_overrides(cfg)
        # Shouldn't mutate input
        assert "quality_floor" not in cfg.get("ranking", {}).get(
            "panel_scoring", {})

    def test_gate_b_only_enables_quality_floor(self):
        out = _apply_overrides(self._base_cfg(), gate_b_threshold=0.20)
        qf = out["ranking"]["panel_scoring"]["quality_floor"]
        assert qf["enabled"] is True
        assert qf["edge_sharpe_floor"]["enabled"] is True
        assert qf["edge_sharpe_floor"]["threshold"] == 0.20

    def test_gate_a_with_default_lookback(self):
        out = _apply_overrides(self._base_cfg(), gate_a_pct=85)
        qf = out["ranking"]["panel_scoring"]["quality_floor"]
        df = qf["distribution_floor"]
        assert df["enabled"] is True
        assert df["percentile"] == 85

    def test_qp_solver_flag(self):
        out = _apply_overrides(self._base_cfg(), qp_solver=True)
        ja = out["rotation"]["joint_actions"]
        assert ja["enabled"] is True
        assert ja["solver"] == "qp"

    def test_qp_with_advanced_knobs(self):
        out = _apply_overrides(
            self._base_cfg(),
            qp_solver=True,
            qp_signal_decay=0.5,
            qp_robust_kappa=0.3,
            qp_cvar_lambda=1.0,
        )
        ja = out["rotation"]["joint_actions"]
        assert ja["qp_signal_decay"] == 0.5
        assert ja["qp_robust_mu_kappa"] == 0.3
        assert ja["qp_cvar_lambda"] == 1.0


# ── _build_tag ────────────────────────────────────────────────────────────────

class TestBuildTag:
    def _args(self, **kwargs):
        class _A:
            baseline = False
            gate_b = None
            gate_a_pct = None
            gate_c_gamma = None
            qp_solver = False
            qp_signal_decay = 0.0
            qp_robust_kappa = 0.0
            qp_cvar_lambda = 0.0
        a = _A()
        for k, v in kwargs.items():
            setattr(a, k, v)
        return a

    def test_baseline_returns_baseline(self):
        assert _build_tag(self._args(baseline=True)) == "baseline"

    def test_no_args_defaults_to_baseline(self):
        assert _build_tag(self._args()) == "baseline"

    def test_single_gate(self):
        assert _build_tag(self._args(gate_b=0.20)) == "gate-b0.2"

    def test_combined_gates(self):
        tag = _build_tag(self._args(gate_b=0.20, gate_a_pct=85))
        assert "gate-b0.2" in tag and "gate-a-p85" in tag

    def test_full_stack(self):
        tag = _build_tag(self._args(
            gate_b=0.20, gate_a_pct=85, gate_c_gamma=3.0,
            qp_solver=True, qp_signal_decay=0.5,
        ))
        for substr in ("gate-b0.2", "gate-a-p85", "gate-c-g3", "qp", "qp-decay0.5"):
            assert substr in tag
