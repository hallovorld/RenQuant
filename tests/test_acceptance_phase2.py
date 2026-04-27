"""Phase 2 tests: sim-based gates G9/G10/G11 + sim_smoke helper.

User spec 2026-04-26: "phase 1,2,3,4全做". Phase 2 of the model
selection systematization adds sim-output-based gates that catch the
"OOS IC looks fine, real sim is broken" failure mode that pure-IC
gates can't see.

These tests pin:
- G9 sim_apy ≥ prior - 1pp (default)
- G10 sim_sharpe ≥ prior - 0.1 (default)
- G11 turnover ≤ prior × 1.5 (soft)
- All three skip-pass when smoke metrics absent (default state — no
  retrain runs sim_smoke yet, so gates must NEVER fail-closed on
  missing data)
- Config-driven thresholds for each
- compute_metrics_from_equity_curve correctness
- add_smoke_metrics_to_artifact patches metadata in-place
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.model_acceptance import (   # noqa: E402
    DEFAULT_GATES,
    ModelAcceptanceGate,
    build_gates_from_config,
)
from kernel.sim_smoke import (   # noqa: E402
    add_smoke_metrics_to_artifact,
    compute_metrics_from_equity_curve,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_artifact(path: Path, sim_smoke: dict | None = None,
                    oos_mean_ic: float = 0.05) -> Path:
    md = {
        "oos_mean_ic":     oos_mean_ic,
        "n_unique_prob_y": 10,
        "pool_ic":         0.01,
    }
    if sim_smoke is not None:
        md["sim_smoke"] = sim_smoke
    data = {
        "kind":         "panel_ltr_xgboost",
        "feature_cols": ["a", "b", "c"],
        "metadata":     md,
    }
    path.write_text(json.dumps(data))
    return path


# ── G9 sim_apy ────────────────────────────────────────────────────────────────

class TestG9SimApy:
    def test_skips_when_no_smoke_metrics(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json")
        v = ModelAcceptanceGate().evaluate(s)
        g9 = next(r for r in v.results if r.name == "G9_sim_apy")
        assert g9.passed   # skip-pass

    def test_no_prior_negative_apy_fails(self, tmp_path):
        """Without prior, G9 falls back to apy > -10% (catastrophic floor)."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"apy": -0.15})
        v = ModelAcceptanceGate().evaluate(s)
        g9 = next(r for r in v.results if r.name == "G9_sim_apy")
        assert not g9.passed

    def test_prior_apy_15pct_new_14pct_passes(self, tmp_path):
        """Prior 15%, new 14% = 1pp drop. At threshold (passes)."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"apy": 0.14})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"apy": 0.15})
        v = ModelAcceptanceGate().evaluate(s, a)
        g9 = next(r for r in v.results if r.name == "G9_sim_apy")
        assert g9.passed

    def test_prior_15pct_new_13pct_fails(self, tmp_path):
        """Prior 15%, new 13% = 2pp drop. Exceeds 1pp default → fail."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"apy": 0.13})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"apy": 0.15})
        v = ModelAcceptanceGate().evaluate(s, a)
        g9 = next(r for r in v.results if r.name == "G9_sim_apy")
        assert not g9.passed

    def test_g9_max_pp_drop_configurable(self, tmp_path):
        """Loosen to 5pp: new 10% vs prior 15% (5pp drop) → boundary."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"apy": 0.10})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"apy": 0.15})
        v_default = ModelAcceptanceGate().evaluate(s, a)
        g9_default = next(r for r in v_default.results if r.name == "G9_sim_apy")
        assert not g9_default.passed
        v_loose = ModelAcceptanceGate(config={"g9_max_pp_drop": 5.0}).evaluate(s, a)
        g9_loose = next(r for r in v_loose.results if r.name == "G9_sim_apy")
        assert g9_loose.passed


# ── G10 sim_sharpe ────────────────────────────────────────────────────────────

class TestG10SimSharpe:
    def test_skips_when_no_smoke_metrics(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json")
        v = ModelAcceptanceGate().evaluate(s)
        g10 = next(r for r in v.results if r.name == "G10_sim_sharpe")
        assert g10.passed

    def test_no_prior_negative_sharpe_fails(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"sharpe": -0.5})
        v = ModelAcceptanceGate().evaluate(s)
        g10 = next(r for r in v.results if r.name == "G10_sim_sharpe")
        assert not g10.passed

    def test_prior_15_new_145_passes(self, tmp_path):
        """Prior 1.5, new 1.45 = 0.05 drop, under 0.1 default."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"sharpe": 1.45})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"sharpe": 1.5})
        v = ModelAcceptanceGate().evaluate(s, a)
        g10 = next(r for r in v.results if r.name == "G10_sim_sharpe")
        assert g10.passed

    def test_prior_15_new_13_fails(self, tmp_path):
        """Prior 1.5, new 1.3 = 0.2 drop, exceeds 0.1 default."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"sharpe": 1.3})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"sharpe": 1.5})
        v = ModelAcceptanceGate().evaluate(s, a)
        g10 = next(r for r in v.results if r.name == "G10_sim_sharpe")
        assert not g10.passed


# ── G11 turnover ──────────────────────────────────────────────────────────────

class TestG11Turnover:
    def test_skips_when_no_smoke_metrics(self, tmp_path):
        s = _write_artifact(tmp_path / "s.json")
        v = ModelAcceptanceGate().evaluate(s)
        g11 = next(r for r in v.results if r.name == "G11_turnover")
        assert g11.passed

    def test_g11_is_soft_default(self):
        """G11 is soft by default — turnover increase is suspicious but
        operator may accept (e.g., regime change demands more rebalance)."""
        g11 = next(g for g in DEFAULT_GATES if g.name == "G11_turnover")
        assert g11.severity == "soft"

    def test_prior_2x_new_25x_passes(self, tmp_path):
        """Prior 2.0, new 2.5 → 1.25× multiplier, under 1.5 default."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"turnover_ratio": 2.5})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"turnover_ratio": 2.0})
        v = ModelAcceptanceGate().evaluate(s, a)
        g11 = next(r for r in v.results if r.name == "G11_turnover")
        assert g11.passed

    def test_prior_2x_new_4x_fails(self, tmp_path):
        """Prior 2.0, new 4.0 → 2.0× multiplier, exceeds 1.5 default."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={"turnover_ratio": 4.0})
        a = _write_artifact(tmp_path / "a.json", sim_smoke={"turnover_ratio": 2.0})
        v = ModelAcceptanceGate().evaluate(s, a)
        g11 = next(r for r in v.results if r.name == "G11_turnover")
        assert not g11.passed   # but soft — doesn't block aggregate verdict


# ── Aggregate behavior with phase-2 gates ─────────────────────────────────────

class TestPhase2AggregateVerdict:
    def test_default_gate_count_is_11(self):
        assert len(DEFAULT_GATES) == 11

    def test_no_smoke_metrics_doesnt_block_promotion(self, tmp_path):
        """A model with no sim_smoke must still be promotable — Phase 2
        gates are opt-in, not breaking. This is the default state for
        all retrain pipelines today."""
        s = _write_artifact(tmp_path / "s.json", oos_mean_ic=0.04)
        v = ModelAcceptanceGate().evaluate(s)
        assert v.all_hard_passed

    def test_smoke_failure_blocks_via_g9_or_g10(self, tmp_path):
        """When smoke metrics ARE present and bad, the new gates fire."""
        s = _write_artifact(tmp_path / "s.json", sim_smoke={
            "apy": 0.05, "sharpe": 0.3, "turnover_ratio": 5.0,
        })
        a = _write_artifact(tmp_path / "a.json", sim_smoke={
            "apy": 0.15, "sharpe": 1.5, "turnover_ratio": 2.0,
        })
        v = ModelAcceptanceGate().evaluate(s, a)
        assert not v.all_hard_passed
        # G9 + G10 should both fail; G11 is soft (warns only)
        hard_fails = {r.name for r in v.hard_failures()}
        assert "G9_sim_apy" in hard_fails
        assert "G10_sim_sharpe" in hard_fails


# ── kernel/sim_smoke helpers ──────────────────────────────────────────────────

class TestSimSmokeHelpers:
    def test_compute_metrics_with_synthetic_equity(self):
        """Linear-growth equity curve should yield well-defined APY,
        positive Sharpe (modulo zero-variance edge case), zero drawdown."""
        # Use a noisy positive-drift equity curve so std(returns) > 0
        rng = np.random.default_rng(42)
        n = 252
        rets = rng.normal(0.0008, 0.005, n)   # ~20% annual, 8% vol
        eq = pd.Series(np.cumprod(1.0 + rets) * 1000.0,
                       index=pd.bdate_range("2024-01-02", periods=n))
        m = compute_metrics_from_equity_curve(eq)
        assert m["apy"] > 0.05    # rough — should be ~20%
        assert m["sharpe"] > 0    # should be positive
        assert m["max_drawdown"] <= 0   # should be non-positive

    def test_compute_metrics_handles_short_series(self):
        eq = pd.Series([1000.0])
        m = compute_metrics_from_equity_curve(eq)
        assert m["apy"] == 0.0
        assert m["sharpe"] == 0.0

    def test_compute_metrics_handles_empty_series(self):
        m = compute_metrics_from_equity_curve(pd.Series([], dtype=float))
        assert m["apy"] == 0.0

    def test_compute_metrics_with_trades_dataframe(self):
        eq = pd.Series([1000.0, 1010.0, 1020.0])
        trades = pd.DataFrame({"notional": [500.0, -300.0, 200.0]})
        m = compute_metrics_from_equity_curve(eq, trades)
        assert m["n_trades"] == 3
        # 1000 gross / ~1010 avg_eq ≈ 0.99
        assert 0.95 < m["turnover_ratio"] < 1.05

    def test_add_smoke_metrics_patches_artifact(self, tmp_path):
        path = tmp_path / "panel-ltr.json"
        path.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "metadata": {"oos_mean_ic": 0.04},
        }))
        add_smoke_metrics_to_artifact(path, {
            "apy": 0.12, "sharpe": 1.4, "turnover_ratio": 2.1,
        })
        d = json.loads(path.read_text())
        assert d["metadata"]["sim_smoke"]["apy"] == 0.12
        assert d["metadata"]["oos_mean_ic"] == 0.04   # original preserved
        assert "written_at" in d["metadata"]["sim_smoke"]

    def test_add_smoke_metrics_creates_metadata_when_missing(self, tmp_path):
        """Older flat-format artifacts may not have a metadata block."""
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"kind": "old_format"}))
        add_smoke_metrics_to_artifact(path, {"apy": 0.10})
        d = json.loads(path.read_text())
        assert d["metadata"]["sim_smoke"]["apy"] == 0.10

    def test_add_smoke_metrics_idempotent(self, tmp_path):
        """Re-running overwrites prior block (operator can re-run sim)."""
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"metadata": {}}))
        add_smoke_metrics_to_artifact(path, {"apy": 0.1, "sharpe": 1.0})
        add_smoke_metrics_to_artifact(path, {"apy": 0.12, "sharpe": 1.2})
        d = json.loads(path.read_text())
        assert d["metadata"]["sim_smoke"]["apy"] == 0.12   # newest wins


# ── Strategy config block parsing ─────────────────────────────────────────────

class TestStrategyConfigPhase2Keys:
    def test_phase2_keys_present_in_strategy_config(self):
        cfg_path = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
        cfg = json.loads(cfg_path.read_text())
        acc = cfg.get("acceptance", {})
        # Phase 2 keys
        assert acc.get("g9_max_pp_drop") == 1.0
        assert acc.get("g10_max_sharpe_drop") == 0.1
        assert acc.get("g11_max_multiplier") == 1.5
        assert acc.get("g11_severity") == "soft"
        assert acc.get("run_sim_smoke") is False  # opt-in

    def test_build_gates_with_phase2_config(self):
        config = {
            "g9_max_pp_drop":      2.0,
            "g10_max_sharpe_drop": 0.2,
            "g11_max_multiplier":  2.0,
            "g11_severity":        "hard",
        }
        gates = build_gates_from_config(config)
        # 11 gates total
        assert len(gates) == 11
        g11 = next(g for g in gates if g.name == "G11_turnover")
        assert g11.severity == "hard"   # config override stuck
