"""Tests for the §8 Step 4g CLI driver — orchestration layer pins.

The verdict-assembly logic and the regime-stratification block are
the parts this PR is responsible for. The pieces underneath
(allocators, replay, DSR/PBO) are pinned in their own test files.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.allocator_replay import AllocatorReplayBar  # noqa: E402
from kernel.portfolio_qp.baseline_allocators import (  # noqa: E402
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot  # noqa: E402
from kernel.portfolio_qp.run_ab_replay import (  # noqa: E402
    assemble_verdict,
    get_allocator,
    main,
    paired_comparison_metrics,
    regime_stratified_block,
    register_allocator,
    run_replay,
    violation_report_block,
)


def _snap(n: int) -> ConstraintSnapshot:
    return ConstraintSnapshot(
        n=n, tickers=tuple(f"T{i}" for i in range(n)),
        w_current=np.zeros(n),
        w_upper_hard=np.full(n, 0.50),
        w_upper=np.full(n, 0.50),
        w_lower=0.0,
        dw_max=np.full(n, 1.0),
        cash_reserve=0.0,
        turnover_max=None,
        drawdown=0.0,
        drawdown_limit=0.20,
        gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
    )


def _bars(n_bars: int, *, regime: str | None = "BULL_CALM", seed: int = 0):
    rng = np.random.default_rng(seed)
    bars = []
    for i in range(n_bars):
        bars.append(AllocatorReplayBar(
            bar_date=f"d-{i:03d}",
            snap=_snap(3),
            mu=rng.uniform(0.0, 0.05, 3),
            sigma=rng.uniform(0.10, 0.20, 3),
            fwd_return=rng.normal(0.001, 0.005, 3),
            regime=regime,
            cost_per_trade_bps=0.0,
        ))
    return bars


class TestRegistry:
    def test_default_allocators_resolvable(self):
        assert get_allocator("equal_weight_top_k") is equal_weight_top_k
        assert get_allocator("inverse_vol_top_k") is inverse_vol_top_k
        assert get_allocator("fractional_kelly_top_k") is fractional_kelly_top_k

    def test_unknown_allocator_raises(self):
        with pytest.raises(KeyError, match="not in registry"):
            get_allocator("does_not_exist")

    def test_register_overrides(self):
        def fake(snap, *, mu, sigma=None):  # noqa: ARG001
            return None
        register_allocator("custom_test_only", fake)
        assert get_allocator("custom_test_only") is fake


class TestRunReplayEndToEnd:
    def test_basic_orchestration(self):
        bars = _bars(n_bars=64)
        payload = run_replay(
            bars,
            ["equal_weight_top_k", "inverse_vol_top_k", "fractional_kelly_top_k"],
            incumbent="fractional_kelly_top_k",
        )
        # Every top-level block is present
        for key in (
            "n_bars", "n_unique_dates", "regime_distribution",
            "constraint_snapshot_contract_version", "allocators",
            "per_allocator", "paired_comparisons", "significance",
            "regime_stratified", "violation_report", "verdict",
        ):
            assert key in payload, f"missing {key}"
        # Per-allocator block has each allocator
        assert set(payload["per_allocator"]) == {
            "equal_weight_top_k", "inverse_vol_top_k", "fractional_kelly_top_k",
        }
        # Paired comparisons key incumbent vs each other allocator
        assert "fractional_kelly_top_k_vs_equal_weight_top_k" in payload["paired_comparisons"]
        assert "fractional_kelly_top_k_vs_inverse_vol_top_k" in payload["paired_comparisons"]
        # Significance has DSR + PBO populated (64 bars >= 30 + 16)
        for name in ("equal_weight_top_k", "inverse_vol_top_k", "fractional_kelly_top_k"):
            sig = payload["significance"][name]
            assert sig["dsr"] is not None
            assert "live_promotable_per_clause_7_4" in sig
        # Verdict block has the gate decision
        assert "promotion_candidate" in payload["verdict"]
        assert "next_action" in payload["verdict"]
        # JSON-serialisable
        json.dumps(payload)

    def test_regime_distribution_sums_to_one(self):
        bars = _bars(40, regime="BULL_CALM") + _bars(20, regime="BULL_VOLATILE", seed=1)
        payload = run_replay(
            bars, ["equal_weight_top_k", "inverse_vol_top_k"],
            incumbent="equal_weight_top_k",
        )
        total = sum(payload["regime_distribution"].values())
        assert abs(total - 1.0) < 1e-9


class TestPairedComparisonMetrics:
    def test_delta_sharpe_and_win_rate(self):
        a = np.array([0.01, 0.02, 0.005, -0.01, 0.015])
        b = np.array([0.005, 0.01, 0.01, -0.005, 0.005])
        out = paired_comparison_metrics(a, b, name_a="a", name_b="b")
        assert out["n_bars"] == 5
        # a > b on 3 of 5 bars: idx 0, 1, 4 (a=0.005<b=0.01 and a=-0.01<b=-0.005 fail)
        assert abs(out["win_rate_a_beats_b"] - 0.6) < 1e-9
        assert out["max_delta_daily_return"] > 0
        assert out["min_delta_daily_return"] < 0
        assert out["delta_sharpe_annual"] is not None

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            paired_comparison_metrics(
                np.array([0.01, 0.02]),
                np.array([0.01]),
                name_a="a", name_b="b",
            )


class TestRegimeStratifiedBlock:
    def test_per_regime_n_bars_correct(self):
        bars = _bars(30, regime="BULL_CALM") + _bars(20, regime="BULL_VOLATILE", seed=1)
        from kernel.portfolio_qp.allocator_replay import replay_all
        results = replay_all({"eq": equal_weight_top_k}, bars)
        block = regime_stratified_block(results, bars)
        assert block["BULL_CALM"]["n_bars"] == 30
        assert block["BULL_VOLATILE"]["n_bars"] == 20
        # BULL_VOLATILE has < 30 bars → undersampled
        assert block["BULL_VOLATILE"]["undersampled"] is True
        assert block["BULL_CALM"]["undersampled"] is False


class TestViolationReport:
    def test_zero_violations_passes_gate(self):
        bars = _bars(20)
        from kernel.portfolio_qp.allocator_replay import replay_all
        results = replay_all(
            {"eq": equal_weight_top_k, "iv": inverse_vol_top_k}, bars,
        )
        block = violation_report_block(results)
        # Healthy baselines should have zero violations
        assert block["any_allocator_violated_any_family"] is False
        for name in ("eq", "iv"):
            assert block["by_allocator"][name]["rejected_for_promotion"] is False


class TestAssembleVerdict:
    def test_no_candidate_beats_incumbent_yields_reject_all(self):
        # Incumbent wins on paired bars → no promotion
        significance = {
            "incumbent_qp": {"live_promotable_per_clause_7_4": True},
            "challenger": {"live_promotable_per_clause_7_4": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": 0.5,  # incumbent beats
                "win_rate_a_beats_b": 0.70,  # incumbent wins 70% of bars
            },
        }
        violations = {
            "any_allocator_violated_any_family": False,
            "by_allocator": {
                "incumbent_qp": {"rejected_for_promotion": False},
                "challenger": {"rejected_for_promotion": False},
            },
        }
        verdict = assemble_verdict(
            significance, paired, violations, incumbent="incumbent_qp",
        )
        assert verdict["promotion_candidate"] is None
        assert verdict["next_action"] == "reject_all"

    def test_challenger_wins_but_violates_yields_iterate(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_clause_7_4": True},
            "challenger": {"live_promotable_per_clause_7_4": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.3,  # challenger beats
                "win_rate_a_beats_b": 0.30,    # incumbent wins only 30%, challenger 70%
            },
        }
        violations = {
            "any_allocator_violated_any_family": True,
            "by_allocator": {
                "incumbent_qp": {"rejected_for_promotion": False},
                "challenger": {"rejected_for_promotion": True},
            },
        }
        verdict = assemble_verdict(
            significance, paired, violations, incumbent="incumbent_qp",
        )
        assert verdict["promotion_candidate"] is None
        assert verdict["next_action"] == "iterate"

    def test_clean_challenger_wins_yields_live_shadow(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_clause_7_4": True},
            "challenger": {"live_promotable_per_clause_7_4": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.5,
                "win_rate_a_beats_b": 0.30,
            },
        }
        violations = {
            "any_allocator_violated_any_family": False,
            "by_allocator": {
                "incumbent_qp": {"rejected_for_promotion": False},
                "challenger": {"rejected_for_promotion": False},
            },
        }
        verdict = assemble_verdict(
            significance, paired, violations, incumbent="incumbent_qp",
        )
        assert verdict["promotion_candidate"] == "challenger"
        assert verdict["next_action"] == "live_shadow"


class TestCLISmoke:
    def test_main_with_placeholder_loader_writes_verdict_json(self):
        # Uses the placeholder bars when --loader-module is absent.
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "verdict.json"
            rc = main([
                "--wf-artifact-root", "/dev/null",
                "--start-cut", "2024-01-01",
                "--end-cut", "2024-06-30",
                "--out", str(out),
                "--allocators", "equal_weight_top_k,inverse_vol_top_k,fractional_kelly_top_k",
                "--incumbent", "fractional_kelly_top_k",
            ])
            assert rc == 0
            assert out.exists()
            payload = json.loads(out.read_text())
            # Schema-conforming top-level keys
            for key in ("as_of_date", "cut_range", "wf_artifact_root",
                        "per_allocator", "verdict"):
                assert key in payload
            assert payload["cut_range"] == ["2024-01-01", "2024-06-30"]
