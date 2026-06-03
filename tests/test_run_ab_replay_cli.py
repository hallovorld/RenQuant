"""Tests for the §8 Step 4g CLI driver — orchestration layer pins.

The verdict-assembly logic and the regime-stratification block are
the parts this PR is responsible for. The pieces underneath
(allocators, replay, DSR/PBO) are pinned in their own test files.
"""
from __future__ import annotations

import json
import sqlite3
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
    constraint_fidelity_block,
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


def _snap_with_sector(n: int) -> ConstraintSnapshot:
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
        sector_indicator=np.ones((1, n)),
        sector_cap_vec=np.array([1.0]),
        sector_names=("All",),
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


def _write_cli_fixture_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE score_distribution (
            run_id TEXT,
            date TEXT,
            ticker TEXT,
            raw_panel REAL,
            rank_score REAL,
            mu REAL,
            sigma REAL,
            regime TEXT,
            is_holding INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE ticker_forward_returns (
            as_of_date TEXT,
            ticker TEXT,
            close_price REAL,
            fwd_1d REAL,
            fwd_5d REAL,
            fwd_10d REAL,
            fwd_20d REAL,
            fwd_60d REAL,
            updated_at TEXT
        )
    """)
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "GOOG"]
    for day in range(1, 31):
        date = f"2024-01-{day:02d}"
        for rank, ticker in enumerate(tickers):
            mu = 0.03 - rank * 0.005 + float(rng.normal(0.0, 0.001))
            sigma = 0.10 + rank * 0.02
            fwd = 0.004 - rank * 0.001 + float(rng.normal(0.0, 0.002))
            cur.execute(
                "INSERT INTO score_distribution "
                "(run_id, date, ticker, mu, sigma, regime) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("run-test", date, ticker, mu, sigma, "BULL_CALM"),
            )
            cur.execute(
                "INSERT INTO ticker_forward_returns "
                "(as_of_date, ticker, fwd_60d) VALUES (?, ?, ?)",
                (date, ticker, fwd),
            )
    conn.commit()
    conn.close()


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
            "regime_stratified", "violation_report", "constraint_fidelity",
            "verdict",
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
            assert "live_promotable_per_section_8" in sig
            assert "live_promotable_per_clause_7_4" in sig
        # Verdict block has the gate decision
        assert "promotion_candidate" in payload["verdict"]
        assert "next_action" in payload["verdict"]
        # Synthetic bars omit sector constraints, so the run is not
        # decision-grade and must fail closed for promotion.
        assert payload["constraint_fidelity"]["decision_grade"] is False
        assert payload["verdict"]["promotion_candidate"] is None
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
        assert out["win_rate_a_beats_b_z_score"] is not None
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


class TestConstraintFidelity:
    def test_missing_sector_cap_is_not_decision_grade(self):
        block = constraint_fidelity_block(_bars(3))
        assert block["decision_grade"] is False
        assert block["missing_critical_families"] == ["sector_cap"]

    def test_sector_cap_present_is_decision_grade(self):
        bars = [
            AllocatorReplayBar(
                bar_date="d-001",
                snap=_snap_with_sector(3),
                mu=np.array([0.1, 0.2, 0.3]),
                sigma=np.array([0.2, 0.2, 0.2]),
                fwd_return=np.array([0.01, 0.02, 0.03]),
                regime="BULL_CALM",
                cost_per_trade_bps=0.0,
            )
        ]
        block = constraint_fidelity_block(bars)
        assert block["decision_grade"] is True
        assert block["missing_critical_families"] == []


class TestAssembleVerdict:
    def test_no_candidate_beats_incumbent_yields_keep_incumbent(self):
        # Incumbent wins on paired bars → no promotion
        significance = {
            "incumbent_qp": {"live_promotable_per_section_8": True},
            "challenger": {"live_promotable_per_section_8": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": 0.5,  # incumbent beats
                "win_rate_a_beats_b": 0.70,  # incumbent wins 70% of bars
                "n_bars": 30,
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
        assert verdict["next_action"] == "keep_incumbent"

    def test_challenger_wins_but_violates_yields_iterate(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_section_8": True},
            "challenger": {"live_promotable_per_section_8": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.3,  # challenger beats
                "win_rate_a_beats_b": 0.10,    # incumbent wins only 10%, challenger 90%
                "n_bars": 30,
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

    def test_clean_challenger_wins_yields_promote_to_shadow(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_section_8": True},
            "challenger": {
                "live_promotable_per_section_8": True,
                "dsr": 0.99,
                "pbo": 0.2,
                "pbo_se": None,
            },
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.5,
                "win_rate_a_beats_b": 0.10,
                "n_bars": 30,
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
        assert verdict["next_action"] == "promote_to_shadow"

    def test_unrelated_allocator_violation_does_not_flip_promoted_candidate_gate(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_section_8": True},
            "challenger": {
                "live_promotable_per_section_8": True,
                "dsr": 0.99,
                "pbo": 0.2,
                "pbo_se": None,
            },
            "bad_other": {"live_promotable_per_section_8": False},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.5,
                "win_rate_a_beats_b": 0.10,
                "n_bars": 30,
            },
        }
        violations = {
            "any_allocator_violated_any_family": True,
            "by_allocator": {
                "incumbent_qp": {"rejected_for_promotion": False},
                "challenger": {"rejected_for_promotion": False},
                "bad_other": {"rejected_for_promotion": True},
            },
        }
        verdict = assemble_verdict(
            significance, paired, violations, incumbent="incumbent_qp",
        )

        assert verdict["promotion_candidate"] == "challenger"
        assert verdict["next_action"] == "promote_to_shadow"
        assert verdict["non_negotiable_gate_passed"]["zero_hard_constraint_regressions"] is True

    def test_incomplete_constraints_block_promotion(self):
        significance = {
            "incumbent_qp": {"live_promotable_per_section_8": True},
            "challenger": {"live_promotable_per_section_8": True},
        }
        paired = {
            "incumbent_qp_vs_challenger": {
                "delta_sharpe_annual": -0.5,
                "win_rate_a_beats_b": 0.10,
                "n_bars": 30,
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
            significance,
            paired,
            violations,
            incumbent="incumbent_qp",
            constraints_decision_grade=False,
        )
        assert verdict["promotion_candidate"] is None
        assert verdict["next_action"] == "iterate"
        assert verdict["non_negotiable_gate_passed"]["decision_grade_constraints"] is False


class TestCLISmoke:
    def test_main_with_default_wf_loader_writes_verdict_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "sim_runs.db"
            _write_cli_fixture_db(db)
            out = Path(td) / "verdict.json"
            rc = main([
                "--wf-artifact-root", td,
                "--start-cut", "2024-01-01",
                "--end-cut", "2024-01-30",
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
            assert payload["cut_range"] == ["2024-01-01", "2024-01-30"]
            assert payload["n_bars"] == 30
            assert payload["constraint_fidelity"]["decision_grade"] is False
            assert payload["verdict"]["promotion_candidate"] is None
