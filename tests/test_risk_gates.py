"""Regression tests for per-ticker runtime risk gates.

Two gates added 2026-05-03:
  * RealizedVolGateTask — drop candidates with trailing realized vol > cap.
  * PositionConcentrationGateTask — drop candidates already ≥ cap%.

Invariant: no candidate buys a ticker over realized-vol cap or already
holding ≥ concentration cap of portfolio weight.
"""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.pipeline.task_risk_gates import (  # noqa: E402
    PositionConcentrationGateTask,
    RealizedVolGateTask,
)


@dataclass
class _Cand:
    ticker: str
    raw_score: float = 1.0
    rank_score: float = 1.0
    rs_score: float = 1.0


@dataclass
class _Held:
    shares: float = 0.0
    prev_close: float = 0.0


def _ohlcv_with_vol(annualized_vol: float, n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build close series with a target realized annualized vol."""
    rng = np.random.default_rng(seed)
    daily_sigma = annualized_vol / math.sqrt(252.0)
    rets = rng.normal(0.0, daily_sigma, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(end="2026-05-01", periods=n, freq="B")
    return pd.DataFrame({"close": close}, index=idx)


def _make_ctx(**kw):
    return SimpleNamespace(
        config=kw.get("config", {}),
        candidates=kw.get("candidates", []),
        ohlcv=kw.get("ohlcv", {}),
        holdings=kw.get("holdings", {}),
        prices=kw.get("prices", {}),
        portfolio_value=kw.get("portfolio_value", 0.0),
        cash=kw.get("cash", 0.0),
        counters=kw.get("counters", {}),
    )


class TestRealizedVolGate(unittest.TestCase):
    def test_drops_high_vol_keeps_low_vol(self) -> None:
        ohlcv = {
            "LOW": _ohlcv_with_vol(0.20),   # 20% — keep
            "MID": _ohlcv_with_vol(0.45),   # 45% — keep (under 60% default)
            "HIGH": _ohlcv_with_vol(0.90, seed=1),  # 90% — drop
            "EXTREME": _ohlcv_with_vol(1.20, seed=2),  # 120% — drop
        }
        ctx = _make_ctx(
            candidates=[_Cand(t) for t in ohlcv],
            ohlcv=ohlcv,
        )
        RealizedVolGateTask().run(ctx)
        kept = [c.ticker for c in ctx.candidates]
        self.assertIn("LOW", kept)
        self.assertIn("MID", kept)
        self.assertNotIn("HIGH", kept)
        self.assertNotIn("EXTREME", kept)
        self.assertEqual(ctx.counters.get("risk_gate_vol_dropped"), 2)

    def test_custom_cap_via_config(self) -> None:
        ohlcv = {"AAA": _ohlcv_with_vol(0.40)}
        ctx = _make_ctx(
            config={"risk_gates": {"realized_vol": {"max_annualized": 0.30}}},
            candidates=[_Cand("AAA")],
            ohlcv=ohlcv,
        )
        RealizedVolGateTask().run(ctx)
        # 40% > 30% cap → dropped
        self.assertEqual(len(ctx.candidates), 0)

    def test_disabled_via_config_keeps_all(self) -> None:
        ohlcv = {"VOLATILE": _ohlcv_with_vol(2.0, seed=3)}
        ctx = _make_ctx(
            config={"risk_gates": {"realized_vol": {"enabled": False}}},
            candidates=[_Cand("VOLATILE")],
            ohlcv=ohlcv,
        )
        RealizedVolGateTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["VOLATILE"])

    def test_insufficient_history_keeps_candidate(self) -> None:
        # Only 10 bars — too few for 60d default window
        df = _ohlcv_with_vol(0.50, n=10)
        ctx = _make_ctx(candidates=[_Cand("X")], ohlcv={"X": df})
        RealizedVolGateTask().run(ctx)
        # Permissive: don't drop on insufficient history
        self.assertEqual(len(ctx.candidates), 1)

    def test_missing_ohlcv_keeps_candidate(self) -> None:
        ctx = _make_ctx(candidates=[_Cand("MISSING")], ohlcv={})
        RealizedVolGateTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 1)

    def test_empty_candidates_noop(self) -> None:
        ctx = _make_ctx()
        RealizedVolGateTask().run(ctx)
        self.assertEqual(ctx.candidates, [])

    def test_window_days_config(self) -> None:
        df = _ohlcv_with_vol(0.50, n=40)
        # Default window 60 → not enough → keep
        ctx = _make_ctx(candidates=[_Cand("X")], ohlcv={"X": df})
        RealizedVolGateTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 1)
        # window=30 → enough → 50% > 60% default? No, 50% < 60% → kept anyway
        ctx2 = _make_ctx(
            config={"risk_gates": {"realized_vol": {"window_days": 30, "max_annualized": 0.40}}},
            candidates=[_Cand("X")], ohlcv={"X": df},
        )
        RealizedVolGateTask().run(ctx2)
        self.assertEqual(len(ctx2.candidates), 0, "50% > 40% cap → drop")


class TestPositionConcentrationGate(unittest.TestCase):
    def test_drops_already_at_cap(self) -> None:
        # Portfolio = $10,000. Holding LITE 10 shares @ $200 = $2,000 = 20%
        ctx = _make_ctx(
            portfolio_value=10000.0,
            candidates=[_Cand("LITE"), _Cand("WMT")],
            holdings={
                "LITE": _Held(shares=10, prev_close=200.0),
                "WMT": _Held(shares=5, prev_close=100.0),  # = 5% — well under 15%
            },
            prices={"LITE": 200.0, "WMT": 100.0},
        )
        PositionConcentrationGateTask().run(ctx)
        kept = [c.ticker for c in ctx.candidates]
        self.assertNotIn("LITE", kept)  # 20% ≥ 15% default cap
        self.assertIn("WMT", kept)       # 5% < 15%

    def test_no_existing_position_passes(self) -> None:
        ctx = _make_ctx(
            portfolio_value=10000.0,
            candidates=[_Cand("FRESH")],
            holdings={},
        )
        PositionConcentrationGateTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["FRESH"])

    def test_custom_cap_via_config(self) -> None:
        # 8% holding, cap=5% → drop. cap=10% → keep.
        held = {"X": _Held(shares=8, prev_close=100.0)}
        prices = {"X": 100.0}
        ctx_drop = _make_ctx(
            config={"risk_gates": {"position_concentration": {"max_pct": 0.05}}},
            portfolio_value=10000.0, candidates=[_Cand("X")],
            holdings=held, prices=prices,
        )
        PositionConcentrationGateTask().run(ctx_drop)
        self.assertEqual(len(ctx_drop.candidates), 0)

        ctx_keep = _make_ctx(
            config={"risk_gates": {"position_concentration": {"max_pct": 0.10}}},
            portfolio_value=10000.0, candidates=[_Cand("X")],
            holdings=held, prices=prices,
        )
        PositionConcentrationGateTask().run(ctx_keep)
        self.assertEqual(len(ctx_keep.candidates), 1)

    def test_disabled_via_config_keeps_all(self) -> None:
        ctx = _make_ctx(
            config={"risk_gates": {"position_concentration": {"enabled": False}}},
            portfolio_value=10000.0, candidates=[_Cand("OVER")],
            holdings={"OVER": _Held(shares=100, prev_close=100.0)},  # 100% portfolio
            prices={"OVER": 100.0},
        )
        PositionConcentrationGateTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 1)

    def test_zero_portfolio_value_skips_check(self) -> None:
        ctx = _make_ctx(portfolio_value=0.0, candidates=[_Cand("ANY")], holdings={})
        PositionConcentrationGateTask().run(ctx)
        self.assertEqual(len(ctx.candidates), 1)

    def test_falls_back_to_cash_plus_holdings_value(self) -> None:
        ctx = _make_ctx(
            portfolio_value=0.0, cash=5000.0,
            candidates=[_Cand("X")],
            holdings={
                "X": _Held(shares=10, prev_close=100.0),  # $1k
                "Y": _Held(shares=10, prev_close=400.0),  # $4k
            },
            prices={"X": 100.0, "Y": 400.0},
        )
        # Equity = 5k cash + 1k + 4k = 10k. X at $1k = 10% < 15% → keep.
        PositionConcentrationGateTask().run(ctx)
        self.assertEqual([c.ticker for c in ctx.candidates], ["X"])

    def test_counters_record_drop(self) -> None:
        ctx = _make_ctx(
            portfolio_value=10000.0,
            candidates=[_Cand("BIG"), _Cand("SMALL")],
            holdings={
                "BIG": _Held(shares=20, prev_close=100.0),  # 20%
                "SMALL": _Held(shares=1, prev_close=100.0),  # 1%
            },
            prices={"BIG": 100.0, "SMALL": 100.0},
        )
        PositionConcentrationGateTask().run(ctx)
        self.assertEqual(ctx.counters.get("risk_gate_concentration_dropped"), 1)


class TestPipelineWiring(unittest.TestCase):
    """Both gates must be wired into InferencePipeline AFTER candidate scoring."""

    def test_inference_pipeline_imports_both_gates(self) -> None:
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "pp_inference.py"
        src = path.read_text()
        idx_inf = src.find("class InferencePipeline")
        idx_sell = src.find("class SellOnlyPipeline")
        body = src[idx_inf:idx_sell]
        # Both gate calls must appear in InferencePipeline body
        self.assertIn("RealizedVolGateTask().run(ctx)", body)
        self.assertIn("PositionConcentrationGateTask().run(ctx)", body)

    def test_gates_run_after_candidate_phase_before_ranking(self) -> None:
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "pp_inference.py"
        src = path.read_text()
        idx_inf = src.find("class InferencePipeline")
        idx_sell = src.find("class SellOnlyPipeline")
        body = src[idx_inf:idx_sell]
        idx_buy_scan = body.find("Phase 2b")
        idx_vol = body.find("RealizedVolGateTask().run(ctx)")
        idx_conc = body.find("PositionConcentrationGateTask().run(ctx)")
        idx_phase3 = body.find("phase3_jobs")
        self.assertGreater(idx_buy_scan, 0)
        self.assertGreater(idx_vol, idx_buy_scan)
        self.assertGreater(idx_conc, idx_buy_scan)
        self.assertLess(idx_vol, idx_phase3)
        self.assertLess(idx_conc, idx_phase3)


if __name__ == "__main__":
    unittest.main()
