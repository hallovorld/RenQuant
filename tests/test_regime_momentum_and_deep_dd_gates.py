"""Tests for the 2026-05-15 buy-quality gates (Upgrades A + B).

Both gates are OFF by default. Tests pin:
  * Off-by-default behavior (no candidates dropped / scores unchanged)
  * Mismatch detection (regime says momentum, individual r60 negative
    → shrink score)
  * Regime mismatch in NON-momentum regime → no-op
  * Deep-drawdown veto with NO fundamental confirmation → veto
  * Deep-drawdown veto WITH SUE confirmation → kept
  * Pure helpers behave correctly (_trailing_return, _dd_from_high)

Both gates would have changed today's META trade (META: r60d=-4.66%
in MOM regime → score shrunk; dd_from_52w_high=-22.1%, no SUE → veto).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _ohlcv_with_r60(symbol, r60, sigma_daily=0.01, hist_days=300):
    """Build a synthetic OHLCV with target 60-day trailing return.

    Final close = 100 * (1 + r60); 60 days earlier close = 100.
    Older history filled with random walk; recent 61 bars are linear.
    """
    np.random.seed(hash(symbol) & 0xFFFF)
    start = 100.0
    end60 = start * (1 + r60)
    older = np.cumsum(np.random.randn(hist_days - 61) * sigma_daily) + np.log(start)
    recent = np.linspace(np.log(start), np.log(end60), 61)
    closes = np.exp(np.concatenate([older, recent]))
    return pd.DataFrame({"close": closes})


def _ohlcv_with_dd(symbol, dd_from_high, window=252):
    """Build a synthetic OHLCV with target drawdown from 252d max."""
    np.random.seed(hash(symbol) & 0xFFFF)
    hi = 100.0
    final = hi * (1 + dd_from_high)
    closes = np.linspace(hi * 1.02, final, window)
    return pd.DataFrame({"close": closes})


def _cand(ticker, rank_score=0.50, panel_score=0.50, **kw):
    return SimpleNamespace(
        ticker=ticker,
        rank_score=rank_score,
        panel_score=panel_score,
        expected_return=None, mu=None, sigma=None,
        features=kw.get("features", {}),
    )


class TestRegimeMomentumAlignment:

    def test_default_off_no_op(self):
        from kernel.pipeline.task_buy_quality_gates import (
            RegimeMomentumAlignmentTask,
        )
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {}}},
            regime="BULL_CALM", hurst=0.75,
            candidates=[_cand("AAA"), _cand("BBB")],
            ohlcv={
                "AAA": _ohlcv_with_r60("AAA", -0.05),
                "BBB": _ohlcv_with_r60("BBB", -0.05),
            },
            counters={},
        )
        RegimeMomentumAlignmentTask().run(ctx)
        for c in ctx.candidates:
            assert c.rank_score == 0.50

    def test_momentum_regime_shrinks_negative_r60(self):
        from kernel.pipeline.task_buy_quality_gates import (
            RegimeMomentumAlignmentTask,
        )
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "regime_momentum": {"enabled": True, "mismatch_scale": 0.5}
            }}},
            regime="BULL_CALM", hurst=0.77,
            candidates=[
                _cand("WINNER", rank_score=0.80),
                _cand("LOSER",  rank_score=0.80),
                _cand("META",   rank_score=0.99),
            ],
            ohlcv={
                "WINNER": _ohlcv_with_r60("WINNER", +0.10),
                "LOSER":  _ohlcv_with_r60("LOSER",  -0.05),
                "META":   _ohlcv_with_r60("META",   -0.0466),
            },
            counters={},
        )
        RegimeMomentumAlignmentTask().run(ctx)
        assert abs(ctx.candidates[0].rank_score - 0.80) < 1e-9
        assert abs(ctx.candidates[1].rank_score - 0.40) < 1e-9
        assert abs(ctx.candidates[2].rank_score - 0.495) < 1e-9
        assert ctx.counters.get("regime_momentum_shrunk") == 2

    def test_non_momentum_regime_no_op(self):
        from kernel.pipeline.task_buy_quality_gates import (
            RegimeMomentumAlignmentTask,
        )
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "regime_momentum": {"enabled": True}
            }}},
            regime="BEAR", hurst=0.77,
            candidates=[_cand("X", rank_score=0.80)],
            ohlcv={"X": _ohlcv_with_r60("X", -0.10)},
            counters={},
        )
        RegimeMomentumAlignmentTask().run(ctx)
        assert ctx.candidates[0].rank_score == 0.80

    def test_low_hurst_no_op(self):
        from kernel.pipeline.task_buy_quality_gates import (
            RegimeMomentumAlignmentTask,
        )
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "regime_momentum": {"enabled": True}
            }}},
            regime="BULL_CALM", hurst=0.40,  # below floor
            candidates=[_cand("X", rank_score=0.80)],
            ohlcv={"X": _ohlcv_with_r60("X", -0.10)},
            counters={},
        )
        RegimeMomentumAlignmentTask().run(ctx)
        assert ctx.candidates[0].rank_score == 0.80


class TestDeepDrawdownVeto:

    def test_default_off_no_op(self):
        from kernel.pipeline.task_buy_quality_gates import DeepDrawdownVetoTask
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {}}},
            candidates=[_cand("DEEP")],
            ohlcv={"DEEP": _ohlcv_with_dd("DEEP", -0.30)},
            counters={},
        )
        DeepDrawdownVetoTask().run(ctx)
        assert len(ctx.candidates) == 1

    def test_deep_dd_no_fund_vetoed(self):
        from kernel.pipeline.task_buy_quality_gates import DeepDrawdownVetoTask
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "deep_drawdown_veto": {"enabled": True, "dd_threshold": 0.20}
            }}},
            candidates=[
                _cand("OK",    features={"sue_signal": 0.5, "pead_signal": 0.0}),
                _cand("DEEP",  features={"sue_signal": 0.0, "pead_signal": 0.0}),
                _cand("META",  features={}),
            ],
            ohlcv={
                "OK":   _ohlcv_with_dd("OK",   -0.05),
                "DEEP": _ohlcv_with_dd("DEEP", -0.30),
                "META": _ohlcv_with_dd("META", -0.221),
            },
            counters={},
        )
        DeepDrawdownVetoTask().run(ctx)
        survivors = [c.ticker for c in ctx.candidates]
        assert "OK"   in survivors
        assert "DEEP" not in survivors
        assert "META" not in survivors
        assert ctx.counters.get("deep_dd_vetoed") == 2

    def test_deep_dd_with_sue_confirmation_kept(self):
        from kernel.pipeline.task_buy_quality_gates import DeepDrawdownVetoTask
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "deep_drawdown_veto": {"enabled": True}
            }}},
            candidates=[
                _cand("CONFIRMED",   features={"sue_signal": 1.5, "pead_signal": 0.0}),
                _cand("UNCONFIRMED", features={"sue_signal": 0.0, "pead_signal": 0.0}),
            ],
            ohlcv={
                "CONFIRMED":   _ohlcv_with_dd("CONFIRMED",   -0.25),
                "UNCONFIRMED": _ohlcv_with_dd("UNCONFIRMED", -0.25),
            },
            counters={},
        )
        DeepDrawdownVetoTask().run(ctx)
        survivors = [c.ticker for c in ctx.candidates]
        assert "CONFIRMED"   in survivors
        assert "UNCONFIRMED" not in survivors

    def test_no_ohlcv_keeps_candidate_permissive(self):
        from kernel.pipeline.task_buy_quality_gates import DeepDrawdownVetoTask
        ctx = SimpleNamespace(
            config={"ranking": {"buy_quality_gates": {
                "deep_drawdown_veto": {"enabled": True}
            }}},
            candidates=[_cand("NODATA")],
            ohlcv={},
            counters={},
        )
        DeepDrawdownVetoTask().run(ctx)
        assert any(c.ticker == "NODATA" for c in ctx.candidates)


class TestPureHelpers:

    def test_trailing_return_basic(self):
        from kernel.pipeline.task_buy_quality_gates import _trailing_return
        df = _ohlcv_with_r60("T", +0.20)
        ret = _trailing_return(df, days=60)
        assert ret is not None
        assert abs(ret - 0.20) < 0.01

    def test_trailing_return_insufficient_history(self):
        from kernel.pipeline.task_buy_quality_gates import _trailing_return
        df = pd.DataFrame({"close": np.linspace(100, 110, 30)})
        assert _trailing_return(df, days=60) is None

    def test_dd_from_high(self):
        from kernel.pipeline.task_buy_quality_gates import _dd_from_high
        df = _ohlcv_with_dd("T", -0.25)
        dd = _dd_from_high(df, 252)
        assert dd is not None
        assert dd < -0.20
        assert dd > -0.30
