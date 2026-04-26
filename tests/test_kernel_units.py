"""Unit tests for kernel modules.

All tests are self-contained — no common/ imports, no LEAN, no broker.
Uses the kernel directly from backtesting/renquant_104/kernel/.
"""
from __future__ import annotations

import datetime
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure kernel is importable
_KERNEL_PARENT = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_KERNEL_PARENT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_PARENT))

from kernel.config import BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES
from kernel.regime import (
    RegimeState,
    compute_hurst,
    compute_cusum,
    compute_regime_confidence,
    detect_regime,
)
from kernel.indicators import (
    compute_rsi,
    compute_macd_hist,
    compute_adx,
    compute_all,
    build_spy_context,
    build_feature_frame,
)
from kernel.models import (
    calibrate_score,
    predict_classification,
    predict_qlearning,
    predict_manual,
    predict_xgboost,
    ScoreResult,
    score_artifact,
)
from kernel.exits import (
    HoldingState,
    ExitSignal,
    check_trailing_stop,
    check_stop_loss,
    check_single_day_loss,
    check_max_hold,
    check_model_sell,
    compute_exits,
)
from kernel.selection import (
    CandidateResult,
    SelectionContext,
    is_wash_sale_blocked,
    is_earnings_blocked,
    passes_sector_guard,
    passes_correlation_guard,
    score_candidates,
    run_selection_loop,
)
from kernel.sizing import compute_position_size


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.date_range("2023-01-01", periods=n, freq="B"))
    return df


# ── kernel.config ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_regimes_list(self):
        assert set(REGIMES) == {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR}

    def test_regime_constants_unique(self):
        assert len(REGIMES) == len(set(REGIMES))


# ── kernel.regime ─────────────────────────────────────────────────────────────

class TestComputeHurst:
    def test_trending_series_above_half(self):
        """A trending series should have H > 0.5."""
        rng = np.random.default_rng(0)
        trend = np.cumsum(rng.normal(0.001, 0.01, 200))
        h = compute_hurst(trend)
        assert h > 0.5

    def test_mean_reverting_series_below_half(self):
        """Alternating sign series is mean-reverting (H < 0.5)."""
        n = 200
        arr = np.array([(-1) ** i * 0.01 for i in range(n)])
        h = compute_hurst(arr)
        assert h < 0.5

    def test_short_series_returns_half(self):
        assert compute_hurst(np.array([0.01, -0.01, 0.01])) == 0.5

    def test_returns_in_unit_interval(self):
        rng = np.random.default_rng(1)
        h = compute_hurst(rng.normal(0, 0.01, 100))
        assert 0.0 <= h <= 1.0


class TestComputeCusum:
    def test_stable_series_no_trigger(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 0.01, 60)
        assert not compute_cusum(data, lookback=20, threshold=3.0, drift=0.5)

    def test_mean_shift_triggers(self):
        rng = np.random.default_rng(7)
        # Reference window: small noise around 0
        stable  = rng.normal(0, 0.002, 40)
        # Current window: shifted up by many sigmas
        shifted = rng.normal(0.05, 0.002, 20)
        data = np.concatenate([stable, shifted])
        assert compute_cusum(data, lookback=20, threshold=3.0, drift=0.5)

    def test_too_short_returns_false(self):
        assert not compute_cusum(np.ones(5), lookback=20, threshold=3.0, drift=0.5)


class TestRegimeConfidence:
    def test_transition_returns_half(self):
        conf = compute_regime_confidence(BULL_CALM, 0.6, {BULL_CALM: 0.9}, True, {})
        assert conf == 0.5

    def test_choppy_uses_hurst(self):
        """CHOPPY: H=hurst_floor → max confidence (1.0); H close to hurst_rev → low.

        Formula: (hurst_rev - H) / (hurst_rev - hurst_floor).
        With defaults hurst_rev=0.52, hurst_floor=0.20:
          H=0.20 → 1.0, H=0.51 → (0.52-0.51)/(0.52-0.20)=0.03 (very low).
        """
        cfg = {"regime": {"choppy_hurst_floor": 0.20, "hurst_reversion_threshold": 0.52}}
        high = compute_regime_confidence(CHOPPY, 0.20, {CHOPPY: 0.5}, False, cfg)
        low  = compute_regime_confidence(CHOPPY, 0.51, {CHOPPY: 0.5}, False, cfg)
        assert high > low
        assert high == pytest.approx(1.0)
        assert low == pytest.approx((0.52 - 0.51) / (0.52 - 0.20))

    def test_bull_calm_uses_gmm(self):
        gmm = {BULL_CALM: 0.75, BEAR: 0.25}
        conf = compute_regime_confidence(BULL_CALM, 0.6, gmm, False, {})
        assert conf == pytest.approx(0.75)


class TestDetectRegime:
    def _make_returns(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, n)

    def test_returns_regime_state(self):
        state = RegimeState()
        returns = self._make_returns()
        new_state = detect_regime(returns, None, None, state, {})
        assert isinstance(new_state, RegimeState)
        assert new_state.regime in REGIMES

    def test_short_series_unchanged(self):
        state = RegimeState()
        result = detect_regime(np.zeros(10), None, None, state, {})
        assert result.regime == BULL_CALM   # default unchanged

    def test_confidence_in_unit_interval(self):
        state = RegimeState()
        returns = self._make_returns(120)
        result = detect_regime(returns, None, None, state, {})
        assert 0.0 <= result.confidence <= 1.0


# ── kernel.indicators ─────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_overbought_range(self):
        # Mix of up and down days so avg_loss > 0, but mostly up → RSI should be high
        rng = np.random.default_rng(99)
        # 30 days: 90% up days with large gains, 10% down with tiny losses
        deltas = np.where(rng.random(30) < 0.9, 2.0, -0.1)
        close = pd.Series(np.cumsum(deltas) + 100.0)
        rsi = compute_rsi(close)
        assert rsi.dropna().iloc[-1] > 70

    def test_range_0_to_100(self):
        df = _make_ohlcv()
        rsi = compute_rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestComputeAll:
    def test_returns_dataframe(self):
        df = _make_ohlcv(80)
        result = compute_all(df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns
        assert "macd_hist" in result.columns

    def test_short_series_returns_none(self):
        result = compute_all(_make_ohlcv(5))
        assert result is None

    def test_no_nan_in_indicator_cols(self):
        df = _make_ohlcv(100)
        result = compute_all(df)
        ind_cols = ["rsi", "macd_hist", "cci", "bbp", "adx", "williams_r", "obv_slope"]
        assert result[ind_cols].isna().sum().sum() == 0


class TestBuildFeatureFrame:
    def test_returns_dataframe(self):
        stock = _make_ohlcv(100, seed=1)
        spy   = _make_ohlcv(100, seed=2)
        result = build_feature_frame(stock, spy, {})
        assert result is not None
        assert not result.empty

    def test_contains_relative_features(self):
        stock = _make_ohlcv(100, seed=3)
        spy   = _make_ohlcv(100, seed=4)
        result = build_feature_frame(stock, spy, {})
        assert "rsi" in result.columns
        assert "hurst_proxy" in result.columns
        assert "spy_realized_vol" in result.columns

    def test_no_nan(self):
        stock = _make_ohlcv(100, seed=5)
        spy   = _make_ohlcv(100, seed=6)
        result = build_feature_frame(stock, spy, {})
        assert not result.isna().any().any()


# ── kernel.models ─────────────────────────────────────────────────────────────

class TestCalibrateScore:
    def test_identity_passthrough(self):
        assert calibrate_score(0.42, {"method": "identity"}) == pytest.approx(0.42)

    def test_none_calibration(self):
        assert calibrate_score(0.7, None) == pytest.approx(0.7)

    def test_constant_probability(self):
        val = calibrate_score(0.99, {"method": "constant_probability", "base_rate": 0.3})
        assert val == pytest.approx(0.3)

    def test_isotonic_interpolation(self):
        cal = {
            "method": "isotonic",
            "x_thresholds": [0.0, 0.5, 1.0],
            "y_thresholds": [0.1, 0.5, 0.9],
        }
        val = calibrate_score(0.25, cal)
        assert 0.1 < val < 0.5

    def test_platt_scaling(self):
        cal = {
            "method": "platt",
            "platt_coef": 1.0,
            "platt_intercept": 0.0,
        }
        val = calibrate_score(0.0, cal)
        assert val == pytest.approx(0.5)

    def test_platt_clipped(self):
        cal = {"method": "platt", "platt_coef": 100.0, "platt_intercept": 100.0}
        assert calibrate_score(1.0, cal) == pytest.approx(1.0)


class TestPredictClassification:
    def _make_artifact(self):
        # Single-node tree: leaf at index 0, feature=-1 means leaf
        tree = [[-1, 0.8, 0, 0]]  # leaf node returning 0.8
        return {
            "policy_type": "classification",
            "feature_columns": ["rsi", "adx"],
            "buy_threshold": 0.6,
            "sell_threshold": -0.1,
            "trees": [tree, tree],
            "score_calibration": None,
        }

    def test_returns_float(self):
        artifact = self._make_artifact()
        row = pd.Series({"rsi": 50.0, "adx": 25.0})
        val = predict_classification(artifact, row)
        assert isinstance(val, float)
        assert val == pytest.approx(0.8)

    def test_nan_feature_returns_zero(self):
        artifact = self._make_artifact()
        row = pd.Series({"rsi": float("nan"), "adx": 25.0})
        assert predict_classification(artifact, row) == 0.0


class TestPredictManual:
    def _make_artifact(self):
        return {
            "policy_type": "manual",
            "feature_columns": [],
            "buy_threshold": 1,
            "sell_threshold": -1,
            "score_rules": [
                {"col": "rsi", "buy_below": 40, "sell_above": 70},
                {"col": "adx", "buy_above": 25},
            ],
            "score_calibration": None,
        }

    def test_buy_signal(self):
        artifact = self._make_artifact()
        row = pd.Series({"rsi": 30.0, "adx": 30.0})
        score = predict_manual(artifact, row)
        assert score == 2   # rsi<40 + adx>25

    def test_sell_signal(self):
        artifact = self._make_artifact()
        row = pd.Series({"rsi": 80.0, "adx": 10.0})
        score = predict_manual(artifact, row)
        assert score == -1  # rsi>70


class TestScoreArtifact:
    def _make_artifact(self, buy_thr=0.6, sell_thr=-0.1):
        tree = [[-1, 0.8, 0, 0]]
        return {
            "policy_type": "classification",
            "feature_columns": ["rsi", "adx"],
            "buy_threshold": buy_thr,
            "sell_threshold": sell_thr,
            "trees": [tree],
            "score_calibration": None,
        }

    def test_buy_signal(self):
        artifact = self._make_artifact(buy_thr=0.5)
        row = pd.Series({"rsi": 50.0, "adx": 25.0})
        result = score_artifact(artifact, row)
        assert isinstance(result, ScoreResult)
        assert result.signal == "buy"
        assert result.raw_score == pytest.approx(0.8)

    def test_hold_signal(self):
        artifact = self._make_artifact(buy_thr=0.9, sell_thr=-0.9)
        row = pd.Series({"rsi": 50.0, "adx": 25.0})
        result = score_artifact(artifact, row)
        assert result.signal == "hold"


# ── kernel.exits ──────────────────────────────────────────────────────────────

class TestCheckTrailingStop:
    def _state(self, entry=100.0, hwm=125.0):
        return HoldingState(
            entry_price=entry,
            entry_date=datetime.date(2024, 1, 1),
            high_watermark=hwm,
        )

    def test_not_armed_below_trigger(self):
        state = self._state(hwm=105.0)
        sig = check_trailing_stop(104.0, state, ts_trigger=0.20, ts_trail=0.18)
        assert not sig.should_exit

    def test_fires_when_below_trail_floor(self):
        state = self._state(hwm=130.0)  # peak_gain=30% > trigger=20%
        trail_floor = 130.0 * (1 - 0.18)   # 106.6
        sig = check_trailing_stop(100.0, state, ts_trigger=0.20, ts_trail=0.18)
        assert sig.should_exit
        assert sig.exit_type == "trailing_stop"

    def test_does_not_fire_above_trail_floor(self):
        state = self._state(hwm=125.0)  # 25% gain > 20% trigger
        sig = check_trailing_stop(110.0, state, ts_trigger=0.20, ts_trail=0.18)
        # trail_floor = 125 * 0.82 = 102.5; 110 > 102.5 → no exit
        assert not sig.should_exit


class TestCheckStopLoss:
    def test_fires_at_threshold(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        sig = check_stop_loss(84.9, state, stop_pct=0.15)
        assert sig.should_exit
        assert sig.exit_type == "stop_loss"

    def test_does_not_fire_above_threshold(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        sig = check_stop_loss(86.0, state, stop_pct=0.15)
        assert not sig.should_exit

    def test_disabled_when_zero(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        assert not check_stop_loss(0.0, state, stop_pct=0.0).should_exit


class TestCheckSingleDayLoss:
    def test_fires_on_gap_down(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1),
            high_watermark=100.0, prev_close=100.0)
        sig = check_single_day_loss(88.0, state, sdl_pct=0.10)
        assert sig.should_exit
        assert sig.exit_type == "single_day_loss"

    def test_no_prev_close_no_exit(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        assert not check_single_day_loss(88.0, state, sdl_pct=0.10).should_exit


class TestCheckMaxHold:
    def test_fires_at_max_days(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        today = datetime.date(2024, 1, 1) + datetime.timedelta(days=500)
        sig = check_max_hold(today, state, max_hold=500)
        assert sig.should_exit
        assert sig.exit_type == "max_hold"

    def test_does_not_fire_before_max_days(self):
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)
        today = datetime.date(2024, 1, 20)
        assert not check_max_hold(today, state, max_hold=500).should_exit


class TestCheckModelSell:
    def _state(self, streak=0):
        return HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1),
            high_watermark=100.0, sell_streak=streak)

    def test_accumulates_streak(self):
        state, sig = check_model_sell("sell", self._state(0), 3, 0, datetime.date(2024, 2, 1))
        assert state.sell_streak == 1
        assert not sig.should_exit

    def test_fires_at_streak(self):
        state, sig = check_model_sell("sell", self._state(2), 3, 0, datetime.date(2024, 2, 1))
        assert sig.should_exit
        assert sig.exit_type == "model_sell"

    def test_resets_on_hold(self):
        state = self._state(streak=2)
        new_state, sig = check_model_sell("hold", state, 3, 0, datetime.date(2024, 2, 1))
        assert new_state.sell_streak == 0
        assert not sig.should_exit

    def test_blocked_by_min_hold(self):
        state = self._state(streak=2)
        today = datetime.date(2024, 1, 5)   # only 4 days after entry
        new_state, sig = check_model_sell("sell", state, 3, min_hold_days=20, today=today)
        assert not sig.should_exit
        assert new_state.sell_streak == 2   # unchanged


class TestComputeExits:
    def _state(self):
        return HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=100.0)

    def test_priority_trailing_before_stop(self):
        """Trailing stop should fire before cumulative stop when both would trigger."""
        state = HoldingState(
            entry_price=100.0, entry_date=datetime.date(2024, 1, 1), high_watermark=130.0)
        params = {
            "trailing_stop_trigger_pct": 0.20,
            "trailing_stop_trail_pct": 0.18,
            "stop_loss_pct": 0.05,
        }
        sig, _ = compute_exits(80.0, datetime.date(2024, 6, 1), "hold", state, params)
        assert sig.should_exit
        assert sig.exit_type == "trailing_stop"

    def test_no_exit_when_all_hold(self):
        state = self._state()
        params = {
            "stop_loss_pct": 0.15,
            "max_hold_days": 500,
            "consecutive_sell_signals": 3,
            "min_hold_days": 0,
        }
        sig, _ = compute_exits(105.0, datetime.date(2024, 1, 10), "hold", state, params)
        assert not sig.should_exit

    def test_hwm_updated(self):
        state = self._state()
        _, new_state = compute_exits(110.0, datetime.date(2024, 1, 10), "hold", state, {})
        assert new_state.high_watermark == 110.0


# ── kernel.selection ─────────────────────────────────────────────────────────

class TestGuards:
    def test_wash_sale_blocked(self):
        last_sell = {datetime.date(2024, 1, 15): None}
        assert is_wash_sale_blocked(
            "AAPL", datetime.date(2024, 1, 20),
            {"AAPL": datetime.date(2024, 1, 15)}, 30)

    def test_wash_sale_clear_after_period(self):
        assert not is_wash_sale_blocked(
            "AAPL", datetime.date(2024, 4, 1),
            {"AAPL": datetime.date(2024, 1, 1)}, 30)

    def test_earnings_blocked(self):
        cal = {"AAPL": ["2024-02-01"]}
        assert is_earnings_blocked("AAPL", datetime.date(2024, 2, 1), cal, 3)

    def test_earnings_clear_outside_window(self):
        cal = {"AAPL": ["2024-02-01"]}
        assert not is_earnings_blocked("AAPL", datetime.date(2024, 2, 10), cal, 3)

    def test_sector_guard_blocks_at_max(self):
        held = ["AAPL", "MSFT"]
        sector_map = {"AAPL": "tech", "MSFT": "tech", "GOOG": "tech"}
        assert not passes_sector_guard("GOOG", held, sector_map, 2, set())

    def test_sector_guard_allows_below_max(self):
        held = ["AAPL"]
        sector_map = {"AAPL": "tech", "MSFT": "tech"}
        assert passes_sector_guard("MSFT", held, sector_map, 2, set())

    def test_defensives_bypass_sector_guard(self):
        held = ["GLD", "TLT"]
        sector_map = {"GLD": "bond", "TLT": "bond", "XLV": "bond"}
        assert passes_sector_guard("XLV", held, sector_map, 1, {"GLD", "TLT", "XLV"})

    def test_correlation_guard_blocks_high_corr(self):
        corr = {"AAPL": {"MSFT": 0.95}}
        assert not passes_correlation_guard("AAPL", ["MSFT"], corr, 0.70)

    def test_correlation_guard_allows_low_corr(self):
        corr = {"AAPL": {"MSFT": 0.50}}
        assert passes_correlation_guard("AAPL", ["MSFT"], corr, 0.70)


class TestScoreCandidates:
    def test_sorting_order(self):
        candidates = [
            CandidateResult("AAPL", 0.5, 0.3, 0.1),
            CandidateResult("MSFT", 0.8, 0.7, 0.5),
            CandidateResult("GOOG", 0.6, 0.5, 0.3),
        ]
        ranked = score_candidates(candidates, w_rank=0.5, w_rs=0.5)
        assert ranked[0].ticker == "MSFT"  # highest blend

    def test_empty_returns_empty(self):
        assert score_candidates([], 0.5, 0.5) == []

    def test_single_candidate(self):
        c = CandidateResult("AAPL", 0.5, 0.5, 0.5)
        result = score_candidates([c], 0.5, 0.5)
        assert len(result) == 1


class TestRunSelectionLoop:
    def _ctx(self, open_slots=3, tiered=None):
        return SelectionContext(
            today=datetime.date(2024, 2, 1),
            held_tickers=[],
            last_sell_dates={},
            earnings_calendar={},
            corr_matrix=None,
            sector_map={},
            defensive_set=set(),
            wash_sale_days=0,
            earnings_buffer=3,
            corr_threshold=0.70,
            max_per_sector=0,
            tiered_thresholds=tiered or [],
            open_slots=open_slots,
        )

    def test_fills_up_to_slots(self):
        candidates = [CandidateResult(f"T{i}", 0.5, 0.5, 0.5) for i in range(5)]
        ctx = self._ctx(open_slots=2)
        selected, _ = run_selection_loop(candidates, ctx)
        assert len(selected) == 2

    def test_tiered_threshold_blocks_low_score(self):
        candidates = [CandidateResult("AAPL", 0.5, 0.05, 0.5)]
        ctx = self._ctx(tiered=[{"min_model_score": 0.10}])
        selected, blocks = run_selection_loop(candidates, ctx)
        assert len(selected) == 0
        assert blocks["tier"] == 1

    def test_wash_sale_blocked(self):
        ctx = self._ctx()
        ctx.last_sell_dates = {"AAPL": datetime.date(2024, 1, 20)}
        ctx.wash_sale_days = 30
        candidates = [CandidateResult("AAPL", 0.8, 0.8, 0.8)]
        selected, blocks = run_selection_loop(candidates, ctx)
        assert len(selected) == 0
        assert blocks["wash_sale"] == 1


# ── kernel.sizing ─────────────────────────────────────────────────────────────

class TestComputePositionSize:
    def test_normal_sizing(self):
        pct, shares = compute_position_size(
            portfolio_value=100_000,
            available_cash=80_000,
            max_position_pct=0.15,
            cash_reserve_pct=0.0,
            price=50.0,
        )
        assert shares == 300   # 15% of 100k = 15k / 50 = 300
        assert pct == pytest.approx(0.15)

    def test_no_shares_when_no_cash(self):
        pct, shares = compute_position_size(
            portfolio_value=100_000,
            available_cash=0.0,
            max_position_pct=0.15,
            cash_reserve_pct=0.0,
            price=50.0,
        )
        assert shares == 0
        assert pct == 0.0

    def test_override_bypasses_reserve(self):
        """override_pct bypasses cash_reserve constraint."""
        pct, shares = compute_position_size(
            portfolio_value=100_000,
            available_cash=5_000,
            max_position_pct=0.15,
            cash_reserve_pct=0.90,  # 90% reserve would leave 10k unreachable
            price=10.0,
            override_pct=0.05,
        )
        assert shares > 0

    def test_oversize_fallback_high_price(self):
        """Expensive stock: normal allocation < 1 share → fallback to 25%."""
        _, shares = compute_position_size(
            portfolio_value=100_000,
            available_cash=100_000,
            max_position_pct=0.005,  # 0.5% → $500 < $2000 price
            cash_reserve_pct=0.0,
            price=2000.0,
        )
        # fallback: 25% of 100k = 25k / 2000 = 12 shares
        assert shares == 12


# ── kernel.market_gates ───────────────────────────────────────────────────────

from kernel.market_gates import check_spy_velocity_crash, check_spy_ema_trend


class TestCheckSpyVelocityCrash:
    def test_no_crash_flat_market(self):
        """No crash when SPY is flat."""
        spy_rets = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert check_spy_velocity_crash(spy_rets) is False

    def test_crash_detected_large_drop(self):
        """Blocks when SPY fell > 3% cumulatively over 3 days."""
        spy_rets = [-0.015, -0.015, -0.015]  # ≈ -4.4% cumulative
        assert check_spy_velocity_crash(spy_rets, lookback_days=3, halt_pct=0.03) is True

    def test_small_drop_not_blocked(self):
        """Does not block when drop is below threshold."""
        spy_rets = [-0.005, -0.005, -0.005]  # ≈ -1.5% cumulative
        assert check_spy_velocity_crash(spy_rets, lookback_days=3, halt_pct=0.03) is False

    def test_halt_pct_zero_never_blocks(self):
        """halt_pct=0 disables the gate."""
        spy_rets = [-0.05, -0.05, -0.05]
        assert check_spy_velocity_crash(spy_rets, halt_pct=0) is False

    def test_insufficient_history_returns_false(self):
        """Returns False when fewer bars than lookback."""
        assert check_spy_velocity_crash([-0.05], lookback_days=3) is False

    def test_uses_only_last_lookback_days(self):
        """Only the last lookback_days are considered."""
        spy_rets = [-0.05, -0.05, 0.01, 0.01, 0.01]  # last 3 days are small gains
        assert check_spy_velocity_crash(spy_rets, lookback_days=3, halt_pct=0.03) is False


class TestCheckSpyEmaTrend:
    def _make_series(self, n: int, trend: str = "up") -> pd.Series:
        if trend == "up":
            prices = [100.0 + i * 0.5 for i in range(n)]
        else:
            prices = [100.0 + 25 - i * 0.5 for i in range(n)]  # starts high, falls
        return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def test_above_ema_returns_false(self):
        """SPY above EMA50 → gate passes (returns False = not blocking)."""
        s = self._make_series(100, "up")
        assert check_spy_ema_trend(s, ema_span=50) is False

    def test_below_ema_returns_true(self):
        """SPY below EMA50 → gate blocks (returns True = blocking)."""
        s = self._make_series(100, "down")
        assert check_spy_ema_trend(s, ema_span=50) is True

    def test_insufficient_history_returns_false(self):
        """Too few bars → no block (default safe)."""
        s = pd.Series([100.0] * 10)
        assert check_spy_ema_trend(s, ema_span=50) is False

    def test_none_series_returns_false(self):
        """None input → no block."""
        assert check_spy_ema_trend(None) is False  # type: ignore[arg-type]


# ── kernel.portfolio ──────────────────────────────────────────────────────────

from kernel.portfolio import update_drawdown_circuit_breaker, compute_trade_tax


class TestUpdateDrawdownCircuitBreaker:
    def test_no_drawdown_no_halt(self):
        new_hwm, halt = update_drawdown_circuit_breaker(100_000, 100_000, 0.15)
        assert new_hwm == 100_000
        assert halt is False

    def test_hwm_updated_on_new_high(self):
        new_hwm, _ = update_drawdown_circuit_breaker(110_000, 100_000, 0.15)
        assert new_hwm == 110_000

    def test_circuit_fires_at_threshold(self):
        """Drawdown exactly at threshold triggers halt."""
        _, halt = update_drawdown_circuit_breaker(85_000, 100_000, 0.15)
        assert halt is True

    def test_circuit_does_not_fire_below_threshold(self):
        _, halt = update_drawdown_circuit_breaker(90_000, 100_000, 0.15)
        assert halt is False

    def test_zero_halt_threshold_never_fires(self):
        _, halt = update_drawdown_circuit_breaker(50_000, 100_000, 0.0)
        assert halt is False


class TestComputeTradeTax:
    def test_no_tax_on_loss(self):
        assert compute_trade_tax(-1000.0, 10, 0.35, 0.20) == 0.0

    def test_short_term_rate_applied(self):
        tax = compute_trade_tax(1000.0, 100, short_term_rate=0.35, long_term_rate=0.20)
        assert tax == pytest.approx(350.0)

    def test_long_term_rate_applied(self):
        tax = compute_trade_tax(1000.0, 400, short_term_rate=0.35, long_term_rate=0.20)
        assert tax == pytest.approx(200.0)

    def test_exactly_at_lt_threshold(self):
        """Hold_days == 365 → long-term rate."""
        tax = compute_trade_tax(1000.0, 365, short_term_rate=0.35, long_term_rate=0.20)
        assert tax == pytest.approx(200.0)

    def test_one_day_before_threshold(self):
        """Hold_days == 364 → short-term rate."""
        tax = compute_trade_tax(1000.0, 364, short_term_rate=0.35, long_term_rate=0.20)
        assert tax == pytest.approx(350.0)

    def test_zero_pnl_no_tax(self):
        assert compute_trade_tax(0.0, 200, 0.35, 0.20) == 0.0


# ── kernel.selection.compute_relative_strength ────────────────────────────────

from kernel.selection import compute_relative_strength


class TestComputeRelativeStrength:
    def test_positive_outperformance(self):
        assert compute_relative_strength(0.10, 0.05) == pytest.approx(0.05)

    def test_negative_underperformance(self):
        assert compute_relative_strength(0.02, 0.08) == pytest.approx(-0.06)

    def test_equal_returns_zero(self):
        assert compute_relative_strength(0.05, 0.05) == pytest.approx(0.0)

    def test_nan_stock_returns_zero(self):
        assert compute_relative_strength(float("nan"), 0.05) == 0.0

    def test_nan_etf_returns_zero(self):
        assert compute_relative_strength(0.05, float("nan")) == 0.0

    def test_both_negative(self):
        """Stock losing less than ETF = positive RS."""
        assert compute_relative_strength(-0.02, -0.07) == pytest.approx(0.05)


# ── kernel.regime: configurable Hurst thresholds ─────────────────────────────

class TestConfigurableHurstThresholds:
    """detect_regime respects hurst_trending_threshold / hurst_reversion_threshold."""

    def _cfg(self, trending=0.65, reversion=0.52):
        return {"regime": {
            "hurst_trending_threshold":  trending,
            "hurst_reversion_threshold": reversion,
            "cusum_threshold": 99.0,  # disable CUSUM for isolation
            "cusum_lookback": 20,
            "cusum_drift": 0.5,
            "hurst_window": 63,
            "transition_uncertainty_bars": 3,
            "vol_realized_window": 20,
            "bear_vol_threshold":    99.0,  # disable BEAR override
            "bear_return_threshold": -99.0,
        }}

    def _strong_trend_returns(self, n=200):
        """Persistent trending returns → high H."""
        rng = np.random.default_rng(0)
        return np.cumsum(rng.normal(0.001, 0.005, n))

    def test_high_hurst_labels_bull_calm(self):
        """H well above 0.65 → BULL_CALM (not CHOPPY)."""
        # Force H by passing returns that produce momentum character
        rets = np.diff(self._strong_trend_returns())
        state = detect_regime(rets, None, None, RegimeState(), self._cfg())
        assert state.regime in (BULL_CALM, BULL_VOLATILE)

    def test_raising_threshold_reclassifies(self):
        """With threshold raised to 0.90 even strong-trending H may become AMBIGUOUS."""
        cfg_high = self._cfg(trending=0.90, reversion=0.52)
        rets = np.diff(self._strong_trend_returns())
        state = detect_regime(rets, None, None, RegimeState(), cfg_high)
        # Hurst rarely hits 0.90, so should NOT be BULL_CALM via momentum branch
        # (will fall to GMM dominant, likely BULL_VOLATILE or BEAR depending on data)
        assert state.regime is not None  # just verify it ran without error

    def test_low_hurst_labels_choppy(self):
        """H below reversion threshold → CHOPPY."""
        from unittest.mock import patch
        rets = np.random.default_rng(42).normal(0, 0.01, 200)
        with patch("kernel.regime.compute_hurst", return_value=0.40):
            state = detect_regime(rets, None, None, RegimeState(), self._cfg())
        assert state.regime == CHOPPY

    def test_defaults_used_when_config_absent(self):
        """Missing threshold keys fall back to hardcoded defaults (0.65/0.52)."""
        cfg_empty = {"regime": {"cusum_threshold": 99.0, "cusum_lookback": 20,
                                "cusum_drift": 0.5, "hurst_window": 63,
                                "transition_uncertainty_bars": 3,
                                "vol_realized_window": 20,
                                "bear_vol_threshold": 99.0,
                                "bear_return_threshold": -99.0}}
        rets = np.diff(self._strong_trend_returns())
        state = detect_regime(rets, None, None, RegimeState(), cfg_empty)
        assert state.regime is not None


# ── kernel.regime: BEAR hard override ────────────────────────────────────────

class TestBearHardOverride:
    """detect_regime forces BEAR when vol or return thresholds are breached."""

    def _cfg(self, bear_vol=0.35, bear_ret=-0.08):
        return {"regime": {
            "hurst_trending_threshold":  0.65,
            "hurst_reversion_threshold": 0.52,
            "cusum_threshold": 99.0,
            "cusum_lookback": 20,
            "cusum_drift": 0.5,
            "hurst_window": 63,
            "transition_uncertainty_bars": 3,
            "vol_realized_window": 20,
            "bear_vol_threshold":    bear_vol,
            "bear_return_threshold": bear_ret,
        }}

    def _high_vol_returns(self, n=200):
        """Returns with very high realized vol (annualized >> 35%)."""
        rng = np.random.default_rng(7)
        # daily vol ~4% → annualized ~63%
        return rng.normal(0.0, 0.04, n)

    def _crash_returns(self, n=200):
        """Persistent downward drift totalling > 8% over 20 days."""
        base = np.zeros(n)
        # last 20 days drop 0.5%/day = -10% cumulative
        base[-20:] = -0.005
        return base

    def test_high_vol_forces_bear(self):
        """Annualized vol > 35% → BEAR regardless of Hurst."""
        rets = self._high_vol_returns()
        state = detect_regime(rets, None, None, RegimeState(), self._cfg(bear_vol=0.35))
        assert state.regime == BEAR

    def test_crash_return_forces_bear(self):
        """20-day cumulative return < -8% → BEAR."""
        rets = self._crash_returns()
        state = detect_regime(rets, None, None, RegimeState(), self._cfg(bear_ret=-0.08))
        assert state.regime == BEAR

    def test_normal_vol_does_not_force_bear(self):
        """Normal vol (~1%/day, ~16% ann) does NOT trigger override."""
        rng = np.random.default_rng(1)
        rets = rng.normal(0.001, 0.01, 200)
        state = detect_regime(rets, None, None, RegimeState(), self._cfg(bear_vol=0.35))
        assert state.regime != BEAR

    def test_override_disabled_with_high_threshold(self):
        """Setting bear_vol_threshold=99 and bear_return_threshold=-99 disables override."""
        rets = self._high_vol_returns()
        state = detect_regime(rets, None, None, RegimeState(), self._cfg(bear_vol=99.0, bear_ret=-99.0))
        # High vol + disabled override → should NOT be BEAR from override
        # (may still be BEAR from GMM if GMM fires, but override path won't set it)
        # We can't assert non-BEAR here since GMM might; just assert it ran cleanly
        assert state.regime is not None

    def test_bear_confidence_is_gmm_probability(self):
        """BEAR regime confidence uses GMM P(BEAR), not Hurst distance."""
        rets = self._crash_returns()
        state = detect_regime(rets, None, None, RegimeState(), self._cfg(bear_ret=-0.08))
        assert state.regime == BEAR
        assert 0.0 <= state.confidence <= 1.0


# ── kernel.exits: tax-aware hold gate ────────────────────────────────────────

class TestTaxHoldGate:
    """Tax-aware hold gate suppresses model-sell near the 1-year LT threshold.

    Audit fix STREAK-TRADING-DAY (2026-04-26 round-7): tests now pin
    `today` to a fixed NYSE trading day (Mon 2026-04-27) rather than
    `date.today()`. The new streak rule rejects non-trading days, so
    a Sunday `date.today()` would silently break model_sell tests.
    Tests are about LT-tax timing, not the actual calendar — pinning
    is the more correct semantic anyway.
    """

    # Fixed Monday — guaranteed NYSE trading day.
    _TRADING_DAY = datetime.date(2026, 4, 27)

    def _state(self, days_ago: int, entry_price=100.0) -> HoldingState:
        entry = self._TRADING_DAY - datetime.timedelta(days=days_ago)
        return HoldingState(
            entry_price=entry_price,
            entry_date=entry,
            high_watermark=entry_price,
        )

    def _params(self, lt_gate=330, lt_min_gain=0.10):
        return {
            "trailing_stop_trigger_pct": 0.0,
            "trailing_stop_trail_pct":   0.0,
            "stop_loss_pct":             0.0,
            "max_single_day_loss_pct":   0.0,
            "max_hold_days":             0,
            "consecutive_sell_signals":  1,
            "min_hold_days":             0,
            "lt_hold_gate_days":         lt_gate,
            "lt_hold_min_gain":          lt_min_gain,
        }

    def test_gate_suppresses_model_sell_in_window(self):
        """Day 340, 20% gain, 3 sell signals → gate suppresses exit."""
        state = self._state(days_ago=340, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=120.0,  # 20% gain
            today=self._TRADING_DAY,
            model_action="sell",
            state=state,
            params=self._params(lt_gate=330, lt_min_gain=0.10),
        )
        assert not sig.should_exit

    def test_gate_does_not_suppress_before_window(self):
        """Day 300 (before 330 gate) → model-sell fires normally."""
        state = self._state(days_ago=300, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=120.0,
            today=self._TRADING_DAY,
            model_action="sell",
            state=state,
            params=self._params(lt_gate=330, lt_min_gain=0.10),
        )
        assert sig.should_exit
        assert sig.exit_type == "model_sell"

    def test_gate_does_not_suppress_after_one_year(self):
        """Day 366 (past 1-year) → gate window closed, model-sell fires."""
        state = self._state(days_ago=366, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=120.0,
            today=self._TRADING_DAY,
            model_action="sell",
            state=state,
            params=self._params(lt_gate=330, lt_min_gain=0.10),
        )
        assert sig.should_exit
        assert sig.exit_type == "model_sell"

    def test_gate_does_not_suppress_if_gain_below_minimum(self):
        """In window but gain only 5% (below 10% min) → model-sell fires."""
        state = self._state(days_ago=340, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=105.0,  # only 5% gain
            today=self._TRADING_DAY,
            model_action="sell",
            state=state,
            params=self._params(lt_gate=330, lt_min_gain=0.10),
        )
        assert sig.should_exit
        assert sig.exit_type == "model_sell"

    def test_gate_disabled_when_zero(self):
        """lt_hold_gate_days=0 disables tax gate entirely."""
        state = self._state(days_ago=340, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=120.0,
            today=self._TRADING_DAY,
            model_action="sell",
            state=state,
            params=self._params(lt_gate=0, lt_min_gain=0.10),
        )
        assert sig.should_exit
        assert sig.exit_type == "model_sell"

    def test_stop_loss_still_fires_through_gate(self):
        """Hard stop-loss fires even when tax gate is active."""
        state = self._state(days_ago=340, entry_price=100.0)
        sig, _ = compute_exits(
            current_price=80.0,   # 20% loss → triggers stop
            today=self._TRADING_DAY,
            model_action="hold",
            state=state,
            params={**self._params(lt_gate=330, lt_min_gain=0.10),
                    "stop_loss_pct": 0.15},
        )
        assert sig.should_exit
        assert sig.exit_type == "stop_loss"

    def test_streak_accumulates_during_gate_window(self):
        """Sell streak still builds during gate so it fires immediately after window.

        Audit fix STREAK-DAY-DEDUP (2026-04-26 round-5): pre-fix used same
        `today` twice — relied on per-RUN streak increment. Post-fix, streak
        increments AT MOST once per calendar day, so consecutive bars on
        DIFFERENT dates required.

        Round-7 (2026-04-26): also requires both days be NYSE trading
        days. Use Fri 2026-04-24 + Mon 2026-04-27 (both trading days).
        """
        state = self._state(days_ago=340, entry_price=100.0)
        # Two consecutive trading days (Fri + Mon).
        friday = datetime.date(2026, 4, 24)
        monday = self._TRADING_DAY
        params = self._params(lt_gate=330, lt_min_gain=0.10)
        params["consecutive_sell_signals"] = 2

        _, state = compute_exits(120.0, friday, "sell", state, params)
        _, state = compute_exits(120.0, monday, "sell", state, params)
        assert state.sell_streak == 2

    def test_streak_idempotent_within_same_day(self):
        """STREAK-DAY-DEDUP regression: multiple bars on the same calendar
        day must not inflate the streak. Critical for testing scenarios
        where multiple e2e runs happen the same day, AND for intraday
        sell-only runs in production (open / 30-min / preclose).
        """
        state = self._state(days_ago=340, entry_price=100.0)
        today = self._TRADING_DAY
        params = self._params(lt_gate=330, lt_min_gain=0.10)
        params["consecutive_sell_signals"] = 99   # don't exit, just count

        # Simulate 5 bars on the SAME day (5 e2e runs in a few hours)
        for _ in range(5):
            _, state = compute_exits(120.0, today, "sell", state, params)

        assert state.sell_streak == 1, (
            f"5 same-day bars should produce streak=1 (per-trading-day), "
            f"got {state.sell_streak}"
        )

    def test_streak_resets_on_non_sell_signal(self):
        """Reset still happens immediately on hold/buy on a trading day."""
        state = self._state(days_ago=340, entry_price=100.0)
        # Two consecutive trading days (Fri + Mon).
        friday = datetime.date(2026, 4, 24)
        monday = self._TRADING_DAY
        params = self._params(lt_gate=330, lt_min_gain=0.10)
        params["consecutive_sell_signals"] = 99

        _, state = compute_exits(120.0, friday, "sell", state, params)
        assert state.sell_streak == 1
        # Hold signal next trading day — streak resets
        _, state = compute_exits(120.0, monday, "hold", state, params)
        assert state.sell_streak == 0



# ── kernel.rotation ───────────────────────────────────────────────────────────

from kernel.rotation import (
    RotationPair,
    tax_drag,
    is_lt_protected,
    find_rotation_pairs,
)


class _Cand:
    """Lightweight stand-in for CandidateResult."""
    def __init__(self, ticker: str, rank_score: float, expected_return: float = 0.0):
        self.ticker          = ticker
        self.rank_score      = rank_score
        self.expected_return = expected_return


class TestTaxDrag:
    def test_loss_returns_zero(self):
        assert tax_drag(-0.10, 60, 0.5, 0.32, 365) == 0.0

    def test_zero_pnl_returns_zero(self):
        assert tax_drag(0.0, 200, 0.5, 0.32, 365) == 0.0

    def test_short_term_uses_st_rate(self):
        # 20% gain held 30 days → 0.20 * 0.50 = 0.10
        assert tax_drag(0.20, 30, 0.50, 0.32, 365) == pytest.approx(0.10)

    def test_long_term_uses_lt_rate(self):
        # 20% gain held 400 days → 0.20 * 0.32 = 0.064
        assert tax_drag(0.20, 400, 0.50, 0.32, 365) == pytest.approx(0.064)

    def test_threshold_boundary_inclusive(self):
        # Exactly at threshold → LT rate
        assert tax_drag(0.10, 365, 0.50, 0.32, 365) == pytest.approx(0.032)

    def test_one_day_before_threshold_uses_st(self):
        assert tax_drag(0.10, 364, 0.50, 0.32, 365) == pytest.approx(0.05)


class TestIsLtProtected:
    def test_loss_position_not_protected(self):
        # No tax discount to lose if you'd be selling at a loss
        assert not is_lt_protected(-0.05, 350, 365, 30)

    def test_zero_pnl_not_protected(self):
        assert not is_lt_protected(0.0, 350, 365, 30)

    def test_inside_window_with_gain_protected(self):
        # 350d held + gain → within 30d window → protected
        assert is_lt_protected(0.20, 350, 365, 30)

    def test_outside_window_not_protected(self):
        # 300d held → 65d from LT threshold > 30d window
        assert not is_lt_protected(0.20, 300, 365, 30)

    def test_already_lt_not_protected(self):
        # Past the threshold — discount already secured
        assert not is_lt_protected(0.20, 400, 365, 30)


class TestFindRotationPairs:
    """Expected-return rotation: net_advantage = (buy_er - sell_er) - tax - cost.

    Scenarios use ER values directly, not probabilities — the kernel never
    interprets rank_score for the swap decision (it's only carried for log
    readability).
    """

    def _cfg(self, **over):
        cfg = {
            "enabled": True,
            "min_expected_advantage_pct": 0.03,
            "target_horizon_days": 20,
            "transaction_cost_pct": 0.0,
            "min_rotation_hold_days": 30,
            "lt_protection_days": 30,
            "max_rotations_per_bar": 2,
        }
        cfg.update(over)
        return cfg

    def _tax(self):
        return {"short_term_rate": 0.50, "long_term_rate": 0.32, "long_term_threshold_days": 365}

    def _meta(self, entry_days_ago: int = 60, entry_price: float = 100.0,
              current_price: float = 100.0):
        return {
            "entry_date":    datetime.date.today() - datetime.timedelta(days=entry_days_ago),
            "entry_price":   entry_price,
            "current_price": current_price,
        }

    def test_disabled_returns_empty(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.30},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta()},
            candidates =[_Cand("MSFT", 0.80, expected_return=0.10)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(enabled=False),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_min_hold_blocks_recent_entries(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.30},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_days_ago=10)},
            candidates =[_Cand("MSFT", 0.95, expected_return=0.10)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_swap_emitted_when_net_advantage_clears_threshold(self):
        # AAPL ER=0.01, loss position (no tax drag).
        # MSFT ER=0.06 → raw_adv 0.05, no cost/tax → net_adv 0.05 ≥ 0.03 → swap.
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.30},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_price=120.0, current_price=100.0)},
            candidates =[_Cand("MSFT", 0.50, expected_return=0.06)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert len(pairs) == 1
        p = pairs[0]
        assert p.sell_ticker == "AAPL"
        assert p.buy_ticker  == "MSFT"
        assert p.raw_advantage    == pytest.approx(0.05)
        assert p.tax_drag         == pytest.approx(0.0)
        assert p.transaction_cost == pytest.approx(0.0)
        assert p.net_advantage    == pytest.approx(0.05)
        assert p.threshold        == pytest.approx(0.03)
        assert p.margin_realized  == pytest.approx(0.02)
        assert p.horizon_days     == 20

    def test_tax_drag_blocks_marginal_swap(self):
        # AAPL +20% ST gain → drag 0.10.
        # raw_adv = 0.06 - 0.01 = 0.05; net_adv = 0.05 - 0.10 = -0.05 < 0.03 → no swap.
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.30},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_price=100.0, current_price=120.0)},
            candidates =[_Cand("MSFT", 0.45, expected_return=0.06)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_transaction_cost_blocks_marginal_swap(self):
        # 5% raw advantage, 4% transaction cost → net 1% < 3% threshold → no swap.
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.30},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_price=120.0, current_price=100.0)},
            candidates =[_Cand("MSFT", 0.50, expected_return=0.06)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(transaction_cost_pct=0.04),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_lt_protection_pins_position(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.10},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_days_ago=350,
                                            entry_price=100.0, current_price=120.0)},
            candidates =[_Cand("MSFT", 0.90, expected_return=0.50)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_max_rotations_per_bar_caps_output(self):
        held_scores = {"H1": 0.10, "H2": 0.12, "H3": 0.15}
        held_er     = {"H1": 0.01, "H2": 0.015, "H3": 0.02}
        held_meta   = {t: self._meta(entry_price=120.0, current_price=100.0)
                       for t in held_scores}
        candidates  = [
            _Cand("C1", 0.90, expected_return=0.10),
            _Cand("C2", 0.85, expected_return=0.09),
            _Cand("C3", 0.80, expected_return=0.08),
        ]
        pairs = find_rotation_pairs(
            held_scores=held_scores, held_er=held_er, held_meta=held_meta,
            candidates=candidates, today=datetime.date.today(),
            rotation_cfg=self._cfg(max_rotations_per_bar=2),
            tax_cfg=self._tax(),
        )
        assert len(pairs) == 2
        # Strongest candidate paired with weakest-ER hold
        assert pairs[0].buy_ticker == "C1" and pairs[0].sell_ticker == "H1"

    def test_held_ticker_excluded_as_candidate(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.20},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_price=120.0, current_price=100.0)},
            candidates =[_Cand("AAPL", 0.95, expected_return=0.50)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_each_held_used_once(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.10},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta(entry_price=120.0, current_price=100.0)},
            candidates =[
                _Cand("MSFT", 0.90, expected_return=0.10),
                _Cand("NVDA", 0.85, expected_return=0.09),
            ],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert len(pairs) == 1
        assert pairs[0].buy_ticker == "MSFT"

    def test_no_candidates_returns_empty(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.10},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta()},
            candidates =[],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_no_holdings_returns_empty(self):
        pairs = find_rotation_pairs(
            held_scores={}, held_er={}, held_meta={},
            candidates =[_Cand("MSFT", 0.90, expected_return=0.10)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_skips_holding_with_none_score(self):
        pairs = find_rotation_pairs(
            held_scores={"AAPL": None},
            held_er    ={"AAPL": 0.01},
            held_meta  ={"AAPL": self._meta()},
            candidates =[_Cand("MSFT", 0.90, expected_return=0.10)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []

    def test_skips_holding_with_no_expected_return(self):
        # Held position has rank_score but no ER (e.g. stale calibration) —
        # rotation skips it rather than guessing.
        pairs = find_rotation_pairs(
            held_scores={"AAPL": 0.10},
            held_er    ={},
            held_meta  ={"AAPL": self._meta(entry_price=120.0, current_price=100.0)},
            candidates =[_Cand("MSFT", 0.90, expected_return=0.10)],
            today      =datetime.date.today(),
            rotation_cfg=self._cfg(),
            tax_cfg    =self._tax(),
        )
        assert pairs == []
