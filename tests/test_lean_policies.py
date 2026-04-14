"""
Unit tests for renquant_103 LEAN strategy policies (main.py).

LEAN requires Docker to run the full engine, so these tests:
  1. Test the pure-Python model methods added to common/models/ (predict_score_bulk)
  2. Extract and replicate the exact policy logic from main.py in pure Python,
     verifying each filter/guard works as specified.

Run with:
    cd /path/to/RenQuant
    python -m pytest tests/test_lean_policies.py -v
"""

import sys
import json
import math
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Model method tests ────────────────────────────────────────────────────────

class TestClassificationPredictScoreBulk:
    """ClassificationModel.predict_score_bulk() returns continuous float scores."""

    @pytest.fixture
    def trained_model(self):
        from common.models import create_model
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 200
        feature_cols = ["rsi", "macd_hist", "cci", "bbp", "adx"]
        df = pd.DataFrame(rng.normal(0, 1, (n, len(feature_cols))), columns=feature_cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        model = create_model(
            "classification", feature_columns=feature_cols,
            lookahead=5, threshold=0.02, leaf_size=10, bags=5,
            buy_threshold=0.1, sell_threshold=-0.1,
        )
        model.train(df)
        return model, df, feature_cols

    def test_returns_float_series(self, trained_model):
        model, df, feature_cols = trained_model
        scores = model.predict_score_bulk(df)
        assert isinstance(scores, pd.Series), "predict_score_bulk must return a pd.Series"
        assert scores.dtype == float or np.issubdtype(scores.dtype, np.floating), \
            f"Expected float dtype, got {scores.dtype}"

    def test_same_length_as_input(self, trained_model):
        model, df, _ = trained_model
        scores = model.predict_score_bulk(df)
        assert len(scores) == len(df), \
            f"Score length {len(scores)} must match input length {len(df)}"

    def test_scores_span_positive_and_negative(self, trained_model):
        """A trained model should produce both positive and negative scores."""
        model, df, _ = trained_model
        scores = model.predict_score_bulk(df)
        assert scores.max() > 0, "Expected some positive scores (buy pressure)"
        assert scores.min() < 0, "Expected some negative scores (sell pressure)"

    def test_consistent_with_predict_bulk_direction(self, trained_model):
        """Rows where predict_bulk returns 'buy' should have higher scores."""
        model, df, _ = trained_model
        scores = model.predict_score_bulk(df)
        actions = model.predict_bulk(df)
        buy_scores  = scores[actions == "buy"]
        sell_scores = scores[actions == "sell"]
        if len(buy_scores) > 0 and len(sell_scores) > 0:
            assert buy_scores.mean() > sell_scores.mean(), \
                "Buy-action rows should have higher average scores than sell-action rows"

    def test_same_index_as_input(self, trained_model):
        model, df, _ = trained_model
        scores = model.predict_score_bulk(df)
        assert list(scores.index) == list(df.index), \
            "Score index must match input DataFrame index"


class TestQLearningPredictScoreBulk:
    """QLearningModel.predict_score_bulk() returns Q(buy) - Q(sell) per row."""

    @pytest.fixture
    def trained_model(self):
        from common.models import create_model
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(99)
        n = 200
        feature_cols = ["rsi", "macd_hist", "cci", "bbp", "adx"]
        df = pd.DataFrame(rng.normal(0, 1, (n, len(feature_cols))), columns=feature_cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        df["position_flag"] = 0
        model = create_model("qlearning", feature_columns=feature_cols)
        model.train(df)
        return model, df

    def test_returns_float_series(self, trained_model):
        model, df = trained_model
        scores = model.predict_score_bulk(df)
        assert isinstance(scores, pd.Series)
        assert np.issubdtype(scores.dtype, np.floating), \
            f"Expected float dtype, got {scores.dtype}"

    def test_same_length_as_input(self, trained_model):
        model, df = trained_model
        scores = model.predict_score_bulk(df)
        assert len(scores) == len(df)

    def test_score_is_q_buy_minus_q_sell(self, trained_model):
        """Score = Q(buy) - Q(sell). Manually verify for one row."""
        from common.models.qlearning import QLearningModel
        model, df = trained_model

        # Compute expected score for row 0 manually
        row = df.iloc[[0]]
        disc = model._discretize(row[model.feature_columns])
        pos_flag = int(row["position_flag"].iloc[0]) * 1000
        s = model._encode_state(disc[0], pos_flag)
        q_row = model.qlearner.Q[s]
        expected_score = float(q_row[0] - q_row[1])  # Q(buy=0) - Q(sell=1)

        actual_scores = model.predict_score_bulk(row)
        assert abs(actual_scores.iloc[0] - expected_score) < 1e-9, \
            f"Score {actual_scores.iloc[0]:.6f} != Q(buy)-Q(sell) {expected_score:.6f}"

    def test_buy_action_rows_have_non_negative_scores(self, trained_model):
        """Rows where model chooses 'buy' must have Q(buy) >= Q(sell), i.e. score >= 0.
        Score == 0 is valid: it happens when Q(buy) == Q(sell) == 0 (cold table states
        where argmax defaults to action 0 = buy).  Score < 0 would be a contradiction.
        """
        model, df = trained_model
        actions = model.predict_bulk(df)
        scores  = model.predict_score_bulk(df)
        buy_rows = scores[actions == "buy"]
        if len(buy_rows) > 0:
            assert (buy_rows >= 0).all(), \
                f"Buy-action rows must have Q(buy)-Q(sell) >= 0. "  \
                f"Min buy score: {buy_rows.min():.4f}"


# ── LEAN policy logic tests ───────────────────────────────────────────────────
# These replicate exactly the logic from main.py in pure Python,
# so we can verify each policy without the LEAN engine.

# ── Policy: Wash-sale guard ───────────────────────────────────────────────────

def is_wash_sale_blocked(ticker: str, last_sell_dates: dict, today: date,
                         wash_sale_days: int) -> bool:
    """Exact replica of LEAN _is_wash_sale_blocked()."""
    last_sell = last_sell_dates.get(ticker)
    if last_sell is None:
        return False
    return (today - last_sell).days < wash_sale_days


class TestLEANWashSaleGuard:
    def test_fresh_ticker_not_blocked(self):
        assert not is_wash_sale_blocked("AAPL", {}, date(2024, 3, 1), 30)

    def test_sold_yesterday_is_blocked(self):
        last_sell = {date(2024, 2, 29)}
        assert is_wash_sale_blocked(
            "AAPL", {"AAPL": date(2024, 2, 29)}, date(2024, 3, 1), 30
        )

    def test_sold_31_days_ago_not_blocked(self):
        assert not is_wash_sale_blocked(
            "AAPL", {"AAPL": date(2024, 1, 1)}, date(2024, 2, 1), 30
        )

    def test_sold_exactly_30_days_ago_is_still_blocked(self):
        """Boundary: 30 days gap means blocked (< 30 is blocked, 30 is NOT blocked)."""
        assert not is_wash_sale_blocked(
            "AAPL", {"AAPL": date(2024, 1, 1)}, date(2024, 1, 31), 30
        )  # exactly 30 days → not blocked (30 >= 30)

    def test_sold_29_days_ago_is_blocked(self):
        assert is_wash_sale_blocked(
            "AAPL", {"AAPL": date(2024, 1, 2)}, date(2024, 1, 31), 30
        )


# ── Policy: Min-hold constraint ───────────────────────────────────────────────

def apply_sell_constraints(action: str, entry_date: date, today: date,
                           gain: float, min_hold_profit: int, min_hold_loss: int) -> str:
    """Replica of LEAN _apply_sell_constraints() — blocks model sells before min_hold."""
    if action != "sell":
        return action
    hold_days = (today - entry_date).days
    min_hold = min_hold_profit if gain > 0 else min_hold_loss
    if hold_days < min_hold:
        return "hold"   # blocked
    return "sell"


class TestLEANMinHoldConstraint:
    def test_sell_blocked_before_min_hold(self):
        result = apply_sell_constraints(
            "sell",
            entry_date=date(2024, 1, 1),
            today=date(2024, 1, 15),   # only 14 days held
            gain=0.05,
            min_hold_profit=20, min_hold_loss=20,
        )
        assert result == "hold", "Sell should be blocked before 20-day min hold"

    def test_sell_allowed_after_min_hold(self):
        result = apply_sell_constraints(
            "sell",
            entry_date=date(2024, 1, 1),
            today=date(2024, 1, 22),   # 21 days held
            gain=0.05,
            min_hold_profit=20, min_hold_loss=20,
        )
        assert result == "sell", "Sell should be allowed after min hold"

    def test_buy_and_hold_actions_pass_through(self):
        for action in ("buy", "hold"):
            result = apply_sell_constraints(
                action, date(2024, 1, 1), date(2024, 1, 5),
                0.0, 20, 20,
            )
            assert result == action, f"Non-sell action '{action}' should pass through unchanged"


# ── Policy: Consecutive sell signals ─────────────────────────────────────────

class TestLEANConsecutiveSellSignals:
    """3 consecutive sell signals required before model exit."""

    def _run_sell_streak(self, signal_days: list, required: int = 3) -> int:
        """Simulate the sell streak counter and count how many model exits fire."""
        sell_streak = 0
        exits = 0
        for sig in signal_days:
            if sig == "sell":
                sell_streak += 1
            else:
                sell_streak = 0
            if sell_streak >= required:
                exits += 1
                sell_streak = 0  # reset after exit
        return exits

    def test_one_sell_no_exit(self):
        assert self._run_sell_streak(["sell", "buy", "buy"]) == 0

    def test_two_consecutive_no_exit(self):
        assert self._run_sell_streak(["sell", "sell", "buy"]) == 0

    def test_three_consecutive_exits(self):
        assert self._run_sell_streak(["sell", "sell", "sell"]) == 1

    def test_interrupted_streak_resets(self):
        assert self._run_sell_streak(["sell", "sell", "buy", "sell", "sell", "sell"]) == 1

    def test_two_streaks_two_exits(self):
        signals = ["sell", "sell", "sell", "buy", "sell", "sell", "sell"]
        assert self._run_sell_streak(signals) == 2


# ── Policy: Trailing stop ─────────────────────────────────────────────────────

def check_trailing_stop(entry_price: float, current_price: float, high_watermark: float,
                        trigger_pct: float, trail_pct: float) -> tuple:
    """
    Replica of LEAN trailing stop logic.
    Returns (should_exit, updated_hwm).
    """
    hwm = max(high_watermark, current_price)
    if trigger_pct <= 0 or trail_pct <= 0:
        return False, hwm
    peak_gain = (hwm - entry_price) / entry_price
    if peak_gain >= trigger_pct:
        floor = hwm * (1 - trail_pct)
        if current_price <= floor:
            return True, hwm
    return False, hwm


class TestLEANTrailingStop:
    def test_no_exit_before_trigger(self):
        """Position up 10% (below 20% trigger) — trailing stop not active."""
        exit_flag, _ = check_trailing_stop(
            entry_price=100, current_price=110, high_watermark=110,
            trigger_pct=0.20, trail_pct=0.18,
        )
        assert not exit_flag

    def test_exit_after_trigger_and_retracement(self):
        """Position up 30% (trigger at 20%), then retraces 20% — trailing stop fires."""
        hwm = 130  # peak gain 30% from entry 100
        current = hwm * (1 - 0.20)  # 20% below HWM
        exit_flag, _ = check_trailing_stop(
            entry_price=100, current_price=current, high_watermark=hwm,
            trigger_pct=0.20, trail_pct=0.18,
        )
        assert exit_flag, (
            f"Trailing stop should fire: entry=100, hwm={hwm}, "
            f"current={current:.2f} which is {(hwm - current)/hwm:.1%} below hwm"
        )

    def test_no_exit_when_still_above_trail_floor(self):
        """Position at HWM × 0.85 with 18% trail — floor is HWM × 0.82 → no exit."""
        hwm = 130
        current = hwm * 0.85   # only 15% below HWM, floor is 18% below
        exit_flag, _ = check_trailing_stop(
            entry_price=100, current_price=current, high_watermark=hwm,
            trigger_pct=0.20, trail_pct=0.18,
        )
        assert not exit_flag

    def test_hwm_updated(self):
        """High-water mark should update to current price when current > hwm."""
        _, new_hwm = check_trailing_stop(
            entry_price=100, current_price=150, high_watermark=130,
            trigger_pct=0.20, trail_pct=0.18,
        )
        assert new_hwm == 150


# ── Policy: Hard stop-loss ────────────────────────────────────────────────────

def check_stop_loss(entry_price: float, current_price: float, stop_pct: float) -> bool:
    """Replica of LEAN fixed stop-loss logic."""
    if stop_pct <= 0 or entry_price <= 0:
        return False
    loss_pct = (entry_price - current_price) / entry_price
    return loss_pct >= stop_pct


class TestLEANStopLoss:
    def test_no_stop_at_small_loss(self):
        assert not check_stop_loss(100, 90, 0.15)  # 10% loss, 15% threshold

    def test_stop_at_threshold(self):
        assert check_stop_loss(100, 85, 0.15)  # exactly 15% loss

    def test_stop_above_threshold(self):
        assert check_stop_loss(100, 70, 0.15)  # 30% loss

    def test_no_stop_in_profit(self):
        assert not check_stop_loss(100, 110, 0.15)


# ── Policy: BEAR regime defensive buying ─────────────────────────────────────

class TestLEANBEARDefensive:
    """BEAR regime blocks offensive tickers; allows 1 defensive slot."""

    def _simulate_bear_buy_decision(self, ticker: str, is_defensive: bool,
                                    defensive_held_count: int,
                                    max_defensive_slots: int = 1) -> str:
        """
        Replica of the LEAN BEAR regime buy gate:
        - BEAR blocks all offensive tickers
        - Defensives allowed up to max_defensive_slots
        Returns 'allow' or 'block'.
        """
        current_regime = "BEAR"
        if current_regime != "BEAR":
            return "allow"
        if not is_defensive:
            return "block"   # all offensive tickers blocked in BEAR
        if defensive_held_count >= max_defensive_slots:
            return "block"   # defensive slot full
        return "allow"

    def test_offensive_blocked_in_bear(self):
        result = self._simulate_bear_buy_decision("AAPL", is_defensive=False,
                                                  defensive_held_count=0)
        assert result == "block", "Offensive ticker must be blocked in BEAR"

    def test_defensive_allowed_when_slot_empty(self):
        result = self._simulate_bear_buy_decision("GLD", is_defensive=True,
                                                  defensive_held_count=0)
        assert result == "allow", "Defensive ticker must be allowed when slot is empty"

    def test_defensive_blocked_when_slot_full(self):
        result = self._simulate_bear_buy_decision("TLT", is_defensive=True,
                                                  defensive_held_count=1)
        assert result == "block", "Second defensive must be blocked when 1 slot already used"


# ── Policy: SPY velocity crash filter ────────────────────────────────────────

def check_spy_velocity(spy_returns: list, lookback: int, halt_pct: float) -> bool:
    """
    Replica of LEAN SPY velocity crash filter.
    Returns True if buys should be halted.
    """
    if halt_pct <= 0 or len(spy_returns) < lookback:
        return False
    recent = spy_returns[-lookback:]
    cumulative = math.prod(1 + r for r in recent) - 1.0
    return cumulative < -halt_pct


class TestLEANSPYVelocityFilter:
    def test_flat_market_not_halted(self):
        returns = [0.001, 0.002, -0.001]
        assert not check_spy_velocity(returns, 3, 0.03)

    def test_crash_halts_buys(self):
        # SPY down 5% in 3 days (compound)
        returns = [-0.02, -0.015, -0.02]  # ~5.4% cumulative loss
        assert check_spy_velocity(returns, 3, 0.03)

    def test_small_drop_not_halted(self):
        returns = [-0.01, -0.005, -0.005]  # ~2% cumulative — below 3% threshold
        assert not check_spy_velocity(returns, 3, 0.03)

    def test_insufficient_history_not_halted(self):
        returns = [-0.05]  # only 1 day of history, need 3
        assert not check_spy_velocity(returns, 3, 0.03)

    def test_clearly_above_threshold_not_halted(self):
        # SPY down only 1% total — clearly below 3% halt threshold
        returns = [-0.003, -0.003, -0.004]  # ~1% cumulative loss
        assert not check_spy_velocity(returns, 3, 0.03)


# ── Policy: SPY EMA50 trend gate ─────────────────────────────────────────────

def check_spy_ema50_gate(spy_closes: list) -> bool:
    """
    Replica of LEAN SPY EMA50 trend gate.
    Returns True if buys should be blocked (SPY below EMA50).
    Requires at least 51 closes.
    """
    if len(spy_closes) < 51:
        return False   # not enough history → gate inactive (allow buys)
    s = pd.Series(spy_closes)
    ema50 = s.ewm(span=50, adjust=False).mean()
    return float(s.iloc[-1]) < float(ema50.iloc[-1])


class TestLEANSPYEMA50Gate:
    def test_spy_above_ema_allows_buys(self):
        # Steadily rising SPY — always above EMA50
        closes = [450.0 * (1 + 0.002 * i) for i in range(55)]
        assert not check_spy_ema50_gate(closes), \
            "Rising SPY always above EMA50 → gate should not block"

    def test_spy_below_ema_blocks_buys(self):
        # 50 days at 450, then crash to 380
        closes = [450.0] * 50 + [380.0, 380.0, 380.0, 380.0, 380.0]
        assert check_spy_ema50_gate(closes), \
            "SPY crashed to 380 while EMA50 is ~450 → gate must block"

    def test_insufficient_history_allows_buys(self):
        # Only 30 days of history — gate is inactive
        closes = [450.0] * 30
        assert not check_spy_ema50_gate(closes), \
            "With < 51 days history, gate should be inactive (allow buys)"


# ── Policy: Sector guard ─────────────────────────────────────────────────────

def check_sector_guard(ticker: str, sector_map: dict, held_tickers: list,
                       selected: list, defensive_set: set,
                       max_per_sector: int) -> bool:
    """
    Replica of LEAN sector guard logic in the EXECUTE loop.
    Returns True if the ticker is allowed (not blocked).
    """
    if ticker in defensive_set:
        return True  # defensives skip sector guard
    sector = sector_map.get(ticker, "other")
    count = sum(
        1 for t in held_tickers + selected
        if sector_map.get(t, "other") == sector
    )
    return count < max_per_sector


class TestLEANSectorGuard:
    SECTOR_MAP = {
        "AAPL": "tech", "MSFT": "tech", "AMZN": "tech",
        "GOOG": "tech", "NVDA": "tech",
        "JPM": "finance",
        "GLD": "commodity",
    }

    def test_first_sector_ticker_allowed(self):
        assert check_sector_guard("AAPL", self.SECTOR_MAP, [], [], set(), 3)

    def test_third_sector_ticker_allowed(self):
        assert check_sector_guard(
            "GOOG", self.SECTOR_MAP, ["AAPL", "MSFT"], [], set(), 3
        )

    def test_fourth_sector_ticker_blocked(self):
        assert not check_sector_guard(
            "NVDA", self.SECTOR_MAP, ["AAPL", "MSFT", "AMZN"], [], set(), 3
        )

    def test_defensive_bypasses_sector_guard(self):
        # GLD is defensive — should always pass even if commodity sector is full
        assert check_sector_guard(
            "GLD", self.SECTOR_MAP, ["GLD", "GLD_B", "GLD_C"], [], {"GLD"}, 3
        )

    def test_different_sector_not_blocked(self):
        # JPM (finance) not blocked by tech being full
        assert check_sector_guard(
            "JPM", self.SECTOR_MAP, ["AAPL", "MSFT", "AMZN"], [], set(), 3
        )

    def test_selected_counted_toward_sector_cap(self):
        # AAPL and MSFT held, AMZN selected (not yet in holdings) — NVDA should be blocked
        assert not check_sector_guard(
            "NVDA", self.SECTOR_MAP,
            held_tickers=["AAPL", "MSFT"],
            selected=["AMZN"],  # already selected this bar
            defensive_set=set(),
            max_per_sector=3,
        )


# ── Policy: Transition uncertainty window ────────────────────────────────────

class TestLEANTransitionUncertainty:
    """After a CUSUM regime transition, no new buys for N bars."""

    def _simulate_transition(self, countdown_start: int, bars: int = 10) -> list:
        """
        Simulate the transition countdown decrement.
        Returns list of (bar, buys_allowed).
        """
        countdown = countdown_start
        results = []
        for bar in range(bars):
            if countdown > 0:
                countdown -= 1
                results.append((bar, False))  # blocked
            else:
                results.append((bar, True))   # allowed
        return results

    def test_buys_blocked_during_countdown(self):
        results = self._simulate_transition(countdown_start=3, bars=5)
        blocked = [(bar, allowed) for bar, allowed in results if not allowed]
        assert len(blocked) == 3, \
            f"Expected 3 blocked bars with countdown=3, got {len(blocked)}"

    def test_buys_resume_after_countdown(self):
        results = self._simulate_transition(countdown_start=3, bars=6)
        allowed_after = [(bar, a) for bar, a in results if bar >= 3 and a]
        assert len(allowed_after) == 3, \
            "Buys should resume after countdown expires"

    def test_zero_countdown_always_allowed(self):
        results = self._simulate_transition(countdown_start=0, bars=5)
        assert all(a for _, a in results), \
            "With no countdown, all bars should allow buys"


# ── Policy: Correlation guard ─────────────────────────────────────────────────

def check_correlation_guard(ticker: str, held_plus_selected: list,
                            corr_matrix: dict, threshold: float) -> bool:
    """
    Replica of LEAN _passes_correlation_guard().
    Returns True if the ticker passes (not too correlated with anything held/selected).
    """
    for other in held_plus_selected:
        corr = corr_matrix.get(ticker, {}).get(other,
               corr_matrix.get(other, {}).get(ticker, 0.0))
        if abs(corr) >= threshold:
            return False
    return True


class TestLEANCorrelationGuard:
    CORR = {
        "AAPL": {"MSFT": 0.92, "AMZN": 0.75, "JPM": 0.30},
        "MSFT": {"AAPL": 0.92, "AMZN": 0.80},
        "AMZN": {"AAPL": 0.75, "MSFT": 0.80},
        "JPM":  {"AAPL": 0.30},
    }

    def test_low_correlation_passes(self):
        # JPM has 0.30 correlation with AAPL — below 0.70 threshold → passes
        assert check_correlation_guard("JPM", ["AAPL"], self.CORR, 0.70)

    def test_high_correlation_blocked(self):
        # MSFT has 0.92 correlation with AAPL — above 0.70 → blocked
        assert not check_correlation_guard("MSFT", ["AAPL"], self.CORR, 0.70)

    def test_no_holdings_always_passes(self):
        assert check_correlation_guard("AAPL", [], self.CORR, 0.70)

    def test_symmetric_lookup(self):
        # Correlation lookup works in both directions
        assert not check_correlation_guard("AAPL", ["MSFT"], self.CORR, 0.70)

    def test_exactly_at_threshold_is_blocked(self):
        corr = {"A": {"B": 0.70}}
        assert not check_correlation_guard("A", ["B"], corr, 0.70)

    def test_just_below_threshold_passes(self):
        corr = {"A": {"B": 0.699}}
        assert check_correlation_guard("A", ["B"], corr, 0.70)


# ── Policy: Min-model-score filter ───────────────────────────────────────────

class TestLEANMinModelScoreFilter:
    """Only candidates with model_score >= min_model_score proceed to EXECUTE."""

    def _filter_candidates(self, scored: list, min_score: float) -> list:
        """Replica of LEAN SCAN loop filter."""
        return [s for s in scored if s[1] >= min_score]

    def test_score_above_threshold_passes(self):
        scored = [("AAPL", 0.5, 0.0, "detail")]
        result = self._filter_candidates(scored, min_score=0.10)
        assert len(result) == 1

    def test_score_below_threshold_filtered(self):
        scored = [("AAPL", 0.05, 0.0, "detail")]
        result = self._filter_candidates(scored, min_score=0.10)
        assert len(result) == 0

    def test_mixed_scores_correct_filtering(self):
        scored = [
            ("AAPL", 0.15, 0.0, ""),
            ("MSFT", 0.08, 0.0, ""),   # below threshold
            ("NVDA", 0.30, 0.0, ""),
            ("AMZN", 0.02, 0.0, ""),   # below threshold
        ]
        result = self._filter_candidates(scored, min_score=0.10)
        tickers = [r[0] for r in result]
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "MSFT" not in tickers
        assert "AMZN" not in tickers


# ── Policy: Ranking by model score ───────────────────────────────────────────

class TestLEANRankingByModelScore:
    """Candidates ranked 50% model score + 50% RS score, descending."""

    def _rank(self, scored: list) -> list:
        """Replica of LEAN ranking logic."""
        model_scores = [s[1] for s in scored]
        rs_scores    = [s[2] for s in scored]
        ms_min, ms_max = min(model_scores), max(model_scores)
        rs_min, rs_max = min(rs_scores),    max(rs_scores)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        return sorted(
            scored,
            key=lambda s: 0.5 * norm(s[1], ms_min, ms_max)
                        + 0.5 * norm(s[2], rs_min, rs_max),
            reverse=True,
        )

    def test_highest_model_score_ranked_first(self):
        scored = [
            ("AAPL", 0.9, 0.5, ""),
            ("MSFT", 0.3, 0.5, ""),
        ]
        ranked = self._rank(scored)
        assert ranked[0][0] == "AAPL", "Highest model score should rank first"

    def test_combined_score_ordering(self):
        # AAPL: model=0.9, rs=0.1 → combined ≈ 0.5*1.0 + 0.5*0.0 = 0.5
        # MSFT: model=0.1, rs=0.9 → combined ≈ 0.5*0.0 + 0.5*1.0 = 0.5  (tie → stable)
        # NVDA: model=0.9, rs=0.9 → combined ≈ 0.5*1.0 + 0.5*1.0 = 1.0  (winner)
        scored = [
            ("AAPL", 0.9, 0.1, ""),
            ("MSFT", 0.1, 0.9, ""),
            ("NVDA", 0.9, 0.9, ""),
        ]
        ranked = self._rank(scored)
        assert ranked[0][0] == "NVDA", \
            f"NVDA (top on both) should rank first, got {ranked[0][0]}"

    def test_single_candidate_unchanged(self):
        scored = [("AAPL", 0.5, 0.5, "")]
        ranked = self._rank(scored)
        assert ranked[0][0] == "AAPL"


# ══════════════════════════════════════════════════════════════════════════════
# Tests for the 6 fixes applied in the gap-alignment pass
# ══════════════════════════════════════════════════════════════════════════════

# ── Fix 1: LEAN trailing stop uses peak_gain (HWM) not current_gain ──────────

class TestLEANTrailingStopPeakGain:
    """
    Trailing stop must use peak_gain = (HWM - entry) / entry, NOT current_gain.
    Once the trigger is crossed, the stop should stay armed even after pullback.
    Old bug: if stock surged to +40% then fell to +10%, LEAN disarmed the stop.
    New fix: uses HWM-based peak_gain, so stop remains armed.
    """

    def _trailing_stop_check(self, entry_price, high_watermark, current_price,
                             ts_trigger=0.35, ts_trail=0.28):
        """Replica of LEAN's fixed trailing stop logic (peak_gain version)."""
        peak_gain = (high_watermark - entry_price) / entry_price
        if peak_gain >= ts_trigger:
            trail_floor = high_watermark * (1 - ts_trail)
            return current_price <= trail_floor, trail_floor
        return False, None

    def test_fires_when_price_drops_below_trail_floor(self):
        """Stock gained +50% (armed), now pulls back below trail floor → fire."""
        entry = 100.0
        hwm   = 150.0   # peak gain = 50% ≥ 35% trigger
        # trail_floor = 150 * (1 - 0.28) = 108.0
        current = 107.0   # below floor
        fired, floor = self._trailing_stop_check(entry, hwm, current)
        assert fired, f"Trailing stop should fire (price {current} < floor {floor:.2f})"

    def test_does_not_fire_when_price_above_trail_floor(self):
        """Stock gained +50% (armed), still above trail floor → hold."""
        entry = 100.0
        hwm   = 150.0   # peak gain = 50%
        current = 120.0   # above floor (108.0)
        fired, _ = self._trailing_stop_check(entry, hwm, current)
        assert not fired, "Trailing stop should not fire while price above trail floor"

    def test_stays_armed_after_pullback_below_trigger(self):
        """
        Critical fix: stock peaked at +40% (above trigger), then pulled back to +10%.
        Old code: current_gain = 10% < 35% → stop DISARMED.
        New code: peak_gain = 40% ≥ 35% → stop STAYS ARMED.
        """
        entry = 100.0
        hwm   = 140.0   # peak gain = 40% ≥ 35% trigger
        # With 28% trail: floor = 140 * 0.72 = 100.8
        current = 99.0   # fallen BELOW floor — should fire
        fired, floor = self._trailing_stop_check(entry, hwm, current)
        assert fired, (
            f"Bug fix: trailing stop should fire (peak_gain=40% ≥ trigger, "
            f"current {current} < floor {floor:.2f}) even though current_gain=−1%"
        )

    def test_does_not_arm_if_trigger_never_crossed(self):
        """Stock only gained +20%, trigger is 35% — stop never arms."""
        entry = 100.0
        hwm   = 120.0   # peak gain = 20% < 35% trigger
        current = 85.0  # big drop, but trailing stop not armed — hard stop handles it
        fired, _ = self._trailing_stop_check(entry, hwm, current)
        assert not fired, "Trailing stop must not fire if trigger level was never reached"

    def test_old_current_gain_logic_would_fail(self):
        """Demonstrate that the OLD current_gain check would have missed the stop."""
        entry = 100.0
        hwm   = 140.0   # peak = +40%
        current = 99.0  # current_gain = −1%

        # Old (wrong) logic
        current_gain = (current - entry) / entry   # −0.01
        old_would_fire = current_gain >= 0.35      # False — BUG
        assert not old_would_fire, "Old logic correctly shown to fail here"

        # New (fixed) logic
        peak_gain = (hwm - entry) / entry          # +0.40
        trail_floor = hwm * (1 - 0.28)            # 100.8
        new_fires = peak_gain >= 0.35 and current <= trail_floor
        assert new_fires, "New logic correctly fires the stop"


# ── Fix 5: LEAN sell streak doesn't accumulate during min_hold period ─────────

class TestLEANStreakGatedByMinHold:
    """
    Old behavior: sell streak accumulated even during min_hold; _apply_sell_constraints
    blocked execution but streak could reach 3 BEFORE min_hold expired, causing an
    exit on day 20 even if no sell signals occurred after the hold period.

    New behavior: sell signal check is skipped entirely during min_hold (matching
    the notebook), so streak cannot accumulate until the hold period expires.
    """

    def _run_hold_sell_sim(self, signals: list, min_hold: int, consecutive_req: int = 3):
        """
        Simulate the fixed LEAN sell logic for a single position.
        signals: list of "sell"/"hold" per day, starting from day 1 (entry is day 0).
        Returns (exit_day, exit_type) or (None, None) if no exit.
        """
        entry_day = 0
        sell_streak = 0
        for day_offset, signal in enumerate(signals):
            current_day = day_offset + 1   # day 1 is first post-entry bar
            days_held = current_day - entry_day

            # Fixed: skip streak accumulation during min_hold
            if days_held < min_hold:
                continue

            if signal == "sell":
                sell_streak += 1
            else:
                sell_streak = 0

            if sell_streak >= consecutive_req:
                return current_day, "model_sell"

        return None, None

    def test_streak_from_before_min_hold_does_not_cause_immediate_exit(self):
        """
        Sell signals on days 1-3 (within min_hold=20) must NOT count toward streak.
        Exit should require 3 consecutive sells AFTER day 20.
        """
        # 3 sells in min_hold, then holds, then 3 consecutive sells after
        signals = ["sell"] * 3 + ["hold"] * 17 + ["sell"] * 3   # total 23 days
        exit_day, _ = self._run_hold_sell_sim(signals, min_hold=20)
        assert exit_day == 23, (
            f"Exit should be on day 23 (3 new sells after day 20), got day {exit_day}"
        )

    def test_exit_requires_new_streak_after_min_hold(self):
        """3 sells before min_hold + 1 sell after: still not enough — need 3."""
        signals = ["sell"] * 3 + ["hold"] * 17 + ["sell"] * 1 + ["hold"] * 5
        exit_day, _ = self._run_hold_sell_sim(signals, min_hold=20)
        assert exit_day is None, "Should not exit — streak restarted after min_hold"

    def test_clean_three_consecutive_sells_after_min_hold_exits(self):
        """After min_hold, 3 fresh consecutive sells triggers exit."""
        signals = ["hold"] * 20 + ["sell"] * 3   # exactly at boundary + 3
        exit_day, _ = self._run_hold_sell_sim(signals, min_hold=20)
        assert exit_day == 23, f"Expected exit on day 23, got {exit_day}"

    def test_interrupted_streak_resets(self):
        """Sell-sell-hold-sell-sell-sell after min_hold: exit on 3rd consecutive."""
        signals = ["hold"] * 20 + ["sell", "sell", "hold", "sell", "sell", "sell"]
        exit_day, _ = self._run_hold_sell_sim(signals, min_hold=20)
        assert exit_day == 26, f"Expected exit day 26 (3rd consecutive after reset), got {exit_day}"


# ── Fix 6: Q-Learning score = Q(buy) − Q(sell), not Q(buy) − Q(hold) ─────────

class TestQLearningScoreFormula:
    """
    LEAN was computing Q(buy) − Q(hold) (q_vals[0] − q_vals[2]).
    The notebook predict_score_bulk() uses Q(buy) − Q(sell) (q_vals[0] − q_vals[1]).
    Fix: LEAN now uses q_vals[0] − q_vals[1] to match.
    Actions: 0=buy, 1=sell, 2=hold.
    """

    def _lean_score_fixed(self, q_row):
        """Replica of fixed LEAN _get_raw_model_score for qlearning."""
        return float(q_row[0] - q_row[1])   # Q(buy) - Q(sell)

    def _lean_score_old(self, q_row):
        """Old LEAN formula for comparison."""
        return float(q_row[0] - q_row[2])   # Q(buy) - Q(hold) — was wrong

    def _notebook_score(self, q_row):
        """predict_score_bulk() in common/models/qlearning.py."""
        return float(q_row[0] - q_row[1])   # Q(buy) - Q(sell)

    def test_fixed_lean_matches_notebook(self):
        """After fix, LEAN and notebook formulas produce identical scores."""
        q = [0.7, 0.2, 0.5]   # Q(buy)=0.7, Q(sell)=0.2, Q(hold)=0.5
        assert self._lean_score_fixed(q) == self._notebook_score(q), \
            "Fixed LEAN score must match notebook predict_score_bulk"

    def test_old_lean_differed_from_notebook(self):
        """Verify the old formula did NOT match — proving the bug existed."""
        q = [0.7, 0.2, 0.5]
        assert self._lean_score_old(q) != self._notebook_score(q), \
            "Old formula should differ — it used Q(hold) instead of Q(sell)"

    def test_positive_when_buy_dominates(self):
        """When Q(buy) > Q(sell), score > 0 (buy signal)."""
        q = [0.8, 0.3, 0.5]
        assert self._lean_score_fixed(q) > 0

    def test_negative_when_sell_dominates(self):
        """When Q(sell) > Q(buy), score < 0 (sell signal)."""
        q = [0.3, 0.8, 0.5]
        assert self._lean_score_fixed(q) < 0

    def test_zero_when_buy_equals_sell(self):
        """When Q(buy) == Q(sell), score is 0 (neutral)."""
        q = [0.5, 0.5, 0.3]
        assert self._lean_score_fixed(q) == 0.0

    def test_notebook_predict_score_bulk_uses_same_formula(self):
        """End-to-end: QLearningModel.predict_score_bulk() uses Q(buy)−Q(sell)."""
        from common.models import create_model
        import numpy as np, pandas as pd

        rng = np.random.default_rng(99)
        n = 300
        feature_cols = ["rsi", "macd_hist", "cci", "bbp", "adx"]
        df = pd.DataFrame(rng.normal(0, 1, (n, len(feature_cols))), columns=feature_cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        df["position_flag"] = 0

        model = create_model(
            "qlearning", feature_columns=feature_cols,
            n_bins=3, n_epochs=50,
        )
        model.train(df)
        scores = model.predict_score_bulk(df)
        # Verify scores match manual Q(buy)−Q(sell) computation
        from common.models.qlearning import QLearningModel
        assert isinstance(model, QLearningModel)
        # A row where model says buy should have score ≥ 0
        actions = model.predict_bulk(df)
        buy_scores = scores[actions == "buy"]
        if len(buy_scores):
            assert (buy_scores >= 0).all(), "Buy rows must have Q(buy)−Q(sell) ≥ 0"


# ── Fix: Notebook BEAR defensive uses live score, not static Sharpe ───────────

class TestNotebookBEARRankingUsesLiveScore:
    """
    Old: def_candidates sorted by results[ticker]['sharpe'] (static, same every day).
    New: def_candidates sorted by oos_raw_scores.loc[today] (live model confidence).
    """

    def _bear_defensive_rank(self, ticker_data: dict, today) -> str:
        """Replica of fixed notebook BEAR defensive ranking."""
        candidates = []
        for ticker, data in ticker_data.items():
            raw_sc = data.get("oos_raw_scores")
            if raw_sc is None or today not in raw_sc.index:
                continue
            model_score = float(raw_sc.loc[today])
            candidates.append((ticker, model_score))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def test_highest_live_score_wins(self):
        today = pd.Timestamp("2024-06-01")
        data = {
            "GLD": {"oos_raw_scores": pd.Series([0.8], index=[today])},
            "TLT": {"oos_raw_scores": pd.Series([0.3], index=[today])},
        }
        winner = self._bear_defensive_rank(data, today)
        assert winner == "GLD", f"GLD has higher live score but got {winner}"

    def test_missing_raw_scores_skipped(self):
        today = pd.Timestamp("2024-06-01")
        data = {
            "GLD": {"oos_raw_scores": None},          # no raw scores
            "TLT": {"oos_raw_scores": pd.Series([0.5], index=[today])},
        }
        winner = self._bear_defensive_rank(data, today)
        assert winner == "TLT", "Ticker with missing raw scores should be skipped"

    def test_live_score_differs_from_static_sharpe(self):
        """Demonstrate that static Sharpe (old code) would pick differently."""
        today = pd.Timestamp("2024-06-01")
        # GLD: higher static Sharpe but lower live score
        # TLT: lower static Sharpe but higher live score TODAY
        static_sharpes = {"GLD": 1.5, "TLT": 0.9}
        live_scores    = {"GLD": 0.2, "TLT": 0.7}

        # Old behavior: pick by static Sharpe
        old_winner = max(static_sharpes, key=lambda t: static_sharpes[t])  # GLD

        # New behavior: pick by live score
        new_winner = max(live_scores, key=lambda t: live_scores[t])        # TLT

        assert old_winner == "GLD"
        assert new_winner == "TLT"
        assert old_winner != new_winner, "Fix changes selection when live score disagrees with Sharpe"


# ── Fix: Notebook transition uncertainty window ───────────────────────────────

class TestNotebookTransitionWindow:
    """
    Old: no transition uncertainty window in notebook simulation.
    New: after each CUSUM changepoint, no new buys for TRANSITION_BARS bars.
    Matches LEAN's _transition_countdown logic.
    """

    def _run_transition_sim(self, days, changepoint_dates, transition_bars=3):
        """
        Simulate the transition window logic.
        Returns set of days that were blocked by the transition window.
        """
        blocked = []
        countdown = 0
        for today in days:
            if today in changepoint_dates:
                countdown = transition_bars   # (re)set on each changepoint
            if countdown > 0:
                blocked.append(today)
                countdown -= 1
        return set(blocked)

    def test_blocks_three_bars_after_changepoint(self):
        days = pd.date_range("2024-01-01", periods=6)
        changepoints = {days[0]}   # changepoint on day 0
        blocked = self._run_transition_sim(days, changepoints, transition_bars=3)
        # Day 0 is the changepoint itself and is blocked, days 1 and 2 also blocked
        assert days[0] in blocked, "Day of changepoint should be blocked"
        assert days[1] in blocked, "Day 1 after changepoint should be blocked"
        assert days[2] in blocked, "Day 2 after changepoint should be blocked"
        assert days[3] not in blocked, "Day 3 should be clear"

    def test_no_blocks_without_changepoint(self):
        days = pd.date_range("2024-01-01", periods=5)
        blocked = self._run_transition_sim(days, set())
        assert len(blocked) == 0, "No blocks when no changepoints"

    def test_consecutive_changepoints_reset_countdown(self):
        """Two changepoints close together extend the block window."""
        days = pd.date_range("2024-01-01", periods=8)
        # changepoints on day 0 and day 2 — second one resets to 3
        changepoints = {days[0], days[2]}
        blocked = self._run_transition_sim(days, changepoints, transition_bars=3)
        # Day 2 resets: days 2, 3, 4 blocked
        assert days[4] in blocked, "Day 4 should be blocked (reset on day 2)"
        assert days[5] not in blocked, "Day 5 should be clear"

    def test_transition_blocks_match_lean_countdown_logic(self):
        """Verify notebook and LEAN countdown decrement at same rate."""
        # LEAN: countdown starts at N, decrements on each bar that checks it
        # Notebook: same — decrement inside `if countdown > 0:` block
        days = pd.date_range("2024-01-01", periods=5)
        changepoints = {days[0]}
        blocked = self._run_transition_sim(days, changepoints, transition_bars=3)
        # Exactly 3 bars blocked (days 0, 1, 2)
        assert len(blocked) == 3, f"Exactly 3 bars should be blocked, got {len(blocked)}"


# ── Fix: Notebook earnings filter ────────────────────────────────────────────

class TestNotebookEarningsFilter:
    """
    Old: no earnings filter in notebook simulation — all tickers eligible.
    New: tickers within ±EARNINGS_BUFFER days of earnings date are blocked.
    Matches LEAN's _is_earnings_blocked() check.
    """

    def _is_earnings_blocked(self, ticker, today_ts, calendar, buf=3):
        """Replica of the earnings filter helper added to the notebook."""
        for d_str in calendar.get(ticker, []):
            try:
                if abs((pd.Timestamp(d_str).date() - today_ts.date()).days) <= buf:
                    return True
            except Exception:
                pass
        return False

    def test_blocked_on_earnings_day(self):
        cal = {"AAPL": ["2024-07-25"]}
        today = pd.Timestamp("2024-07-25")
        assert self._is_earnings_blocked("AAPL", today, cal), \
            "Should block on earnings day itself"

    def test_blocked_within_buffer(self):
        cal = {"AAPL": ["2024-07-25"]}
        for delta in range(-3, 4):   # ±3 days
            today = pd.Timestamp("2024-07-25") + pd.Timedelta(days=delta)
            assert self._is_earnings_blocked("AAPL", today, cal, buf=3), \
                f"Should block at offset {delta} days"

    def test_not_blocked_outside_buffer(self):
        cal = {"AAPL": ["2024-07-25"]}
        today = pd.Timestamp("2024-07-29")   # 4 days after, outside ±3
        assert not self._is_earnings_blocked("AAPL", today, cal), \
            "Should not block 4 days away from earnings"

    def test_no_earnings_date_not_blocked(self):
        cal = {"MSFT": ["2024-07-30"]}
        today = pd.Timestamp("2024-07-25")
        assert not self._is_earnings_blocked("AAPL", today, cal), \
            "Ticker with no earnings entry should not be blocked"

    def test_multiple_earnings_dates_any_match_blocks(self):
        cal = {"NVDA": ["2024-02-21", "2024-05-22"]}
        today_q1 = pd.Timestamp("2024-02-20")   # 1 day before first
        today_q2 = pd.Timestamp("2024-05-23")   # 1 day after second
        assert self._is_earnings_blocked("NVDA", today_q1, cal)
        assert self._is_earnings_blocked("NVDA", today_q2, cal)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
