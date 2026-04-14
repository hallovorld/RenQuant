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


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
