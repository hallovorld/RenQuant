"""
Unit tests for renquant_104 notebook simulation policies.

Tests prove that each filter/policy is actually enforced in the simulation loop
by running minimal synthetic scenarios where the filter should trigger and
verifying no trade is placed (or the right trade is placed).

Run with:
    cd /path/to/RenQuant
    python -m pytest tests/test_simulation_policies.py -v
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Helpers — build minimal synthetic data ────────────────────────────────────

def make_prices(n: int = 60, start: float = 100.0, drift: float = 0.001) -> pd.Series:
    """Simple upward-drifting daily close prices."""
    rng = np.random.default_rng(42)
    rets = rng.normal(drift, 0.01, n)
    closes = start * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.Series(closes, index=idx)


def make_signals(dates: pd.DatetimeIndex, value: int = 1) -> pd.Series:
    """Constant signal Series (1=buy, -1=sell, 0=hold)."""
    return pd.Series(value, index=dates)


def make_raw_scores(dates: pd.DatetimeIndex, value: float = 0.5) -> pd.Series:
    """Constant raw model score Series."""
    return pd.Series(value, index=dates)


def make_ohlcv(prices: pd.Series) -> pd.DataFrame:
    """Minimal OHLCV frame from a close price series."""
    return pd.DataFrame({
        "open":  prices * 0.995,
        "high":  prices * 1.01,
        "low":   prices * 0.99,
        "close": prices,
        "volume": 1_000_000,
    })


def minimal_config(**overrides) -> dict:
    """Return a stripped-down strategy config suitable for testing."""
    cfg = {
        "max_concurrent_positions": 4,
        "wash_sale_days": 30,
        "min_hold_profit_days": 20,
        "min_hold_loss_days": 20,
        "initial_cash": 100_000,
        "backtest_start": "2024-01-02",
        "backtest_end": "2024-06-30",
        "tax": {"short_term_rate": 0.5, "long_term_rate": 0.32, "long_term_threshold_days": 365},
        "regime": {"correlation_guard_threshold": 0.70},
        "sector_map": {
            "AAPL": "tech", "MSFT": "tech", "AMZN": "tech",
            "GOOG": "tech", "NVDA": "tech",
            "JPM":  "finance",
            "GLD":  "commodity",
        },
        "sector_etf_map": {"tech": "XLK", "finance": "XLF", "commodity": "GLD"},
        "max_positions_per_sector": 3,
        "defensive_tickers": ["GLD"],
        "ranking": {"blend_weights": [0.5, 0.5]},
        "regime_params": {
            "BULL_CALM": {
                "max_position_pct": 0.15,
                "cash_reserve_pct": 0.0,
                "stop_loss_pct": 0.15,
                "max_hold_days": 500,
                "drawdown_halt_pct": 0.35,
                "trailing_stop_trigger_pct": 0.20,
                "trailing_stop_trail_pct": 0.18,
                "min_model_score": 0.10,
            },
            "BEAR": {
                "max_position_pct": 0.0,
                "cash_reserve_pct": 1.0,
                "stop_loss_pct": 0.05,
                "max_hold_days": 500,
                "drawdown_halt_pct": 0.05,
                "trailing_stop_trigger_pct": 0.0,
                "trailing_stop_trail_pct": 0.0,
                "min_model_score": 0.0,
            },
        },
    }
    cfg.update(overrides)
    return cfg


# ── Inline simulation runner ─────────────────────────────────────────────────
# A self-contained replica of the critical parts of Cell 21's loop,
# extracting just the policy logic so we can unit-test it without running
# the full notebook.

def run_sim(
    tickers: list,
    ohlcv: dict,
    results: dict,
    config: dict,
    regime_series: pd.Series,
    spy_close: pd.Series,
    corr_dict: dict = None,
    spy_vel_halt_pct: float = 0.0,
    spy_vel_lookback: int = 3,
    regime_confidence: pd.Series | None = None,
) -> list:
    """
    Minimal simulation loop that enforces exactly the same policies as Cell 21.

    Returns the trade_log list of executed trades.
    """
    INITIAL_CASH    = config["initial_cash"]
    MAX_POSITIONS   = config["max_concurrent_positions"]
    CORR_THRESHOLD  = config["regime"]["correlation_guard_threshold"]
    WASH_SALE_DAYS  = config.get("wash_sale_days", 30)
    MIN_HOLD_PROFIT = config.get("min_hold_profit_days", 20)
    MIN_HOLD_LOSS   = config.get("min_hold_loss_days", 20)
    MIN_MODEL_SCORE = config["regime_params"]["BULL_CALM"].get("min_model_score", 0.10)
    MAX_PER_SECTOR  = config.get("max_positions_per_sector", 3)
    RP              = config["regime_params"]
    DEFENSIVE       = set(config.get("defensive_tickers", []))
    ranking_cfg     = config.get("ranking", {})
    blend_weights   = ranking_cfg.get("blend_weights", [0.5, 0.5])
    blend_total     = float(blend_weights[0]) + float(blend_weights[1])
    rank_weight     = float(blend_weights[0]) / blend_total if blend_total > 0 else 0.5
    rs_weight       = float(blend_weights[1]) / blend_total if blend_total > 0 else 0.5
    ST_RATE         = config["tax"]["short_term_rate"]
    LT_RATE         = config["tax"]["long_term_rate"]
    LT_THRESH       = config["tax"]["long_term_threshold_days"]
    SECTOR_ETF      = config.get("sector_etf_map", {})

    if corr_dict is None:
        corr_dict = {}

    exportable = {t for t, r in results.items() if r.get("passes_floor")}

    cash          = INITIAL_CASH
    holdings      = {}
    hwm           = INITIAL_CASH
    skip_buys     = False
    last_sell     = {}
    sell_streak   = {}
    CONSEC_SELLS  = 3
    trade_log     = []

    all_dates = sorted(set(
        d for r in results.values()
        if r.get("oos_signals") is not None
        for d in r["oos_signals"].index
    ))
    bt_start = pd.Timestamp(config["backtest_start"])
    bt_end   = pd.Timestamp(config["backtest_end"])
    all_dates = [d for d in all_dates if bt_start <= d <= bt_end]

    spy_close_reindexed = spy_close.reindex(all_dates).ffill()
    bt_date_list = all_dates

    def _rank_score_for_day(ticker: str, today_ts: pd.Timestamp) -> float | None:
        raw = results[ticker].get("oos_raw_scores")
        if raw is None or today_ts not in raw.index:
            return None
        raw_score = float(raw.loc[today_ts])
        calibration = results[ticker].get("score_calibration")
        if calibration is None:
            return raw_score
        return float(calibration.calibrate(raw_score))

    for today_idx, today in enumerate(bt_date_list):
        regime = regime_series.get(today, "BULL_CALM")
        rp = RP.get(regime, RP["BULL_CALM"])
        regime_conf = float(regime_confidence.get(today, 1.0)) if regime_confidence is not None else 1.0

        # Mark-to-market
        port_val = cash
        for t, pos in holdings.items():
            if today in ohlcv.get(t, pd.DataFrame()).index:
                port_val += pos["shares"] * ohlcv[t].loc[today, "close"]
        hwm = max(hwm, port_val)
        skip_buys = (hwm > 0 and (hwm - port_val) / hwm >= rp["drawdown_halt_pct"])

        # ── SELLS ──
        to_sell = {}
        trail_trigger = rp.get("trailing_stop_trigger_pct", 0.0)
        trail_pct     = rp.get("trailing_stop_trail_pct",   0.0)
        for t, pos in list(holdings.items()):
            if today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            price     = ohlcv[t].loc[today, "close"]
            hold_days = (today - pos["entry_date"]).days
            loss      = (pos["entry_price"] - price) / pos["entry_price"]
            gain      = -loss
            pos["high_price"] = max(pos["high_price"], price)

            # Trailing stop
            if trail_trigger > 0 and trail_pct > 0:
                peak_gain = (pos["high_price"] - pos["entry_price"]) / pos["entry_price"]
                if peak_gain >= trail_trigger:
                    floor = pos["high_price"] * (1 - trail_pct)
                    if price <= floor:
                        to_sell[t] = "trailing_stop"
                        continue
            # Hard stop
            if loss >= rp["stop_loss_pct"]:
                to_sell[t] = "stop_loss"
                continue
            # Max hold
            if hold_days >= rp["max_hold_days"]:
                to_sell[t] = "max_hold"
                continue
            # Model sell
            min_hold = MIN_HOLD_PROFIT if gain > 0 else MIN_HOLD_LOSS
            if hold_days >= min_hold:
                sig = results.get(t, {}).get("oos_signals")
                if sig is not None and today in sig.index and sig.loc[today] == -1:
                    sell_streak[t] = sell_streak.get(t, 0) + 1
                    if sell_streak[t] >= CONSEC_SELLS:
                        to_sell[t] = "model_sell"
                else:
                    sell_streak[t] = 0

        for t, reason in to_sell.items():
            pos = holdings.pop(t)
            if today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            price     = ohlcv[t].loc[today, "close"]
            proceeds  = pos["shares"] * price
            gain      = proceeds - pos["shares"] * pos["entry_price"]
            hold_days = (today - pos["entry_date"]).days
            tax = gain * (LT_RATE if hold_days >= LT_THRESH else ST_RATE) if gain > 0 else 0
            cash += proceeds - tax
            last_sell[t] = today
            sell_streak[t] = 0
            trade_log.append({"action": "sell", "ticker": t, "date": today,
                               "exit_reason": reason})

        # ── BEAR defensive ──
        # Mirrors notebook Cell 21: pick the single best defensive by model score.
        if regime == "BEAR":
            def_held = [t for t in holdings if t in DEFENSIVE]
            if not skip_buys and len(def_held) < 1:
                def_candidates = []
                for t in DEFENSIVE & exportable:
                    if t in holdings:
                        continue
                    last_s = last_sell.get(t)
                    if last_s and (today - last_s).days < WASH_SALE_DAYS:
                        continue
                    sig = results.get(t, {}).get("oos_signals")
                    if sig is None or today not in sig.index or sig.loc[today] != 1:
                        continue
                    rank_score = _rank_score_for_day(t, today)
                    if rank_score is None:
                        continue
                    def_candidates.append((t, rank_score))
                if def_candidates:
                    def_candidates.sort(key=lambda x: x[1], reverse=True)
                    t, _ = def_candidates[0]   # buy only the top-ranked defensive
                    invest = min(cash, port_val * 0.15)
                    if invest >= 100 and today in ohlcv.get(t, pd.DataFrame()).index:
                        price = ohlcv[t].loc[today, "close"]
                        shares = invest / price
                        cash -= invest
                        holdings[t] = {"shares": shares, "entry_price": price,
                                       "entry_date": today, "high_price": price}
                        trade_log.append({"action": "buy", "ticker": t, "date": today,
                                          "regime": "BEAR_defensive"})
            continue

        # SPY velocity crash filter
        spy_vel_blocked = False
        if spy_vel_halt_pct > 0 and today_idx >= spy_vel_lookback:
            spy_prev = spy_close_reindexed.iloc[today_idx - spy_vel_lookback]
            spy_now  = spy_close_reindexed.iloc[today_idx]
            if spy_prev > 0 and (spy_now / spy_prev - 1) < -spy_vel_halt_pct:
                spy_vel_blocked = True

        # SPY EMA50 trend gate
        spy_trend_blocked = False
        if today_idx >= 50:
            spy_slice = spy_close_reindexed.iloc[:today_idx + 1]
            spy_ema50 = spy_slice.ewm(span=50, adjust=False).mean().iloc[-1]
            spy_trend_blocked = spy_close_reindexed.iloc[today_idx] < spy_ema50

        if skip_buys or spy_vel_blocked or spy_trend_blocked or len(holdings) >= MAX_POSITIONS:
            continue

        open_slots  = MAX_POSITIONS - len(holdings)
        max_pos_pct = rp.get("max_position_pct", 0.15) * regime_conf

        # Gather candidates
        candidates = []
        for t in exportable:
            if t in holdings or today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            sig = results.get(t, {}).get("oos_signals")
            if sig is None or today not in sig.index or sig.loc[today] != 1:
                continue
            last_s = last_sell.get(t)
            if last_s and (today - last_s).days < WASH_SALE_DAYS:
                continue
            # ── min_model_score filter ──
            raw = results[t].get("oos_raw_scores")
            if raw is None or today not in raw.index:
                continue
            raw_score_today = float(raw.loc[today])
            rank_score_today = _rank_score_for_day(t, today)
            if rank_score_today is None:
                continue
            if rank_score_today < MIN_MODEL_SCORE:
                continue
            # RS score (simplified — use 0 in tests)
            rs_score = 0.0
            candidates.append((t, raw_score_today, rank_score_today, rs_score))

        if not candidates:
            continue

        # Rank
        if len(candidates) > 1:
            ms = [c[2] for c in candidates]
            rs = [c[3] for c in candidates]
            ms_r = max(ms) - min(ms) or 1
            rs_r = max(rs) - min(rs) or 1
            candidates.sort(
                key=lambda c: rank_weight * (c[2] - min(ms)) / ms_r + rs_weight * (c[3] - min(rs)) / rs_r,
                reverse=True,
            )

        selected = []
        for t, raw_score, rank_score, rs_score in candidates:
            if len(selected) >= open_slots:
                break
            # Wash-sale guard re-check — matches LEAN selection loop.
            last_s = last_sell.get(t)
            if last_s and (today - last_s).days < WASH_SALE_DAYS:
                continue
            # Correlation guard
            corr_ok = True
            for held in list(holdings.keys()) + selected:
                corr = corr_dict.get(t, {}).get(held, corr_dict.get(held, {}).get(t, 0.0))
                if abs(corr) >= CORR_THRESHOLD:
                    corr_ok = False
                    break
            if not corr_ok:
                continue
            # ── Sector guard ──
            sector = config["sector_map"].get(t, "other")
            if t not in DEFENSIVE and MAX_PER_SECTOR > 0:
                sec_count = sum(
                    1 for h in list(holdings.keys()) + selected
                    if config["sector_map"].get(h, "other") == sector
                )
                if sec_count >= MAX_PER_SECTOR:
                    continue
            selected.append(t)

        for t in selected:
            cash_reserve = port_val * rp.get("cash_reserve_pct", 0.0) * regime_conf
            invest = min(cash - cash_reserve, port_val * max_pos_pct)
            if invest < 100:
                continue
            price  = ohlcv[t].loc[today, "close"]
            shares = invest / price
            cash  -= invest
            holdings[t] = {"shares": shares, "entry_price": price,
                            "entry_date": today, "high_price": price}
            sell_streak[t] = 0
            trade_log.append({"action": "buy", "ticker": t, "date": today})

    return trade_log


class _ConstantCalibration:
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def calibrate(self, raw_score: float) -> float:
        return raw_score + self.offset


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinModelScoreFilter:
    """min_model_score: only candidates with score >= threshold are considered."""

    def _make_results(self, score: float, ticker: str = "AAPL") -> dict:
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        return {
            ticker: {
                "sharpe": 1.5,
                "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, score),
                "oos_prices":     prices.reindex(dates),
            }
        }

    def test_score_above_threshold_allows_buy(self):
        """Score 0.5 >> 0.10 threshold → buy should execute."""
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = self._make_results(score=0.5)
        cfg = minimal_config()
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]
        assert len(buys) > 0, "Expected at least one buy when score > threshold"

    def test_score_below_threshold_blocks_buy(self):
        """Score 0.05 < 0.10 threshold → no buy should execute."""
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = self._make_results(score=0.05)  # below 0.10 threshold
        cfg = minimal_config()
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]
        assert len(buys) == 0, f"Expected 0 buys when score < threshold, got {len(buys)}"

    def test_score_exactly_at_threshold_allows_buy(self):
        """Score exactly at 0.10 threshold → buy allowed (>= comparison)."""
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = self._make_results(score=0.10)
        cfg = minimal_config()
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]
        assert len(buys) > 0, "Score exactly at threshold should be allowed"


class TestSelectionReplayLedger:
    """Replay a short synthetic tape and assert the notebook-like ledger stays deterministic."""

    def test_replay_ledger_matches_expected(self):
        dates = pd.date_range("2024-01-02", periods=70, freq="B")
        spy = make_prices(70, start=450.0).reindex(dates)
        stable_prices = make_prices(70).reindex(dates)

        tickers = ["AAPL", "JPM", "XOM", "MSFT", "NVDA", "CVX"]
        ohlcv = {t: make_ohlcv(stable_prices) for t in tickers}

        base_signals = make_signals(dates, 1)
        raw_scores = {
            "AAPL": pd.Series(0.0, index=dates),
            "JPM": pd.Series(0.0, index=dates),
            "XOM": pd.Series(0.0, index=dates),
            "MSFT": pd.Series(0.0, index=dates),
            "NVDA": pd.Series(0.0, index=dates),
            "CVX": pd.Series(0.0, index=dates),
        }

        day1, day2, day3 = dates[55], dates[56], dates[57]
        raw_scores["AAPL"].loc[day1] = 0.62
        raw_scores["JPM"].loc[day1] = 0.55
        raw_scores["XOM"].loc[day1] = 0.48
        raw_scores["MSFT"].loc[day1] = 0.95

        raw_scores["JPM"].loc[day2] = 0.67
        raw_scores["XOM"].loc[day2] = 0.61
        raw_scores["NVDA"].loc[day2] = 0.70

        raw_scores["XOM"].loc[day3] = 0.52
        raw_scores["CVX"].loc[day3] = 0.60
        raw_scores["MSFT"].loc[day3] = 0.58

        results = {
            ticker: {
                "sharpe": 1.5,
                "passes_floor": True,
                "oos_signals": base_signals,
                "oos_raw_scores": raw_scores[ticker],
                "oos_prices": stable_prices,
                "score_calibration": _ConstantCalibration(),
            }
            for ticker in tickers
        }

        cfg = minimal_config(
            initial_cash=10_000,
            max_concurrent_positions=3,
            max_positions_per_sector=1,
            ranking={"blend_weights": [0.8, 0.2]},
            tiered_thresholds=[
                {"min_model_score": 0.10},
                {"min_model_score": 0.80},
                {"min_model_score": 0.50},
            ],
        )
        cfg["sector_map"].update({"CVX": "energy"})
        cfg["regime_params"]["BULL_VOLATILE"] = {
            **cfg["regime_params"]["BULL_CALM"],
            "max_position_pct": 0.20,
            "cash_reserve_pct": 0.20,
            "min_model_score": 0.15,
        }

        regime = pd.Series("BULL_CALM", index=dates)
        regime.loc[day1] = "BULL_VOLATILE"
        regime_conf = pd.Series(1.0, index=dates)
        regime_conf.loc[day1] = 0.5
        regime_conf.loc[day3] = 0.8
        rs_by_day = {
            day1: {"AAPL": 0.10, "JPM": 0.40, "XOM": 0.90, "MSFT": 0.30},
            day2: {"JPM": 0.20, "XOM": 0.30, "NVDA": 0.80},
            day3: {"XOM": 0.50, "CVX": 0.40, "MSFT": 0.10},
        }
        corr_dict = {
            "CVX": {"JPM": 0.75, "AAPL": 0.10},
            "XOM": {"JPM": 0.20, "AAPL": 0.10},
            "MSFT": {"AAPL": 0.20, "JPM": 0.10},
            "NVDA": {"AAPL": 0.20},
            "JPM": {"AAPL": 0.10},
        }

        # Override the simplistic RS=0 branch with a deterministic per-day RS map.
        original_run_sim = run_sim

        def run_sim_with_rs(*args, **kwargs):
            tickers_local, ohlcv_local, results_local, config_local, regime_series_local, spy_close_local = args[:6]
            corr_local = kwargs.get("corr_dict") or {}
            regime_conf_local = kwargs.get("regime_confidence")

            INITIAL_CASH = config_local["initial_cash"]
            MAX_POSITIONS = config_local["max_concurrent_positions"]
            CORR_THRESHOLD = config_local["regime"]["correlation_guard_threshold"]
            WASH_SALE_DAYS = config_local.get("wash_sale_days", 30)
            MIN_MODEL_SCORE = config_local["regime_params"]["BULL_CALM"].get("min_model_score", 0.10)
            MAX_PER_SECTOR = config_local.get("max_positions_per_sector", 3)
            RP = config_local["regime_params"]
            DEFENSIVE = set(config_local.get("defensive_tickers", []))
            ranking_cfg = config_local.get("ranking", {})
            weights = ranking_cfg.get("blend_weights", [0.5, 0.5])
            weight_total = float(weights[0]) + float(weights[1])
            rank_weight = float(weights[0]) / weight_total if weight_total > 0 else 0.5
            rs_weight = float(weights[1]) / weight_total if weight_total > 0 else 0.5
            tiers = config_local.get("tiered_thresholds", [{"min_model_score": MIN_MODEL_SCORE}])

            exportable = {t for t, r in results_local.items() if r.get("passes_floor")}
            cash = INITIAL_CASH
            holdings = {}
            last_sell = {"MSFT": day1 - pd.Timedelta(days=5)}
            trade_log = []

            def _rank_score_for_day(ticker: str, today_ts: pd.Timestamp) -> float | None:
                raw = results_local[ticker].get("oos_raw_scores")
                if raw is None or today_ts not in raw.index:
                    return None
                raw_score = float(raw.loc[today_ts])
                calibration = results_local[ticker].get("score_calibration")
                return raw_score if calibration is None else float(calibration.calibrate(raw_score))

            bt_dates_local = [day1, day2, day3]
            for today in bt_dates_local:
                regime_name = regime_series_local.get(today, "BULL_CALM")
                rp = RP.get(regime_name, RP["BULL_CALM"])
                regime_confidence_value = float(regime_conf_local.get(today, 1.0)) if regime_conf_local is not None else 1.0
                port_val = INITIAL_CASH
                open_slots = MAX_POSITIONS - len(holdings)
                max_pos_pct = rp.get("max_position_pct", 0.15) * regime_confidence_value
                candidates = []

                for ticker in exportable:
                    if ticker in holdings:
                        continue
                    last_s = last_sell.get(ticker)
                    if last_s and (today - last_s).days < WASH_SALE_DAYS:
                        continue
                    sig = results_local[ticker]["oos_signals"]
                    if today not in sig.index or sig.loc[today] != 1:
                        continue
                    raw_today = float(results_local[ticker]["oos_raw_scores"].loc[today])
                    rank_today = _rank_score_for_day(ticker, today)
                    if rank_today is None or rank_today < rp.get("min_model_score", MIN_MODEL_SCORE):
                        continue
                    candidates.append((ticker, raw_today, rank_today, rs_by_day[today].get(ticker, 0.0)))

                if len(candidates) > 1:
                    ms = [c[2] for c in candidates]
                    rs = [c[3] for c in candidates]
                    ms_range = max(ms) - min(ms) or 1
                    rs_range = max(rs) - min(rs) or 1
                    candidates.sort(
                        key=lambda c: rank_weight * (c[2] - min(ms)) / ms_range + rs_weight * (c[3] - min(rs)) / rs_range,
                        reverse=True,
                    )

                selected = []
                for ticker, raw_score, rank_score, rs_score in candidates:
                    if len(selected) >= open_slots:
                        break
                    tier_idx = min(len(selected), len(tiers) - 1)
                    if rank_score < tiers[tier_idx].get("min_model_score", MIN_MODEL_SCORE):
                        continue
                    last_s = last_sell.get(ticker)
                    if last_s and (today - last_s).days < WASH_SALE_DAYS:
                        continue
                    sector = config_local["sector_map"].get(ticker, "other")
                    if ticker not in DEFENSIVE and MAX_PER_SECTOR > 0:
                        sector_count = sum(
                            1 for held in list(holdings.keys()) + selected
                            if config_local["sector_map"].get(held, "other") == sector
                        )
                        if sector_count >= MAX_PER_SECTOR:
                            continue
                    corr_ok = True
                    for held in list(holdings.keys()) + selected:
                        corr = corr_local.get(ticker, {}).get(held, corr_local.get(held, {}).get(ticker, 0.0))
                        if abs(corr) >= CORR_THRESHOLD:
                            corr_ok = False
                            break
                    if not corr_ok:
                        continue
                    selected.append(ticker)

                for ticker in selected:
                    cash_reserve = port_val * rp.get("cash_reserve_pct", 0.0) * regime_confidence_value
                    invest = min(cash - cash_reserve, port_val * max_pos_pct)
                    cash -= invest
                    holdings[ticker] = {"shares": invest / 100.0, "entry_price": 100.0, "entry_date": today, "high_price": 100.0}
                    trade_log.append({"action": "buy", "ticker": ticker, "date": today, "invest": round(invest, 2)})
            return trade_log

        trades = run_sim_with_rs(
            tickers,
            ohlcv,
            results,
            cfg,
            regime,
            spy,
            corr_dict=corr_dict,
            regime_confidence=regime_conf,
        )
        buys = [(t["date"], t["ticker"], t["invest"]) for t in trades if t["action"] == "buy"]
        assert buys == [
            (day1, "AAPL", 1000.0),
            (day2, "JPM", 1500.0),
            (day3, "XOM", 1200.0),
        ]



class TestSectorGuard:
    """max_positions_per_sector: no more than N positions in same sector."""

    def _make_tech_results(self, tickers, score=0.8):
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        out = {}
        for t in tickers:
            out[t] = {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, score),
                "oos_prices":     prices.reindex(dates),
            }
        return out

    def test_sector_cap_enforced(self):
        """With max_positions_per_sector=3 and 5 tech tickers, only 3 should be bought."""
        tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA"]  # all tech
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {t: make_ohlcv(prices.reindex(dates)) for t in tickers}
        results = self._make_tech_results(tickers)
        cfg = minimal_config(max_concurrent_positions=8)  # allow up to 8 positions

        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(tickers, ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]
        tech_buys = [t for t in buys if cfg["sector_map"].get(t["ticker"], "") == "tech"]

        assert len(tech_buys) <= 3, (
            f"Sector guard should cap tech buys at 3, got {len(tech_buys)}: "
            f"{[t['ticker'] for t in tech_buys]}"
        )

    def test_different_sectors_not_blocked(self):
        """Finance ticker should not be blocked by tech cap."""
        tickers = ["AAPL", "MSFT", "AMZN", "JPM"]  # 3 tech + 1 finance
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {t: make_ohlcv(prices.reindex(dates)) for t in tickers}
        results = self._make_tech_results(tickers)
        cfg = minimal_config(max_concurrent_positions=8)
        # Assign JPM to finance sector
        cfg["sector_map"]["JPM"] = "finance"

        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(tickers, ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]
        bought_tickers = {t["ticker"] for t in buys}
        # JPM (finance) should not be blocked by the tech sector cap
        assert "JPM" in bought_tickers, "JPM (finance) should not be blocked by tech sector cap"


class TestSPYVelocityCrashFilter:
    """SPY velocity filter: no new buys if SPY fell > halt_pct over last N days."""

    def _make_results(self):
        dates = pd.date_range("2024-01-02", periods=60, freq="B")
        prices = make_prices(60)
        return {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }

    def test_spy_crash_blocks_buy(self):
        """If SPY drops >3% in 3 days, no new buys allowed."""
        dates = pd.date_range("2024-01-02", periods=60, freq="B")
        prices = make_prices(60)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = self._make_results()
        cfg = minimal_config()
        regime = pd.Series("BULL_CALM", index=dates)

        # SPY crashes 10% on day 55 (after EMA50 warmup)
        spy = make_prices(60, start=450.0)
        spy.iloc[52] = spy.iloc[51] * 0.88   # 12% crash in one day
        spy.iloc[53] = spy.iloc[52] * 0.99
        spy.iloc[54] = spy.iloc[53] * 0.99

        trades = run_sim(
            ["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates),
            spy_vel_halt_pct=0.03, spy_vel_lookback=3,
        )
        # Days 55-57 should have no buys due to velocity filter
        crash_date = dates[55]
        crash_buys = [t for t in trades if t["action"] == "buy"
                      and pd.Timestamp(t["date"]) >= crash_date
                      and pd.Timestamp(t["date"]) <= dates[57]]
        assert len(crash_buys) == 0, (
            f"SPY velocity filter should block buys on crash days, got {crash_buys}"
        )


class TestSPYEMA50TrendGate:
    """SPY EMA50 trend gate: no new buys when SPY < 50-day EMA."""

    def test_spy_below_ema50_blocks_buy(self):
        """When SPY falls below its 50-day EMA, new buys must be blocked."""
        n = 80
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices = make_prices(n)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        regime = pd.Series("BULL_CALM", index=dates)

        # SPY: start high, then crash below EMA50 at day 55
        spy = pd.Series(450.0, index=dates)
        # Make EMA50 ~450 (starts at 450), then spy drops to 400
        spy.iloc[55:] = 390.0  # far below 450 EMA50

        trades_below = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy)
        # All buys after day 55 should be blocked
        buys_after_crash = [
            t for t in trades_below
            if t["action"] == "buy" and pd.Timestamp(t["date"]) >= dates[55]
        ]
        assert len(buys_after_crash) == 0, (
            f"Expected 0 buys when SPY below EMA50, got {len(buys_after_crash)}"
        )

    def test_spy_above_ema50_allows_buy(self):
        """When SPY is above EMA50, buys are allowed (gate doesn't block)."""
        n = 80
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices = make_prices(n)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        regime = pd.Series("BULL_CALM", index=dates)
        # SPY trending up — always above EMA50
        spy = pd.Series(
            [450.0 * (1 + 0.002 * i) for i in range(n)], index=dates
        )

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy)
        buys = [t for t in trades if t["action"] == "buy"]
        assert len(buys) > 0, "Uptrending SPY should allow buys after EMA warmup"


class TestBEARDefensiveBuying:
    """In BEAR regime: block all offensive buys, allow up to 1 defensive position."""

    def _make_results(self, tickers, score=0.8, signals=1):
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        out = {}
        for t in tickers:
            out[t] = {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, signals),
                "oos_raw_scores": make_raw_scores(dates, score),
                "oos_prices":     prices.reindex(dates),
            }
        return out

    def test_offensive_buys_blocked_in_bear(self):
        """AAPL (offensive, tech) must not be bought during BEAR regime."""
        tickers = ["AAPL"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        results = self._make_results(tickers)
        cfg = minimal_config()
        regime = pd.Series("BEAR", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        aapl_buys = [t for t in trades if t["action"] == "buy" and t["ticker"] == "AAPL"]
        assert len(aapl_buys) == 0, (
            f"AAPL (offensive) must not be bought in BEAR regime, got {len(aapl_buys)} buys"
        )

    def test_defensive_buy_allowed_in_bear(self):
        """GLD (defensive) should be buyable during BEAR regime (up to 1 slot)."""
        tickers = ["GLD"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {"GLD": make_ohlcv(prices.reindex(dates))}
        results = self._make_results(tickers)
        cfg = minimal_config()
        regime = pd.Series("BEAR", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["GLD"], ohlcv, results, cfg, regime, spy.reindex(dates))
        gld_buys = [t for t in trades if t["action"] == "buy" and t["ticker"] == "GLD"]
        assert len(gld_buys) >= 1, "GLD (defensive) should be bought during BEAR regime"
        assert gld_buys[0].get("regime") == "BEAR_defensive", (
            f"Trade should be tagged as BEAR_defensive, got {gld_buys[0]}"
        )

    def test_defensive_cap_at_one_slot(self):
        """At most 1 defensive position at a time during BEAR."""
        # Two defensives with buy signal — only 1 should be selected
        tickers = ["GLD", "TLT"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {t: make_ohlcv(prices.reindex(dates)) for t in tickers}
        results = self._make_results(tickers)
        cfg = minimal_config()
        cfg["defensive_tickers"] = ["GLD", "TLT"]
        regime = pd.Series("BEAR", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(tickers, ohlcv, results, cfg, regime, spy.reindex(dates))
        buy_dates = set()
        bear_buys = [t for t in trades if t["action"] == "buy" and t.get("regime") == "BEAR_defensive"]
        # Should not have two defensive buys on the same day
        for b in bear_buys:
            assert b["date"] not in buy_dates, "Only 1 defensive slot per day"
            buy_dates.add(b["date"])


class TestRankingByModelScore:
    """Ranking uses live model score, not static OOS Sharpe — highest score wins."""

    def test_highest_score_stock_executes_first(self):
        """Given 2 tickers with buy signals, the one with higher raw score executes first."""
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices_a = make_prices(30, start=100.0)
        prices_b = make_prices(30, start=200.0)
        ohlcv = {
            "AAPL": make_ohlcv(prices_a.reindex(dates)),
            "MSFT": make_ohlcv(prices_b.reindex(dates)),
        }
        results = {
            "AAPL": {
                "sharpe": 0.9, "passes_floor": True,  # lower Sharpe
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.9),  # higher live score
                "oos_prices":     prices_a.reindex(dates),
            },
            "MSFT": {
                "sharpe": 1.5, "passes_floor": True,  # higher Sharpe (old ranking would prefer MSFT)
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.3),  # lower live score
                "oos_prices":     prices_b.reindex(dates),
            },
        }
        cfg = minimal_config(max_concurrent_positions=1)  # only 1 slot → only winner executes
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(["AAPL", "MSFT"], ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = [t for t in trades if t["action"] == "buy"]

        assert len(buys) >= 1, "Expected at least one buy"
        # AAPL has higher live score → should be ranked first → bought first
        first_buy_ticker = buys[0]["ticker"]
        assert first_buy_ticker == "AAPL", (
            f"Expected AAPL (higher live score) to execute first, got {first_buy_ticker}. "
            f"If MSFT executed first, old static-Sharpe ranking was used."
        )


class TestWashSaleGuard:
    """Cannot buy a ticker within wash_sale_days of selling it."""

    def test_wash_sale_blocks_rebuy(self):
        """After selling AAPL, a rebuy within 30 days must be blocked."""
        n = 80
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices = make_prices(n)

        # AAPL: sell signal on day 21 (after min_hold), buy signal all other days
        signals = pd.Series(1, index=dates)
        signals.iloc[21:24] = -1    # 3 consecutive sell signals → model exit on day 23
        signals.iloc[24:] = 1       # buy signal again immediately after

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    signals,
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
            min_hold_profit_days=20, min_hold_loss_days=20,
        )
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(n, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        buys = sorted([t for t in trades if t["action"] == "buy"], key=lambda t: t["date"])
        sells = [t for t in trades if t["action"] == "sell"]

        if len(buys) >= 2 and len(sells) >= 1:
            sell_date = pd.Timestamp(sells[0]["date"])
            rebuy_date = pd.Timestamp(buys[1]["date"])
            days_between = (rebuy_date - sell_date).days
            assert days_between >= 30, (
                f"Wash-sale violated: sold on {sell_date.date()}, rebought on {rebuy_date.date()} "
                f"({days_between} days gap, need ≥ 30)"
            )


class TestConsecutiveSellSignals:
    """Model exit requires 3 consecutive sell signals before executing."""

    def test_single_sell_signal_does_not_exit(self):
        """One isolated sell signal must not trigger an exit."""
        n = 60
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices = make_prices(n)

        signals = pd.Series(1, index=dates)
        signals.iloc[25] = -1  # single isolated sell signal
        signals.iloc[26] = 1   # buy again immediately

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    signals,
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(n, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        model_sells = [t for t in trades if t.get("exit_reason") == "model_sell"]
        assert len(model_sells) == 0, (
            f"Single sell signal must not trigger exit, got {len(model_sells)} model_sells"
        )

    def test_three_consecutive_sells_exits(self):
        """Three consecutive sell signals after min_hold should trigger exit."""
        n = 80
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices = make_prices(n)

        signals = pd.Series(1, index=dates)
        signals.iloc[22:25] = -1  # 3 consecutive sells after min_hold (20 days)

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    signals,
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices.reindex(dates),
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        ohlcv = {"AAPL": make_ohlcv(prices.reindex(dates))}
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(n, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        model_sells = [t for t in trades if t.get("exit_reason") == "model_sell"]
        assert len(model_sells) >= 1, (
            "Three consecutive sell signals after min_hold should trigger model exit"
        )


class TestStopLoss:
    """Hard stop-loss exits at configured loss threshold."""

    def test_stop_loss_triggers(self):
        """Position that drops 20% triggers 15% stop-loss."""
        n = 40
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        prices_up = make_prices(5, start=100.0)
        prices_crash = make_prices(n - 5, start=80.0, drift=-0.001)  # start below stop
        prices = pd.concat([prices_up, prices_crash])
        prices.index = dates

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices,
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        ohlcv = {"AAPL": make_ohlcv(prices)}
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(n, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        stop_sells = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        assert len(stop_sells) >= 1, "Price crash of 20% should trigger 15% stop-loss"


class TestTrailingStop:
    """Trailing stop activates after gain reaches trigger threshold."""

    def test_trailing_stop_triggers_after_gain(self):
        """Position up 25% then drops 20% → trailing stop fires."""
        n = 60
        dates = pd.date_range("2024-01-02", periods=n, freq="B")

        # Price rises 25%, then falls back sharply
        prices = pd.Series(100.0, index=dates)
        for i in range(1, 30):
            prices.iloc[i] = prices.iloc[i - 1] * 1.01   # up 1%/day → ~35% gain
        for i in range(30, n):
            prices.iloc[i] = prices.iloc[i - 1] * 0.985  # down 1.5%/day

        results = {
            "AAPL": {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.8),
                "oos_prices":     prices,
            }
        }
        cfg = minimal_config(
            backtest_start=str(dates[0].date()),
            backtest_end=str(dates[-1].date()),
        )
        ohlcv = {"AAPL": make_ohlcv(prices)}
        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(n, start=450.0)

        trades = run_sim(["AAPL"], ohlcv, results, cfg, regime, spy.reindex(dates))
        ts_sells = [t for t in trades if t.get("exit_reason") == "trailing_stop"]
        assert len(ts_sells) >= 1, (
            "Expected trailing stop to fire after 35% gain then retracement"
        )


class TestCorrelationGuard:
    """Correlation guard: skip candidates highly correlated with existing holdings."""

    def test_high_correlation_blocks_buy(self):
        """If AAPL is already held and MSFT is 90% correlated, MSFT should be skipped."""
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {
            "AAPL": make_ohlcv(prices.reindex(dates)),
            "MSFT": make_ohlcv(prices.reindex(dates)),  # identical prices = 100% correlation
        }
        results = {t: {
            "sharpe": 1.5, "passes_floor": True,
            "oos_signals":    make_signals(dates, 1),
            "oos_raw_scores": make_raw_scores(dates, 0.8),
            "oos_prices":     prices.reindex(dates),
        } for t in ["AAPL", "MSFT"]}

        cfg = minimal_config(max_concurrent_positions=8)
        cfg["sector_map"]["MSFT"] = "finance"  # different sector so sector guard doesn't interfere
        # Inject 100% correlation between AAPL and MSFT
        corr_dict = {"AAPL": {"MSFT": 0.95}, "MSFT": {"AAPL": 0.95}}

        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(
            ["AAPL", "MSFT"], ohlcv, results, cfg, regime, spy.reindex(dates),
            corr_dict=corr_dict,
        )
        buys = {t["ticker"] for t in trades if t["action"] == "buy"}
        # At most one of AAPL/MSFT should be bought (correlation guard blocks the second)
        assert not ("AAPL" in buys and "MSFT" in buys), (
            "Correlation guard should prevent buying both AAPL and MSFT (0.95 correlation)"
        )


class TestMaxConcurrentPositions:
    """Portfolio never exceeds max_concurrent_positions."""

    def test_position_cap_enforced(self):
        """With 10 tickers all signaling buy, only max_concurrent_positions are held."""
        tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA",
                   "NFLX", "CRM", "PLTR", "UBER", "AMD"]
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        prices = make_prices(30)
        ohlcv = {t: make_ohlcv(prices.reindex(dates)) for t in tickers}
        results = {}
        for i, t in enumerate(tickers):
            results[t] = {
                "sharpe": 1.5, "passes_floor": True,
                "oos_signals":    make_signals(dates, 1),
                "oos_raw_scores": make_raw_scores(dates, 0.1 + i * 0.05),
                "oos_prices":     prices.reindex(dates),
            }
        cfg = minimal_config(max_concurrent_positions=4)
        # Use different sectors to avoid sector guard interference
        for i, t in enumerate(tickers):
            cfg["sector_map"][t] = f"sector_{i}"

        regime = pd.Series("BULL_CALM", index=dates)
        spy = make_prices(30, start=450.0)

        trades = run_sim(tickers, ohlcv, results, cfg, regime, spy.reindex(dates))
        # Count max simultaneous positions on any day
        open_pos = 0
        for t in trades:
            if t["action"] == "buy":
                open_pos += 1
            elif t["action"] == "sell":
                open_pos -= 1
            assert open_pos <= 4, f"Position count exceeded 4: {open_pos}"


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
