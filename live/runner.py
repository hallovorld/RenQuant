"""Live trading runner.

Usage::

    python -m live.runner --strategy renquant_101 --broker paper --once
    python -m live.runner --strategy renquant_102 --broker alpaca-paper --once
    python -m live.runner --strategy renquant_102 --broker alpaca --once  # real money

Broker options: paper, alpaca, alpaca-paper, ibkr

Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables for Alpaca.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.config import load_strategy_config
from common.data import fetch_ohlcv
from common.indicators import compute_indicators
from common.models import create_model
from common.strategy import StrategyConfig

from .broker import BaseBroker
from .alpaca_broker import AlpacaBroker
from .ibkr_broker import IBKRBroker
from .paper_broker import PaperBroker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("live.runner")


def _load_strategy(strategy_name: str):
    """Load strategy config and model from backtesting/{strategy_name}/."""
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_path = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config = StrategyConfig.load(config_path)

    # Load the trained model
    model = create_model(config.model_type, **config.model_params)
    model.load(strategy_dir, config.name)

    return config, model, strategy_dir


def _get_broker(broker_type: str, initial_cash: float = 100_000) -> BaseBroker:
    if broker_type == "paper":
        return PaperBroker(initial_cash=initial_cash)
    elif broker_type == "alpaca":
        return AlpacaBroker(paper=False)
    elif broker_type == "alpaca-paper":
        return AlpacaBroker(paper=True)
    elif broker_type == "ibkr":
        return IBKRBroker()
    else:
        raise ValueError(f"Unknown broker: {broker_type}")


def _log_trade(strategy_dir: Path, strategy_name: str, record: dict) -> None:
    """Append trade record to daily log file."""
    log_dir = REPO_ROOT / "live" / "logs" / strategy_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"

    entries = []
    if log_file.exists():
        entries = json.loads(log_file.read_text())
    entries.append(record)
    log_file.write_text(json.dumps(entries, indent=2, default=str))


def run_once(
    config: StrategyConfig,
    model,
    broker: BaseBroker,
    strategy_dir: Path,
) -> None:
    """Execute one trading cycle: fetch data → compute signal → place order."""
    log.info("Running strategy %s on %s", config.name, config.symbol)

    # Fetch latest data
    df = fetch_ohlcv(config.symbol, provider=config.data_provider)
    df = compute_indicators(df, config.indicator_spec).dropna()

    if df.empty:
        log.warning("No data after indicator computation")
        return

    # Build current state
    latest = df.iloc[-1].copy()
    position = broker.get_position(config.symbol)
    latest["position_flag"] = 1 if position > 0 else 0

    # For FQI models that need gate signals
    if len(df) >= 2:
        prev = df.iloc[-2]
        latest["buy_signal"] = int(
            latest.get("macd_line", 0) > latest.get("macd_signal", 0)
            and prev.get("macd_line", 0) <= prev.get("macd_signal", 0)
            and latest.get("rsi", 50) > 50
        )
        latest["sell_signal"] = int(
            latest.get("macd_line", 0) < latest.get("macd_signal", 0)
            and prev.get("macd_line", 0) >= prev.get("macd_signal", 0)
            and latest.get("rsi", 50) < 50
        )
    else:
        latest["buy_signal"] = 0
        latest["sell_signal"] = 0

    signal = model.predict(latest)
    log.info("Signal: %s (position=%.0f)", signal, position)

    record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": config.symbol,
        "signal": signal,
        "position_before": position,
        "date": str(df.index[-1].date()),
    }

    # Execute
    if signal == "buy" and position <= 0:
        cash = broker.get_account_value()
        price = float(latest["close"])
        shares = int(cash * 0.95 / price)  # 95% of cash
        if shares > 0:
            result = broker.place_order(config.symbol, "BUY", shares)
            record["order"] = result
            log.info("Bought %d shares at ~$%.2f", shares, price)
    elif signal == "sell" and position > 0:
        result = broker.place_order(config.symbol, "SELL", abs(position))
        record["order"] = result
        log.info("Sold %.0f shares", abs(position))
    else:
        log.info("No action taken")

    _log_trade(strategy_dir, config.name, record)


# ── Multi-stock support ──────────────────────────────────────────────────


def _get_model_score(model, row) -> float:
    """Return a continuous float score for *row* using the best available method.

    Priority:
      1. ``predict_score_bulk``  — present on Classification, QLearning, XGBoost, Manual
      2. ``predict_score``       — XGBoost legacy name (kept for backwards compat)
      3. Fallback: map predict() string → {buy: 1.0, hold: 0.0, sell: -1.0}
    """
    df_row = row.to_frame().T
    if hasattr(model, "predict_score_bulk"):
        return float(model.predict_score_bulk(df_row).iloc[0])
    if hasattr(model, "predict_score"):
        return float(model.predict_score(df_row).iloc[0])
    return {"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(model.predict(row), 0.0)


def _load_strategy_multi(strategy_name: str) -> tuple[dict[str, Any], dict, Path]:
    """Load multi-stock strategy config and per-stock models.

    Each symbol has its own model type defined in its policy-metadata.json,
    so we read the metadata first and create the correct model per symbol.
    """
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_path = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    staleness_days = int(config.get("model_staleness_days", 30))
    models_dir = strategy_dir / "models"

    models = {}
    for symbol in config["watchlist"]:
        meta_path = models_dir / symbol / f"{symbol}-policy-metadata.json"
        if not meta_path.exists():
            log.warning("No model metadata for %s, skipping", symbol)
            continue

        metadata = json.loads(meta_path.read_text())

        # Check staleness
        trained_date = metadata.get("trained_date")
        if trained_date and staleness_days > 0:
            from datetime import date
            age = (date.today() - datetime.strptime(trained_date, "%Y-%m-%d").date()).days
            if age > staleness_days:
                log.warning("%s model is %d days old (limit=%d), skipping", symbol, age, staleness_days)
                continue

        policy_type = metadata["policy_type"]
        model = create_model(policy_type)
        model.load(models_dir / symbol, symbol)
        models[symbol] = model

    log.info("Loaded models for %d/%d symbols: %s",
             len(models), len(config["watchlist"]), sorted(models.keys()))
    return config, models, strategy_dir


def _build_relative_features(
    df_stock, df_spy, feature_columns, indicator_spec,
):
    """Compute indicators and relative features for a stock vs SPY."""
    import numpy as np
    import pandas as pd

    df_stock = compute_indicators(df_stock, indicator_spec)
    df_spy = compute_indicators(df_spy, indicator_spec)

    common_idx = df_stock.index.intersection(df_spy.index)
    if len(common_idx) == 0:
        return None

    df_stock = df_stock.loc[common_idx]
    df_spy = df_spy.loc[common_idx]

    ratio_features = {"rsi", "adx"}
    diff_features = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

    # ── SPY regime-context features (renquant_103 models trained with these) ──
    # Computed from SPY series using the same formulas as the notebook training cell.
    # Without these, any model whose feature_columns includes spy_realized_vol /
    # spy_adx / spy_trend / hurst_proxy would raise KeyError at inference time.
    spy_close_full = df_spy["close"]
    spy_rets_full  = spy_close_full.pct_change()
    spy_ema50_full = spy_close_full.ewm(span=50, adjust=False).mean()

    def _hurst_proxy(x):
        """Lag-1 autocorr of a 20-day window of SPY daily returns."""
        if len(x) <= 2 or np.std(x) == 0:
            return 0.0
        cc = np.corrcoef(x[:-1], x[1:])
        v = cc[0, 1]
        return float(v) if not np.isnan(v) else 0.0

    spy_regime_features = {
        "spy_realized_vol": spy_rets_full.rolling(20).std() * np.sqrt(252),
        "spy_adx":   df_spy["adx"] if "adx" in df_spy.columns
                     else pd.Series(25.0, index=df_spy.index),
        "spy_trend": spy_close_full / spy_ema50_full.replace(0, np.nan),
        "hurst_proxy": spy_rets_full.rolling(20).apply(_hurst_proxy, raw=True),
    }

    result = pd.DataFrame(index=common_idx)
    result["close"] = df_stock["close"]
    for col in feature_columns:
        if col in ratio_features:
            if col in df_stock.columns and col in df_spy.columns:
                result[col] = df_stock[col] / df_spy[col].replace(0, np.nan)
        elif col in diff_features:
            if col in df_stock.columns and col in df_spy.columns:
                result[col] = df_stock[col] - df_spy[col]
        elif col in spy_regime_features:
            result[col] = spy_regime_features[col].reindex(common_idx)
        elif col in df_stock.columns:
            result[col] = df_stock[col]
        # else: unknown column — silently skip (Manual models don't use feature_columns)

    # Trend and relative momentum features (required by Q-Learning and Dual Momentum models)
    close = df_stock.loc[common_idx, "close"]
    spy_close = df_spy.loc[common_idx, "close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    result["trend"] = close / ema50
    result["trend_long"] = close / ema200
    rel_price = close / spy_close.replace(0, np.nan)
    result["rel_mom_20d"] = rel_price.pct_change(20)
    result["rel_mom_60d"] = rel_price.pct_change(60)

    return result.dropna()


def run_once_multi(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool = False,
) -> None:
    """Execute one multi-stock trading cycle: scan → analyze → trade.

    Parameters
    ----------
    sell_only:
        When True, skip the buy phase entirely.  Used for intraday runs
        (market-open and pre-close) where the goal is to exit loss positions
        quickly without placing new entries on incomplete daily bars.
    """
    import numpy as np

    watchlist = config["watchlist"]
    benchmark = config.get("benchmark", "SPY")
    max_positions = config.get("max_concurrent_positions", 3)
    zscore_threshold = float(config.get("volume_zscore_threshold", 1.5))
    zscore_lookback = int(config.get("volume_zscore_lookback", 20))
    vol_filter = config.get("volume_filter", {})
    filter_mode = vol_filter.get("mode", "percentile")
    percentile_threshold = float(vol_filter.get("percentile_threshold", 85))
    indicator_spec = config.get("indicator_spec", {})
    feature_columns = config["model_params"]["feature_columns"]
    pos_sizing = config.get("position_sizing", {})
    max_position_pct = float(pos_sizing.get("max_position_pct", 0.30))
    cash_reserve_pct = float(pos_sizing.get("cash_reserve_pct", 0.00))

    log.info("Running multi-stock strategy %s on %s", config["model_name"], watchlist)

    # Fetch data for all stocks + benchmark
    dfs = {}
    for symbol in watchlist + [benchmark]:
        df = fetch_ohlcv(symbol, provider=config.get("data_src", "yfinance"))
        if df.empty:
            log.warning("No data for %s, skipping", symbol)
            continue
        dfs[symbol] = df

    if benchmark not in dfs:
        log.error("No benchmark data for %s", benchmark)
        return

    df_spy = dfs[benchmark]

    # Risk config
    risk_cfg = config.get("risk", {})
    stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.0))
    drawdown_halt_pct = float(risk_cfg.get("portfolio_drawdown_halt_pct", 0.0))
    regime_cfg = risk_cfg.get("regime_filter", {})
    regime_enabled = bool(regime_cfg.get("enabled", False))
    regime_sma_period = int(regime_cfg.get("sma_period", 200))
    sector_map = config.get("sector_map", {})
    max_per_sector = int(config.get("max_positions_per_sector", 0))

    # Load persisted live state (entry dates, sell streaks, high-water mark)
    state_file = strategy_dir / "live_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    entry_dates: dict     = state.setdefault("entry_dates", {})     # symbol → "YYYY-MM-DD"
    sell_streaks: dict    = state.setdefault("sell_streaks", {})    # symbol → int
    last_sell_dates: dict = state.setdefault("last_sell_dates", {}) # symbol → "YYYY-MM-DD"

    min_hold_days = int(config.get("min_hold_days", 0))
    consec_sells_required = int(config.get("consecutive_sell_signals", 1))
    wash_sale_days = int(config.get("wash_sale_days", 30))

    # Step 1: Check current positions, process sells first
    held = [s for s in watchlist if broker.get_position(s) > 0]
    log.info("Current positions: %s", held if held else "none")

    for symbol in list(held):
        if symbol not in dfs:
            continue

        # Stop-loss check before model signal
        if stop_loss_pct > 0:
            avg_cost = broker.get_avg_cost(symbol)
            current_price = float(dfs[symbol]["close"].iloc[-1])
            if avg_cost > 0:
                loss_pct = (avg_cost - current_price) / avg_cost
                if loss_pct >= stop_loss_pct:
                    position = broker.get_position(symbol)
                    result = broker.place_order(symbol, "SELL", abs(position))
                    log.info("STOP LOSS %s: loss=%.1f%% avg=%.2f current=%.2f",
                             symbol, loss_pct * 100, avg_cost, current_price)
                    _log_trade(strategy_dir, config["model_name"], {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol, "signal": "stop_loss",
                        "loss_pct": round(loss_pct, 4), "order": result,
                    })
                    entry_dates.pop(symbol, None)
                    sell_streaks.pop(symbol, None)
                    last_sell_dates[symbol] = datetime.now().strftime("%Y-%m-%d")
                    held.remove(symbol)
                    continue

        # Single-day loss gate — catches gap-down days before the cumulative stop fires.
        # Reads max_single_day_loss_pct from BULL_CALM regime params (most relevant regime;
        # other regimes already have tight 5% cumulative stops so the value is 0 there).
        _bull_calm_rp = config.get("regime_params", {}).get("BULL_CALM", {})
        sdl_pct = float(_bull_calm_rp.get("max_single_day_loss_pct", 0.0))
        if sdl_pct > 0 and len(dfs[symbol]) >= 2:
            today_close = float(dfs[symbol]["close"].iloc[-1])
            prev_close  = float(dfs[symbol]["close"].iloc[-2])
            if prev_close > 0:
                daily_drop = (prev_close - today_close) / prev_close
                if daily_drop >= sdl_pct:
                    position = broker.get_position(symbol)
                    result = broker.place_order(symbol, "SELL", abs(position))
                    log.info("SINGLE DAY LOSS %s: drop=%.1f%% prev=%.2f today=%.2f",
                             symbol, daily_drop * 100, prev_close, today_close)
                    _log_trade(strategy_dir, config["model_name"], {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol, "signal": "single_day_loss",
                        "daily_drop_pct": round(daily_drop, 4), "order": result,
                    })
                    entry_dates.pop(symbol, None)
                    sell_streaks.pop(symbol, None)
                    last_sell_dates[symbol] = datetime.now().strftime("%Y-%m-%d")
                    held.remove(symbol)
                    continue

        # min_hold_days guard: block model-sell exits during early hold period.
        # Stop-loss and single-day loss gate are exempt (already processed above).
        today_str = datetime.now().strftime("%Y-%m-%d")
        entry_date_str = entry_dates.get(symbol)
        days_held = 0
        if entry_date_str:
            from datetime import date as _date
            try:
                days_held = (_date.today() - _date.fromisoformat(entry_date_str)).days
            except ValueError:
                days_held = 0
        if min_hold_days > 0 and days_held < min_hold_days:
            log.info("%s min_hold_days=%d, held %d days — skipping model-sell check",
                     symbol, min_hold_days, days_held)
            sell_streaks[symbol] = 0  # reset streak: we're inside the hold window
            continue

        model_feature_cols = getattr(models[symbol], "feature_columns", None) or feature_columns
        rel = _build_relative_features(dfs[symbol], df_spy, model_feature_cols, indicator_spec)
        if rel is None or rel.empty:
            continue
        row = rel.iloc[-1].copy()
        row["position_flag"] = 1  # currently held — needed by Q-Learning state encoding
        signal = models[symbol].predict(row)
        if signal == "sell":
            sell_streaks[symbol] = sell_streaks.get(symbol, 0) + 1
            streak = sell_streaks[symbol]
            if streak < consec_sells_required:
                log.info("%s sell signal (streak %d/%d) — waiting for %d consecutive",
                         symbol, streak, consec_sells_required, consec_sells_required)
            else:
                sell_streaks[symbol] = 0
                position = broker.get_position(symbol)
                result = broker.place_order(symbol, "SELL", abs(position))
                log.info("SELL %s (streak=%d, held=%d days): %.0f shares",
                         symbol, streak, days_held, abs(position))
                _log_trade(strategy_dir, config["model_name"], {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "signal": "sell",
                    "days_held": days_held, "sell_streak": streak,
                    "order": result,
                })
                entry_dates.pop(symbol, None)
                last_sell_dates[symbol] = datetime.now().strftime("%Y-%m-%d")
                held.remove(symbol)
        else:
            sell_streaks[symbol] = 0  # reset streak on non-sell signal

    # Sell-only mode: persist state updates (stop-loss exits may have cleared entry dates)
    # then stop — don't scan for new buys.
    if sell_only:
        log.info("sell_only mode — skipping buy phase")
        state_file.write_text(json.dumps(state, indent=2))
        return

    # Step 2: Check portfolio drawdown circuit breaker
    open_slots = max_positions - len(held)
    if open_slots <= 0:
        log.info("All %d slots filled, no scanning", max_positions)
        return

    if drawdown_halt_pct > 0:
        account_value = broker.get_account_value()
        hwm = float(state.get("high_water_mark", account_value))
        hwm = max(hwm, account_value)
        state["high_water_mark"] = hwm
        if hwm > 0:
            drawdown = (hwm - account_value) / hwm
            if drawdown >= drawdown_halt_pct:
                log.info("Drawdown circuit breaker: %.1f%% >= %.0f%%, halting new buys",
                         drawdown * 100, drawdown_halt_pct * 100)
                return

    # Step 3: Regime filter — suppress new buys when SPY is below its 200-day SMA
    if regime_enabled and benchmark in dfs:
        spy_close = dfs[benchmark]["close"].astype(float)
        if len(spy_close) >= regime_sma_period:
            sma200 = float(spy_close.iloc[-regime_sma_period:].mean())
            if float(spy_close.iloc[-1]) <= sma200:
                log.info("Regime filter: SPY(%.2f) below 200-SMA(%.2f), suppressing new buys",
                         float(spy_close.iloc[-1]), sma200)
                return

    vol_candidates = []
    for symbol in watchlist:
        if symbol in held or symbol not in dfs:
            continue
        df = dfs[symbol]
        if len(df) < zscore_lookback + 1:
            continue
        vol = df["volume"].astype(float)
        today_vol = float(vol.iloc[-1])
        hist_vol = vol.iloc[-(zscore_lookback + 1):-1]

        if filter_mode == "percentile":
            # Percentile rank: what fraction of lookback days had lower volume?
            vol_score = float((hist_vol < today_vol).sum()) / len(hist_vol) * 100
            triggered = vol_score >= percentile_threshold
        else:
            # Legacy z-score mode
            roll_mean = float(hist_vol.mean())
            roll_std = float(hist_vol.std())
            vol_score = (today_vol - roll_mean) / roll_std if roll_std > 0 else 0
            triggered = vol_score >= zscore_threshold

        if triggered:
            # Bullish filter: only enter on up-close days
            close = df["close"].astype(float)
            if len(close) >= 2 and close.iloc[-1] <= close.iloc[-2]:
                continue
            vol_candidates.append((symbol, vol_score))

    score_label = "pct" if filter_mode == "percentile" else "z"
    log.info("Volume candidates: %s",
             [(s, f"{score_label}={v:.1f}") for s, v in vol_candidates] or "none")

    # Step 4: Score all volume candidates with their models, then rank by model score.
    # This ensures the highest-conviction signal gets priority over the largest volume spike.
    #
    # Tiered thresholds: each successive order placed in a single run requires a
    # progressively higher bar. Configure via "tiered_thresholds" in strategy_config.json:
    #   [{"min_volume_pct": 85, "min_model_score": 0.0},   <- slot 1 (easiest)
    #    {"min_volume_pct": 90, "min_model_score": 0.30},  <- slot 2
    #    {"min_volume_pct": 95, "min_model_score": 0.50}]  <- slot 3 (strictest)
    # When not configured, a single tier uses the base thresholds for all slots.
    vol_key = "min_volume_pct" if filter_mode == "percentile" else "min_volume_zscore"
    base_vol_thresh = percentile_threshold if filter_mode == "percentile" else zscore_threshold
    base_score_thresh = float(config.get("min_model_score", 0.0))

    raw_tiers = config.get("tiered_thresholds", [])
    if raw_tiers:
        tiers = [
            (float(t.get(vol_key, base_vol_thresh)), float(t.get("min_model_score", 0.0)))
            for t in raw_tiers
        ]
    else:
        tiers = [(base_vol_thresh, base_score_thresh)]

    # Pre-score all candidates that pass the most lenient tier (tier 1)
    tier1_vol_min, tier1_score_min = tiers[0]
    scored_candidates = []
    for symbol, vol_score in vol_candidates:
        if symbol not in models:
            continue
        if vol_score < tier1_vol_min:
            continue  # didn't even meet slot-1 bar

        # Wash-sale guard: block re-buy within wash_sale_days of last sell
        if wash_sale_days > 0 and symbol in last_sell_dates:
            from datetime import date as _date
            try:
                days_since_sell = (_date.today() - _date.fromisoformat(last_sell_dates[symbol])).days
                if days_since_sell < wash_sale_days:
                    log.info("%s wash-sale blocked: sold %d days ago (limit=%d)",
                             symbol, days_since_sell, wash_sale_days)
                    continue
            except ValueError:
                pass
        model_feature_cols = getattr(models[symbol], "feature_columns", None) or feature_columns
        rel = _build_relative_features(dfs[symbol], df_spy, model_feature_cols, indicator_spec)
        if rel is None or rel.empty:
            continue
        row = rel.iloc[-1].copy()
        row["position_flag"] = 0
        signal = models[symbol].predict(row)
        model_score = _get_model_score(models[symbol], row)
        log.info("%s %s=%.1f signal=%s model_score=%.4f",
                 symbol, score_label, vol_score, signal, model_score)
        if signal == "buy" and model_score >= tier1_score_min:
            scored_candidates.append((symbol, vol_score, model_score, row))

    # Sort by model score descending — highest conviction first
    scored_candidates.sort(key=lambda x: x[2], reverse=True)

    slots_filled_this_run = 0
    for symbol, vol_score, model_score, row in scored_candidates:
        if open_slots <= 0:
            break
        # Determine which tier applies for the slot we're about to fill
        tier_idx = min(slots_filled_this_run, len(tiers) - 1)
        tier_min_vol, tier_min_score = tiers[tier_idx]
        slot_num = slots_filled_this_run + 1

        if vol_score < tier_min_vol:
            log.info("Slot %d: %s vol_score=%.1f below tier min %.1f, skipping",
                     slot_num, symbol, vol_score, tier_min_vol)
            continue
        if model_score < tier_min_score:
            log.info("Slot %d: %s model_score=%.4f below tier min %.4f, skipping",
                     slot_num, symbol, model_score, tier_min_score)
            continue

        # Sector concentration guard (applied after ranking so sector count stays current)
        if max_per_sector > 0:
            sector = sector_map.get(symbol, "other")
            sector_count = sum(1 for h in held if sector_map.get(h, "other") == sector)
            if sector_count >= max_per_sector:
                log.info("Sector limit: %s (%s) already at %d/%d, skipping",
                         symbol, sector, sector_count, max_per_sector)
                continue

        account_value = broker.get_account_value()
        price = float(dfs[symbol]["close"].iloc[-1])
        cash_reserve = account_value * cash_reserve_pct
        available = broker.get_account_value() - cash_reserve
        max_invest = account_value * max_position_pct
        invest = min(max_invest, max(available, 0))
        shares = int(invest / price)
        if shares > 0:
            result = broker.place_order(symbol, "BUY", shares)
            log.info("BUY %s (slot %d/%d): %d shares at ~$%.2f (%s=%.1f model_score=%.4f)",
                     symbol, slot_num, max_positions, shares, price,
                     score_label, vol_score, model_score)
            _log_trade(strategy_dir, config["model_name"], {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol, "signal": "buy",
                "slot": slot_num,
                "volume_score": vol_score,
                "volume_filter_mode": filter_mode,
                "model_score": model_score,
                "order": result,
            })
            # Persist entry date so min_hold_days can be enforced on future runs
            entry_dates[symbol] = datetime.now().strftime("%Y-%m-%d")
            sell_streaks.pop(symbol, None)
            last_sell_dates.pop(symbol, None)  # reset wash-sale clock on new entry
            held.append(symbol)
            slots_filled_this_run += 1
            open_slots -= 1

    # Persist updated state (entry dates, sell streaks, high-water mark)
    state_file.write_text(json.dumps(state, indent=2))


def _is_multi_stock(strategy_name: str) -> bool:
    """Check if a strategy uses multi-stock watchlist config."""
    config_path = REPO_ROOT / "backtesting" / strategy_name / "strategy_config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    return "watchlist" in config


def main():
    parser = argparse.ArgumentParser(description="RenQuant live trading runner")
    parser.add_argument("--strategy", required=True, help="Strategy directory name")
    parser.add_argument("--broker", choices=["paper", "alpaca", "alpaca-paper", "ibkr"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--sell-only", action="store_true",
                        help="Process exits only — skip buy scan (for intraday runs)")
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400)")
    args = parser.parse_args()

    multi = _is_multi_stock(args.strategy)

    if multi:
        config, models, strategy_dir = _load_strategy_multi(args.strategy)
        initial_cash = config.get("initial_cash", 100_000)
        broker = _get_broker(args.broker, initial_cash=initial_cash)
        broker.connect()
        run_fn = lambda: run_once_multi(config, models, broker, strategy_dir,
                                        sell_only=args.sell_only)
    else:
        config, model, strategy_dir = _load_strategy(args.strategy)
        broker = _get_broker(args.broker, initial_cash=config.initial_cash)
        broker.connect()
        run_fn = lambda: run_once(config, model, broker, strategy_dir)

    try:
        if args.once:
            run_fn()
        else:
            log.info("Starting scheduled mode (interval=%ds)", args.interval)
            while True:
                try:
                    run_fn()
                except Exception:
                    log.exception("Error in trading cycle")
                log.info("Sleeping %ds until next cycle...", args.interval)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
