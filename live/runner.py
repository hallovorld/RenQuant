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

import numpy as np

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.config import load_strategy_config
from common.data import fetch_ohlcv
from common.indicators import compute_indicators
from common.models import create_model
from common.models.scoring import (
    ScoreCalibration,
    evaluate_row,
    extract_raw_score,
    extract_raw_scores_bulk,
    fit_probability_calibration,
    raw_score_kind_for_model,
)
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
    return extract_raw_score(model, row)


def _get_rank_score(model, row) -> float:
    calibration = getattr(model, "_score_calibration", None)
    return evaluate_row(model, row, calibration).rank_score


def _build_score_calibration(
    model,
    metadata: dict[str, Any],
    df_stock,
    df_spy,
    indicator_spec,
    feature_columns,
) -> ScoreCalibration | None:
    model_feature_cols = getattr(model, "feature_columns", None) or feature_columns
    rel = _build_relative_features(df_stock, df_spy, model_feature_cols, indicator_spec)
    if rel is None or rel.empty:
        return None

    history_features = rel.copy()
    history_features["position_flag"] = 0
    raw_scores = extract_raw_scores_bulk(model, history_features)

    stock_close = df_stock.loc[rel.index, "close"].astype(float)
    spy_close = df_spy.loc[rel.index, "close"].astype(float).replace(0, np.nan)
    relative_price = stock_close / spy_close

    lookahead = int(metadata.get("lookahead", getattr(model, "lookahead", 5)))
    threshold = float(metadata.get("threshold", getattr(model, "threshold", 0.0)))
    future_relative_returns = relative_price.shift(-lookahead) / relative_price - 1.0

    return fit_probability_calibration(
        raw_scores,
        future_relative_returns,
        lookahead=lookahead,
        threshold=threshold,
        score_kind=raw_score_kind_for_model(model),
    )


def _ensure_model_score_calibrations(config: dict[str, Any], models: dict, dfs: dict, df_spy) -> None:
    indicator_spec = config.get("indicator_spec", {})
    feature_columns = config["model_params"]["feature_columns"]

    for symbol, model in models.items():
        if getattr(model, "_score_calibration", None) is not None:
            continue
        if symbol not in dfs:
            continue

        metadata = getattr(model, "_policy_metadata", {})
        calibration = ScoreCalibration.from_dict(metadata.get("score_calibration"))
        if calibration is None:
            calibration = _build_score_calibration(
                model,
                metadata,
                dfs[symbol],
                df_spy,
                indicator_spec,
                feature_columns,
            )
        model._score_calibration = calibration


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
        model._policy_metadata = metadata
        model._score_symbol = symbol
        model._score_calibration = ScoreCalibration.from_dict(metadata.get("score_calibration"))
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

    watchlist      = config["watchlist"]
    benchmark      = config.get("benchmark", "SPY")
    max_positions  = config.get("max_concurrent_positions", 3)
    indicator_spec = config.get("indicator_spec", {})
    feature_columns = config["model_params"]["feature_columns"]
    pos_sizing     = config.get("position_sizing", {})
    max_position_pct  = float(pos_sizing.get("max_position_pct", 0.30))
    cash_reserve_pct  = float(pos_sizing.get("cash_reserve_pct", 0.00))
    sector_map     = config.get("sector_map", {})
    max_per_sector = int(config.get("max_positions_per_sector", 0))
    risk_cfg       = config.get("risk", {})
    drawdown_halt_pct = float(risk_cfg.get("portfolio_drawdown_halt_pct", 0.0))

    # Regime params — use BULL_CALM as default (runner doesn't have live GMM)
    regime_params  = config.get("regime_params", {})
    bull_calm_rp   = regime_params.get("BULL_CALM", {})
    stop_loss_pct  = float(bull_calm_rp.get("stop_loss_pct",
                           risk_cfg.get("stop_loss_pct", 0.0)))
    sdl_pct        = float(bull_calm_rp.get("max_single_day_loss_pct", 0.0))
    spy_vel_halt   = float(bull_calm_rp.get("spy_velocity_halt_pct", 0.03))
    spy_vel_days   = int(bull_calm_rp.get("spy_velocity_lookback_days", 3))

    # Trading constraint params
    min_hold_days        = int(config.get("min_hold_days", 0))
    consec_sells_required = int(config.get("consecutive_sell_signals", 1))
    wash_sale_days       = int(config.get("wash_sale_days", 30))
    tiered_thresholds    = config.get("tiered_thresholds", [])
    base_score_thresh    = float(bull_calm_rp.get("min_model_score",
                                 config.get("min_model_score", 0.0)))
    tiers = [float(t.get("min_model_score", base_score_thresh)) for t in tiered_thresholds] \
            if tiered_thresholds else [base_score_thresh]
    ranking_cfg = config.get("ranking", {})
    _bw = ranking_cfg.get("blend_weights", [0.5, 0.5])
    _bw_total = float(_bw[0]) + float(_bw[1])
    w_rank = float(_bw[0]) / _bw_total if _bw_total > 0 else 0.5
    w_rs   = float(_bw[1]) / _bw_total if _bw_total > 0 else 0.5

    # ── HEADER ────────────────────────────────────────────────────────────────
    run_mode = "sell-only" if sell_only else "full"
    sep = "=" * 62
    log.info(sep)
    log.info("RENQUANT-103  %s  [%s]", datetime.now().strftime("%Y-%m-%d %H:%M PT"), run_mode.upper())
    log.info(sep)

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

    df_spy     = dfs[benchmark]
    _ensure_model_score_calibrations(config, models, dfs, df_spy)
    spy_close  = df_spy["close"].astype(float)
    spy_price  = float(spy_close.iloc[-1])
    spy_ret1d  = (spy_price / float(spy_close.iloc[-2]) - 1) if len(spy_close) >= 2 else 0.0
    spy_ret5d  = (spy_price / float(spy_close.iloc[-6]) - 1) if len(spy_close) >= 6 else 0.0

    # SPY EMA50 gate
    spy_ema50       = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
    spy_above_ema50 = spy_price >= spy_ema50

    # SPY velocity crash filter
    spy_vel_ret = 0.0
    if len(spy_close) > spy_vel_days:
        spy_vel_ret = float(spy_close.iloc[-1] / spy_close.iloc[-1 - spy_vel_days] - 1)
    spy_vel_ok = spy_vel_ret >= -spy_vel_halt

    # ── MARKET CONTEXT ────────────────────────────────────────────────────────
    log.info("MARKET CONTEXT")
    log.info("  SPY  $%.2f  |  1d %+.1f%%  |  5d %+.1f%%", spy_price, spy_ret1d*100, spy_ret5d*100)
    log.info("  EMA50 $%.2f  |  SPY %s EMA50  →  gate %s",
             spy_ema50,
             ">" if spy_above_ema50 else "<",
             "CLEAR" if spy_above_ema50 else "BLOCKING buys")
    log.info("  Velocity (%dd): %+.1f%%  →  crash filter %s",
             spy_vel_days, spy_vel_ret*100,
             "CLEAR" if spy_vel_ok else f"BLOCKING buys (threshold -{spy_vel_halt*100:.0f}%)")

    # ── ACCOUNT + POSITIONS (single batch fetch) ──────────────────────────────
    account_value = broker.get_account_value()
    try:
        cash_avail = broker.get_cash()
    except Exception:
        cash_avail = account_value
    # Fetch all positions in ONE API call — avoids N individual round-trips
    try:
        all_pos = broker.get_all_positions()
    except Exception as e:
        log.warning("  get_all_positions() failed, falling back to empty: %s", e)
        all_pos = []
    positions_cache: dict[str, dict] = {p["symbol"]: p for p in all_pos}
    log.info("ACCOUNT")
    log.info("  Equity $%.2f  |  Cash $%.2f", account_value, cash_avail)

    # Load persisted live state
    state_file = strategy_dir / "live_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    entry_dates: dict     = state.setdefault("entry_dates", {})
    sell_streaks: dict    = state.setdefault("sell_streaks", {})
    last_sell_dates: dict = state.setdefault("last_sell_dates", {})

    # HWM + drawdown
    hwm = float(state.get("high_water_mark", account_value))
    hwm = max(hwm, account_value)
    state["high_water_mark"] = hwm
    drawdown = (hwm - account_value) / hwm if hwm > 0 else 0.0
    circuit_open = drawdown_halt_pct > 0 and drawdown >= drawdown_halt_pct
    log.info("  HWM $%.2f  |  Drawdown %.1f%%  |  Circuit breaker %s",
             hwm, drawdown*100, "OPEN — halting buys" if circuit_open else "CLEAR")

    # ── BROKER RECONCILIATION ─────────────────────────────────────────────────
    from datetime import date as _date
    today_str = _date.today().isoformat()
    sixty_days_ago = (_date.today() - __import__("datetime").timedelta(days=60)).isoformat()
    try:
        filled_orders = broker.get_filled_orders(after=sixty_days_ago)
        last_buy: dict = {}
        last_sell_hist: dict = {}
        for o in filled_orders:
            sym = o["symbol"]
            filled_day = o["filled_at"][:10] if o.get("filled_at") else None
            if not filled_day:
                continue
            if o["action"] == "BUY":
                if sym not in last_buy or filled_day > last_buy[sym]:
                    last_buy[sym] = filled_day
            else:
                if sym not in last_sell_hist or filled_day > last_sell_hist[sym]:
                    last_sell_hist[sym] = filled_day
        for sym, buy_day in last_buy.items():
            if sym not in entry_dates and positions_cache.get(sym, {}).get("qty", 0.0) > 0:
                entry_dates[sym] = buy_day
                log.info("  Reconcile: %s entry_date=%s (from Alpaca history)", sym, buy_day)
        for sym, sell_day in last_sell_hist.items():
            if sell_day == today_str and last_sell_dates.get(sym) != sell_day:
                last_sell_dates[sym] = sell_day
                log.info("  Reconcile: %s last_sell=%s (from Alpaca history)", sym, sell_day)
    except Exception as e:
        log.warning("  Broker reconciliation failed (non-fatal): %s", e)

    # ── SELL PHASE ────────────────────────────────────────────────────────────
    held = [s for s in watchlist if positions_cache.get(s, {}).get("qty", 0.0) > 0]
    log.info("─" * 62)
    log.info("SELL PHASE  (%d held: %s)", len(held), held if held else "none")

    for symbol in list(held):
        if symbol not in dfs:
            continue
        current_price = float(dfs[symbol]["close"].iloc[-1])
        avg_cost      = float(positions_cache.get(symbol, {}).get("avg_entry_price", 0.0))
        entry_date_str = entry_dates.get(symbol)
        days_held = 0
        if entry_date_str:
            try:
                days_held = (_date.today() - _date.fromisoformat(entry_date_str)).days
            except ValueError:
                pass
        unrealized = ((current_price - avg_cost) / avg_cost) if avg_cost > 0 else 0.0

        log.info("  %s  $%.2f  held=%dd  entry=$%.2f  P&L %+.1f%%",
                 symbol, current_price, days_held, avg_cost, unrealized*100)

        # [EXIT 1] Trailing stop — not tracked in runner (no per-position HWM yet)

        # [EXIT 2] Cumulative stop-loss
        if stop_loss_pct > 0 and avg_cost > 0:
            loss_pct = (avg_cost - current_price) / avg_cost
            if loss_pct >= stop_loss_pct:
                position = positions_cache.get(symbol, {}).get("qty", 0.0)
                result = broker.place_order(symbol, "SELL", abs(position))
                log.info("    → SELL [STOP LOSS] loss=%.1f%% >= %.0f%%",
                         loss_pct*100, stop_loss_pct*100)
                _log_trade(strategy_dir, config["model_name"], {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "signal": "stop_loss",
                    "loss_pct": round(loss_pct, 4), "order": result,
                })
                entry_dates.pop(symbol, None)
                sell_streaks.pop(symbol, None)
                last_sell_dates[symbol] = today_str
                held.remove(symbol)
                continue
            else:
                log.info("    stop-loss: loss=%.1f%% < threshold %.0f%%  HOLD",
                         loss_pct*100, stop_loss_pct*100)

        # [EXIT 2b] Single-day loss gate
        if sdl_pct > 0 and len(dfs[symbol]) >= 2:
            prev_close  = float(dfs[symbol]["close"].iloc[-2])
            daily_drop  = (prev_close - current_price) / prev_close if prev_close > 0 else 0.0
            if daily_drop >= sdl_pct:
                position = positions_cache.get(symbol, {}).get("qty", 0.0)
                result = broker.place_order(symbol, "SELL", abs(position))
                log.info("    → SELL [SINGLE DAY LOSS] drop=%.1f%% >= %.0f%%",
                         daily_drop*100, sdl_pct*100)
                _log_trade(strategy_dir, config["model_name"], {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "signal": "single_day_loss",
                    "daily_drop_pct": round(daily_drop, 4), "order": result,
                })
                entry_dates.pop(symbol, None)
                sell_streaks.pop(symbol, None)
                last_sell_dates[symbol] = today_str
                held.remove(symbol)
                continue
            else:
                log.info("    single-day gate: drop=%.1f%% < threshold %.0f%%  HOLD",
                         daily_drop*100, sdl_pct*100)

        # [EXIT 4] Model sell — gated by min_hold + consecutive streak
        if min_hold_days > 0 and days_held < min_hold_days:
            log.info("    model-sell: min_hold=%dd, held=%dd  BLOCKED (in hold window)",
                     min_hold_days, days_held)
            sell_streaks[symbol] = 0
            continue

        model_feature_cols = getattr(models[symbol], "feature_columns", None) or feature_columns
        rel = _build_relative_features(dfs[symbol], df_spy, model_feature_cols, indicator_spec)
        if rel is None or rel.empty:
            continue
        row = rel.iloc[-1].copy()
        row["position_flag"] = 1
        score_eval = evaluate_row(models[symbol], row, getattr(models[symbol], "_score_calibration", None))
        signal = score_eval.signal
        raw_score = score_eval.raw_score
        rank_score = score_eval.rank_score

        if signal == "sell":
            sell_streaks[symbol] = sell_streaks.get(symbol, 0) + 1
            streak = sell_streaks[symbol]
            if streak < consec_sells_required:
                log.info("    model-sell: signal=sell  raw=%.4f  rank=%.4f  streak=%d/%d  WAITING",
                         raw_score, rank_score, streak, consec_sells_required)
            else:
                sell_streaks[symbol] = 0
                position = positions_cache.get(symbol, {}).get("qty", 0.0)
                result = broker.place_order(symbol, "SELL", abs(position))
                log.info("    → SELL [MODEL streak=%d] raw=%.4f rank=%.4f  held=%dd",
                         streak, raw_score, rank_score, days_held)
                _log_trade(strategy_dir, config["model_name"], {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "signal": "sell",
                    "days_held": days_held, "sell_streak": streak,
                    "raw_model_score": raw_score,
                    "rank_model_score": rank_score,
                    "order": result,
                })
                entry_dates.pop(symbol, None)
                last_sell_dates[symbol] = today_str
                held.remove(symbol)
        else:
            sell_streaks[symbol] = 0
            log.info("    model-sell: signal=%s  raw=%.4f rank=%.4f  HOLD", signal, raw_score, rank_score)

    # Persist state after sell phase and return if sell-only
    state_file.write_text(json.dumps(state, indent=2))
    if sell_only:
        log.info("sell_only mode — buy phase skipped")
        log.info(sep)
        return

    # ── BUY GATES ─────────────────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("BUY PHASE  (slots %d/%d open)", max_positions - len(held), max_positions)

    open_slots = max_positions - len(held)
    if open_slots <= 0:
        log.info("  All %d slots filled — no scanning", max_positions)
        log.info(sep)
        return

    if circuit_open:
        log.info("  Drawdown circuit breaker OPEN (%.1f%%) — halting buys", drawdown*100)
        log.info(sep)
        return

    if not spy_above_ema50:
        log.info("  SPY EMA50 gate BLOCKING (SPY $%.2f < EMA50 $%.2f) — no buys",
                 spy_price, spy_ema50)
        log.info(sep)
        return

    if not spy_vel_ok:
        log.info("  SPY velocity crash BLOCKING (%+.1f%% over %dd) — no buys",
                 spy_vel_ret*100, spy_vel_days)
        log.info(sep)
        return

    # ── FULL TICKER SCAN ──────────────────────────────────────────────────────
    log.info("  SPY gates clear — scanning all %d tickers", len(watchlist))
    log.info("─" * 62)
    log.info("FULL TICKER SCAN")

    buy_candidates = []   # (symbol, raw_score, rank_score, rs_score, row)

    for symbol in watchlist:
        if symbol in held:
            log.info("  %-6s  SKIP  [already held]", symbol)
            continue
        if symbol not in dfs or symbol not in models:
            log.info("  %-6s  SKIP  [no data/model]", symbol)
            continue

        price   = float(dfs[symbol]["close"].iloc[-1])
        ret1d   = float(dfs[symbol]["close"].iloc[-1] / dfs[symbol]["close"].iloc[-2] - 1) \
                  if len(dfs[symbol]) >= 2 else 0.0
        ret5d   = float(dfs[symbol]["close"].iloc[-1] / dfs[symbol]["close"].iloc[-6] - 1) \
                  if len(dfs[symbol]) >= 6 else 0.0
        spy5d_r = float(spy_close.iloc[-1] / spy_close.iloc[-6] - 1) \
                  if len(spy_close) >= 6 else 0.0
        rs5d    = ret5d - spy5d_r

        # Wash-sale guard
        if wash_sale_days > 0 and symbol in last_sell_dates:
            try:
                days_since_sell = (_date.today() - _date.fromisoformat(last_sell_dates[symbol])).days
                if days_since_sell < wash_sale_days:
                    log.info("  %-6s  $%.2f  1d%+.1f%%  5d%+.1f%%  RS%+.1f%%  SKIP  "
                             "[wash-sale: sold %dd ago, limit %dd]",
                             symbol, price, ret1d*100, ret5d*100, rs5d*100,
                             days_since_sell, wash_sale_days)
                    continue
            except ValueError:
                pass

        # Build features and run model
        model_feature_cols = getattr(models[symbol], "feature_columns", None) or feature_columns
        rel = _build_relative_features(dfs[symbol], df_spy, model_feature_cols, indicator_spec)
        if rel is None or rel.empty:
            log.info("  %-6s  $%.2f  SKIP  [feature build failed]", symbol, price)
            continue
        row = rel.iloc[-1].copy()
        row["position_flag"] = 0
        score_eval = evaluate_row(models[symbol], row, getattr(models[symbol], "_score_calibration", None))
        signal = score_eval.signal
        raw_score = score_eval.raw_score
        rank_score = score_eval.rank_score

        # Min model score (use tier-1 bar as baseline)
        min_score = tiers[0]
        if signal != "buy":
            log.info("  %-6s  $%.2f  1d%+.1f%%  RS%+.1f%%  signal=%-4s  raw=%+.4f  rank=%+.4f  SKIP  [signal=%s]",
                     symbol, price, ret1d*100, rs5d*100, signal, raw_score, rank_score, signal)
            continue
        if rank_score < min_score:
            log.info("  %-6s  $%.2f  1d%+.1f%%  RS%+.1f%%  signal=buy  raw=%+.4f  rank=%+.4f  SKIP  "
                     "[score < min %.2f]",
                     symbol, price, ret1d*100, rs5d*100, raw_score, rank_score, min_score)
            continue

        # Relative strength vs sector ETF (20d)
        sector_etfs = {"tech": "XLK", "finance": "XLF", "healthcare": "XLV",
                       "energy": "XLE", "industrial": "XLI", "utility": "XLU"}
        sector  = sector_map.get(symbol, "other")
        etf     = sector_etfs.get(sector)
        rs_score = 0.0
        if etf and etf in dfs and len(dfs[etf]) >= 21:
            try:
                stock_r = float(dfs[symbol]["close"].iloc[-1] / dfs[symbol]["close"].iloc[-21] - 1)
                etf_r   = float(dfs[etf]["close"].iloc[-1]   / dfs[etf]["close"].iloc[-21]   - 1)
                rs_score = stock_r - etf_r
            except Exception:
                pass

        log.info("  %-6s  $%.2f  1d%+.1f%%  RS%+.1f%%  signal=BUY  raw=%+.4f  rank=%+.4f  rs=%+.4f  "
                 "→ CANDIDATE",
             symbol, price, ret1d*100, rs5d*100, raw_score, rank_score, rs_score)
        buy_candidates.append((symbol, raw_score, rank_score, rs_score, row))

    # ── RANKING ───────────────────────────────────────────────────────────────
    log.info("─" * 62)
    if not buy_candidates:
        log.info("BUY CANDIDATES: none — no buys placed")
        log.info(sep)
        return

    import pandas as _pd
    if len(buy_candidates) > 1:
        rank_vals = [c[2] for c in buy_candidates]
        rs_vals = [c[3] for c in buy_candidates]
        rank_lo, rank_hi = min(rank_vals), max(rank_vals)
        rs_lo, rs_hi = min(rs_vals), max(rs_vals)
        def _norm(v, lo, hi): return (v - lo) / (hi - lo) if hi > lo else 0.5
        buy_candidates.sort(
            key=lambda c: w_rank * _norm(c[2], rank_lo, rank_hi) + w_rs * _norm(c[3], rs_lo, rs_hi),
            reverse=True,
        )
    log.info("RANKED CANDIDATES (%.0f%% calibrated rank + %.0f%% RS):", w_rank * 100, w_rs * 100)
    for rank, (sym, raw_ms, rank_ms, rs, _) in enumerate(buy_candidates, 1):
        log.info("  #%d  %-6s  raw=%+.4f  rank=%+.4f  rs=%+.4f", rank, sym, raw_ms, rank_ms, rs)

    # ── SELECTION LOOP ────────────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("SELECTION (tiered thresholds: %s)", tiers)
    slots_filled = 0
    for symbol, raw_score, rank_score, rs_score, row in buy_candidates:
        if open_slots <= 0:
            break

        tier_idx  = min(slots_filled, len(tiers) - 1)
        tier_min  = tiers[tier_idx]
        slot_num  = slots_filled + 1

        if rank_score < tier_min:
            log.info("  %-6s  SKIP  [slot %d tier min %.2f, rank %+.4f too low]",
                     symbol, slot_num, tier_min, rank_score)
            continue

        # Sector guard
        if max_per_sector > 0:
            sector       = sector_map.get(symbol, "other")
            sector_count = sum(1 for h in held if sector_map.get(h, "other") == sector)
            if sector_count >= max_per_sector:
                log.info("  %-6s  SKIP  [sector %s at %d/%d]",
                         symbol, sector, sector_count, max_per_sector)
                continue

        # Position sizing
        account_value = broker.get_account_value()
        price         = float(dfs[symbol]["close"].iloc[-1])
        cash_reserve  = account_value * cash_reserve_pct
        available     = max(cash_avail - cash_reserve, 0)
        max_invest    = account_value * max_position_pct
        invest        = min(max_invest, available)
        shares        = int(invest / price)

        if shares <= 0:
            log.info("  %-6s  SKIP  [insufficient cash: invest=$%.0f, price=$%.2f]",
                     symbol, invest, price)
            continue

        result = broker.place_order(symbol, "BUY", shares)
        log.info("  %-6s  BUY  slot=%d  %d shares @ $%.2f  invest=$%.0f  "
                 "raw=%+.4f  rank=%+.4f  rs=%+.4f",
                 symbol, slot_num, shares, price, invest, raw_score, rank_score, rs_score)
        _log_trade(strategy_dir, config["model_name"], {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol, "signal": "buy",
            "slot": slot_num,
            "raw_model_score": raw_score,
            "rank_model_score": rank_score,
            "rs_score": rs_score,
            "shares": shares, "price": price, "invest": invest, "order": result,
        })
        entry_dates[symbol] = today_str
        sell_streaks.pop(symbol, None)
        last_sell_dates.pop(symbol, None)
        held.append(symbol)
        slots_filled += 1
        open_slots   -= 1

    if slots_filled == 0:
        log.info("  No buys placed (all candidates filtered out)")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("END OF RUN  |  positions held: %d/%d  |  %s",
             len(held), max_positions, held if held else "none")
    log.info(sep)

    # Persist updated state
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
