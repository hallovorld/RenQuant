"""Live trading runner.

Usage::

    python -m live.runner --strategy renquant_101 --broker paper --once
    python -m live.runner --strategy renquant_101 --broker ibkr
    python -m live.runner --strategy renquant_102 --broker paper --once  # multi-stock
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


def _load_strategy_multi(strategy_name: str) -> tuple[dict[str, Any], dict, Path]:
    """Load multi-stock strategy config and per-stock models."""
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_path = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config = json.loads(config_path.read_text())

    models = {}
    for symbol in config["watchlist"]:
        model = create_model(config["model_type"], **config["model_params"])
        artifact_name = f"{config['model_name']}-{symbol}"
        model.load(strategy_dir, artifact_name)
        models[symbol] = model

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

    result = pd.DataFrame(index=common_idx)
    result["close"] = df_stock["close"]
    for col in feature_columns:
        if col in ratio_features:
            result[col] = df_stock[col] / df_spy[col].replace(0, np.nan)
        elif col in diff_features:
            result[col] = df_stock[col] - df_spy[col]
        else:
            result[col] = df_stock[col]

    return result.dropna()


def run_once_multi(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
) -> None:
    """Execute one multi-stock trading cycle: scan → analyze → trade."""
    import numpy as np

    watchlist = config["watchlist"]
    benchmark = config.get("benchmark", "SPY")
    max_positions = config.get("max_concurrent_positions", 3)
    zscore_threshold = float(config.get("volume_zscore_threshold", 2.0))
    zscore_lookback = int(config.get("volume_zscore_lookback", 15))
    indicator_spec = config.get("indicator_spec", {})
    feature_columns = config["model_params"]["feature_columns"]
    pos_sizing = config.get("position_sizing", {})
    max_position_pct = float(pos_sizing.get("max_position_pct", 0.33))
    cash_reserve_pct = float(pos_sizing.get("cash_reserve_pct", 0.10))

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

    # Step 1: Check current positions, process sells first
    held = [s for s in watchlist if broker.get_position(s) > 0]
    log.info("Current positions: %s", held if held else "none")

    for symbol in list(held):
        if symbol not in dfs:
            continue
        rel = _build_relative_features(dfs[symbol], df_spy, feature_columns, indicator_spec)
        if rel is None or rel.empty:
            continue
        signal = models[symbol].predict(rel.iloc[-1])
        if signal == "sell":
            position = broker.get_position(symbol)
            result = broker.place_order(symbol, "SELL", abs(position))
            log.info("SELL %s: %.0f shares", symbol, abs(position))
            _log_trade(strategy_dir, config["model_name"], {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol, "signal": "sell", "order": result,
            })
            held.remove(symbol)

    # Step 2: Scan volume for non-held stocks
    open_slots = max_positions - len(held)
    if open_slots <= 0:
        log.info("All %d slots filled, no scanning", max_positions)
        return

    candidates = []
    for symbol in watchlist:
        if symbol in held or symbol not in dfs:
            continue
        df = dfs[symbol]
        if len(df) < zscore_lookback + 1:
            continue
        vol = df["volume"].astype(float)
        roll_mean = vol.rolling(zscore_lookback).mean().iloc[-1]
        roll_std = vol.rolling(zscore_lookback).std().iloc[-1]
        today_vol = float(vol.iloc[-1])
        zscore = (today_vol - roll_mean) / roll_std if roll_std > 0 else 0
        if zscore >= zscore_threshold:
            candidates.append((symbol, zscore))

    candidates.sort(key=lambda x: x[1], reverse=True)
    log.info("Volume z-score candidates: %s", [(s, f"z={v:.2f}") for s, v in candidates] or "none")

    # Step 3: Run models on top candidates
    for symbol, zscore in candidates[:open_slots]:
        rel = _build_relative_features(dfs[symbol], df_spy, feature_columns, indicator_spec)
        if rel is None or rel.empty:
            continue

        signal = models[symbol].predict(rel.iloc[-1])
        log.info("%s zscore=%.2f signal=%s", symbol, zscore, signal)

        if signal == "buy":
            account_value = broker.get_account_value()
            price = float(dfs[symbol]["close"].iloc[-1])
            cash_reserve = account_value * cash_reserve_pct
            available = broker.get_account_value() - cash_reserve
            max_invest = account_value * max_position_pct
            invest = min(max_invest, max(available, 0))
            shares = int(invest / price)
            if shares > 0:
                result = broker.place_order(symbol, "BUY", shares)
                log.info("BUY %s: %d shares at ~$%.2f (zscore=%.2f)",
                         symbol, shares, price, zscore)
                _log_trade(strategy_dir, config["model_name"], {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "signal": "buy",
                    "volume_zscore": zscore, "order": result,
                })
                open_slots -= 1
                if open_slots <= 0:
                    break


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
    parser.add_argument("--broker", choices=["paper", "ibkr"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400)")
    args = parser.parse_args()

    multi = _is_multi_stock(args.strategy)

    if multi:
        config, models, strategy_dir = _load_strategy_multi(args.strategy)
        initial_cash = config.get("initial_cash", 100_000)
        broker = _get_broker(args.broker, initial_cash=initial_cash)
        broker.connect()
        run_fn = lambda: run_once_multi(config, models, broker, strategy_dir)
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
