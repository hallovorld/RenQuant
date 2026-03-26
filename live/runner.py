"""Live trading runner.

Usage::

    python -m live.runner --strategy renquant_101 --broker paper --once
    python -m live.runner --strategy renquant_101 --broker ibkr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="RenQuant live trading runner")
    parser.add_argument("--strategy", required=True, help="Strategy directory name")
    parser.add_argument("--broker", choices=["paper", "ibkr"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400)")
    args = parser.parse_args()

    config, model, strategy_dir = _load_strategy(args.strategy)
    broker = _get_broker(args.broker, initial_cash=config.initial_cash)
    broker.connect()

    try:
        if args.once:
            run_once(config, model, broker, strategy_dir)
        else:
            log.info("Starting scheduled mode (interval=%ds)", args.interval)
            while True:
                try:
                    run_once(config, model, broker, strategy_dir)
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
