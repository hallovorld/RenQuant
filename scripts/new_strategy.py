#!/usr/bin/env python
"""Scaffold a new strategy: creates config + backtesting dir.

Usage::

    python scripts/new_strategy.py --name nvda_rf --symbol NVDA --type classification
    python scripts/new_strategy.py --name aapl_fqi --symbol AAPL --type fqi
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new RenQuant strategy")
    parser.add_argument("--name", required=True, help="Strategy name (e.g. nvda_rf)")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. NVDA)")
    parser.add_argument(
        "--type",
        choices=["manual", "classification", "qlearning", "fqi", "optimization"],
        default="fqi",
        help="Model type (default: fqi)",
    )
    parser.add_argument("--start", default="2022-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2023-01-01", help="Backtest end date")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    args = parser.parse_args()

    # Create backtesting directory
    bt_dir = REPO_ROOT / "backtesting" / args.name
    if bt_dir.exists():
        print(f"Error: {bt_dir} already exists", file=sys.stderr)
        sys.exit(1)
    bt_dir.mkdir(parents=True)

    # Default indicator spec
    indicator_spec = {
        "rsi": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "cci": {"period": 20},
    }

    # Model-specific params
    model_params = {}
    if args.type == "fqi":
        model_params = {"n_iter": 8, "gamma": 0.95, "transaction_cost_bps": 5}
    elif args.type == "classification":
        model_params = {"lookahead": 10, "threshold": 0.04}
    elif args.type == "qlearning":
        model_params = {"n_epochs": 100, "n_bins": 10}
    elif args.type == "optimization":
        model_params = {"max_iter": 30}

    config = {
        "model_name": args.name,
        "stock_symbol": args.symbol,
        "model_type": args.type,
        "data_src": "yfinance",
        "initial_cash": args.cash,
        "backtest_start": args.start,
        "backtest_end": args.end,
        "indicator_spec": indicator_spec,
        "model_params": model_params,
    }

    config_path = bt_dir / "strategy_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Create LEAN config.json
    lean_config = {
        "algorithm-type-name": "main",
        "algorithm-language": "Python",
        "parameters": {},
    }
    (bt_dir / "config.json").write_text(json.dumps(lean_config, indent=2))

    print(f"Strategy scaffolded at: {bt_dir}")
    print(f"  Config: {config_path}")
    print(f"  Model type: {args.type}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Period: {args.start} -> {args.end}")
    print()
    print("Next steps:")
    print(f"  1. Open a research notebook and train the model")
    print(f"  2. Export artifacts to {bt_dir}")
    print(f"  3. cd {bt_dir} && lean backtest .")


if __name__ == "__main__":
    main()
