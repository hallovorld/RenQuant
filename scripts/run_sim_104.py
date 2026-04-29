#!/usr/bin/env python
"""Run a 27-month OOS sim for renquant_104 with a named strategy config.

Usage::

    python scripts/run_sim_104.py
    python scripts/run_sim_104.py --strategy-config-name strategy_config.h60_103.json
    python scripts/run_sim_104.py --start 2024-01-01 --end 2026-03-28

Outputs APY, Sharpe, MaxDD, n_trades, and compares to the golden config.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("run-sim-104")

STRATEGY   = "renquant_104"
SIM_START  = "2024-01-02"
SIM_END    = "2026-03-28"   # ~27 months


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy-config-name", default="strategy_config.json",
                   help="Config filename (default: strategy_config.json)")
    p.add_argument("--start", default=SIM_START)
    p.add_argument("--end",   default=SIM_END)
    p.add_argument("--compare-to", default="strategy_config.golden.json",
                   help="Golden config to compare against (default: strategy_config.golden.json)")
    p.add_argument("--initial-cash", type=float, default=100_000)
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / STRATEGY
    sys.path.insert(0, str(strategy_dir))

    cfg_path = strategy_dir / args.strategy_config_name
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    config = json.loads(cfg_path.read_text())
    config["_strategy_dir"]         = str(strategy_dir)
    config["_strategy_config_name"] = args.strategy_config_name
    config["initial_cash"]          = args.initial_cash
    config["backtest_start"]        = args.start
    config["backtest_end"]          = args.end

    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from sim.runner import run_backtest   # noqa: PLC0415

    # Load benchmark + sector ETFs
    log.info("Fetching SPY + sector ETFs …")
    benchmark = config.get("benchmark", "SPY")
    spy_df    = fetch_ohlcv(benchmark)
    etf_map   = config.get("sector_etf_map", {})
    ohlcv: dict = {benchmark: spy_df}
    for sym in sorted(set(config.get("watchlist", [])) | set(etf_map.values())):
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception as exc:
            log.warning("  %s: %s", sym, exc)

    log.info("Running sim: %s → %s  config=%s",
             args.start, args.end, args.strategy_config_name)
    result = run_backtest(
        config        = config,
        strategy_dir  = strategy_dir,
        ohlcv         = ohlcv,
        spy_df        = spy_df,
        sector_etf_map = etf_map,
        snapshot      = False,
    )
    result.print_summary()

    # Compare to golden if available
    golden_path = strategy_dir / args.compare_to
    if golden_path.exists() and args.compare_to != args.strategy_config_name:
        log.info("Running golden comparison: %s", args.compare_to)
        golden_cfg = json.loads(golden_path.read_text())
        golden_cfg["_strategy_dir"]  = str(strategy_dir)
        golden_cfg["initial_cash"]   = args.initial_cash
        golden_cfg["backtest_start"] = args.start
        golden_cfg["backtest_end"]   = args.end
        golden = run_backtest(
            config        = golden_cfg,
            strategy_dir  = strategy_dir,
            ohlcv         = ohlcv,
            spy_df        = spy_df,
            sector_etf_map = etf_map,
            snapshot      = False,
        )
        r_apy = result.apy * 100
        g_apy = golden.apy  * 100
        delta = r_apy - g_apy
        print()
        print("=" * 50)
        print(f"  {args.strategy_config_name:<35} APY={r_apy:+.2f}%  WR={result.win_rate:.0%}  trades={len(result.buys)}")
        print(f"  {args.compare_to:<35} APY={g_apy:+.2f}%  WR={golden.win_rate:.0%}  trades={len(golden.buys)}")
        print(f"  Delta vs golden                         APY={delta:+.2f} pp")
        verdict = "PROMOTE ✓" if delta >= 0 else "REJECT ✗"
        print(f"  Verdict: {verdict}")
        print("=" * 50)


if __name__ == "__main__":
    main()
