#!/usr/bin/env python
"""Dump full walkforward-eval sim metrics (including geometric Sharpe) to JSON.

Used by Track P3.2 (2026-05-10) to produce the honest 27-mo OOS report.
Re-runs run_backtest with strategy_config.sim_baseline.json, captures
the SimResult, recomputes geometric Sharpe from equity_df (not stored on
SimResult), and writes a single JSON to data/logs/.

Usage::
    python scripts/dump_walkforward_sim_metrics.py
    python scripts/dump_walkforward_sim_metrics.py --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dump-wf-sim")


def _to_float(x):
    """Convert numpy / pandas scalars to plain float for JSON."""
    if x is None:
        return None
    try:
        f = float(x)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy-config-name",
                   default="strategy_config.sim_baseline.json")
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--end",   default="2026-03-28")
    p.add_argument("--initial-cash", type=float, default=100_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: data/logs/walkforward_metrics_<ts>.json)")
    args = p.parse_args()

    strategy_dir = REPO / "backtesting" / "renquant_104"
    sys.path.insert(0, str(strategy_dir))

    cfg_path = strategy_dir / args.strategy_config_name
    config = json.loads(cfg_path.read_text())
    config["_strategy_dir"] = str(strategy_dir)
    config["_strategy_config_name"] = args.strategy_config_name
    config["initial_cash"] = args.initial_cash
    config["backtest_start"] = args.start
    config["backtest_end"]   = args.end
    config["persistence"] = {"enabled": False}

    from kernel.data import fetch_ohlcv          # noqa: PLC0415
    from sim.runner import run_backtest          # noqa: PLC0415
    from kernel.risk_metrics import (            # noqa: PLC0415
        compute_risk_metrics,
        daily_returns_from_equity,
        geometric_sharpe_ratio,
    )

    benchmark = config.get("benchmark", "SPY")
    log.info("Fetching SPY + sector ETFs …")
    spy_df = fetch_ohlcv(benchmark)
    etf_map = config.get("sector_etf_map", {})
    ohlcv = {benchmark: spy_df}
    for sym in sorted(set(config.get("watchlist", [])) | set(etf_map.values())):
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception as exc:
            log.warning("  %s: %s", sym, exc)

    log.info("Running walkforward-eval sim …")
    result = run_backtest(
        config        = config,
        strategy_dir  = strategy_dir,
        ohlcv         = ohlcv,
        spy_df        = spy_df,
        sector_etf_map = etf_map,
        initial_cash  = args.initial_cash,
        backtest_start = args.start,
        backtest_end   = args.end,
        snapshot      = False,
        seed          = args.seed,
    )
    result.print_summary()

    # Recompute geometric Sharpe from equity_df (SimResult doesn't carry it).
    sharpe_geom = float("nan")
    rf_annual = float(
        config.get("performance", {}).get("risk_free_rate_annual", 0.0)
    )
    if (not result.equity_df.empty
            and "portfolio" in result.equity_df.columns):
        rets = daily_returns_from_equity(result.equity_df["portfolio"]).dropna()
        if len(rets) >= 2:
            risk = compute_risk_metrics(
                result.equity_df["portfolio"],
                apy=result.apy,
                risk_free_rate=rf_annual,
                include_geometric=True,
            )
            sharpe_geom = float(risk.get("sharpe_geometric", float("nan")))

    out_data = {
        "run_timestamp":    datetime.now().isoformat(),
        "strategy_config":  args.strategy_config_name,
        "backtest_start":   args.start,
        "backtest_end":     args.end,
        "initial_cash":     args.initial_cash,
        "seed":             args.seed,
        "risk_free_rate_annual": rf_annual,
        # Returns
        "final_value":      _to_float(result.final_value),
        "total_return":     _to_float(result.total_return),
        "apy":              _to_float(result.apy),
        # Risk-adjusted
        "sharpe_arithmetic": _to_float(result.sharpe),
        "sharpe_geometric":  _to_float(sharpe_geom),
        "sortino":           _to_float(result.sortino),
        "calmar":            _to_float(result.calmar),
        "max_dd":            _to_float(result.max_dd),
        "ann_vol":           _to_float(result.ann_vol),
        # Falsifiability triple
        "dsr":               _to_float(result.dsr),
        "pbo":               _to_float(result.pbo),
        "n_trials":          int(result.n_trials),
        # vs SPY
        "beta_vs_spy":               _to_float(result.beta_vs_spy),
        "alpha_vs_spy":              _to_float(result.alpha_vs_spy),
        "information_ratio_vs_spy":  _to_float(result.information_ratio_vs_spy),
        # Activity
        "n_buys":               len(result.buys),
        "n_sells":              len(result.sells),
        "win_rate":             _to_float(result.win_rate),
        "avg_hold_days":        _to_float(result.avg_hold),
        "avg_pnl_pct":          _to_float(result.avg_pnl),
        "total_tax_usd":        _to_float(result.total_tax),
        "exit_reasons":         dict(result.exit_reasons),
        "longest_no_trade_streak": int(result.longest_no_trade_streak),
        "first_trade_date":     result.first_trade_date,
        "last_activity_date":   result.last_activity_date,
        # Sample size for falsifiability assessment
        "n_bars":               int(len(result.equity_df)),
    }

    out_path = Path(args.out) if args.out else (
        REPO / "data" / "logs"
        / f"walkforward_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    log.info("metrics JSON → %s", out_path)
    print(f"\nMetrics written to: {out_path}")


if __name__ == "__main__":
    main()
