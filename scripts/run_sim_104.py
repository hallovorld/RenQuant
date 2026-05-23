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
    # 2026-05-09 audit FIX-G: per-seed isolation for parallel multi-seed runs.
    # Without these, multiple sims clobber data/sim_runs.db (single SQLite
    # writer) → race + TRUNCATE conflicts.
    p.add_argument("--sim-db-path", default=None,
                   help="Override persistence.sim_db_path so parallel "
                        "multi-seed runs use isolated DBs.")
    p.add_argument("--no-persist", action="store_true",
                   help="Disable persistence entirely (fastest; no DB writes).")
    p.add_argument("--equity-json", default=None,
                   help="Write daily equity curve to JSON (for paired-returns analysis)")
    p.add_argument("--trade-log-json", default=None,
                   help="Write raw SimResult.trade_log events to JSON.")
    p.add_argument("--trade-log-csv", default=None,
                   help="Write raw SimResult.trade_log events to CSV.")
    p.add_argument("--round-trips-csv", default=None,
                   help="Write FIFO-matched round trips to CSV.")
    p.add_argument("--trade-report-md", default=None,
                   help="Write a Markdown trade-forensics report.")
    p.add_argument("--no-compare", action="store_true",
                   help="Skip the golden-config comparison run.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the static-path preflight on side configs. "
                        "ONLY use when intentionally running a no-op (e.g. "
                        "re-baselining against new artifacts).")
    p.add_argument("--allow-raw-qp-mu", action="store_true",
                   help="Emergency/debug override: allow QP configs that do "
                        "not have a strict expected-return μ contract.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / STRATEGY
    sys.path.insert(0, str(strategy_dir))

    cfg_path = strategy_dir / args.strategy_config_name
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    config = json.loads(cfg_path.read_text())
    from qp_contracts import validate_qp_contract_config  # noqa: PLC0415
    qp_contract = validate_qp_contract_config(config)
    if not qp_contract.passed and not args.allow_raw_qp_mu:
        log.error(qp_contract.summary())
        log.error("QP contract evidence: %s", qp_contract.evidence)
        sys.exit(3)
    if qp_contract.qp_enabled:
        log.info("QP contract: %s  evidence=%s",
                 qp_contract.summary(), qp_contract.evidence)
    # Historical sims/WF cuts are not live inference. The live freshness
    # guard correctly requires every symbol to include the latest completed
    # NYSE close, but old windows can contain IPO/new-listing gaps and would
    # be falsely rejected. Live runner keeps the default enabled.
    data_freshness = config.setdefault("data_freshness", {})
    if "enabled" not in data_freshness:
        data_freshness["enabled"] = False
        log.info("data_freshness.enabled=false by default for historical sim")

    # 2026-05-16: gate on static-path preflight for any side config to
    # prevent the recurrence of the 5/15 no-op build script bug (5h of
    # compute on configs whose knobs didn't reach the kernel). See
    # scripts/validate_sim_config_active.py and the 2026-05-16 entry
    # in doc/research/failed-experiments-log.md.
    SIDE_CFG_BASELINE = "strategy_config.sim_baseline_hmm.json"
    is_side = (args.strategy_config_name.startswith("strategy_config.sim_")
               and not args.strategy_config_name.startswith("strategy_config.sim_baseline"))
    if is_side and not args.skip_preflight:
        import subprocess
        validator = REPO_ROOT / "scripts" / "validate_sim_config_active.py"
        if validator.exists():
            log.info("preflight: static-path validator vs %s", SIDE_CFG_BASELINE)
            r = subprocess.run(
                [sys.executable, str(validator),
                 "--baseline", SIDE_CFG_BASELINE,
                 "--candidate", args.strategy_config_name],
                cwd=str(strategy_dir), capture_output=True, text=True,
            )
            if r.returncode != 0:
                log.error("PREFLIGHT FAILED for %s — config writes to a path "
                          "the kernel does not read (NO-OP). Aborting to "
                          "prevent wasted compute. Pass --skip-preflight to "
                          "override.", args.strategy_config_name)
                log.error("validator output:\n%s", r.stdout)
                sys.exit(2)
            log.info("preflight: ACTIVE — knob reaches kernel")
        else:
            log.warning("preflight skipped — validator not found at %s", validator)

    config["_strategy_dir"]         = str(strategy_dir)
    config["_strategy_config_name"] = args.strategy_config_name
    config["initial_cash"]          = args.initial_cash
    config["backtest_start"]        = args.start
    config["backtest_end"]          = args.end

    # Per-seed DB isolation
    if args.no_persist:
        config["persistence"] = {"enabled": False}
    elif args.sim_db_path:
        config.setdefault("persistence", {})["sim_db_path"] = args.sim_db_path

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

    # Emit daily equity curve for paired-returns analysis (industry-standard
    # eval per doc/research/evaluation-protocol.md). Records date + nav so
    # downstream paired t-test + Newey-West HAC + block-bootstrap have the
    # raw daily P&L stream rather than the noisy per-window APY estimate.
    if args.equity_json:
        from pathlib import Path as _P
        eq = result.equity_df.copy()
        eq.index = eq.index.astype(str)
        payload = {
            "config":        args.strategy_config_name,
            "start":         args.start,
            "end":           args.end,
            "initial_cash":  args.initial_cash,
            "final_value":   float(result.final_value),
            "total_return":  float(result.total_return),
            "apy":           float(result.apy),
            "sharpe":        float(result.sharpe) if result.sharpe == result.sharpe else None,
            "event_level_apy": float(result.apy),
            "event_level_sharpe": (
                float(result.sharpe) if result.sharpe == result.sharpe else None
            ),
            "event_level_tax_debited": float(result.event_level_tax_debited),
            "event_level_tax_estimate": float(result.event_level_tax_estimate),
            "tax_cash_debited": float(result.tax_cash_debited),
            "tax_cash_debit_mode": str(result.tax_cash_debit_mode),
            "annual_net_tax_estimate": float(result.annual_net_tax_estimate),
            "tax_overstatement_vs_annual_net": (
                float(result.tax_overstatement_vs_annual_net)
            ),
            "annual_net_final_value": (
                float(result.annual_net_final_value_estimate)
                if result.annual_net_final_value_estimate
                == result.annual_net_final_value_estimate else None
            ),
            "annual_net_total_return": (
                float(result.annual_net_total_return_estimate)
                if result.annual_net_total_return_estimate
                == result.annual_net_total_return_estimate else None
            ),
            "annual_net_apy": (
                float(result.annual_net_apy_estimate)
                if result.annual_net_apy_estimate
                == result.annual_net_apy_estimate else None
            ),
            "annual_net_sharpe": (
                float(result.annual_net_sharpe_estimate)
                if result.annual_net_sharpe_estimate
                == result.annual_net_sharpe_estimate else None
            ),
            "annual_net_ann_vol": (
                float(result.annual_net_ann_vol_estimate)
                if result.annual_net_ann_vol_estimate
                == result.annual_net_ann_vol_estimate else None
            ),
            "annual_net_max_dd": (
                float(result.annual_net_max_dd_estimate)
                if result.annual_net_max_dd_estimate
                == result.annual_net_max_dd_estimate else None
            ),
            "ann_vol":       float(result.ann_vol) if result.ann_vol == result.ann_vol else None,
            "max_dd":        float(result.max_dd) if result.max_dd == result.max_dd else None,
            "equity":        eq["portfolio"].astype(float).to_dict(),
        }
        annual_eq = result.annual_net_equity_df_estimate.copy()
        if (not annual_eq.empty and "portfolio" in annual_eq.columns):
            annual_eq.index = annual_eq.index.astype(str)
            payload["annual_net_equity"] = (
                annual_eq["portfolio"].astype(float).to_dict()
            )
        _P(args.equity_json).parent.mkdir(parents=True, exist_ok=True)
        _P(args.equity_json).write_text(json.dumps(payload, indent=2))
        log.info("Wrote daily equity → %s (%d days)", args.equity_json, len(eq))

    if any([args.trade_log_json, args.trade_log_csv,
            args.round_trips_csv, args.trade_report_md]):
        from sim_trade_ledger import write_trade_outputs  # noqa: PLC0415
        end_prices = {}
        for sym, df in ohlcv.items():
            try:
                hist = df.loc[:args.end]
                if not hist.empty and "close" in hist.columns:
                    end_prices[sym] = float(hist["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        written = write_trade_outputs(
            result           = result,
            config           = config,
            trade_json       = args.trade_log_json,
            trade_csv        = args.trade_log_csv,
            round_trips_csv  = args.round_trips_csv,
            report_md        = args.trade_report_md,
            end_prices       = end_prices,
            title            = (
                f"renquant_104 sim trade forensics "
                f"({args.strategy_config_name}, {args.start} to {args.end})"
            ),
            extra_metrics    = {
                "config": args.strategy_config_name,
                "start": args.start,
                "end": args.end,
            },
        )
        for kind, path in sorted(written.items()):
            log.info("Wrote %s → %s", kind, path)

    # Compare to golden if available (skip with --no-compare to halve runtime)
    if args.no_compare:
        return
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
