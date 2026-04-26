#!/usr/bin/env python
"""Validate buy-logic + portfolio-QP knobs via sim A/B vs golden config.

Bridges the gap between "code shipped" (unit tests pass) and "production-
ready" (sim performance validated). Loads the golden v4.1 config,
optionally flips one or more flags, runs run_backtest over the
27-month OOS window, and compares APY / Sharpe / DD / trade count.

Usage:
    # Baseline only (golden v4.1, all new gates OFF) — sanity reference
    python scripts/validate_buy_logic.py --baseline

    # Stage 2: enable Gate B (Edge-Sharpe floor)
    python scripts/validate_buy_logic.py --gate-b 0.20

    # Stage 3: enable Gate B + Gate A
    python scripts/validate_buy_logic.py --gate-b 0.20 --gate-a-pct 85

    # Stage 4: enable all three gates
    python scripts/validate_buy_logic.py --gate-b 0.20 --gate-a-pct 85 \
        --gate-c-gamma 3.0 --gate-c-tau 0.001

    # Portfolio-QP solver (Stage 1)
    python scripts/validate_buy_logic.py --qp-solver

    # Full stack
    python scripts/validate_buy_logic.py --gate-b 0.20 --gate-a-pct 85 \
        --qp-solver

A/B: each run compares against the most recent baseline run cached on
disk (logs/sim_validations/baseline.json).

Output:
    logs/sim_validations/{date}-{tag}.md     human-readable report
    logs/sim_validations/{date}-{tag}.json   machine-readable result

Returns exit 0 if the run completed (no error). Promotion verdict
(APY ≥ baseline, Sharpe ≥ baseline) is reported in the markdown but
not enforced — that's the operator's call.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("validate-buy-logic")


def _apply_overrides(cfg: dict, *,
                     gate_a_pct: int | None = None,
                     gate_a_lookback: int = 20,
                     gate_b_threshold: float | None = None,
                     gate_c_gamma: float | None = None,
                     gate_c_tau: float | None = None,
                     qp_solver: bool = False,
                     qp_signal_decay: float = 0.0,
                     qp_robust_kappa: float = 0.0,
                     qp_cvar_lambda: float = 0.0) -> dict:
    """Mutate a config copy with the requested gate / QP flag flips."""
    cfg = deepcopy(cfg)

    # Quality-floor gates A/B/C
    qf = (cfg.setdefault("ranking", {})
              .setdefault("panel_scoring", {})
              .setdefault("quality_floor", {}))
    any_gate = (gate_a_pct is not None or gate_b_threshold is not None
                or gate_c_gamma is not None)
    if any_gate:
        qf["enabled"] = True

    if gate_a_pct is not None:
        qf["distribution_floor"] = {
            "enabled": True,
            "percentile": int(gate_a_pct),
            "lookback_days": int(gate_a_lookback),
            "min_history_days": 5,
        }
    if gate_b_threshold is not None:
        qf["edge_sharpe_floor"] = {
            "enabled": True,
            "threshold": float(gate_b_threshold),
        }
    if gate_c_gamma is not None:
        qf["no_trade_band"] = {
            "enabled": True,
            "risk_aversion": float(gate_c_gamma),
            "round_trip_cost": float(gate_c_tau or 0.001),
            "band_constant": 1.5,
        }

    # Portfolio-QP solver
    if qp_solver:
        ja = (cfg.setdefault("rotation", {})
                  .setdefault("joint_actions", {}))
        ja["enabled"] = True
        ja["solver"] = "qp"
        ja["qp_risk_aversion"] = 3.0
        ja["qp_cost_kappa"] = 0.0001
        ja["qp_dw_max"] = 0.50
        ja["qp_min_dw_pct"] = 0.005
        ja["qp_signal_decay"]    = float(qp_signal_decay)
        ja["qp_robust_mu_kappa"] = float(qp_robust_kappa)
        ja["qp_cvar_lambda"]     = float(qp_cvar_lambda)

    return cfg


def _build_tag(args) -> str:
    parts = []
    if args.gate_b is not None:
        parts.append(f"gate-b{args.gate_b:g}")
    if args.gate_a_pct is not None:
        parts.append(f"gate-a-p{args.gate_a_pct}")
    if args.gate_c_gamma is not None:
        parts.append(f"gate-c-g{args.gate_c_gamma:g}")
    if args.qp_solver:
        parts.append("qp")
    if args.qp_signal_decay > 0:
        parts.append(f"qp-decay{args.qp_signal_decay:g}")
    if args.qp_robust_kappa > 0:
        parts.append(f"qp-robust{args.qp_robust_kappa:g}")
    if args.qp_cvar_lambda > 0:
        parts.append(f"qp-cvar{args.qp_cvar_lambda:g}")
    if args.baseline or not parts:
        return "baseline"
    return "+".join(parts)


def _summarise(result, tag: str) -> dict:
    """Distil SimResult into a comparable dict."""
    out = {
        "tag":         tag,
        "apy":         float(getattr(result, "apy", 0.0) or 0.0),
        "total_return": float(getattr(result, "total_return", 0.0) or 0.0),
        "sharpe":      float(getattr(result, "sharpe", 0.0) or 0.0),
        "max_dd":      float(getattr(result, "max_dd", 0.0) or 0.0),
        "n_trades":    int(getattr(result, "n_trades", 0) or 0),
        "n_buys":      len(result.buys()) if hasattr(result, "buys") else 0,
        "n_sells":     len(result.sells()) if hasattr(result, "sells") else 0,
        "longest_no_trade_streak": int(
            getattr(result, "longest_no_trade_streak", 0) or 0,
        ),
    }
    return out


def _diff_table(baseline: dict, candidate: dict) -> str:
    """Render a markdown diff between baseline and candidate metrics."""
    rows = ["| Metric | Baseline | Candidate | Δ | Verdict |",
            "|---|---:|---:|---:|---|"]
    for k in ("apy", "total_return", "sharpe", "max_dd", "n_trades",
               "n_buys", "n_sells", "longest_no_trade_streak"):
        bv = baseline.get(k, 0.0)
        cv = candidate.get(k, 0.0)
        delta = cv - bv
        # Verdict heuristic
        if k in ("apy", "total_return", "sharpe"):
            verdict = "✅" if delta > 0 else "🟡" if delta >= -0.005 else "❌"
        elif k in ("max_dd", "longest_no_trade_streak"):
            verdict = "✅" if delta < 0 else "🟡" if delta < 1 else "❌"
        else:
            verdict = "—"
        if isinstance(cv, float):
            rows.append(f"| {k} | {bv:+.4f} | {cv:+.4f} | "
                        f"{delta:+.4f} | {verdict} |")
        else:
            rows.append(f"| {k} | {bv} | {cv} | {delta:+d} | {verdict} |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--baseline", action="store_true",
                   help="Run baseline (no flag flips) — saves a fresh "
                        "reference for future comparisons.")
    p.add_argument("--gate-a-pct", type=int, default=None,
                   help="Gate A percentile threshold (e.g. 85)")
    p.add_argument("--gate-b", type=float, default=None,
                   help="Gate B Edge-Sharpe threshold (e.g. 0.20)")
    p.add_argument("--gate-c-gamma", type=float, default=None,
                   help="Gate C risk-aversion (e.g. 3.0)")
    p.add_argument("--gate-c-tau", type=float, default=None,
                   help="Gate C round-trip cost (e.g. 0.001)")
    p.add_argument("--qp-solver", action="store_true",
                   help="Use portfolio QP solver instead of greedy")
    p.add_argument("--qp-signal-decay", type=float, default=0.0)
    p.add_argument("--qp-robust-kappa", type=float, default=0.0)
    p.add_argument("--qp-cvar-lambda",  type=float, default=0.0)
    p.add_argument("--start", default="2024-01-01",
                   help="Backtest start date (default 2024-01-01)")
    p.add_argument("--end",   default=None,
                   help="Backtest end date (default = today)")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    cfg    = _apply_overrides(
        config,
        gate_a_pct       = args.gate_a_pct,
        gate_b_threshold = args.gate_b,
        gate_c_gamma     = args.gate_c_gamma,
        gate_c_tau       = args.gate_c_tau,
        qp_solver        = args.qp_solver,
        qp_signal_decay  = args.qp_signal_decay,
        qp_robust_kappa  = args.qp_robust_kappa,
        qp_cvar_lambda   = args.qp_cvar_lambda,
    )

    tag = _build_tag(args)
    log.info("validation run: tag=%s  start=%s  end=%s",
             tag, args.start, args.end)

    # Load OHLCV via the same path the sim adapter uses
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    benchmark = cfg.get("benchmark", "SPY")
    sector_etf_map = cfg.get("sector_etf_map", {})
    watchlist = cfg["watchlist"]
    needed = sorted(set(watchlist) | {benchmark} | set(sector_etf_map.values()))
    log.info("Fetching OHLCV for %d symbols", len(needed))
    ohlcv = {}
    for sym in needed:
        try:
            df = fetch_ohlcv(sym)
            if df is not None and not df.empty:
                ohlcv[sym] = df
        except Exception as exc:
            log.warning("  %-6s fetch failed: %s", sym, exc)
    spy_df = ohlcv.get(benchmark)
    if spy_df is None:
        log.error("benchmark %s missing — cannot run validation", benchmark)
        return 1

    from sim.runner import run_backtest  # noqa: PLC0415
    log.info("Running sim — this takes ~10-30 min depending on universe size")
    result = run_backtest(
        config         = cfg,
        strategy_dir   = strategy_dir,
        ohlcv          = ohlcv,
        spy_df         = spy_df,
        sector_etf_map = sector_etf_map,
        initial_cash   = 100_000.0,
        backtest_start = args.start,
        backtest_end   = args.end,
        snapshot       = True,
    )
    summary = _summarise(result, tag)
    log.info("Sim done: APY=%+.4f Sharpe=%+.3f DD=%.2f%% trades=%d",
             summary["apy"], summary["sharpe"],
             summary["max_dd"] * 100, summary["n_trades"])

    # Persist + compare
    out_dir = REPO_ROOT / "logs" / "sim_validations"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    json_path = out_dir / f"{today}-{tag}.json"
    md_path   = out_dir / f"{today}-{tag}.md"
    json_path.write_text(json.dumps(summary, indent=2))

    baseline_path = out_dir / "baseline.json"
    if args.baseline or not baseline_path.exists():
        baseline_path.write_text(json.dumps(summary, indent=2))
        log.info("Saved as baseline → %s", baseline_path)
        diff_md = "_baseline run — no comparison_"
    else:
        baseline = json.loads(baseline_path.read_text())
        diff_md = _diff_table(baseline, summary)

    md_path.write_text(
        f"# Buy-logic / Portfolio-QP validation — {tag}\n\n"
        f"**Date**: {today}  \n"
        f"**Backtest window**: {args.start} → {args.end or 'today'}  \n"
        f"**Tag**: `{tag}`\n\n"
        "## Result\n\n"
        f"```\nAPY    = {summary['apy']:+.4f}\n"
        f"Sharpe = {summary['sharpe']:+.3f}\n"
        f"MaxDD  = {summary['max_dd']:.2%}\n"
        f"Trades = {summary['n_trades']}  "
        f"(buys={summary['n_buys']} sells={summary['n_sells']})\n"
        f"NoTradeStreak = {summary['longest_no_trade_streak']}d\n```\n\n"
        "## Diff vs baseline\n\n"
        f"{diff_md}\n"
    )
    log.info("Report → %s", md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
