#!/usr/bin/env python
"""Write the walkforward-eval 27-mo OOS Markdown report (Track P3.2).

Reads a metrics JSON produced by dump_walkforward_sim_metrics.py (OR parses
the print_summary output of run_sim_104.py from a log file), then emits
data/logs/walkforward_oos_report_<timestamp>.md following the spec from
the task brief.

Usage::
    python scripts/write_walkforward_report.py --metrics data/logs/walkforward_metrics_*.json
    python scripts/write_walkforward_report.py --sim-log /tmp/walkforward_sim.log
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _fmt_pct(x, default="—"):
    if x is None:
        return default
    try:
        return f"{float(x) * 100:+.2f}%"
    except Exception:
        return default


def _fmt_pct_unsigned(x, default="—"):
    if x is None:
        return default
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return default


def _fmt_num(x, fmt="+.4f", default="—"):
    if x is None:
        return default
    try:
        return f"{float(x):{fmt}}"
    except Exception:
        return default


def parse_sim_log(log_path: Path) -> dict:
    """Parse run_sim_104.py's stdout for the print_summary block.

    Robust to minor formatting drift — uses regex anchors for each
    metric line emitted by SimResult.print_summary().
    """
    text = log_path.read_text(errors="ignore")
    out = {}

    m = re.search(r"Final value: \$([\d,]+).*?Return: ([\-\d.]+)%.*?APY: ([\-\d.]+)%", text)
    if m:
        out["final_value"] = float(m.group(1).replace(",", ""))
        out["total_return"] = float(m.group(2)) / 100.0
        out["apy"] = float(m.group(3)) / 100.0

    m = re.search(
        r"Risk: Sharpe=([+\-\d.—]+)\s+Sortino=([+\-\d.—]+)\s+"
        r"Calmar=([+\-\d.—]+)\s+MaxDD=([+\-\d.%—]+)\s+Vol=([+\-\d.%—]+)",
        text,
    )
    if m:
        def _f(s):
            s = s.strip()
            if s in ("—", "-"):
                return None
            s = s.rstrip("%")
            try:
                v = float(s)
                # MaxDD / Vol printed as %; Sharpe/Sortino/Calmar dimensionless
                return v
            except Exception:
                return None
        out["sharpe_arithmetic"] = _f(m.group(1))
        out["sortino"] = _f(m.group(2))
        out["calmar"] = _f(m.group(3))
        # MaxDD / Vol were printed as percentages — convert to fraction
        mdd = _f(m.group(4))
        vol = _f(m.group(5))
        out["max_dd"] = (mdd / 100.0) if mdd is not None else None
        out["ann_vol"] = (vol / 100.0) if vol is not None else None

    m = re.search(
        r"Falsifiability: DSR=([+\-\d.—]+) \(n_trials=(\d+)\)\s+PBO=([+\-\d.—]+)",
        text,
    )
    if m:
        def _f(s):
            s = s.strip()
            if s in ("—", "-"):
                return None
            try:
                return float(s)
            except Exception:
                return None
        out["dsr"] = _f(m.group(1))
        out["n_trials"] = int(m.group(2))
        out["pbo"] = _f(m.group(3))

    m = re.search(
        r"vs SPY: Beta=([+\-\d.—]+)\s+Alpha=([+\-\d.%—/yr ]+)\s+InfoRatio=([+\-\d.—]+)",
        text,
    )
    if m:
        def _f(s):
            s = s.strip().rstrip("/yr").rstrip("%").strip()
            if s in ("—", "-"):
                return None
            try:
                return float(s)
            except Exception:
                return None
        out["beta_vs_spy"] = _f(m.group(1))
        a = _f(m.group(2))
        # Alpha printed as %/yr → store as annualized fraction
        out["alpha_vs_spy"] = (a / 100.0) if a is not None else None
        out["information_ratio_vs_spy"] = _f(m.group(3))

    m = re.search(r"Trades:\s+(\d+) buys,\s+(\d+) sells", text)
    if m:
        out["n_buys"] = int(m.group(1))
        out["n_sells"] = int(m.group(2))
    m = re.search(r"Win rate:\s+(\d+)%", text)
    if m:
        out["win_rate"] = float(m.group(1)) / 100.0
    m = re.search(r"Avg hold:\s+([\d.]+)d", text)
    if m:
        out["avg_hold_days"] = float(m.group(1))
    m = re.search(r"Avg P&L/trade:\s+([\-\d.]+)%", text)
    if m:
        out["avg_pnl_pct"] = float(m.group(1)) / 100.0
    m = re.search(r"Total tax: \$([\d,]+)", text)
    if m:
        out["total_tax_usd"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Exit reasons:\s+(\{[^}]+\})", text)
    if m:
        try:
            # Python dict repr; eval-safe alt via ast
            import ast
            out["exit_reasons"] = ast.literal_eval(m.group(1))
        except Exception:
            pass
    m = re.search(r"Longest no-trade streak:\s+(\d+)d", text)
    if m:
        out["longest_no_trade_streak"] = int(m.group(1))
    m = re.search(r"Simulation complete:\s+(\d+) days", text)
    if m:
        out["n_bars"] = int(m.group(1))
    return out


def build_report(metrics: dict, *, sim_log_path: str | None = None) -> str:
    apy = metrics.get("apy")
    sharpe_arith = metrics.get("sharpe_arithmetic")
    sharpe_geom = metrics.get("sharpe_geometric")
    sortino = metrics.get("sortino")
    calmar = metrics.get("calmar")
    max_dd = metrics.get("max_dd")
    ann_vol = metrics.get("ann_vol")
    dsr = metrics.get("dsr")
    pbo = metrics.get("pbo")
    n_trials = metrics.get("n_trials", 38)
    beta_spy = metrics.get("beta_vs_spy")
    alpha_spy = metrics.get("alpha_vs_spy")
    ir_spy = metrics.get("information_ratio_vs_spy")

    apy_s = _fmt_pct(apy)
    sharpe_a_s = _fmt_num(sharpe_arith, "+.4f")
    sharpe_g_s = _fmt_num(sharpe_geom, "+.4f")
    sortino_s = _fmt_num(sortino, "+.4f")
    calmar_s = _fmt_num(calmar, "+.4f")
    max_dd_s = _fmt_pct_unsigned(max_dd)
    ann_vol_s = _fmt_pct_unsigned(ann_vol)
    dsr_s = _fmt_num(dsr, "+.4f")
    pbo_s = _fmt_num(pbo, ".4f")
    beta_s = _fmt_num(beta_spy, "+.4f")
    alpha_s = _fmt_pct(alpha_spy)
    ir_s = _fmt_num(ir_spy, "+.4f")

    # Sharpe delta vs in-sample baseline
    in_sample_sharpe_high = 0.40    # 2026-05-09 morning
    in_sample_sharpe_low = 0.20     # 2026-05-09 evening
    in_sample_apy_high = 0.0677     # +6.77%
    in_sample_apy_low = 0.0197      # +1.97%
    if sharpe_arith is not None:
        sharpe_delta_high = sharpe_arith - in_sample_sharpe_high
        sharpe_delta_low = sharpe_arith - in_sample_sharpe_low
        sharpe_delta_s = (f"{sharpe_arith:+.4f} vs +{in_sample_sharpe_high:.2f} morning "
                          f"(Δ {sharpe_delta_high:+.2f}) / +{in_sample_sharpe_low:.2f} evening "
                          f"(Δ {sharpe_delta_low:+.2f})")
    else:
        sharpe_delta_s = "— (sharpe NaN)"
    if apy is not None:
        apy_delta_high = (apy - in_sample_apy_high) * 100
        apy_delta_low = (apy - in_sample_apy_low) * 100
        apy_delta_s = (f"{apy*100:+.2f}% vs +6.77% morning (Δ {apy_delta_high:+.2f} pp) / "
                       f"+1.97% evening (Δ {apy_delta_low:+.2f} pp)")
    else:
        apy_delta_s = "— (apy NaN)"

    n_buys = metrics.get("n_buys", 0)
    n_sells = metrics.get("n_sells", 0)
    win_rate = metrics.get("win_rate")
    avg_hold = metrics.get("avg_hold_days")
    n_bars = metrics.get("n_bars", 0)
    longest_idle = metrics.get("longest_no_trade_streak", 0)
    exit_reasons = metrics.get("exit_reasons", {})
    seed = metrics.get("seed")
    seed_s = str(seed) if seed is not None else "None (legacy non-deterministic)"

    ts = metrics.get("run_timestamp") or datetime.now().isoformat()

    body = f"""# Walk-forward OOS 27-mo report

Run timestamp: {ts}
Backtest window: {metrics.get('backtest_start', '2024-01-02')} → {metrics.get('backtest_end', '2026-03-28')}
Walkforward retrains used: 38/39 (1 skipped per §5.13.10 undertrain guard)
Seed: {seed_s}
Risk-free rate: {(metrics.get('risk_free_rate_annual') or 0.05)*100:.1f}% annual (2026 SOFR proxy)
Execution model: industrial (commission + slippage + configured T+N settlement)
N bars: {n_bars}
Source: {sim_log_path or "metrics JSON dump"}

## Headline metrics (HONEST OOS — labels never seen at retrain time)
- APY                     : {apy_s}
- Sharpe (arithmetic)     : {sharpe_a_s}
- Sharpe (geometric)      : {sharpe_g_s}
- Sortino (ddof=1)        : {sortino_s}
- Calmar                  : {calmar_s}
- MaxDD                   : {max_dd_s}
- Annualized vol          : {ann_vol_s}

## §5.13.4 Falsifiability triple
- DSR (n_trials={n_trials})       : {dsr_s}
- PBO                     : {pbo_s} (NaN if single-seed)
- Action consistency      : — (only computed in multi-seed K≥2 mode)

## vs SPY (CAPM regression on daily returns)
- β                       : {beta_s}
- α annualized            : {alpha_s}
- Information Ratio       : {ir_s}

## Comparison to in-sample +6.77/+1.97 (2026-05-09 audit)
- Sharpe delta            : {sharpe_delta_s}
- APY delta               : {apy_delta_s}
- Expected gap (audit estimated): -0.2 to -0.4 Sharpe from removing
  leakage + execution costs

## Pre-existing audit predictions
- E27 walk-forward 3-cut alpha: -15.62% ± 10.21% (negative).
- This run with proper walkforward + industrial execution should be
  in the same band or tighter.

## Activity summary
- Trades                  : {n_buys} buys / {n_sells} sells
- Win rate                : {_fmt_pct_unsigned(win_rate)}
- Avg hold (days)         : {avg_hold if avg_hold else '—'}
- Avg P&L / trade         : {_fmt_pct(metrics.get('avg_pnl_pct'))}
- Total tax (USD)         : {metrics.get('total_tax_usd', '—')}
- Longest no-trade streak : {longest_idle} days
- Exit reasons            : {dict(exit_reasons) if exit_reasons else '{{}}'}

## Run notes
- Missing cutoffs: 2024-12-23 (undertrain guard); loader falls back to 2024-12-02.
- Side config: strategy_config.sim_baseline.json (artifact paths aliased
  to walkforward_eval side paths per §5.13.13; sim is read-only).
- Manifest: artifacts/walkforward_manifest.json (38 retrain entries, cadence_days=21,
  training_window_years=3.0, merged from A/B/C chunks).

## §5.13.4 disclaimer
This is a **single-seed** run. APY and Sharpe numbers above are point estimates
with unknown σ — they should NOT be quoted as authoritative until paired with a
multi-seed (K≥5) run that produces mean ± std. The DSR alone deflates for
selection bias across n_trials=38 retrains but does not characterize seed-level
sampling noise. Multi-seed harness (`run_backtest_multi_seed(seeds=5)`) is the
follow-up step.
"""
    return body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", help="Path to metrics JSON (preferred)")
    p.add_argument("--sim-log", help="Path to run_sim_104.py stdout log (fallback)")
    p.add_argument("--out", help="Output report path (default data/logs/walkforward_oos_report_<ts>.md)")
    args = p.parse_args()

    metrics: dict = {}
    sim_log_path = None
    if args.metrics:
        metrics = json.loads(Path(args.metrics).read_text())
    elif args.sim_log:
        sim_log_path = args.sim_log
        metrics = parse_sim_log(Path(args.sim_log))
        # Override fields from defaults
        metrics.setdefault("run_timestamp", datetime.now().isoformat())
    else:
        raise SystemExit("Must provide --metrics or --sim-log")

    report = build_report(metrics, sim_log_path=sim_log_path)

    out_path = Path(args.out) if args.out else (
        REPO / "data" / "logs"
        / f"walkforward_oos_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)
    print(f"\n[wrote → {out_path}]")


if __name__ == "__main__":
    main()
