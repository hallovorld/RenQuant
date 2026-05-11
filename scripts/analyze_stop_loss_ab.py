#!/usr/bin/env python
"""Aggregate the 6 stop-loss A/B sim logs and pick the min-total-loss winner.

Reads ``data/logs/wf_sim_{label}_{ts}.log`` files matching a timestamp
prefix; extracts the final-summary block; computes Total_loss metric
``MaxDD% + Σ|realized_loss_pct|`` for ranking. Per user direction:
the production baseline is whichever config minimises Total_loss subject
to APY ≥ 0 (industry-leading rigor — no negative-APY configs allowed
even if they have low DD).

Usage::

    python scripts/analyze_stop_loss_ab.py --timestamp 231055

Output: prints comparison table + winner; writes
``data/logs/stop_loss_ab_report_<ts>.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "data" / "logs"


_RE_FINAL_VALUE = re.compile(r"Final value:\s+\$([\d,]+)")
_RE_RETURN = re.compile(r"Return:\s+([+-]?\d+\.\d+)%")
_RE_APY = re.compile(r"APY:\s+([+-]?\d+\.\d+)%")
_RE_SHARPE = re.compile(r"Sharpe=([+-]?\d+\.\d+)")
_RE_SORTINO = re.compile(r"Sortino=([+-]?\d+\.\d+)")
_RE_CALMAR = re.compile(r"Calmar=([+-]?\d+\.\d+)")
_RE_MAXDD = re.compile(r"MaxDD=([+-]?\d+\.\d+)%")
_RE_VOL = re.compile(r"Vol=([+-]?\d+\.\d+)%")
_RE_TRADES = re.compile(r"Trades:\s+(\d+)\s+buys,\s+(\d+)\s+sells")
_RE_WINRATE = re.compile(r"Win rate:\s+(\d+)%")
_RE_AVG_PNL = re.compile(r"Avg P&L/trade:\s+([+-]?\d+\.\d+)%")
_RE_AVG_HOLD = re.compile(r"Avg hold:\s+(\d+)d")
_RE_DSR = re.compile(r"DSR=([+-]?\d+\.\d+)")
_RE_BETA = re.compile(r"Beta=([+-]?\d+\.\d+)")
_RE_ALPHA = re.compile(r"Alpha=([+-]?\d+\.\d+)%")
_RE_IR = re.compile(r"InfoRatio=([+-]?\d+\.\d+)")
_RE_TAX = re.compile(r"Total tax:\s+\$([\d,]+)")
_RE_EXIT_REASONS = re.compile(r"Exit reasons:\s+(\{[^}]+\})")


def _maybe_float(m, default=None):
    return float(m.group(1).replace(",", "")) if m else default


def parse_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    txt = path.read_text()
    tail = txt[-6000:]  # Final-summary block is near the end.
    out = {
        "log": str(path),
        "final_value": _maybe_float(_RE_FINAL_VALUE.search(tail)),
        "return_pct": _maybe_float(_RE_RETURN.search(tail)),
        "apy_pct": _maybe_float(_RE_APY.search(tail)),
        "sharpe": _maybe_float(_RE_SHARPE.search(tail)),
        "sortino": _maybe_float(_RE_SORTINO.search(tail)),
        "calmar": _maybe_float(_RE_CALMAR.search(tail)),
        "maxdd_pct": _maybe_float(_RE_MAXDD.search(tail)),
        "vol_pct": _maybe_float(_RE_VOL.search(tail)),
        "winrate_pct": _maybe_float(_RE_WINRATE.search(tail)),
        "avg_pnl_per_trade_pct": _maybe_float(_RE_AVG_PNL.search(tail)),
        "avg_hold_d": _maybe_float(_RE_AVG_HOLD.search(tail)),
        "dsr": _maybe_float(_RE_DSR.search(tail)),
        "beta_spy": _maybe_float(_RE_BETA.search(tail)),
        "alpha_pct_yr": _maybe_float(_RE_ALPHA.search(tail)),
        "info_ratio": _maybe_float(_RE_IR.search(tail)),
        "total_tax": _maybe_float(_RE_TAX.search(tail)),
    }
    m_tr = _RE_TRADES.search(tail)
    if m_tr:
        out["n_buys"], out["n_sells"] = int(m_tr.group(1)), int(m_tr.group(2))
    m_ex = _RE_EXIT_REASONS.search(tail)
    if m_ex:
        out["exit_reasons_raw"] = m_ex.group(1)
    return out


def total_loss_score(row: dict) -> float:
    """User-spec min metric: MaxDD + cumulative-loss proxy.

    For a per-trade loss view we have only summary stats; use
    ``MaxDD% + (1-winrate) × n_sells × |avg_loss_proxy|``.

    Without per-trade loss decomposition in logs, approximate
    Σ realized_loss as ``(1 - winrate) × n_sells × |avg_pnl|`` —
    losing trades contribute their average magnitude.
    """
    maxdd = row.get("maxdd_pct") or 0.0
    wr = (row.get("winrate_pct") or 0.0) / 100.0
    n_sells = row.get("n_sells") or 0
    avg_pnl = row.get("avg_pnl_per_trade_pct") or 0.0
    # Crude proxy — without per-trade log details, model losers as having
    # 1.5x average magnitude (typical win-loss asymmetry).
    losing_share = max(0.0, 1.0 - wr) * n_sells
    realized_loss_proxy = losing_share * abs(avg_pnl) * 1.5 / 100.0  # to "% of capital" units
    return float(maxdd) + float(realized_loss_proxy)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timestamp", required=True,
                   help="Timestamp suffix of the sim logs (e.g. 231055).")
    p.add_argument("--labels", nargs="+",
                   default=[
                       "sim_baseline", "L1_choppy_no_stop",
                       "L2_post_stop_cooldown", "L3_dd_rebalance",
                       "L4_sigma_revival", "L5_atr_trail",
                   ],
                   help="Label strings matching wf_sim_{label}_{ts}.log.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for label in args.labels:
        path = LOG_DIR / f"wf_sim_{label}_{args.timestamp}.log"
        row = parse_log(path)
        if row is None:
            print(f"  ⚠ missing log: {path}", file=sys.stderr)
            continue
        row["label"] = label
        row["total_loss"] = total_loss_score(row)
        rows.append(row)
    if not rows:
        print("No logs parsed; aborting.", file=sys.stderr)
        return 2

    # Sort by total_loss ASC; tag winner (must have APY ≥ 0)
    rows.sort(key=lambda r: r["total_loss"])
    positive_apy = [r for r in rows if (r.get("apy_pct") or -1) >= 0]
    winner = positive_apy[0] if positive_apy else None

    # Pretty print
    cols = ["label", "apy_pct", "maxdd_pct", "sharpe", "sortino", "calmar",
            "vol_pct", "winrate_pct", "n_sells", "avg_pnl_per_trade_pct",
            "total_loss", "dsr", "alpha_pct_yr"]
    head = " | ".join(f"{c:>12s}" for c in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("    n/a")
            elif isinstance(v, float):
                cells.append(f"{v:>12.2f}")
            elif isinstance(v, int):
                cells.append(f"{v:>12d}")
            else:
                cells.append(f"{str(v):>12s}")
        print(" | ".join(cells))

    print()
    if winner:
        print(f"🏆 winner (min total_loss & APY ≥ 0): {winner['label']}")
        print(f"   APY={winner['apy_pct']:+.2f}%  MaxDD={winner['maxdd_pct']:.1f}%  "
              f"Sharpe={winner['sharpe']:+.2f}  total_loss={winner['total_loss']:.2f}")
    else:
        print("⚠ No config has APY ≥ 0 — none qualify for promotion.")

    # Write report
    out = (
        Path(args.out) if args.out
        else LOG_DIR / f"stop_loss_ab_report_{args.timestamp}.md"
    )
    lines = [
        "# Stop-loss A/B sim report",
        f"\nTimestamp: {args.timestamp}\n",
        "| label | APY% | MaxDD% | Sharpe | Sortino | Calmar | Vol% | WR% | n_sells | avg_pnl% | total_loss | DSR | Alpha%/yr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join(str(r.get(c) if r.get(c) is not None else "n/a")
                              for c in cols) + " |"
        )
    if winner:
        lines.append(f"\n**Winner**: {winner['label']} — APY={winner['apy_pct']:+.2f}%, "
                     f"MaxDD={winner['maxdd_pct']:.1f}%, total_loss={winner['total_loss']:.2f}")
    out.write_text("\n".join(lines))
    print(f"\n📄 Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
