#!/usr/bin/env python3
"""Live-only realized win-rate / expectancy report — separate LIVE from SIM.

WHY (2026-06-17): the `runs.{broker}.db` `trades` table commingles live broker
fills with simulation/backtest rows (most have `source=NULL`). Reading win rate
off the whole table is misleading — it is dominated by ~6000 sim rows and reports
a backtest number (~76%) that is NOT the live track record. The reliable split is
the parent run's `run_type` (`pipeline_runs.run_type = 'live'` vs not).

This tool reports LIVE and SIM side-by-side so the two are never conflated, and
surfaces the metric that actually matters — **payoff ratio** (avg win / |avg loss|)
and **expectancy/trade** — not just hit rate. A high win rate with payoff < 1
(small winners, larger losers) is the classic "looks great, bleeds slowly" trap.

Usage:
    python scripts/live_win_rate.py [--db data/runs.alpaca.db] [--by-exit] [--json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass


@dataclass
class Stats:
    n: int
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff: float          # avg_win / |avg_loss|
    expectancy: float      # win_rate*avg_win + (1-win_rate)*avg_loss
    date_min: str | None
    date_max: str | None


def compute(pnls: list[float]) -> Stats | None:
    pnls = [p for p in pnls if p is not None]
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    payoff = (aw / abs(al)) if al else float("inf")
    return Stats(len(pnls), wr, aw, al, payoff, wr * aw + (1 - wr) * al, None, None)


def _rows(cur: sqlite3.Cursor, live: bool) -> list[tuple]:
    op = "=" if live else "!="
    # COALESCE: live trades sometimes carry NULL trade_date; fall back to run_date.
    return cur.execute(
        f"""SELECT t.pnl_pct, COALESCE(t.trade_date, pr.run_date) AS d
            FROM trades t JOIN pipeline_runs pr ON t.run_id = pr.run_id
            WHERE t.action='sell' AND t.pnl_pct IS NOT NULL
              AND pr.run_type {op} 'live'""",
    ).fetchall()


def report(db: str, by_exit: bool = False) -> dict:
    con = sqlite3.connect(db)
    cur = con.cursor()
    out: dict = {"db": db}
    for label, is_live in (("live", True), ("sim", False)):
        rows = _rows(cur, is_live)
        s = compute([r[0] for r in rows])
        if s is not None:
            dates = sorted(r[1] for r in rows if r[1])
            s.date_min = dates[0] if dates else None
            s.date_max = dates[-1] if dates else None
        out[label] = asdict(s) if s else None
    if by_exit:
        out["live_by_exit"] = [
            {"exit_reason": er, "n": n, "win_rate": wr, "avg_pnl": avg, "avg_hold_days": h}
            for er, n, wr, avg, h in cur.execute(
                """SELECT t.exit_reason, COUNT(*),
                          AVG(CASE WHEN t.pnl_pct>0 THEN 1.0 ELSE 0 END),
                          AVG(t.pnl_pct), AVG(t.hold_days)
                   FROM trades t JOIN pipeline_runs pr ON t.run_id=pr.run_id
                   WHERE t.action='sell' AND t.pnl_pct IS NOT NULL AND pr.run_type='live'
                   GROUP BY t.exit_reason ORDER BY 2 DESC""",
            ).fetchall()
        ]
    con.close()
    return out


def _fmt(label: str, s: dict | None) -> str:
    if not s:
        return f"  {label:5}: (no closed trades)"
    return (
        f"  {label:5}: n={s['n']:<5} win={s['win_rate']*100:5.1f}%  "
        f"avgW={s['avg_win']*100:+.2f}%  avgL={s['avg_loss']*100:+.2f}%  "
        f"payoff={s['payoff']:.2f}  exp/trade={s['expectancy']*100:+.3f}%"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live-only realized win-rate report")
    ap.add_argument("--db", default="data/runs.alpaca.db")
    ap.add_argument("--by-exit", action="store_true", help="break LIVE down by exit_reason")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    data = report(args.db, by_exit=args.by_exit)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print(f"=== realized win-rate report ({args.db}) ===")
    print(_fmt("LIVE", data.get("live")))
    print(_fmt("SIM", data.get("sim")))
    live = data.get("live")
    if live:
        print(f"  LIVE dates: {live['date_min']} → {live['date_max']}")
        # The headline interpretation: payoff, not hit rate, is the lever.
        if live["win_rate"] >= 0.6 and live["payoff"] < 1.0:
            print(
                f"\n  NOTE: win rate is high ({live['win_rate']*100:.0f}%) but payoff < 1 "
                f"({live['payoff']:.2f}) — winners are smaller than losers. The lever is "
                f"PAYOFF (let winners run / cut losers), not win rate."
            )
        if live["n"] < 50:
            print(f"  CAVEAT: only {live['n']} live trades — low statistical power; treat as directional.")
    if args.by_exit and data.get("live_by_exit"):
        print("\n  LIVE by exit_reason:")
        for e in data["live_by_exit"]:
            print(f"    {str(e['exit_reason']):16} n={e['n']:<3} "
                  f"win={e['win_rate']*100:5.1f}% avg={e['avg_pnl']*100:+6.2f}% "
                  f"hold={(e['avg_hold_days'] or 0):.0f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
