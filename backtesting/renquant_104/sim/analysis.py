"""Post-backtest analysis helpers — robustness checks on SimResult.

User ask 2026-04-24: "在 notebook 模拟的时候，帮我取掉收益最高的3笔
trade（从买到卖），我想看更可期待的收益，而不是碰巧遇到了大牛股。"

Translation: "during notebook sim, strip the top-N most profitable
trades so I can see the EXPECTED-case return instead of the lucky
tail." This is the classic "stripped alpha" robustness check — if
your strategy's APY depends on 3 lottery winners, the real expected
return is much lower than the headline.

Usage from notebook::

    from sim.runner import run_backtest
    from sim.analysis import strip_top_n_trades, compare_strip_levels

    result = run_backtest(...)
    stripped = strip_top_n_trades(result, n=3)
    print(f"APY with top-3 removed: {stripped['apy']:.1%}")

    # Or show a sensitivity ladder:
    compare_strip_levels(result, levels=[0, 1, 3, 5, 10])
"""
from __future__ import annotations

from typing import Any


def _completed_trades(trade_log: list[dict]) -> list[dict]:
    """Return only completed (sell) entries — these carry realized pnl_pct."""
    return [t for t in trade_log
            if t.get("action") == "sell" and "pnl_pct" in t]


def strip_top_n_trades(result, n: int = 3) -> dict[str, Any]:
    """Re-derive portfolio metrics after removing the top-N realized-return
    trades from `result.trade_log`.

    We don't re-run the sim — that would change every downstream decision
    (different cash, different entries). Instead we de-compound: the
    portfolio final return is the product of all per-trade gross returns
    (1 + pnl_pct); dividing by the top-N trades' gross returns removes
    their contribution as if those trades had broken even (0% P&L).

    Returns a dict::

        {"n_stripped":      int,
         "stripped_trades": list[dict],     # the N removed
         "original_total":  float,          # matches result.total_return
         "stripped_total":  float,          # return excluding the N
         "apy":             float,          # annualised stripped return
         "years":           float,
         "median_trade":    float,
         "median_stripped": float}
    """
    trades = _completed_trades(result.trade_log)
    if not trades:
        return {
            "n_stripped":      0,
            "stripped_trades": [],
            "original_total":  result.total_return,
            "stripped_total":  result.total_return,
            "apy":             result.apy,
            "years":           0.0,
            "median_trade":    0.0,
            "median_stripped": 0.0,
        }

    n = min(max(int(n), 0), len(trades))
    sorted_trades = sorted(trades, key=lambda t: t.get("pnl_pct", 0.0),
                           reverse=True)
    stripped = sorted_trades[:n]

    # Annualisation factor from the real equity curve (dates + years don't
    # change when we strip trades)
    if len(result.equity_df) >= 2:
        first = result.equity_df.index[0]
        last  = result.equity_df.index[-1]
        years = max(1e-9, (last - first).days / 365.25)
    else:
        years = 1.0

    # Reconstruct total as product of gross returns on completed trades.
    # This is an approximation — the real portfolio has overlapping
    # positions, partial sells, open-at-end positions, and cash drag — so
    # we anchor to result.total_return and strip the top-N's product out
    # of the growth factor.
    original_growth = 1.0 + result.total_return
    strip_growth    = 1.0
    for t in stripped:
        strip_growth *= (1.0 + float(t.get("pnl_pct", 0.0)))
    if strip_growth <= 0:
        # Shouldn't happen for positive-PnL trades but guard
        strip_growth = 1e-9

    # "Replace each top trade with a break-even (0%) trade" semantics:
    stripped_growth = original_growth / strip_growth
    stripped_total  = stripped_growth - 1.0
    stripped_apy    = (stripped_growth ** (1 / years)) - 1.0 if years > 0 else 0.0

    # Descriptive: median (not mean) gives more robust "typical trade"
    pnl_list = sorted(t["pnl_pct"] for t in trades)
    median_all = pnl_list[len(pnl_list) // 2]

    remaining = [t for t in sorted_trades[n:]]
    remaining_pnl = sorted(t["pnl_pct"] for t in remaining)
    median_stripped = (remaining_pnl[len(remaining_pnl) // 2]
                       if remaining_pnl else 0.0)

    return {
        "n_stripped":      n,
        "stripped_trades": stripped,
        "original_total":  result.total_return,
        "stripped_total":  stripped_total,
        "apy":             stripped_apy,
        "years":           years,
        "median_trade":    median_all,
        "median_stripped": median_stripped,
    }


def compare_strip_levels(result, levels: list[int] | None = None) -> None:
    """Print a ladder of metrics as we progressively strip top trades.

    Quick eyeball of alpha robustness — if APY drops a lot going from
    N=0 → N=3, strategy depends on tail luck.
    """
    if levels is None:
        levels = [0, 1, 3, 5, 10]

    print(f"{'strip':>5}  {'total':>9}  {'apy':>9}  "
          f"{'median_trade':>14}  {'top_trade_pnl':>14}")
    print("-" * 62)
    for n in levels:
        s = strip_top_n_trades(result, n=n)
        top_pnl = (s["stripped_trades"][-1]["pnl_pct"]
                   if s["stripped_trades"] else 0.0)
        marker = " ⭐" if n == 0 else ""
        print(f"{n:>5}  {s['stripped_total']*100:>8.2f}% "
              f"{s['apy']*100:>8.2f}% "
              f"{s['median_stripped']*100:>13.2f}% "
              f"{top_pnl*100:>13.2f}%{marker}")
    print()
    print("⭐ = original; each step removes the next-most-profitable trade.")
    print("If APY plunges between rows, headline return is tail-driven "
          "(lucky mega-winners carrying the strategy).")


__all__ = ["strip_top_n_trades", "compare_strip_levels"]
