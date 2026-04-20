"""DrawdownJob — mark-to-market, HWM update, drawdown circuit breaker."""
from __future__ import annotations

import pandas as pd

from ..base import Job
from ..context import InferenceContext
from ...portfolio import update_drawdown_circuit_breaker


class DrawdownJob(Job):
    """Marks the portfolio to market and trips the buy-halt circuit breaker.

    Reads:  ctx.holdings, ctx.pos_shares, ctx.ohlcv, ctx.cash, ctx.regime_params, ctx.hwm
    Writes: ctx.portfolio_value, ctx.hwm, ctx.skip_buys, ctx.equity_point
    """

    def run(self, ctx: InferenceContext) -> None:
        today_ts = pd.Timestamp(ctx.today)

        port_val = ctx.cash + sum(
            ctx.pos_shares[t] * float(ctx.ohlcv[t].loc[today_ts, "close"])
            for t in ctx.holdings
            if t in ctx.ohlcv and today_ts in ctx.ohlcv[t].index
        )
        ctx.portfolio_value = port_val

        drawdown_halt = float(ctx.regime_params.get("drawdown_halt_pct", 0.35))
        ctx.hwm, ctx.skip_buys = update_drawdown_circuit_breaker(
            port_val, ctx.hwm, drawdown_halt
        )

        ctx.equity_point = {
            "date": ctx.today,
            "portfolio": port_val,
            "regime": ctx.regime,
        }
