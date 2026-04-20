"""BuyGatesJob — transition window, BEAR branch, SPY velocity, SPY EMA50.

Any failed gate sets ctx.skip_buys = True and marks ctx.orders with a BEAR
defensive buy if applicable, then returns early so CandidateJob/RankingJob/
SelectionJob are all skipped.
"""
from __future__ import annotations

import pandas as pd

from ..base import Job
from ..context import InferenceContext
from ...market_gates import check_spy_velocity_crash, check_spy_ema_trend
from ...sizing import compute_position_size
from ...exits import HoldingState


class BuyGatesJob(Job):
    """Evaluates all buy-blocking gates and handles the BEAR defensive branch.

    Reads:  ctx.regime, ctx.in_transition, ctx.skip_buys, ctx.holdings,
            ctx.spy_returns, ctx.ohlcv["SPY"], ctx.regime_params, ctx.config,
            ctx.action_fn, ctx.score_fn, ctx.last_sell_dates
    Writes: ctx.skip_buys (may be set True),
            ctx.orders (BEAR defensive buy appended if applicable)

    If any gate fires or this is a BEAR bar, sets ctx.skip_buys = True so that
    CandidateJob.should_skip() returns True and the rest of the buy pipeline
    is bypassed.
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        # Already halted — nothing to gate
        return False

    def run(self, ctx: InferenceContext) -> None:
        cfg = ctx.config

        # ── Gate 1: open slots ─────────────────────────────────────────────────
        max_pos = cfg.get("max_concurrent_positions", 8)
        open_slots = max_pos - len(ctx.holdings)
        if open_slots <= 0 or ctx.skip_buys:
            ctx.skip_buys = True
            return

        # ── Gate 2: transition uncertainty window ──────────────────────────────
        if ctx.in_transition:
            ctx.skip_buys = True
            return

        # ── Gate 3: BEAR branch (defensives only) ─────────────────────────────
        if ctx.regime == "BEAR":
            ctx.skip_buys = True   # block offensive buys
            self._try_bear_defensive(ctx)
            return

        # ── Gate 4: SPY velocity crash filter ─────────────────────────────────
        rp = ctx.regime_params
        vel_lookback = int(rp.get("spy_velocity_lookback_days", 3))
        vel_halt     = float(rp.get("spy_velocity_halt_pct", 0.03))
        spy_rets_list = list(ctx.spy_returns)
        if check_spy_velocity_crash(spy_rets_list, vel_lookback, vel_halt):
            ctx.skip_buys = True
            return

        # ── Gate 5: SPY EMA50 trend gate ──────────────────────────────────────
        spy_df = ctx.ohlcv.get("SPY")
        if spy_df is not None:
            today_ts = pd.Timestamp(ctx.today)
            spy_close_hist = spy_df["close"].loc[:today_ts]
            if not check_spy_ema_trend(spy_close_hist):
                ctx.skip_buys = True
                return

    # ─────────────────────────────────────────────────────────────────────────
    def _try_bear_defensive(self, ctx: InferenceContext) -> None:
        cfg = ctx.config
        defensive_set   = set(cfg.get("defensive_tickers", []))
        wash_sale_days  = cfg.get("wash_sale_days", 30)
        bear_def_slots  = 1
        bear_def_pct    = 0.15
        today_ts        = pd.Timestamp(ctx.today)
        exportable      = set(cfg.get("_exportable", []))   # set by adapter

        held_defensives = sum(1 for t in ctx.holdings if t in defensive_set)
        if held_defensives >= bear_def_slots:
            return

        # Score candidates
        def_candidates = []
        for t in defensive_set:
            if t not in exportable:
                continue
            if t in ctx.holdings:
                continue
            df = ctx.ohlcv.get(t)
            if df is None or today_ts not in df.index:
                continue
            last_sell = ctx.last_sell_dates.get(t)
            if last_sell and (ctx.today - last_sell).days < wash_sale_days:
                continue
            if ctx.action_fn(t, today_ts) != "buy":
                continue
            rs = ctx.score_fn(t, today_ts)
            if rs is None:
                continue
            def_candidates.append((t, rs))

        def_candidates.sort(key=lambda x: x[1], reverse=True)

        for t, _ in def_candidates:
            price = float(ctx.ohlcv[t].loc[today_ts, "close"])
            _, shares = compute_position_size(
                ctx.portfolio_value, ctx.cash, 0, 0, price,
                override_pct=bear_def_pct,
            )
            if shares < 1:
                continue
            ctx.orders.append({
                "ticker":  t,
                "price":   price,
                "shares":  shares,
                "invest":  shares * price,
                "regime":  "BEAR_defensive",
            })
            break
