"""NotebookAdapter — normalizes notebook simulation state into InferenceContext.

Usage (replaces the ~300-line simulation cell):

    from adapters.notebook import NotebookAdapter
    from kernel.pipeline import InferencePipeline

    adapter  = NotebookAdapter(ohlcv, spy_daily_ret, results, corr_dict, config,
                                gmm_artifact, earnings_cal, strategy_dir)
    pipeline = InferencePipeline()

    for today in bt_dates:
        ctx = adapter.make_context(today)
        pipeline.run(ctx)
        adapter.commit(ctx)

    equity_df = adapter.equity_df
    trade_log = adapter.trade_log
"""
from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

from kernel.exits import HoldingState
from kernel.regime import RegimeState
from kernel.pipeline import InferenceContext
from kernel.pipeline.jobs import (
    RegimeJob, DrawdownJob, SellJob, BuyGatesJob,
    CandidateJob, RankingJob, SelectionJob,
)
from kernel.pipeline.base import Pipeline


class InferencePipeline(Pipeline):
    """Standard 7-job inference pipeline."""
    def __init__(self) -> None:
        super().__init__([
            RegimeJob(),
            DrawdownJob(),
            SellJob(),
            BuyGatesJob(),
            CandidateJob(),
            RankingJob(),
            SelectionJob(),
        ])


class NotebookAdapter:
    """Manages persistent state across bars and builds InferenceContext per bar.

    All mutable cross-bar state (holdings, cash, hwm, regime_state, …) lives here.
    Each bar: make_context() → pipeline.run() → commit().
    """

    def __init__(
        self,
        ohlcv: dict[str, pd.DataFrame],
        spy_daily_ret: pd.Series,
        results: dict[str, dict],
        corr_dict: dict[str, dict[str, float]],
        config: dict,
        gmm_artifact: dict | None,
        earnings_cal: dict[str, list[str]],
        initial_cash: float | None = None,
    ) -> None:
        self._ohlcv         = ohlcv
        self._spy_daily_ret = spy_daily_ret
        self._results       = results
        self._corr_dict     = corr_dict
        self._config        = config
        self._gmm_artifact  = gmm_artifact
        self._earnings_cal  = earnings_cal

        # Persistent state
        cash = float(initial_cash or config.get("initial_cash", 100_000))
        self._cash            = cash
        self._hwm             = cash
        self._holdings: dict[str, HoldingState] = {}
        self._pos_shares: dict[str, float]      = {}
        self._regime_state    = RegimeState()
        self._last_sell_dates: dict[str, datetime.date] = {}

        # Outputs
        self._equity_curve: list[dict] = []
        self._trade_log: list[dict]    = []

        # Build exportable set (tickers that passed OOS Sharpe floor)
        self._exportable = {t for t, r in results.items() if r.get("passes_floor")}
        # Inject into config so jobs can read it without importing results
        config["_exportable"] = list(self._exportable)

    # ── Per-bar API ───────────────────────────────────────────────────────────

    def make_context(self, today: pd.Timestamp) -> InferenceContext:
        today_date   = today.date() if hasattr(today, "date") else today
        spy_rets_arr = self._spy_daily_ret.loc[:today].values.astype(float)

        # prev_closes from last bar (updated in commit)
        prev_closes = {
            t: float(self._ohlcv[t].loc[:today].iloc[-2]["close"])
            if len(self._ohlcv[t].loc[:today]) >= 2 else float(self._ohlcv[t].loc[:today].iloc[-1]["close"])
            for t in self._holdings
            if t in self._ohlcv and len(self._ohlcv[t].loc[:today]) >= 1
        }

        return InferenceContext(
            today            = today_date,
            ohlcv            = self._ohlcv,
            spy_returns      = spy_rets_arr,
            prev_closes      = prev_closes,
            holdings         = dict(self._holdings),   # shallow copy; jobs mutate HoldingState
            pos_shares       = dict(self._pos_shares),
            cash             = self._cash,
            portfolio_value  = self._cash,  # DrawdownJob will recompute
            action_fn        = self._make_action_fn(today),
            score_fn         = self._make_score_fn(today),
            gmm_artifact     = self._gmm_artifact,
            corr_dict        = self._corr_dict,
            earnings_cal     = self._earnings_cal,
            config           = self._config,
            hwm              = self._hwm,
            regime_state     = self._regime_state,
            last_sell_dates  = dict(self._last_sell_dates),
        )

    def commit(self, ctx: InferenceContext) -> None:
        """Apply pipeline outputs back to persistent state."""
        today_ts = pd.Timestamp(ctx.today)

        # Apply sells
        for act in ctx.exit_actions:
            t = act["ticker"]
            self._holdings.pop(t, None)
            self._pos_shares.pop(t, None)
            self._cash += act["shares"] * act["price"] - act["tax"]
            self._last_sell_dates[t] = ctx.today
            self._trade_log.append({
                "action":      "sell",
                "ticker":      t,
                "date":        today_ts,
                "pnl_pct":     act["pnl_pct"],
                "hold_days":   act["hold_days"],
                "tax":         act["tax"],
                "exit_reason": act["exit_type"],
            })

        # Apply updated HoldingStates (prev_close updated by SellJob)
        for t, st in ctx.holdings.items():
            if t in self._holdings:
                self._holdings[t] = st

        # Apply buys
        for order in ctx.orders:
            t = order["ticker"]
            price, shares, invest = order["price"], order["shares"], order["invest"]
            self._holdings[t]  = HoldingState(
                entry_price=price, entry_date=ctx.today,
                high_watermark=price, prev_close=price,
            )
            self._pos_shares[t] = shares
            # cash already decremented by SelectionJob (BEAR orders: decrement here)
            if order.get("regime") == "BEAR_defensive":
                self._cash -= invest
            self._last_sell_dates.pop(t, None)
            self._trade_log.append({
                "action": "buy", "ticker": t, "date": today_ts,
                "price": price, "shares": shares, "invest": invest,
            })

        # Update persistent state
        self._cash         = ctx.cash
        self._hwm          = ctx.hwm
        self._regime_state = ctx.regime_state
        for t, d in ctx.last_sell_dates.items():
            self._last_sell_dates[t] = d

        # Equity curve
        if ctx.equity_point:
            self._equity_curve.append(ctx.equity_point)

    # ── Output accessors ──────────────────────────────────────────────────────

    @property
    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._equity_curve).set_index("date") if self._equity_curve else pd.DataFrame()

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)

    # ── Signal closures ───────────────────────────────────────────────────────

    def _make_action_fn(self, today: pd.Timestamp):
        results = self._results
        def action_fn(ticker: str, today_ts: Any) -> str:
            sigs = results.get(ticker, {}).get("oos_signals")
            if sigs is None or today_ts not in sigs.index:
                return "hold"
            v = sigs.loc[today_ts]
            return "buy" if v == 1 else ("sell" if v == -1 else "hold")
        return action_fn

    def _make_score_fn(self, today: pd.Timestamp):
        results = self._results
        def score_fn(ticker: str, today_ts: Any) -> float | None:
            r = results.get(ticker, {})
            raw = r.get("oos_raw_scores")
            if raw is None or today_ts not in raw.index:
                return None
            cal = r.get("score_calibration")
            return float(cal.calibrate(float(raw.loc[today_ts]))) if cal else float(raw.loc[today_ts])
        return score_fn
