"""LeanAdapter — wraps QCAlgorithm state into InferenceContext.

LEAN-specific: imports AlgorithmImports and uses History/Portfolio APIs.
Cannot be tested outside LEAN Docker.

Usage in QCAlgorithm.Initialize:
    self._adapter  = LeanAdapter(self)
    self._pipeline = InferencePipeline()

Usage in QCAlgorithm.OnData:
    self._update_spy_buffer(data)          # must run before make_context
    ctx = self._adapter.make_context(data)
    self._pipeline.run(ctx)
    self._adapter.commit(ctx)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from AlgorithmImports import Resolution  # noqa: F401 — LEAN Docker only
except ImportError:
    Resolution = None  # type: ignore  # allows import outside Docker for static analysis

from kernel.models import score_artifact
from kernel.indicators import build_feature_frame
from kernel.pipeline import InferenceContext
from kernel.pipeline.jobs import (
    RegimeJob, DrawdownJob, SellJob, BuyGatesJob,
    CandidateJob, RankingJob, SelectionJob,
)
from kernel.pipeline.base import Pipeline


class InferencePipeline(Pipeline):
    """Standard 7-job inference pipeline — LEAN variant."""
    def __init__(self) -> None:
        super().__init__([
            RegimeJob(), DrawdownJob(), SellJob(),
            BuyGatesJob(), CandidateJob(), RankingJob(), SelectionJob(),
        ])


class LeanAdapter:
    """Translates QCAlgorithm state → InferenceContext → applies ctx back to LEAN.

    One instance per strategy, created in Initialize.  Called once per bar:
        ctx = adapter.make_context(data)
        pipeline.run(ctx)
        adapter.commit(ctx)
    """

    def __init__(self, algo) -> None:
        self._algo = algo

    # ── Per-bar API ───────────────────────────────────────────────────────────

    def make_context(self, data) -> InferenceContext:
        algo     = self._algo
        today    = algo.Time.date()

        # Batch History — one call for all symbols, reused for features + RS
        ohlcv = self._batch_history()

        # action_fn and score_fn use the pre-built ohlcv (avoids per-ticker History calls)
        spy_df    = ohlcv.get("SPY")
        action_fn = self._make_action_fn(spy_df, ohlcv)
        score_fn  = self._make_score_fn(spy_df, ohlcv)

        return InferenceContext(
            today           = today,
            ohlcv           = ohlcv,
            spy_returns     = np.array(algo._spy_returns, dtype=float),
            prev_closes     = dict(algo._prev_closes),
            holdings        = dict(algo._holdings),
            pos_shares      = {
                t: float(algo.Portfolio[algo.symbols[t]].Quantity)
                for t in algo._holdings if t in algo.symbols
            },
            cash            = float(algo.Portfolio.Cash),
            portfolio_value = float(algo.Portfolio.TotalPortfolioValue),
            action_fn       = action_fn,
            score_fn        = score_fn,
            gmm_artifact    = algo._gmm,
            corr_dict       = algo._corr or {},
            earnings_cal    = algo._earnings or {},
            config          = algo._config,
            hwm             = algo._hwm,
            regime_state    = algo._regime_state,
            last_sell_dates = dict(algo._last_sell_dates),
        )

    def commit(self, ctx: InferenceContext) -> None:
        algo = self._algo

        # Apply exits
        for act in ctx.exit_actions:
            ticker = act["ticker"]
            algo._execute_sell(ticker, act["exit_type"])
            # Telemetry by exit type
            et = act["exit_type"]
            if et == "trailing_stop":    algo._trail_exits += 1
            elif et == "stop_loss":       algo._stop_exits += 1
            elif et == "single_day_loss": algo._sdl_exits  += 1

        # Persist updated HoldingStates (streak / HWM / prev_close updated by SellJob)
        for t, hs in ctx.holdings.items():
            if t in algo._holdings:
                algo._holdings[t] = hs

        # Apply buy orders
        for order in ctx.orders:
            t = order["ticker"]
            if order.get("regime") == "BEAR_defensive":
                override_pct = 0.15
            else:
                override_pct = None
            algo._execute_buy(
                t,
                order.get("rank_score", 0.0),
                order.get("rs_score", 0.0),
                order.get("detail", ""),
                ctx.regime,
                ctx.regime_confidence,
                ctx.regime_params,
                override_pct=override_pct,
            )

        # Update persistent state
        algo._hwm          = ctx.hwm
        algo._skip_buys    = ctx.skip_buys
        algo._regime_state = ctx.regime_state
        algo._regime_counts[ctx.regime] = algo._regime_counts.get(ctx.regime, 0) + 1
        for t, d in ctx.last_sell_dates.items():
            algo._last_sell_dates[t] = d

        # Refresh prev_closes from current bar prices
        for t, sym in algo.symbols.items():
            if algo.Securities.ContainsKey(sym):
                algo._prev_closes[t] = float(algo.Securities[sym].Price)
        if algo.Securities.ContainsKey(algo._spy_sym):
            algo._prev_closes["SPY"] = float(algo.Securities[algo._spy_sym].Price)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _batch_history(self) -> dict[str, pd.DataFrame]:
        """One History() call for all symbols → per-ticker DataFrames."""
        algo = self._algo
        all_syms = (
            list(algo.symbols.values())
            + list(algo._sector_etf_symbols.values())
            + [algo._spy_sym]
        )
        try:
            hist = algo.History(all_syms, 62, Resolution.Daily)
        except Exception:
            return {}

        sym_to_ticker = {v: k for k, v in algo.symbols.items()}
        sym_to_ticker.update({v: k for k, v in algo._sector_etf_symbols.items()})
        sym_to_ticker[algo._spy_sym] = "SPY"

        ohlcv: dict[str, pd.DataFrame] = {}
        for sym in all_syms:
            ticker = sym_to_ticker.get(sym)
            if ticker is None:
                continue
            try:
                df = hist.loc[sym].copy()
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 20:
                    ohlcv[ticker] = df
            except (KeyError, Exception):
                pass
        return ohlcv

    def _make_action_fn(self, spy_df, ohlcv):
        algo = self._algo
        vol_window = int(algo._config.get("regime", {}).get("vol_realized_window", 20))
        spec = algo._config.get("indicator_spec", {})

        def action_fn(ticker: str, today_ts) -> str:
            if ticker not in algo._models:
                return "hold"
            stock_df = ohlcv.get(ticker)
            if stock_df is None or len(stock_df) < 40 or spy_df is None:
                return "hold"
            features = build_feature_frame(stock_df, spy_df, spec, vol_window)
            if features is None or features.empty:
                return "hold"
            qty = int(algo.Portfolio[algo.symbols[ticker]].Quantity) if ticker in algo.symbols else 0
            sr = score_artifact(algo._models[ticker], features.iloc[-1], qty)
            return sr.signal

        return action_fn

    def _make_score_fn(self, spy_df, ohlcv):
        algo = self._algo
        vol_window = int(algo._config.get("regime", {}).get("vol_realized_window", 20))
        spec = algo._config.get("indicator_spec", {})

        def score_fn(ticker: str, today_ts) -> float | None:
            if ticker not in algo._models:
                return None
            stock_df = ohlcv.get(ticker)
            if stock_df is None or len(stock_df) < 40 or spy_df is None:
                return None
            features = build_feature_frame(stock_df, spy_df, spec, vol_window)
            if features is None or features.empty:
                return None
            qty = int(algo.Portfolio[algo.symbols[ticker]].Quantity) if ticker in algo.symbols else 0
            sr = score_artifact(algo._models[ticker], features.iloc[-1], qty)
            return sr.rank_score

        return score_fn
