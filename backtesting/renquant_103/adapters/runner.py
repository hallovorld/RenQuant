"""RunnerAdapter — wraps broker + OHLCV + live_state into InferenceContext.

Replaces SignalJob + ExecutionJob in the live runner's _run_once_multi_pipeline.
DataJob still runs first to populate ohlcv and artifacts.

Usage in _run_once_multi_pipeline (after DataJob populates ctx):

    adapter  = RunnerAdapter(config, models, broker, strategy_dir)
    inf_ctx  = adapter.make_context(ctx.ohlcv, ctx.gmm_artifact,
                                    ctx.corr_matrix, ctx.earnings_cal)
    pipeline = InferencePipeline()   # or sell-only Pipeline([RegimeJob, DrawdownJob, SellJob])
    pipeline.run(inf_ctx)
    adapter.commit(inf_ctx)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from kernel.exits     import HoldingState
from kernel.regime    import RegimeState
from kernel.models    import score_artifact
from kernel.indicators import build_feature_frame
from kernel.pipeline  import InferenceContext
from kernel.pipeline.jobs import (
    RegimeJob, DrawdownJob, SellJob, BuyGatesJob,
    CandidateJob, RankingJob, SelectionJob,
)
from kernel.pipeline.base import Pipeline

log = logging.getLogger("adapters.runner")


class InferencePipeline(Pipeline):
    """Standard 7-job inference pipeline for the live runner."""
    def __init__(self) -> None:
        super().__init__([
            RegimeJob(), DrawdownJob(), SellJob(),
            BuyGatesJob(), CandidateJob(), RankingJob(), SelectionJob(),
        ])


class SellOnlyPipeline(Pipeline):
    """Regime + drawdown + sell exits only (for market-open intraday checks)."""
    def __init__(self) -> None:
        super().__init__([RegimeJob(), DrawdownJob(), SellJob()])


class RunnerAdapter:
    """Manages persistent live state and translates broker context → InferenceContext."""

    def __init__(
        self,
        config: dict,
        models: dict[str, Any],
        broker: Any,
        strategy_dir: Path,
        trade_log_path: Path | None = None,
    ) -> None:
        self._config        = config
        self._models        = models
        self._broker        = broker
        self._strategy_dir  = strategy_dir
        self._trade_log_path = trade_log_path
        config["_exportable"] = list(models.keys())

    # ── Per-run API ───────────────────────────────────────────────────────────

    def make_context(
        self,
        ohlcv: dict,
        gmm_artifact: dict | None,
        corr_matrix: dict,
        earnings_cal: dict,
    ) -> InferenceContext:
        config = self._config
        state  = self._load_state()
        today  = date.today()

        # Broker account state
        try:
            account_value = float(self._broker.get_account_value())
            cash          = float(self._broker.get_cash())
            all_positions = {p["symbol"]: p for p in self._broker.get_all_positions()}
        except Exception as exc:
            log.warning("Broker state unavailable: %s", exc)
            account_value, cash, all_positions = 0.0, 0.0, {}

        # Build HoldingState from broker positions + persisted state
        entry_dates   = state.get("entry_dates", {})
        position_hwm  = state.get("position_hwm", {})
        sell_streaks  = state.get("sell_streaks", {})
        holdings: dict[str, HoldingState] = {}
        pos_shares: dict[str, float] = {}
        for ticker, pos in all_positions.items():
            qty = float(pos.get("qty", 0))
            if qty <= 0:
                continue
            entry_d_str = entry_dates.get(ticker)
            entry_d = date.fromisoformat(entry_d_str) if entry_d_str else today
            avg_cost = float(pos.get("avg_cost", pos.get("current_price", 0)))
            cur_price = float(pos.get("current_price", avg_cost))
            holdings[ticker] = HoldingState(
                entry_price    = avg_cost,
                entry_date     = entry_d,
                high_watermark = float(position_hwm.get(ticker, cur_price)),
                sell_streak    = int(sell_streaks.get(ticker, 0)),
                prev_close     = cur_price,
            )
            pos_shares[ticker] = qty

        # last_sell_dates (str → date)
        last_sell_dates: dict[str, date] = {}
        for t, d_str in state.get("last_sell_dates", {}).items():
            try:
                last_sell_dates[t] = date.fromisoformat(d_str)
            except (ValueError, TypeError):
                pass

        # SPY returns for regime detection
        spy_df = ohlcv.get("SPY")
        if spy_df is not None and len(spy_df) > 1:
            spy_returns = spy_df["close"].pct_change().dropna().values[-100:].astype(float)
        else:
            spy_returns = np.zeros(0, dtype=float)

        # Rebuild RegimeState from saved state
        regime_state = RegimeState()
        saved_regime = state.get("regime")
        if saved_regime:
            regime_state.regime     = saved_regime
            regime_state.confidence = float(state.get("regime_confidence", 0.5))
            regime_state.cusum_countdown = int(state.get("transition_countdown", 0))

        action_fn = self._make_action_fn(ohlcv, spy_df)
        score_fn  = self._make_score_fn(ohlcv, spy_df)

        return InferenceContext(
            today           = today,
            ohlcv           = ohlcv,
            spy_returns     = spy_returns,
            prev_closes     = {},
            holdings        = holdings,
            pos_shares      = pos_shares,
            cash            = cash,
            portfolio_value = account_value,
            action_fn       = action_fn,
            score_fn        = score_fn,
            gmm_artifact    = gmm_artifact,
            corr_dict       = corr_matrix or {},
            earnings_cal    = earnings_cal or {},
            config          = config,
            hwm             = float(state.get("high_water_mark", account_value)),
            regime_state    = regime_state,
            last_sell_dates = last_sell_dates,
        )

    def commit(self, ctx: InferenceContext) -> None:
        state = self._load_state()

        # ── Sells ─────────────────────────────────────────────────────────────
        for act in ctx.exit_actions:
            ticker = act["ticker"]
            qty    = act["shares"]
            try:
                result = self._broker.place_order(ticker, "SELL", abs(qty))
                log.info("SELL %s  exit=%s  pnl=%.1f%%  hold=%dd",
                         ticker, act["exit_type"], act["pnl_pct"] * 100, act["hold_days"])
                self._log_trade({
                    "symbol": ticker, "signal": act["exit_type"],
                    "sell_price": act["price"], "qty": qty,
                    "pnl_pct": round(act["pnl_pct"], 4),
                    "hold_days": act["hold_days"],
                    "tax": round(act["tax"], 2), "order": result,
                })
            except Exception as exc:
                log.error("SELL failed %s: %s", ticker, exc)
                continue
            state.setdefault("last_sell_dates", {})[ticker] = ctx.today.isoformat()
            state.setdefault("entry_dates", {}).pop(ticker, None)
            state.setdefault("position_hwm", {}).pop(ticker, None)
            state.setdefault("sell_streaks", {}).pop(ticker, None)

        # ── Persist updated sell streaks for held positions ────────────────────
        exit_tickers = {a["ticker"] for a in ctx.exit_actions}
        for t, hs in ctx.holdings.items():
            if t not in exit_tickers:
                state.setdefault("sell_streaks", {})[t] = hs.sell_streak

        # ── Buys ──────────────────────────────────────────────────────────────
        for order in ctx.orders:
            ticker = order["ticker"]
            shares = int(order["shares"])
            if shares <= 0:
                continue
            try:
                result = self._broker.place_order(ticker, "BUY", shares)
                log.info("BUY  %s  %d shares  regime=%s  rank=%.3f",
                         ticker, shares, order.get("regime"), order.get("rank_score", 0))
                self._log_trade({
                    "symbol": ticker, "signal": "buy",
                    "price": order["price"], "shares": shares,
                    "invest": order["invest"], "regime": order.get("regime"),
                    "rank_score": order.get("rank_score", 0),
                    "rs_score":   order.get("rs_score",   0),
                    "order": result,
                })
            except Exception as exc:
                log.error("BUY failed %s: %s", ticker, exc)
                continue
            state.setdefault("entry_dates", {})[ticker]   = ctx.today.isoformat()
            state.setdefault("position_hwm", {})[ticker]  = order["price"]

        # ── Update persisted state ────────────────────────────────────────────
        state["regime"]               = ctx.regime
        state["regime_confidence"]    = round(ctx.regime_confidence, 4)
        state["high_water_mark"]      = ctx.hwm
        state["transition_countdown"] = getattr(ctx.regime_state, "cusum_countdown", 0)
        for t, d in ctx.last_sell_dates.items():
            state.setdefault("last_sell_dates", {})[t] = (
                d.isoformat() if hasattr(d, "isoformat") else str(d)
            )
        self._save_state(state)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_action_fn(self, ohlcv, spy_df):
        models = self._models
        spec   = self._config.get("indicator_spec", {})

        def action_fn(ticker: str, today_ts) -> str:
            if ticker not in models:
                return "hold"
            stock_df = ohlcv.get(ticker)
            if stock_df is None or len(stock_df) < 40 or spy_df is None:
                return "hold"
            features = build_feature_frame(stock_df, spy_df, spec)
            if features is None or features.empty:
                return "hold"
            try:
                sr = score_artifact(models[ticker], features.iloc[-1])
                return sr.signal
            except Exception:
                return "hold"

        return action_fn

    def _make_score_fn(self, ohlcv, spy_df):
        models = self._models
        spec   = self._config.get("indicator_spec", {})

        def score_fn(ticker: str, today_ts) -> float | None:
            if ticker not in models:
                return None
            stock_df = ohlcv.get(ticker)
            if stock_df is None or len(stock_df) < 40 or spy_df is None:
                return None
            features = build_feature_frame(stock_df, spy_df, spec)
            if features is None or features.empty:
                return None
            try:
                sr = score_artifact(models[ticker], features.iloc[-1])
                return sr.rank_score
            except Exception:
                return None

        return score_fn

    def _load_state(self) -> dict:
        state_file = self._strategy_dir / "live_state.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text())
            except Exception:
                pass
        return {}

    def _save_state(self, state: dict) -> None:
        state_file = self._strategy_dir / "live_state.json"
        state_file.write_text(json.dumps(state, indent=2))

    def _log_trade(self, entry: dict) -> None:
        if not self._trade_log_path:
            return
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        trades: list = []
        if self._trade_log_path.exists():
            try:
                trades = json.loads(self._trade_log_path.read_text())
            except Exception:
                pass
        trades.append(entry)
        self._trade_log_path.write_text(json.dumps(trades, indent=2))
