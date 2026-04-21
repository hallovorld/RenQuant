"""RunnerAdapter — bridges live broker state → InferenceContext → order execution.

Can import kernel/ and common/ (runs on host, not in LEAN Docker).
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("adapters.runner")


class RunnerAdapter:
    """Translate between the live broker/state and InferenceContext.

    Usage::

        adapter = RunnerAdapter(config, models, broker, strategy_dir, sell_only)
        ctx = adapter.make_context()
        InferencePipeline().run(ctx)
        adapter.commit(ctx)
    """

    def __init__(
        self,
        config: dict,
        models: dict,
        broker: Any,
        strategy_dir: Path,
        sell_only: bool = False,
    ) -> None:
        self._config       = config
        self._models       = models
        self._broker       = broker
        self._strategy_dir = strategy_dir
        self._sell_only    = sell_only

    # ── make_context ───────────────────────────────────────────────────────────

    def make_context(self):  # noqa: ANN201
        """Build InferenceContext from broker, parquet cache, and live_state.json."""
        from kernel.pipeline.context import InferenceContext  # noqa: PLC0415
        from kernel.regime import RegimeState                  # noqa: PLC0415
        from kernel.config import REGIMES                      # noqa: PLC0415

        config  = self._config
        today   = datetime.date.today()
        broker  = self._broker

        # ── Load persisted live state ────────────────────────────────────────
        state_file = self._strategy_dir / "live_state.json"
        state      = json.loads(state_file.read_text()) if state_file.exists() else {}

        entry_dates     = state.get("entry_dates",     {})
        sell_streaks    = state.get("sell_streaks",    {})
        last_sell_dates = state.get("last_sell_dates", {})
        position_hwm    = state.get("position_hwm",    {})
        hwm             = float(state.get("high_water_mark", 0.0))

        # ── Broker account ───────────────────────────────────────────────────
        account_value = broker.get_account_value()
        try:
            cash = broker.get_cash()
        except Exception:
            cash = account_value
        hwm = max(hwm, account_value)

        try:
            all_pos = broker.get_all_positions()
        except Exception:
            all_pos = []
        positions_cache = {p["symbol"]: p for p in all_pos}

        # ── Holdings from live state + broker positions ─────────────────────
        from kernel.exits import HoldingState  # noqa: PLC0415

        held_set  = set(s for s in config["watchlist"]
                        if float(positions_cache.get(s, {}).get("qty", 0)) > 0)
        holdings: dict[str, HoldingState] = {}
        for ticker in held_set:
            pos     = positions_cache.get(ticker, {})
            avg_cost = float(pos.get("avg_entry_price", 0.0))
            hwm_pos  = float(position_hwm.get(ticker, avg_cost))
            entry_str = entry_dates.get(ticker, today.isoformat())
            try:
                entry_dt = datetime.date.fromisoformat(entry_str)
            except ValueError:
                entry_dt = today
            holdings[ticker] = HoldingState(
                entry_price    = avg_cost,
                entry_date     = entry_dt,
                high_watermark = hwm_pos,
                sell_streak    = int(sell_streaks.get(ticker, 0)),
            )

        # ── Current prices from broker positions ────────────────────────────
        prices: dict[str, float] = {}
        for ticker, pos in positions_cache.items():
            qty = float(pos.get("qty", 0))
            mkt = float(pos.get("market_value", 0))
            if qty > 0 and mkt > 0:
                prices[ticker] = mkt / qty

        # ── OHLCV from parquet cache ─────────────────────────────────────────
        from kernel.data import fetch_ohlcv  # noqa: PLC0415

        watchlist   = config["watchlist"]
        benchmark   = config.get("benchmark", "SPY")
        sector_etfs = set(config.get("sector_etf_map", {}).values())
        all_symbols = list(dict.fromkeys(watchlist + [benchmark] + sorted(sector_etfs)))

        ohlcv: dict[str, Any] = {}
        for sym in all_symbols:
            try:
                df = fetch_ohlcv(sym)
                if not df.empty:
                    ohlcv[sym] = df
                    # Fill prices from OHLCV last close if broker didn't supply
                    if sym not in prices:
                        prices[sym] = float(df["close"].iloc[-1])
            except Exception as exc:
                log.warning("OHLCV fetch failed for %s: %s", sym, exc)

        spy_df = ohlcv.get(benchmark)
        spy_returns: list[float] = []
        if spy_df is not None:
            spy_close   = spy_df["close"].astype(float)
            spy_returns = list(spy_close.pct_change().dropna().values[-100:])

        # ── Load artifacts ───────────────────────────────────────────────────
        from kernel.regime import load_gmm_artifact  # noqa: PLC0415
        from kernel.config import artifact_path       # noqa: PLC0415

        regime_cfg = config.get("regime", {})
        artifacts_dir = self._strategy_dir / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = self._strategy_dir

        gmm_path  = artifacts_dir / regime_cfg.get("gmm_artifact", "spy-gmm-regime.json")
        gmm       = load_gmm_artifact(gmm_path)

        corr_path = artifacts_dir / regime_cfg.get("correlation_artifact", "watchlist-correlation.json")
        corr      = json.loads(corr_path.read_text()) if corr_path.exists() else None

        earn_path = artifacts_dir / "earnings-calendar.json"
        earnings  = json.loads(earn_path.read_text()) if earn_path.exists() else None

        # Convert last_sell_dates strings to date objects for kernel.selection guards
        last_sells_d: dict[str, datetime.date | None] = {}
        for sym, d_str in last_sell_dates.items():
            try:
                last_sells_d[sym] = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                last_sells_d[sym] = None

        # ── Persisted live state on context for commit() ─────────────────────
        self._state          = state
        self._entry_dates    = entry_dates
        self._sell_streaks   = sell_streaks
        self._last_sell_dates_str = last_sell_dates
        self._position_hwm   = position_hwm
        self._positions_cache = positions_cache
        self._account_value  = account_value

        return InferenceContext(
            config            = config,
            today             = today,
            ohlcv             = ohlcv,
            spy_returns       = spy_returns,
            models            = self._models,
            gmm               = gmm,
            corr_matrix       = corr,
            earnings_calendar = earnings,
            holdings          = holdings,
            last_sell_dates   = last_sells_d,
            portfolio_value   = account_value,
            cash              = cash,
            prices            = prices,
            hwm               = hwm,
            skip_buys         = False,
            regime_state      = RegimeState(),
            regime_counts     = {r: 0 for r in REGIMES},
        )

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx) -> None:  # noqa: ANN001
        """Apply pipeline outputs: execute broker orders, update live_state.json."""
        broker        = self._broker
        today_str     = ctx.today.isoformat()
        pos_cache     = self._positions_cache

        # ── Apply exits ──────────────────────────────────────────────────────
        for ticker, sig in ctx.exits:
            pos = pos_cache.get(ticker, {})
            qty = float(pos.get("qty", 0))
            if qty <= 0:
                continue
            try:
                result = broker.place_order(ticker, "SELL", abs(qty))
            except Exception as exc:
                log.error("SELL failed for %s: %s", ticker, exc)
                continue

            price  = ctx.prices.get(ticker, 0.0)
            log.info("SELL  %s  [%s]  %.0f shares @ %.2f  %s",
                     ticker, sig.exit_type, qty, price, sig.reason)

            self._last_sell_dates_str[ticker] = today_str
            self._entry_dates.pop(ticker, None)
            self._sell_streaks.pop(ticker, None)
            self._position_hwm.pop(ticker, None)
            self._log_trade(ctx, {
                "action":    "SELL",
                "symbol":    ticker,
                "exit_type": sig.exit_type,
                "reason":    sig.reason,
                "price":     price,
                "qty":       qty,
            })

        # ── Apply buys ───────────────────────────────────────────────────────
        if not self._sell_only:
            for order in ctx.orders:
                ticker = order["ticker"]
                shares = order["shares"]
                price  = order["price"]
                # Duplicate-order guard
                try:
                    pending = broker.get_open_orders()
                    if ticker in pending:
                        log.info("BUY skipped: pending order exists for %s", ticker)
                        continue
                except Exception:
                    pass

                try:
                    result = broker.place_order(ticker, "BUY", shares)
                except Exception as exc:
                    log.error("BUY failed for %s: %s", ticker, exc)
                    continue

                invest = shares * price
                log.info("BUY  %s  %d shares @ %.2f  invest=$%.0f", ticker, shares, price, invest)

                self._entry_dates[ticker]       = today_str
                self._sell_streaks.pop(ticker, None)
                self._last_sell_dates_str.pop(ticker, None)
                self._position_hwm[ticker]      = price
                self._log_trade(ctx, {
                    "action":     "BUY",
                    "symbol":     ticker,
                    "shares":     shares,
                    "price":      price,
                    "invest":     invest,
                    "rank_score": order["rank_score"],
                    "rs_score":   order["rs_score"],
                    "regime":     order["regime"],
                })

        # ── Persist updated sell streaks from SellJob ─────────────────────
        for ticker, hs in ctx.holdings.items():
            self._sell_streaks[ticker] = hs.sell_streak
            if ticker in ctx.prices:
                self._position_hwm[ticker] = max(
                    float(self._position_hwm.get(ticker, 0)),
                    ctx.prices[ticker],
                )

        # ── Save live_state.json ──────────────────────────────────────────
        self._state.update({
            "regime":            ctx.regime,
            "regime_confidence": round(ctx.confidence, 4),
            "high_water_mark":   ctx.hwm,
            "entry_dates":       self._entry_dates,
            "sell_streaks":      self._sell_streaks,
            "last_sell_dates":   self._last_sell_dates_str,
            "position_hwm":      self._position_hwm,
        })
        state_file = self._strategy_dir / "live_state.json"
        state_file.write_text(json.dumps(self._state, indent=2))
        log.info("State saved → %s", state_file)

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, ctx, record: dict) -> None:
        import datetime as _dt
        strategy_name = self._config.get("model_name", "renquant_103")
        repo_root     = self._strategy_dir.parent.parent
        log_dir       = repo_root / "live" / "logs" / strategy_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_dt.datetime.now().strftime('%Y-%m-%d')}.json"
        entries = json.loads(log_file.read_text()) if log_file.exists() else []
        record["timestamp"] = _dt.datetime.now().isoformat()
        entries.append(record)
        log_file.write_text(json.dumps(entries, indent=2, default=str))
