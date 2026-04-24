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


# ── Helpers ────────────────────────────────────────────────────────────────────

# Ratio above which a stored high_water_mark is treated as "stale" relative to
# current account value and snapped down. Chosen so that a real 33% drawdown
# (hwm/equity ratio = 1.49) is preserved but the typical stale-seed case
# (hwm=$100k, equity=$10k → ratio 10×) trips the snap.
_HWM_STALE_RATIO = 1.5


def resolve_hwm(stored_hwm: float, account_value: float,
                stale_ratio: float = _HWM_STALE_RATIO) -> tuple[float, bool]:
    """Resolve the effective high_water_mark for the current bar.

    The live-trading DrawdownCircuitTask divides `(hwm - equity) / hwm` and
    compares to `halt_pct`. When `hwm` is stale (e.g. initial-seed $100k
    from a fresh install, actual Alpaca equity a fraction of that), this
    ratio blows up and the drawdown halt latches on every bar — exactly
    the 2026-04-23 "zero orders despite healthy models" bug.

    Rule: if stored_hwm > stale_ratio * account_value, snap to account_value.
    Otherwise ratchet up to max(stored_hwm, account_value) as before.

    Returns (resolved_hwm, was_snapped).
    """
    if account_value > 0 and stored_hwm > stale_ratio * account_value:
        return float(account_value), True
    return float(max(stored_hwm, account_value)), False


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
        use_intraday_prices: bool = False,
    ) -> None:
        self._config              = config
        self._models              = models
        self._broker              = broker
        self._strategy_dir        = strategy_dir
        self._sell_only           = sell_only
        self._use_intraday_prices = use_intraday_prices
        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(config, strategy_dir=strategy_dir)

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
        # Persisted RegimeState across live runs. Without this, each fresh
        # `daily_104.sh` invocation starts countdown=0 → CUSUM re-trips every
        # bar → `transition_window` stays True forever → buys perpetually
        # blocked. (Sim doesn't hit this because state lives in-process.)
        regime_persist  = state.get("regime_state", {}) or {}

        # ── Broker account ───────────────────────────────────────────────────
        account_value = broker.get_account_value()
        try:
            cash = broker.get_cash()
        except Exception:
            cash = account_value

        # Stale-HWM guard (see `resolve_hwm` docstring above). Snaps when
        # stored HWM is wildly above current equity, preserves normal
        # drawdowns otherwise.
        hwm, snapped = resolve_hwm(hwm, account_value)
        if snapped:
            log.warning("Stale HWM: snapped to current equity=%.2f "
                        "(stored HWM was stale; see adapters.runner.resolve_hwm)",
                        account_value)

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
            # entry_dates lookup with **persistent fallback**: if a position
            # is held but missing from entry_dates (e.g. inherited from
            # renquant_103 or manually added), stamp today and persist so
            # hold_days is measured from first-sighting, not from
            # today-minus-today (which made hold_days=0 forever → locked all
            # min_hold_days / rotation gates). Ideally seed from Alpaca's
            # fill timestamp on migration; today is the least-bad fallback.
            if ticker not in entry_dates:
                entry_dates[ticker] = today.isoformat()
            entry_str = entry_dates[ticker]
            try:
                entry_dt = datetime.date.fromisoformat(entry_str)
            except ValueError:
                entry_dt = today
                entry_dates[ticker] = today.isoformat()
            qty_held = float(pos.get("qty", 0))
            holdings[ticker] = HoldingState(
                entry_price    = avg_cost,
                entry_date     = entry_dt,
                high_watermark = hwm_pos,
                sell_streak    = int(sell_streaks.get(ticker, 0)),
                shares         = qty_held,   # broker qty for Kelly top-up sizing
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

        # ── Intraday price overlay (for intraday sell-only checks) ──────────
        if self._use_intraday_prices:
            try:
                from kernel.data import fetch_intraday_bars  # noqa: PLC0415
                ibars = fetch_intraday_bars(
                    list(all_symbols),
                    timeframe="5Min",
                    start=datetime.datetime.combine(
                        today, datetime.datetime.min.time(),
                    ),
                )
                overlaid = 0
                for sym, idf in ibars.items():
                    if idf is None or idf.empty:
                        continue
                    latest_close = float(idf["close"].iloc[-1])
                    prices[sym] = latest_close
                    # Overwrite today's daily bar's close so kernel.exits sees the intraday level
                    if sym in ohlcv and not ohlcv[sym].empty:
                        df = ohlcv[sym]
                        last_day = df.index.max()
                        if last_day.date() == today:
                            df.at[last_day, "close"] = latest_close
                    overlaid += 1
                log.info("Intraday overlay: %d/%d symbols had fresh minute bars",
                         overlaid, len(all_symbols))
            except Exception as exc:
                log.warning("Intraday overlay failed — falling back to daily closes: %s", exc)

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

        ctx = InferenceContext(
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
            regime_state      = RegimeState(
                regime        = regime_persist.get("regime",        "BULL_CALM"),
                confidence    = float(regime_persist.get("confidence",     0.5)),
                in_transition = bool(regime_persist.get("in_transition", False)),
                countdown     = int(regime_persist.get("countdown",          0)),
                cusum_pos     = float(regime_persist.get("cusum_pos",      0.0)),
                cusum_neg     = float(regime_persist.get("cusum_neg",      0.0)),
            ),
            regime_counts     = {r: 0 for r in REGIMES},
            monitor_state     = dict(state.get("monitor_state", {}) or {}),
        )

        # ── Panel scoring prep (optional) ────────────────────────────────────
        panel_cfg = config.get("ranking", {}).get("panel_scoring", {})
        if panel_cfg.get("enabled", False) and not self._sell_only:
            try:
                from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415
                ff, fac = prepare_inference_panel_frames(
                    watchlist=config["watchlist"],
                    ohlcv=ohlcv,
                    ticker_sectors=config.get("sector_map", {}),
                    config=config,
                )
                ctx._panel_feature_frames = ff  # noqa: SLF001
                ctx._panel_factor_frames  = fac  # noqa: SLF001
                log.info("Panel frames prepared: feat=%d  factor=%d",
                         len(ff), len(fac))
            except Exception as exc:
                log.warning("Panel frame prep failed — panel scoring will be skipped: %s",
                            exc)

        return ctx

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
        # Track BUYS as they actually reach the broker vs what the pipeline
        # merely intended. `ctx.orders_placed` = submitted to Alpaca,
        # `ctx.orders_skipped` = blocked locally (with reason). The
        # runner-level ntfy reads these; `ctx.orders` keeps the pipeline
        # intent unchanged for DB / audit.
        if not hasattr(ctx, "orders_placed"):
            ctx.orders_placed = []
        if not hasattr(ctx, "orders_skipped"):
            ctx.orders_skipped = []
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
                        ctx.orders_skipped.append({
                            **order, "skip_reason": "pending_order_exists",
                        })
                        continue
                except Exception:
                    pass

                try:
                    result = broker.place_order(ticker, "BUY", shares)
                except Exception as exc:
                    log.error("BUY failed for %s: %s", ticker, exc)
                    ctx.orders_skipped.append({
                        **order, "skip_reason": f"broker_error:{type(exc).__name__}",
                    })
                    continue
                # Broker accepted — record
                ctx.orders_placed.append(order)

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
        # Snapshot RegimeState (countdown / cusum / in_transition) so the
        # next live invocation resumes mid-cooldown instead of re-tripping
        # CUSUM from scratch. Without this, transition_window=True stays
        # stuck whenever SPY's 20-day window still differs from the 20-day
        # reference — which can last 20+ bars after a genuine regime shift.
        rs = getattr(ctx, "regime_state", None)
        regime_state_out = {
            "regime":        ctx.regime,
            "confidence":    round(ctx.confidence, 4),
            "in_transition": bool(getattr(rs, "in_transition", False)),
            "countdown":     int(getattr(rs, "countdown", 0)),
            "cusum_pos":     float(getattr(rs, "cusum_pos", 0.0)),
            "cusum_neg":     float(getattr(rs, "cusum_neg", 0.0)),
        } if rs is not None else {}
        self._state.update({
            "regime":            ctx.regime,
            "regime_confidence": round(ctx.confidence, 4),
            "high_water_mark":   ctx.hwm,
            "entry_dates":       self._entry_dates,
            "sell_streaks":      self._sell_streaks,
            "last_sell_dates":   self._last_sell_dates_str,
            "position_hwm":      self._position_hwm,
            "regime_state":      regime_state_out,
            # MonitorIdleStreakTask counters — persisted across scheduled runs
            "monitor_state":     dict(getattr(ctx, "monitor_state", {}) or {}),
        })
        state_file = self._strategy_dir / "live_state.json"
        state_file.write_text(json.dumps(self._state, indent=2))
        log.info("State saved → %s", state_file)

        # ── Optional SQLite decision trace ────────────────────────────────
        if self._db is not None:
            from kernel.persistence import (  # noqa: PLC0415
                record_pipeline_run, record_candidate_scores, record_trades,
            )
            # Reconstruct trade events from ctx (live path doesn't keep an
            # in-memory trade list — we synthesise from exits + orders).
            trade_events: list[dict] = []
            for t, sig in ctx.exits:
                hs    = ctx.holdings.get(t)
                price = ctx.prices.get(t, 0.0)
                entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
                trade_events.append({
                    "ticker":      t,
                    "action":      "sell",
                    "price":       price,
                    "exit_reason": sig.exit_type,
                    "pnl_pct":     (price - entry_p) / entry_p if entry_p > 0 else 0.0,
                    "hold_days":   (ctx.today - hs.entry_date).days if hs and hs.entry_date else 0,
                })
            for o in ctx.orders:
                trade_events.append({
                    "ticker":      o.get("ticker"),
                    "action":      "buy",
                    "shares":      o.get("shares"),
                    "price":       o.get("price"),
                    "invest":      o.get("invest"),
                    "target_pct":  o.get("target_pct"),
                    "rank_score":  o.get("rank_score"),
                    "conviction":  o.get("conviction"),
                    "sigma_mult":  o.get("sigma_mult"),
                    "mu":          o.get("mu"),
                    "sigma":       o.get("sigma"),
                })
            run_id = record_pipeline_run(
                self._db,
                run_type        = "live",
                run_date        = ctx.today,
                strategy        = str(self._config.get("model_name", "")),
                regime          = ctx.regime,
                confidence      = float(ctx.confidence) if ctx.confidence is not None else None,
                portfolio_value = float(ctx.portfolio_value) if ctx.portfolio_value else None,
                cash            = float(ctx.cash) if ctx.cash is not None else None,
                n_candidates    = len(ctx.candidates),
                n_exits         = len(ctx.exits),
                n_rotations     = len(ctx.rotations),
                n_buys          = len(ctx.orders),
            )
            selected_tickers = {o["ticker"] for o in ctx.orders}
            record_candidate_scores(
                self._db, run_id, ctx.candidates, ctx.holdings,
                selected_tickers=selected_tickers,
            )
            record_trades(self._db, run_id, trade_events)

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, ctx, record: dict) -> None:
        import datetime as _dt
        strategy_name = self._config.get("model_name", "renquant_104")
        repo_root     = self._strategy_dir.parent.parent
        log_dir       = repo_root / "live" / "logs" / strategy_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_dt.datetime.now().strftime('%Y-%m-%d')}.json"
        entries = json.loads(log_file.read_text()) if log_file.exists() else []
        record["timestamp"] = _dt.datetime.now().isoformat()
        entries.append(record)
        log_file.write_text(json.dumps(entries, indent=2, default=str))
