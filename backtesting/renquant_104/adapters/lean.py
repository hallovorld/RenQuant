"""LeanAdapter — bridges LEAN QCAlgorithm state → InferenceContext → LEAN actions.

LEAN-safe: AlgorithmImports imported only at runtime, guarded by try/except for
static analysis.  No common/ imports.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd

try:
    from AlgorithmImports import Resolution  # type: ignore[import]  # noqa: F401
except ImportError:
    pass  # running outside LEAN Docker (static analysis / tests)

from kernel.pipeline.context import InferenceContext
from kernel.exits import HoldingState
from kernel.portfolio import compute_trade_tax

log = logging.getLogger("adapters.lean")

# Baseline indicator computation only needs ~60 bars; panel-LTR
# neutralization + factor building needs ≥504 bars of history.
_INDICATOR_LOOKBACK = 60
_PANEL_LOOKBACK     = 520


class LeanAdapter:
    """Translate between LEAN API and InferenceContext.

    Usage inside QCAlgorithm::

        self._adapter = LeanAdapter(self)
        ctx = self._adapter.make_context(data)
        self._pipeline.run(ctx)
        self._adapter.commit(ctx)
    """

    def __init__(self, algo: Any) -> None:
        self._algo = algo
        # Audit P-3 (2026-04-24): cache panel feature/factor frames so we
        # only rebuild on a real History-buffer roll-forward, not every
        # bar. Pre-fix, `prepare_inference_panel_frames` was called per
        # OnData → 99 tickers × N bars × full feature pipeline = hours of
        # wasted compute per backtest. Cache is invalidated when the SPY
        # buffer extends past the last cached date.
        self._panel_cache_last_date: "pd.Timestamp | None" = None
        self._panel_cache_ff: "dict | None"  = None
        self._panel_cache_fac: "dict | None" = None

    # ── make_context ───────────────────────────────────────────────────────────

    def make_context(self, data: Any) -> InferenceContext:
        """Build InferenceContext from current LEAN state and *data* slice."""
        algo   = self._algo
        config = algo._config
        today  = algo.Time.date()

        if "_strategy_dir" not in config:
            config["_strategy_dir"] = str(algo._strategy_dir)

        panel_cfg  = config.get("ranking", {}).get("panel_scoring", {})
        panel_on   = bool(panel_cfg.get("enabled", False))
        lookback   = _PANEL_LOOKBACK if panel_on else _INDICATOR_LOOKBACK

        # ── SPY returns buffer ───────────────────────────────────────────────
        spy_sym   = algo._spy_sym
        prev      = algo._prev_closes.get("SPY")
        spy_returns = list(algo._spy_returns)  # copy so pipeline mutations don't alias

        if data.ContainsKey(spy_sym):
            spy_close = float(data[spy_sym].Close)
            if prev and prev > 0:
                spy_returns.append((spy_close - prev) / prev)
            if len(spy_returns) > 100:
                spy_returns = spy_returns[-100:]

        # ── OHLCV via batch History calls ────────────────────────────────────
        ohlcv: dict[str, Any] = {}

        all_tickers = (
            list(algo._models.keys())
            + list(algo._sector_etf_symbols.keys())
            + ["SPY"]
        )
        # Remove duplicates, preserve order
        seen: set[str] = set()
        unique_tickers: list[str] = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)

        spy_df = _get_spy_df(algo, lookback)
        if spy_df is not None:
            ohlcv["SPY"] = spy_df

        # Per-ticker OHLCV (models + ETFs)
        for ticker in unique_tickers:
            if ticker == "SPY":
                continue
            sym = algo.symbols.get(ticker) or algo._sector_etf_symbols.get(ticker)
            if sym is None:
                continue
            try:
                h = algo.History(sym, lookback, Resolution.Daily)
                if not h.empty:
                    ohlcv[ticker] = h.loc[sym].copy()
            except Exception:
                pass

        # ── Portfolio state ──────────────────────────────────────────────────
        prices: dict[str, float] = {}
        for ticker, hs in algo._holdings.items():
            sym = algo.symbols.get(ticker)
            if sym and data.ContainsKey(sym):
                prices[ticker] = float(data[sym].Close)
            elif sym:
                prices[ticker] = float(algo.Securities[sym].Price)

        pv   = float(algo.Portfolio.TotalPortfolioValue)
        cash = float(algo.Portfolio.Cash)

        # ── Prices for ETFs (used by CandidateJob RS scoring) ─────────────
        for ticker, sym in algo._sector_etf_symbols.items():
            if data.ContainsKey(sym):
                prices[ticker] = float(data[sym].Close)

        ctx = InferenceContext(
            config           = config,
            today            = today,
            ohlcv            = ohlcv,
            spy_returns      = spy_returns,
            models           = algo._models,
            gmm              = algo._gmm,
            corr_matrix      = algo._corr,
            earnings_calendar = algo._earnings,
            holdings         = dict(algo._holdings),  # shallow copy; jobs mutate in place
            last_sell_dates  = algo._last_sell_dates,
            portfolio_value  = pv,
            cash             = cash,
            prices           = prices,
            hwm              = algo._hwm,
            skip_buys        = algo._skip_buys,
            regime_state     = algo._regime_state,
            regime_counts    = algo._regime_counts,
        )

        # ── Panel scoring prep ───────────────────────────────────────────────
        # Audit P-3: cache panel frames between bars when the underlying
        # SPY buffer hasn't extended. The full panel pipeline is the
        # heaviest call in OnData (~99 tickers × N feature steps), and
        # day-over-day it produces frames that differ only in their
        # trailing row — not worth a full rebuild every bar.
        if panel_on:
            cache_age_days = int(config.get("panel_cache_age_days", 1))
            need_rebuild = (
                self._panel_cache_ff is None
                or self._panel_cache_last_date is None
                or (pd.Timestamp(today) - self._panel_cache_last_date).days
                   >= cache_age_days
            )
            if need_rebuild:
                try:
                    from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415
                    ff, fac, macro = prepare_inference_panel_frames(
                        watchlist=config["watchlist"],
                        ohlcv=ohlcv,
                        ticker_sectors=config.get("sector_map", {}),
                        config=config,
                    )
                    self._panel_cache_ff = ff
                    self._panel_cache_fac = fac
                    self._panel_cache_macro = macro   # Bug #25
                    self._panel_cache_last_date = pd.Timestamp(today)
                except Exception as exc:
                    log.warning("Panel frame prep failed — panel scoring disabled: %s", exc)
            ctx._panel_feature_frames = self._panel_cache_ff   # noqa: SLF001
            ctx._panel_factor_frames  = self._panel_cache_fac  # noqa: SLF001
            ctx._panel_macro_frame    = getattr(self, "_panel_cache_macro", None)  # noqa: SLF001
        return ctx

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx: InferenceContext) -> None:
        """Apply pipeline outputs: execute exits, place buys, persist state."""
        algo   = self._algo
        config = algo._config

        tax_short      = algo._tax_short
        tax_long       = algo._tax_long
        tax_thresh     = algo._tax_thresh_days

        # ── Apply exits ──────────────────────────────────────────────────────
        # 2026-04-24 partial-sell support: when sig.quantity is set and
        # < current holding, place a market sell for that quantity instead
        # of Liquidate (which always closes the entire position).
        # full_exits tracks tickers that we should actually pop from
        # algo._holdings + stamp last_sell_dates for wash-sale.
        full_exits: set[str] = set()
        for ticker, sig in ctx.exits:
            hs        = ctx.holdings.get(ticker)
            sym       = algo.symbols[ticker]
            gross_pnl = float(algo.Portfolio[sym].UnrealizedProfit)
            days_held = (ctx.today - hs.entry_date).days if hs else 0
            is_lt     = days_held >= tax_thresh

            req_qty = getattr(sig, "quantity", None)
            holding_qty = float(algo.Portfolio[sym].Quantity)
            is_partial = (req_qty is not None and 0 < req_qty < holding_qty)

            if is_partial:
                # Partial sell: pro-rate tax to fraction sold, keep position.
                frac = float(req_qty) / max(holding_qty, 1.0)
                tax  = compute_trade_tax(
                    gross_pnl * frac, days_held, tax_short, tax_long, tax_thresh,
                )
            else:
                tax = compute_trade_tax(
                    gross_pnl, days_held, tax_short, tax_long, tax_thresh,
                )

            algo._total_tax     += tax
            algo._executed_sells += 1
            if is_lt:
                algo._lt_trades += 1
            else:
                algo._st_trades += 1

            # Exit type counters
            if sig.exit_type == "trailing_stop":
                algo._trail_exits += 1
            elif sig.exit_type == "stop_loss":
                algo._stop_exits  += 1
            elif sig.exit_type == "single_day_loss":
                algo._sdl_exits   += 1
            elif sig.exit_type == "rotation":
                algo._rotation_exits += 1

            tag = "TRIM" if is_partial else "SELL"
            algo.Debug(
                f"{ctx.today} {ticker} {tag} pnl=${gross_pnl:.2f} "
                f"held={days_held}d tax=${tax:.2f} "
                f"({'LT' if is_lt else 'ST'}) {sig.reason}"
            )

            if is_partial:
                # Place a market order for -quantity (negative = sell).
                # MarketOrder API: place exactly N shares, leaves remainder.
                algo.MarketOrder(sym, -int(req_qty))
                # DON'T stamp last_sell_dates — partial trim shouldn't
                # block top-up via wash-sale (matches RunnerAdapter +
                # SimAdapter behaviour after the 2026-04-24 fix).
            else:
                algo._last_sell_dates[ticker] = ctx.today
                algo.Liquidate(sym)
                full_exits.add(ticker)

        # Remove fully-exited tickers from algo._holdings (partial trims keep
        # the position open with original entry_date / entry_price preserved).
        for ticker in full_exits:
            algo._holdings.pop(ticker, None)

        # ── Persist updated HoldingStates (streak, HWM) from SellJob ────────
        # Skip only fully-exited tickers; partial trims keep the holding
        # state alive (entry_date / entry_price unchanged).
        for ticker, hs in ctx.holdings.items():
            if ticker not in full_exits:
                algo._holdings[ticker] = hs

        # ── Update SPY return buffer and prev closes ──────────────────────
        algo._spy_returns = ctx.spy_returns
        algo._regime_state  = ctx.regime_state
        algo._regime_counts = ctx.regime_counts
        algo._hwm       = ctx.hwm
        algo._skip_buys = ctx.skip_buys

        # Update prev_closes for all traded symbols
        import importlib
        try:
            AlgImp = importlib.import_module("AlgorithmImports")
            Resolution_cls = AlgImp.Resolution
        except (ImportError, AttributeError):
            Resolution_cls = None

        # prev_closes update — use ohlcv last close from context
        for ticker in list(algo._models.keys()) + ["SPY"]:
            df = ctx.ohlcv.get(ticker)
            if df is not None and not df.empty:
                algo._prev_closes[ticker] = float(df["close"].iloc[-1])

        # ── Apply buy orders ──────────────────────────────────────────────────
        # Top-up support: when ticker already held (TopUpHeldTask emitted
        # an add-to-existing order), preserve entry_date / entry_price /
        # entry_*_score baselines and only adjust shares + HWM. Resetting
        # entry state on every buy used to corrupt hold-day clocks, tax
        # ST/LT classification, and trailing-stop arming.
        # Audit fix LEAN-NaN (Round 2 deep audit, 2026-04-25): defense
        # in depth at the adapter boundary. Pre-fix, no isfinite check
        # on price/shares before mutating cost-basis or HWM, so a NaN
        # leaking through SizeAndEmitTask (pre-SE-1) or TopUpHeldTask
        # (pre-TU-1..TU-4) would corrupt hs.entry_price via the volume-
        # weighted average formula and propagate forever in stop-loss /
        # trailing-stop comparisons. Now: skip + warn on non-finite
        # price, shares, or target_pct so a single bad order doesn't
        # poison the held state.
        import math
        for order in ctx.orders:
            ticker     = order["ticker"]
            shares     = order["shares"]
            target_pct = order["target_pct"]
            price      = order["price"]
            sym        = algo.symbols.get(ticker)
            if sym is None:
                continue
            try:
                price_f      = float(price)
                shares_f     = float(shares)
                target_pct_f = float(target_pct)
            except (TypeError, ValueError):
                algo.Debug(f"{ctx.today} {ticker} SKIP — non-numeric order field")
                continue
            if not (math.isfinite(price_f) and price_f > 0
                    and math.isfinite(shares_f) and shares_f > 0
                    and math.isfinite(target_pct_f) and target_pct_f > 0):
                algo.Debug(
                    f"{ctx.today} {ticker} SKIP — non-finite order "
                    f"(price={price_f} shares={shares_f} target_pct={target_pct_f})"
                )
                continue

            already_held = ticker in algo._holdings

            algo.Debug(
                f"{ctx.today} {ticker} {'TOPUP' if already_held else 'BUY'} "
                f"regime={order['regime']} "
                f"conf={order['confidence']:.2f} rank={order['rank_score']:.3f} "
                f"rs={order['rs_score']:.3f} pct={target_pct:.2%} {order['detail']}"
            )

            if already_held:
                hs = algo._holdings[ticker]
                # Volume-weighted average cost basis on top-up — matches
                # SimAdapter._apply_buy. Without this, kernel.exits' stop-loss
                # / trailing-stop / single-day gates compute against the
                # ORIGINAL entry while the broker's actual cost basis is
                # the average → exits diverge between LEAN and sim.
                # Round 2 audit (#R1).
                try:
                    old_qty = float(algo.Portfolio[sym].Quantity)
                except Exception:
                    old_qty = 0.0
                new_qty = old_qty + shares
                if new_qty > 0 and old_qty > 0:
                    hs.entry_price = (
                        hs.entry_price * old_qty + price * shares
                    ) / new_qty
                # Refresh HWM with today's price; keep entry tenure intact.
                hs.high_watermark = max(hs.high_watermark, price)
            else:
                algo._holdings[ticker] = HoldingState(
                    entry_price    = price,
                    entry_date     = ctx.today,
                    high_watermark = price,
                    # Thesis-degradation baselines (Approach A) — stamp entry
                    # signals so future rotation checks can compare today's
                    # scores vs this fixed baseline. Not recomputed per bar.
                    entry_rank_score       = order.get("rank_score"),
                    entry_panel_score      = order.get("panel_score"),
                    entry_kelly_target_pct = order.get("kelly_target_pct"),
                )
            algo._executed_buys += 1
            algo.SetHoldings(sym, target_pct)

        # ── Telemetry counters from pipeline ─────────────────────────────────
        # Audit #88: also wire blocked_min_hold so OnEndOfAlgorithm displays
        # the real value (was previously initialised to 0 and never bumped,
        # always reported as zero in stats).
        c = ctx.counters
        algo._blocked_streak    += c.get("blocked_streak",    0)
        algo._transition_blocks += c.get("transition_blocks", 0)
        algo._velocity_blocks   += c.get("velocity_blocks",   0)
        algo._earnings_blocks   += c.get("earnings_blocks",   0)
        algo._blocked_wash      += c.get("blocked_wash",      0)
        algo._sector_blocks     += c.get("sector_blocks",     0)
        algo._corr_blocks       += c.get("corr_blocks",       0)
        algo._blocked_min_hold  += c.get("blocked_min_hold",  0)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_spy_df(algo: Any, bars: int):
    """Fetch SPY OHLCV from LEAN History API."""
    try:
        h = algo.History(algo._spy_sym, bars, Resolution.Daily)
        if not h.empty:
            return h.loc[algo._spy_sym].copy()
    except Exception:
        pass
    return None
