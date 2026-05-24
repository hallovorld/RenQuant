"""LeanAdapter — bridges LEAN QCAlgorithm state → InferenceContext → LEAN actions.

LEAN-safe: AlgorithmImports imported only at runtime, guarded by try/except for
static analysis.  No common/ imports.
"""
from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

import pandas as pd

from adapters.panel_runtime import (
    PanelFrameBundle,
    attach_panel_runtime_frames,
    prepare_panel_runtime_frames,
)
from kernel.decision_trace import (
    build_ticker_daily_state_rows,
    candidate_trace_pool,
    model_type_from_artifact as _shared_model_type_from_artifact,
    model_types_from_models,
    qp_trace_maps,
    selected_buy_tickers,
)

try:
    from AlgorithmImports import Resolution  # type: ignore[import]  # noqa: F401
except ImportError:
    pass  # running outside LEAN Docker (static analysis / tests)

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.task_execution import (
    dedupe_exit_signals,
    is_full_liquidate_signal,
)
from kernel.exits import HoldingState
from kernel.portfolio import compute_trade_tax

log = logging.getLogger("adapters.lean")

# Baseline indicator computation only needs ~60 bars; panel-LTR
# neutralization + factor building needs ≥504 bars of history.
_INDICATOR_LOOKBACK = 60
_PANEL_LOOKBACK     = 520


def _symbol_for_ticker(algo: Any, ticker: str):
    """Resolve regular watchlist, sector ETF, or benchmark symbols."""
    sym = algo.symbols.get(ticker)
    if sym is not None:
        return sym
    sym = algo._sector_etf_symbols.get(ticker)
    if sym is not None:
        return sym
    if ticker == getattr(algo, "_benchmark", None):
        return getattr(algo, "_spy_sym", None)
    return None


def _model_type_from_artifact(model: Any) -> str | None:
    """Extract a readable model type for decision-trace rows."""
    return _shared_model_type_from_artifact(model)


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
        # bar. Pre-fix, panel frame prep was called per
        # OnData → 99 tickers × N bars × full feature pipeline = hours of
        # wasted compute per backtest. Cache is invalidated when the SPY
        # buffer extends past the last cached date.
        self._panel_cache_last_date: "pd.Timestamp | None" = None
        self._panel_cache_ff: "dict | None"  = None
        self._panel_cache_fac: "dict | None" = None

        # Execution-model parity bookkeeping (Track Batch A, 2026-05-10):
        # LEAN's brokerage model handles fee + slippage + broker settlement
        # natively when `SetBrokerageModel` is wired (main.py:Initialize).
        # We keep a cumulative-fees mirror here so the runtime statistics
        # report matches sim's `_total_fees` summary. Read from
        # `algo.Portfolio.TotalFees` each commit; the running total comes
        # straight from LEAN's brokerage model — no parallel arithmetic.
        # Per CLAUDE.md §5.13.5 single-source-of-truth: never compute
        # fees on the LEAN side from scratch — always read LEAN's number.
        self._lean_fees_last_seen: float = 0.0

        # ── Meta-label hooks (P-LEAN, 2026-05-11) ─────────────────────
        # Mirror SimAdapter / RunnerAdapter: LeanAdapter loads the
        # XGBoost predictor at construction so backtest results in
        # LEAN are PARITY with sim + live. Without this, LEAN sims
        # would silently no-op meta-label (§5.13.10 fallback) and
        # under-state prod behavior, defeating the point of LEAN
        # backtest validation.
        config = getattr(algo, "_config", None) or {}
        self._universe_rejections = dict(
            getattr(algo, "_universe_rejections", {}) or {}
        )
        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(
            config,
            strategy_dir=getattr(algo, "_strategy_dir", None),
            role="live",
        )

        ml_train_cfg = config.get("meta_label_training", {}) or {}
        if ml_train_cfg.get("enabled", False):
            from kernel.meta_label import SnapshotLogger  # noqa: PLC0415
            self._meta_label_logger = SnapshotLogger()
            self._meta_label_output_path = str(
                ml_train_cfg.get("output_path", "data/position_day_snapshots.parquet")
            )
        else:
            self._meta_label_logger = None
            self._meta_label_output_path = None

        veto_cfg = (config.get("ranking") or {}).get("meta_label") or {}
        if veto_cfg.get("enabled", False):
            from kernel.meta_label.predictor import load_meta_label_predictor  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415
            art_path = veto_cfg.get(
                "artifact_path",
                "backtesting/renquant_104/artifacts/meta-label-exit.json",
            )
            strategy_dir = getattr(algo, "_strategy_dir", None)
            art_resolved = Path(art_path)
            if not art_resolved.is_absolute() and strategy_dir is not None:
                art_resolved = Path(strategy_dir).parent.parent / art_resolved
            self._meta_label_predictor = load_meta_label_predictor(art_resolved)
        else:
            self._meta_label_predictor = None

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
            sym = _symbol_for_ticker(algo, ticker)
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
            last_stop_exit_dates = getattr(algo, "_last_stop_exit_dates", {}) or {},
            portfolio_value  = pv,
            cash             = cash,
            prices           = prices,
            hwm              = algo._hwm,
            skip_buys        = algo._skip_buys,
            regime_state     = algo._regime_state,
            regime_counts    = algo._regime_counts,
        )
        ctx.run_id = f"{today.isoformat()}-lean-{uuid.uuid4().hex[:8]}"

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
                    bundle = prepare_panel_runtime_frames(
                        config=config,
                        ohlcv=ohlcv,
                    )
                    self._panel_cache_ff = bundle.feature_frames
                    self._panel_cache_fac = bundle.factor_frames
                    self._panel_cache_macro = bundle.macro_frame   # Bug #25
                    self._panel_cache_emb = bundle.asset_embeddings  # T2-2
                    self._panel_cache_last_date = pd.Timestamp(today)
                except Exception as exc:
                    msg = f"Panel frame prep failed — panel scoring cannot run: {exc}"
                    log.error(msg)
                    raise RuntimeError(msg) from exc
            attach_panel_runtime_frames(
                ctx,
                PanelFrameBundle(
                    feature_frames=self._panel_cache_ff,
                    factor_frames=self._panel_cache_fac,
                    macro_frame=getattr(self, "_panel_cache_macro", None),
                    asset_embeddings=getattr(self, "_panel_cache_emb", None),
                ),
            )

        # P-LEAN (2026-05-11) — attach meta-label hooks for parity with
        # SimAdapter / RunnerAdapter. snapshot_logger is None unless
        # meta_label_training mode; predictor is None unless artifact
        # loaded successfully (§5.13.10 fallback).
        ctx.snapshot_logger = self._meta_label_logger
        ctx._meta_label_predictor = self._meta_label_predictor  # noqa: SLF001
        return ctx

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx: InferenceContext) -> None:
        """Apply pipeline outputs: execute exits, place buys, persist state."""
        algo   = self._algo
        config = algo._config

        tax_short      = algo._tax_short
        tax_long       = algo._tax_long
        tax_thresh     = algo._tax_thresh_days
        trade_events: list[dict[str, Any]] = []

        # ── Apply exits ──────────────────────────────────────────────────────
        # 2026-04-24 partial-sell support: when sig.quantity is set and
        # < current holding, place a market sell for that quantity instead
        # of Liquidate (which always closes the entire position).
        # full_exits tracks tickers that we should actually pop from
        # algo._holdings + stamp last_sell_dates for wash-sale.
        full_exits: set[str] = set()
        # 2026-05-04 audit Issue 37 fix: NaN gross_pnl (broker disconnect,
        # corrupted UnrealizedProfit) propagated `tax = NaN` → `_total_tax += NaN`
        # → cumulative tax permanently NaN, all post-trade reports broken.
        # Skip the tax addition on non-finite values; sell still proceeds.
        import math as _math_lex
        def _held_qty(t: str) -> float:
            sym_local = _symbol_for_ticker(algo, t)
            if sym_local is None:
                return 0.0
            try:
                return float(algo.Portfolio[sym_local].Quantity)
            except Exception:
                return 0.0

        for ticker, sig in dedupe_exit_signals(ctx.exits, held_qty_for=_held_qty):
            hs        = ctx.holdings.get(ticker)
            sym       = _symbol_for_ticker(algo, ticker)
            if sym is None:
                continue
            gross_pnl = float(algo.Portfolio[sym].UnrealizedProfit)
            days_held = (ctx.today - hs.entry_date).days if hs else 0
            is_lt     = days_held >= tax_thresh

            req_qty = getattr(sig, "quantity", None)
            holding_qty = float(algo.Portfolio[sym].Quantity)
            is_partial = not is_full_liquidate_signal(sig, holding_qty)

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

            if not _math_lex.isfinite(tax):
                log.warning(
                    "lean.commit [%s]: NaN/inf tax (gross_pnl=%s) — "
                    "skipping cumulative add to preserve _total_tax.",
                    ticker, gross_pnl,
                )
                tax = 0.0
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
            try:
                price = float(algo.Securities[sym].Price)
            except Exception:
                price = 0.0
            event_shares = float(req_qty) if is_partial else float(holding_qty)
            event_gross = gross_pnl * frac if is_partial else gross_pnl
            proceeds_basis = (
                price * event_shares - event_gross
                if _math_lex.isfinite(price)
                and _math_lex.isfinite(event_shares)
                and _math_lex.isfinite(event_gross)
                else None
            )
            exit_type = getattr(sig, "exit_type", "") or ""
            reason = getattr(sig, "reason", None)
            source_job = str(getattr(sig, "source_job", None) or "TickerSellJob")
            source_task = str(getattr(sig, "source_task", None) or exit_type or "sell")
            order_source = str(
                getattr(sig, "order_source", None) or f"{source_job}.{source_task}"
            )
            trade_events.append({
                "ticker": ticker,
                "action": "sell",
                "date": ctx.today,
                "shares": event_shares,
                "price": price,
                "gross_pnl": event_gross,
                "proceeds_basis": proceeds_basis,
                "tax": tax,
                "net_pnl_after_tax": event_gross - tax
                if _math_lex.isfinite(event_gross) else None,
                "exit_reason": exit_type,
                "pnl_pct": event_gross / proceeds_basis
                if proceeds_basis and proceeds_basis > 0 else None,
                "hold_days": days_held,
                "rank_score": getattr(hs, "rank_score", None) if hs else None,
                "mu": getattr(hs, "mu", None) if hs else None,
                "sigma": getattr(hs, "sigma", None) if hs else None,
                "order_type": f"SELL_{exit_type}" if exit_type else "SELL",
                "source": str(getattr(sig, "source", None) or "ExitPipeline"),
                "source_job": source_job,
                "source_task": source_task,
                "order_source": order_source,
                "attribution_version": "lean_exit_decision_v1",
                "score_snapshot": {
                    "rank_score": getattr(hs, "rank_score", None) if hs else None,
                    "panel_score": getattr(hs, "panel_score", None) if hs else None,
                    "mu": getattr(hs, "mu", None) if hs else None,
                    "sigma": getattr(hs, "sigma", None) if hs else None,
                    "confidence": ctx.confidence,
                    "regime": ctx.regime,
                },
                "decision_inputs": {
                    "acceptance_reason": exit_type or reason,
                    "exit_reason": exit_type,
                    "signal_reason": reason,
                    "partial": is_partial,
                    "quantity": getattr(sig, "quantity", None),
                    "shares": event_shares,
                    "gross_pnl": event_gross,
                    "tax": tax,
                    "net_pnl_after_tax": event_gross - tax
                    if _math_lex.isfinite(event_gross) else None,
                    "hold_days": days_held,
                },
            })

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
            # G8 (2026-05-04): stamp post-stop blackout date on path-rule
            # exits regardless of partial/full. Tracked separately from
            # last_sell_dates so it survives even when the partial-trim
            # wash-sale exemption applies.
            from kernel.pipeline.task_post_stop_cooldown import (  # noqa: PLC0415
                DEFAULT_STOP_EXIT_TYPES,
            )
            if str(sig.exit_type) in DEFAULT_STOP_EXIT_TYPES:
                if not hasattr(algo, "_last_stop_exit_dates"):
                    algo._last_stop_exit_dates = {}
                algo._last_stop_exit_dates[ticker] = ctx.today

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
        # 2026-05-04 audit Issue 35 fix: NaN close (delisted/suspended ticker)
        # slipped past the empty-check, then `float(NaN) = NaN` corrupted
        # algo._prev_closes[ticker]. Downstream check_single_day_loss sees
        # NaN prev_close, comparison `daily_drop >= sdl_pct` is False
        # → SDL gate silently disabled for that ticker. Skip non-finite.
        import math as _math_le
        for ticker in list(algo._models.keys()) + ["SPY"]:
            df = ctx.ohlcv.get(ticker)
            if df is not None and not df.empty:
                _close = float(df["close"].iloc[-1])
                if _math_le.isfinite(_close):
                    algo._prev_closes[ticker] = _close

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
            sym        = _symbol_for_ticker(algo, ticker)
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
                    entry_regime           = order.get("regime"),
                )
            algo._executed_buys += 1
            algo.SetHoldings(sym, target_pct)
            trade_events.append({
                "ticker": ticker,
                "action": "buy",
                "date": ctx.today,
                "shares": shares_f,
                "price": price_f,
                "invest": shares_f * price_f,
                "target_pct": target_pct_f,
                "rank_score": order.get("rank_score"),
                "conviction": order.get("conviction"),
                "sigma_mult": order.get("sigma_mult"),
                "mu": order.get("mu"),
                "sigma": order.get("sigma"),
                "order_type": order.get("order_type", "BUY"),
                "source": order.get("source"),
                "source_job": order.get("source_job"),
                "source_task": order.get("source_task"),
                "order_source": order.get("order_source"),
                "attribution_version": "lean_buy_decision_v1",
                "score_snapshot": {
                    "rank_score": order.get("rank_score"),
                    "panel_score": order.get("panel_score"),
                    "rs_score": order.get("rs_score"),
                    "mu": order.get("mu"),
                    "sigma": order.get("sigma"),
                    "kelly_target_pct": order.get("kelly_target_pct"),
                    "confidence": order.get("confidence"),
                    "regime": order.get("regime"),
                },
                "decision_inputs": {
                    "acceptance_reason": order.get("detail") or "lean_buy",
                    "target_pct": target_pct_f,
                    "shares": shares_f,
                    "price": price_f,
                    "invest": shares_f * price_f,
                    "order_source": order.get("order_source"),
                    "source_job": order.get("source_job"),
                    "source_task": order.get("source_task"),
                },
            })

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
        self._record_decision_trace(ctx, trade_events)

    def _record_decision_trace(
        self,
        ctx: InferenceContext,
        trade_events: list[dict[str, Any]],
    ) -> None:
        """Persist LEAN bar decisions with the same DB contract as sim/live."""
        if self._db is None:
            return
        from kernel.persistence import (  # noqa: PLC0415
            record_candidate_scores,
            record_pipeline_run,
            record_ticker_daily_state,
            record_trades,
        )

        algo = self._algo
        config = algo._config
        selected_tickers = selected_buy_tickers(trade_events)
        blocked_map = dict(getattr(ctx, "_blocked_by_ticker", None) or {})
        sector_map = config.get("sector_map", {}) or {}
        model_types = model_types_from_models(algo._models)
        panel_artifact = (
            config.get("ranking", {})
                  .get("panel_scoring", {})
                  .get("artifact_path")
        )
        qp_delta_by_ticker, qp_target_by_ticker, qp_status = qp_trace_maps(ctx)

        run_id = record_pipeline_run(
            self._db,
            run_type="lean",
            run_date=ctx.today,
            strategy=str(config.get("model_name", "")),
            regime=ctx.regime,
            confidence=float(ctx.confidence) if ctx.confidence is not None else None,
            portfolio_value=float(ctx.portfolio_value)
            if ctx.portfolio_value is not None else None,
            cash=float(ctx.cash) if ctx.cash is not None else None,
            n_candidates=len(ctx.candidates),
            n_exits=len(ctx.exits),
            n_rotations=int(ctx.counters.get("rotations", 0)),
            n_buys=len(selected_tickers),
            buy_blocked=bool(getattr(ctx, "buy_blocked", False)),
            skip_buys=bool(getattr(ctx, "skip_buys", False)),
            bear_only=bool(getattr(ctx, "bear_only", False)),
            counters=getattr(ctx, "counters", {}) or {},
            run_bundle={"adapter": "lean"},
            run_id=getattr(ctx, "run_id", None),
        )

        cand_pool = candidate_trace_pool(ctx)
        record_candidate_scores(
            self._db,
            run_id,
            cand_pool,
            ctx.holdings,
            selected_tickers=selected_tickers,
            blocked_map=blocked_map,
            sector_map=sector_map,
            model_types=model_types,
            panel_artifact=panel_artifact,
            qp_delta_by_ticker=qp_delta_by_ticker,
            qp_target_by_ticker=qp_target_by_ticker,
            qp_status=qp_status,
        )
        record_trades(self._db, run_id, trade_events)

        rows = build_ticker_daily_state_rows(
            config=config,
            ctx=ctx,
            selected_tickers=selected_tickers,
            blocked_map=blocked_map,
            model_types=model_types,
            universe_rejections=self._universe_rejections,
            model_keys=set(algo._models or {}),
            portfolio_value=(
                float(ctx.portfolio_value)
                if ctx.portfolio_value is not None else None
            ),
            sector_map=sector_map,
            qp_delta_by_ticker=qp_delta_by_ticker,
            qp_target_by_ticker=qp_target_by_ticker,
            qp_status=qp_status,
        )
        record_ticker_daily_state(
            self._db,
            run_date=ctx.today,
            rows=rows,
            run_id=run_id,
        )


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
