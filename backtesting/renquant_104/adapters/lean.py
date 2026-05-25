"""LeanAdapter — bridges LEAN QCAlgorithm state → InferenceContext → LEAN actions.

LEAN-safe: AlgorithmImports imported only at runtime, guarded by try/except for
static analysis.  No common/ imports.
"""
from __future__ import annotations

import datetime
import logging
import math
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
    candidate_score_excluded_holding_tickers,
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
from kernel.exits import (
    HoldingState,
    apply_buy_lot,
    apply_sell_lots_detailed,
    ensure_lots,
)
from kernel.portfolio import compute_disposed_lot_tax, compute_trade_tax
from kernel.trade_events import build_buy_trade_event, build_sell_trade_event

log = logging.getLogger("adapters.lean")

# Baseline indicator computation only needs ~60 bars; panel-LTR
# neutralization + factor building needs ≥504 bars of history.
_INDICATOR_LOOKBACK = 60
_PANEL_LOOKBACK     = 520
_BUYING_POWER_SETTLED = "settled_cash"
_BUYING_POWER_NMBP = "non_marginable_buying_power"
_BUYING_POWER_ALIASES = {
    _BUYING_POWER_SETTLED: _BUYING_POWER_SETTLED,
    "settled": _BUYING_POWER_SETTLED,
    "cash": _BUYING_POWER_SETTLED,
    _BUYING_POWER_NMBP: _BUYING_POWER_NMBP,
    "cash_plus_unsettled": _BUYING_POWER_NMBP,
    "unsettled": _BUYING_POWER_NMBP,
}


def _normalize_buying_power_mode(raw: Any) -> str:
    mode = str(raw or _BUYING_POWER_NMBP).strip().lower()
    if mode not in _BUYING_POWER_ALIASES:
        raise ValueError(
            "execution.buying_power_mode must be one of "
            f"{sorted(_BUYING_POWER_ALIASES)}; got {raw!r}"
        )
    return _BUYING_POWER_ALIASES[mode]


def _finite_attr_float(obj: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _lean_buying_power_snapshot(algo: Any, config: dict) -> dict[str, Any]:
    portfolio = getattr(algo, "Portfolio", None)
    settled_cash = _finite_attr_float(portfolio, "Cash") or 0.0
    exec_cfg = (config or {}).get("execution", {}) or {}
    mode = _normalize_buying_power_mode(
        exec_cfg.get("buying_power_mode", _BUYING_POWER_NMBP)
    )
    if mode == _BUYING_POWER_SETTLED:
        return {
            "cash": settled_cash,
            "settled_cash": settled_cash,
            "pending_settle_cash": 0.0,
            "buying_power_mode": mode,
            "buying_power_source": "portfolio_cash",
        }

    nmbp = _finite_attr_float(
        portfolio,
        "NonMarginableBuyingPower",
        "non_marginable_buying_power",
        "NonMarginableBuyingPowerAmount",
    )
    if nmbp is not None and nmbp >= 0.0:
        pending = max(0.0, nmbp - settled_cash)
        return {
            "cash": nmbp,
            "settled_cash": settled_cash,
            "pending_settle_cash": pending,
            "buying_power_mode": mode,
            "buying_power_source": "portfolio_non_marginable_buying_power",
        }

    pending = _finite_attr_float(algo, "_pending_settle_cash") or 0.0
    pending = max(0.0, pending)
    return {
        "cash": settled_cash + pending,
        "settled_cash": settled_cash,
        "pending_settle_cash": pending,
        "buying_power_mode": mode,
        "buying_power_source": (
            "algo_pending_settle_cash" if pending > 0.0
            else "portfolio_cash_fallback"
        ),
    }


def _lean_post_execution_snapshot(
    algo: Any,
    config: dict,
    ctx: Any,
) -> dict[str, Any]:
    portfolio = getattr(algo, "Portfolio", None)
    pv = _finite_attr_float(portfolio, "TotalPortfolioValue")
    if pv is None:
        try:
            pv = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            pv = None
    bp = _lean_buying_power_snapshot(algo, config)
    if _finite_attr_float(portfolio, "Cash") is None:
        try:
            fallback_cash = float(getattr(ctx, "cash", 0.0) or 0.0)
        except (TypeError, ValueError):
            fallback_cash = 0.0
        if math.isfinite(fallback_cash):
            bp = {
                **bp,
                "cash": fallback_cash,
                "settled_cash": fallback_cash,
                "pending_settle_cash": 0.0,
            }
    holdings = getattr(algo, "_holdings", {}) or {}
    n_holdings = sum(
        1 for hs in holdings.values()
        if _finite_attr_float(hs, "shares") and _finite_attr_float(hs, "shares") > 0
    )
    return {
        "portfolio_value": pv,
        "cash": float(bp["cash"]),
        "settled_cash": float(bp["settled_cash"]),
        "pending_settle_cash": float(bp["pending_settle_cash"]),
        "n_holdings": n_holdings,
    }


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


def _positive_finite_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(price) and price > 0.0:
        return price
    return None


def _order_payload(order: dict, key: str) -> Any:
    value = order.get(key)
    if value is not None:
        return value
    for field in ("score_snapshot", "decision_inputs"):
        payload = order.get(field)
        if isinstance(payload, dict) and payload.get(key) is not None:
            return payload.get(key)
    return None


def _stamp_holding_audit_fields(holding: Any, order: dict) -> None:
    if holding is None or not isinstance(order, dict):
        return
    for key in (
        "model_type",
        "sector",
        "blocked_by",
        "expected_return",
        "expected_return_horizon_days",
        "mu",
        "mu_horizon_days",
        "sigma",
        "panel_score",
        "rank_score",
        "kelly_target_pct",
    ):
        value = _order_payload(order, key)
        if value is not None:
            setattr(holding, key, value)


def _current_price_for_ticker(
    algo: Any,
    data: Any,
    ticker: str,
    ohlcv: dict[str, Any],
) -> float | None:
    """Return the current executable price for any ticker the pipeline may size.

    SimAdapter and RunnerAdapter populate prices for all model/watchlist names.
    LEAN must do the same: a ranked buy candidate with no ``ctx.prices`` entry
    is rejected downstream as ``size_bad_price`` even if the model signal is
    valid. Price source order mirrors the execution surface: current Slice,
    current Security price, then latest OHLCV close.
    """
    sym = _symbol_for_ticker(algo, ticker)
    if sym is not None:
        try:
            if data.ContainsKey(sym):
                px = _positive_finite_price(data[sym].Close)
                if px is not None:
                    return px
        except Exception:
            pass
        securities = getattr(algo, "Securities", None)
        if securities is not None:
            try:
                px = _positive_finite_price(securities[sym].Price)
                if px is not None:
                    return px
            except Exception:
                pass

    df = ohlcv.get(ticker)
    if df is not None and not getattr(df, "empty", True):
        try:
            close = df["close"].dropna()
            if not close.empty:
                return _positive_finite_price(close.iloc[-1])
        except Exception:
            pass
    return None


def _model_type_from_artifact(model: Any) -> str | None:
    """Extract a readable model type for decision-trace rows."""
    return _shared_model_type_from_artifact(model)


def _fmt_debug_float(value: Any, spec: str, missing: str = "NA") -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(value_f):
        return missing
    return format(value_f, spec)


def _lean_ticket_status_text(ticket: Any) -> str:
    status = getattr(ticket, "Status", None)
    return str(status or "").lower()


def _lean_ticket_float(ticket: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(ticket, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _lean_order_execution(
    ticket: Any,
    *,
    requested_qty: float,
    fallback_price: float,
) -> tuple[bool, float, float, str]:
    """Return (filled, qty, avg_price, status) for a LEAN order ticket.

    Real tickets must either expose a filled status or a positive filled
    quantity. A missing ticket is not execution evidence; fail closed so LEAN
    cannot mutate state/tax as if an unconfirmed order filled.
    """
    requested_abs = abs(float(requested_qty))
    fallback = _positive_finite_price(fallback_price) or 0.0
    if ticket is None:
        return False, 0.0, fallback, "missing_order_ticket"
    if isinstance(ticket, (list, tuple)):
        total_qty = 0.0
        total_value = 0.0
        statuses: list[str] = []
        for item in ticket:
            ok, qty, px, status = _lean_order_execution(
                item,
                requested_qty=requested_abs,
                fallback_price=fallback,
            )
            statuses.append(status)
            if ok and qty > 0:
                total_qty += qty
                total_value += qty * (px if px > 0 else fallback)
        if total_qty > 0:
            return True, total_qty, total_value / total_qty, ",".join(statuses)
        return False, 0.0, fallback, ",".join(statuses)

    status = _lean_ticket_status_text(ticket)
    if any(token in status for token in ("reject", "cancel", "invalid", "error")):
        return False, 0.0, fallback, status
    qty = _lean_ticket_float(
        ticket,
        "QuantityFilled",
        "AbsoluteQuantityFilled",
        "FilledQuantity",
    )
    price = _lean_ticket_float(
        ticket,
        "AverageFillPrice",
        "AvgFillPrice",
        "FillPrice",
        "Price",
    )
    if qty is not None and abs(qty) > 0:
        return True, abs(qty), price or fallback, status
    if "filled" in status:
        return True, requested_abs, price or fallback, status
    return False, 0.0, price or fallback, status or "unknown"


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
            list(config.get("watchlist", []) or [])
            + list(algo._models.keys())
            + list(algo._holdings.keys())
            + list(algo._sector_etf_symbols.keys())
            + ["SPY"]
        )
        from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
            benchmark_sleeve_ticker,
            is_benchmark_sleeve_enabled,
        )
        sleeve_ticker = benchmark_sleeve_ticker(config)
        if is_benchmark_sleeve_enabled(config) and sleeve_ticker:
            all_tickers.append(sleeve_ticker)
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
        for ticker in unique_tickers:
            px = _current_price_for_ticker(algo, data, ticker, ohlcv)
            if px is not None:
                prices[ticker] = px

        pv   = float(algo.Portfolio.TotalPortfolioValue)
        bp_snapshot = _lean_buying_power_snapshot(algo, config)
        cash = float(bp_snapshot["cash"])

        holdings = dict(algo._holdings)  # shallow copy; jobs mutate states in place
        for ticker, hs in holdings.items():
            sym = _symbol_for_ticker(algo, ticker)
            if sym is None:
                continue
            try:
                qty = float(algo.Portfolio[sym].Quantity)
            except Exception:
                continue
            hs.shares = qty
            ensure_lots(hs)
            hs.model_type = _model_type_from_artifact(algo._models.get(ticker))
            sector = config.get("sector_map", {}).get(ticker)
            if isinstance(sector, str) and sector:
                hs.sector = sector

        ctx = InferenceContext(
            config           = config,
            today            = today,
            ohlcv            = ohlcv,
            spy_returns      = spy_returns,
            models           = algo._models,
            gmm              = algo._gmm,
            corr_matrix      = algo._corr,
            earnings_calendar = algo._earnings,
            holdings         = holdings,
            last_sell_dates  = algo._last_sell_dates,
            last_sell_pls    = getattr(algo, "_last_sell_pls", {}) or {},
            last_stop_exit_dates = getattr(algo, "_last_stop_exit_dates", {}) or {},
            portfolio_value  = pv,
            cash             = cash,
            prices           = prices,
            hwm              = algo._hwm,
            skip_buys        = algo._skip_buys,
            regime_state     = algo._regime_state,
            regime_counts    = algo._regime_counts,
        )
        ctx.settled_cash = float(bp_snapshot["settled_cash"])
        ctx.pending_settle_cash = float(bp_snapshot["pending_settle_cash"])
        ctx.buying_power_mode = str(bp_snapshot["buying_power_mode"])
        ctx.buying_power_source = str(bp_snapshot["buying_power_source"])
        ctx.run_id = f"{today.isoformat()}-lean-{uuid.uuid4().hex[:8]}"
        if self._db is not None:
            ctx._db = self._db  # noqa: SLF001

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
        ctx._run_type = "lean"  # noqa: SLF001
        return ctx

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx: InferenceContext) -> None:
        """Apply pipeline outputs: execute exits, place buys, persist state."""
        algo   = self._algo
        config = algo._config

        if getattr(ctx, "orders", None) and not bool(getattr(algo, "_preflight_ok", False)):
            raise RuntimeError(
                "LEAN preflight did not complete successfully; refusing to submit buy orders"
            )

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
            gross_pnl_broker = float(algo.Portfolio[sym].UnrealizedProfit)
            days_held = (ctx.today - hs.entry_date).days if hs else 0
            is_lt     = days_held >= tax_thresh

            req_qty = getattr(sig, "quantity", None)
            holding_qty = float(algo.Portfolio[sym].Quantity)
            try:
                fallback_price = float(algo.Securities[sym].Price)
            except Exception:
                fallback_price = 0.0
            is_full_request = is_full_liquidate_signal(sig, holding_qty)
            if is_full_request:
                requested_exit_shares = float(holding_qty)
                ticket = algo.Liquidate(sym)
            else:
                requested_exit_shares = min(float(req_qty), float(holding_qty))
                requested_order_shares = int(requested_exit_shares)
                if requested_order_shares <= 0:
                    algo.Debug(
                        f"{ctx.today} {ticker} EXIT skipped — requested "
                        f"shares round to zero ({requested_exit_shares})"
                    )
                    continue
                requested_exit_shares = float(requested_order_shares)
                ticket = algo.MarketOrder(sym, -requested_order_shares)
            filled, filled_qty, fill_price, status = _lean_order_execution(
                ticket,
                requested_qty=requested_exit_shares,
                fallback_price=fallback_price,
            )
            if not filled or filled_qty <= 0.0:
                algo.Debug(
                    f"{ctx.today} {ticker} EXIT skipped — LEAN order not "
                    f"filled (status={status})"
                )
                continue
            event_shares = min(float(filled_qty), float(holding_qty))
            is_partial = event_shares < float(holding_qty) - 1e-9
            price = fill_price if fill_price > 0 else fallback_price

            if is_partial:
                # Partial sell fallback when lot accounting is unavailable.
                frac = float(event_shares) / max(holding_qty, 1.0)
                event_gross = gross_pnl_broker * frac
            else:
                event_gross = gross_pnl_broker

            proceeds_basis = (
                price * event_shares - event_gross
                if _math_lex.isfinite(price)
                and _math_lex.isfinite(event_shares)
                and _math_lex.isfinite(event_gross)
                else None
            )
            disposed_lots = []
            lot_method = str(
                ((config.get("rotation", {}) or {}).get("joint_actions", {}) or {})
                .get("qp_tax_lot_method", "fifo")
            ).lower()
            if (
                hs is not None
                and _math_lex.isfinite(price)
                and price > 0
                and _math_lex.isfinite(event_shares)
                and event_shares > 0
            ):
                ensure_lots(hs)
                had_lots = bool(getattr(hs, "lots", None))
                lot_basis, _, disposed_lots = apply_sell_lots_detailed(
                    hs, event_shares, lot_method,
                )
                if (
                    had_lots
                    and _math_lex.isfinite(lot_basis)
                    and lot_basis > 0
                ):
                    proceeds_basis = lot_basis
                    event_gross = event_shares * price - proceeds_basis
                else:
                    disposed_lots = []
                if is_partial:
                    hs.shares = max(0.0, float(holding_qty) - event_shares)
                    hs.entry_price = hs.weighted_avg_entry_price()

            lot_tax = (
                compute_disposed_lot_tax(
                    price,
                    ctx.today,
                    disposed_lots,
                    tax_short,
                    tax_long,
                    tax_thresh,
                )
                if disposed_lots else {}
            )
            if lot_tax:
                tax = float(lot_tax["tax"])
                days_held = int(round(float(lot_tax["weighted_hold_days"])))
            else:
                tax = compute_trade_tax(
                    event_gross, days_held, tax_short, tax_long, tax_thresh,
                )

            if not _math_lex.isfinite(tax):
                log.warning(
                    "lean.commit [%s]: NaN/inf tax (gross_pnl=%s) — "
                    "skipping cumulative add to preserve _total_tax.",
                    ticker, event_gross,
                )
                tax = 0.0
            is_lt = days_held >= tax_thresh
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
                f"{ctx.today} {ticker} {tag} pnl=${event_gross:.2f} "
                f"held={days_held}d tax=${tax:.2f} "
                f"({'LT' if is_lt else 'ST'}) {sig.reason}"
            )
            event_net = (
                event_gross - tax if _math_lex.isfinite(event_gross) else None
            )
            event_pnl_pct = (
                event_gross / proceeds_basis
                if proceeds_basis and proceeds_basis > 0 else None
            )
            regime_p = dict(
                (config.get("regime_params", {}) or {}).get(ctx.regime, {}) or {}
            )
            regime_p["tax"] = config.get("tax", {}) or {}
            trade_event = build_sell_trade_event(
                ticker=ticker,
                sig=sig,
                holding=hs,
                price=price,
                today=ctx.today,
                regime=ctx.regime,
                confidence=ctx.confidence,
                regime_params=regime_p,
                config=config,
                shares=event_shares,
                gross_pnl=event_gross,
                proceeds_basis=proceeds_basis,
                tax=tax,
                net_pnl_after_tax=event_net,
                pnl_pct=event_pnl_pct,
                hold_days=days_held,
                attribution_version="lean_exit_decision_v1",
            )
            trade_event["decision_inputs"]["partial"] = is_partial
            trade_event["decision_inputs"]["tax_lot_method"] = lot_method
            if lot_tax:
                trade_event["decision_inputs"]["short_term_gross_pnl"] = lot_tax[
                    "short_term_gross_pnl"
                ]
                trade_event["decision_inputs"]["long_term_gross_pnl"] = lot_tax[
                    "long_term_gross_pnl"
                ]
            trade_events.append(trade_event)

            if not is_partial:
                algo._last_sell_dates[ticker] = ctx.today
                if not hasattr(algo, "_last_sell_pls"):
                    algo._last_sell_pls = {}
                algo._last_sell_pls[ticker] = (
                    float(event_gross) if _math_lex.isfinite(event_gross)
                    else None
                )
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
        from kernel.pipeline.order_dedupe import (  # noqa: PLC0415
            dedupe_buy_orders_first_wins,
        )
        deduped_orders, skipped_duplicate_buys = (
            dedupe_buy_orders_first_wins(ctx.orders)
        )
        for order in skipped_duplicate_buys:
            try:
                ticker = (
                    order.get("ticker")
                    if isinstance(order, dict) else
                    getattr(order, "ticker", "?")
                )
            except Exception:
                ticker = "?"
            algo.Debug(f"{ctx.today} {ticker} SKIP — duplicate same-bar buy intent")

        for order in deduped_orders:
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
                f"conf={_fmt_debug_float(order.get('confidence'), '.2f')} "
                f"rank={_fmt_debug_float(order.get('rank_score'), '.3f')} "
                f"rs={_fmt_debug_float(order.get('rs_score'), '.3f')} "
                f"pct={target_pct_f:.2%} {order.get('detail', '')}"
            )
            order_qty = int(shares_f)
            if order_qty <= 0:
                algo.Debug(
                    f"{ctx.today} {ticker} SKIP — shares round to zero "
                    f"({shares_f})"
                )
                continue
            ticket = algo.MarketOrder(sym, order_qty)
            filled, filled_qty, fill_price, status = _lean_order_execution(
                ticket,
                requested_qty=float(order_qty),
                fallback_price=price_f,
            )
            if not filled or filled_qty <= 0.0:
                algo.Debug(
                    f"{ctx.today} {ticker} BUY skipped — LEAN order not "
                    f"filled (status={status})"
                )
                continue
            shares_f = float(filled_qty)
            price_f = float(fill_price or price_f)
            if ctx.portfolio_value and ctx.portfolio_value > 0:
                target_pct_f = shares_f * price_f / float(ctx.portfolio_value)

            if already_held:
                hs = algo._holdings[ticker]
                # Tax-lot-aware top-up — matches SimAdapter._apply_buy.
                # Keep the original entry date, append a new lot, and refresh
                # legacy avg-cost fields from the lot ledger.
                try:
                    old_qty = float(algo.Portfolio[sym].Quantity)
                except Exception:
                    old_qty = 0.0
                hs.shares = old_qty
                ensure_lots(hs)
                apply_buy_lot(hs, shares_f, price_f, ctx.today)
                hs.shares = old_qty + shares_f
                hs.high_watermark = max(hs.high_watermark, price_f)
            else:
                hs_new = HoldingState(
                    entry_price    = price_f,
                    entry_date     = ctx.today,
                    high_watermark = price_f,
                    shares         = 0.0,
                    # Thesis-degradation baselines (Approach A) — stamp entry
                    # signals so future rotation checks can compare today's
                    # scores vs this fixed baseline. Not recomputed per bar.
                    entry_rank_score       = order.get("rank_score"),
                    entry_panel_score      = order.get("panel_score"),
                    entry_kelly_target_pct = order.get("kelly_target_pct"),
                    entry_regime           = order.get("regime"),
                )
                apply_buy_lot(hs_new, shares_f, price_f, ctx.today)
                hs_new.shares = shares_f
                algo._holdings[ticker] = hs_new
            _stamp_holding_audit_fields(algo._holdings.get(ticker), order)
            algo._executed_buys += 1
            trade_events.append(build_buy_trade_event(
                {
                    **order,
                    "price": price_f,
                    "shares": shares_f,
                    "target_pct": target_pct_f,
                    "invest": shares_f * price_f,
                    "order_type": order.get("order_type", "BUY"),
                },
                date=ctx.today,
                default_regime=ctx.regime,
                default_confidence=ctx.confidence,
                attribution_version="lean_buy_decision_v1",
                default_acceptance_reason="lean_buy",
            ))

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
            validate_decision_trace_integrity,
        )
        from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415

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
        post_snapshot = _lean_post_execution_snapshot(algo, config, ctx)
        run_bundle = build_run_bundle(
            config,
            getattr(algo, "_strategy_dir", None),
            run_id=str(getattr(ctx, "run_id", "")),
            run_type="lean",
            ctx=ctx,
        )

        run_id = record_pipeline_run(
            self._db,
            run_type="lean",
            run_date=ctx.today,
            strategy=str(config.get("model_name", "")),
            regime=ctx.regime,
            confidence=float(ctx.confidence) if ctx.confidence is not None else None,
            portfolio_value=post_snapshot["portfolio_value"],
            cash=post_snapshot["cash"],
            n_candidates=len(ctx.candidates),
            n_exits=len(ctx.exits),
            n_rotations=int(ctx.counters.get("rotations", 0)),
            n_buys=len(selected_tickers),
            buy_blocked=bool(getattr(ctx, "buy_blocked", False)),
            skip_buys=bool(getattr(ctx, "skip_buys", False)),
            bear_only=bool(getattr(ctx, "bear_only", False)),
            counters=getattr(ctx, "counters", {}) or {},
            run_bundle=run_bundle,
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
            excluded_holding_tickers=candidate_score_excluded_holding_tickers(config),
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
            portfolio_value=post_snapshot["portfolio_value"],
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
        validate_decision_trace_integrity(
            self._db,
            run_id,
            config,
            context="LeanAdapter._record_decision_trace",
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
