"""Per-ticker runtime risk gates — RealizedVolGate + PositionConcentrationGate.

Pipeline gap surfaced 2026-05-03: ``BuyGatesJob`` only runs regime-level
gates (drawdown, transition window, confidence veto, BULL_VOL block, BEAR
branch, velocity crash, EMA50). ``TickerCandidateJob`` filters earnings,
wash-sale, and score threshold but has NO per-ticker risk filter for
realized volatility or current-portfolio concentration. Universe-admission
filters (Stage 1 ADV ≥ $10M, age ≥ 3y) keep small/illiquid names out of
the watchlist, but once admitted there's no runtime gate that says "this
high-vol name is too dangerous to add today" or "you already hold 15% of
portfolio in this ticker — refuse to add more".

Today's e2e on stale data sized LITE at 9.6% of portfolio in a single
share — both LITE's realized vol and the concentration deserve a hard
runtime check beyond the QP's soft constraints.

Two task classes here:

  * ``RealizedVolGateTask`` — drops buy candidates whose trailing-N-day
    annualized realized volatility exceeds ``risk_gates.realized_vol.
    max_annualized``. Default cap = 0.60 (60% annualized — Russell 2000
    median is ~0.30, so 0.60 already excludes only the high-tail names).

  * ``PositionConcentrationGateTask`` — drops buy candidates whose
    CURRENT portfolio weight already meets or exceeds ``risk_gates.
    position_concentration.max_pct``. Default cap = 0.15 (15%). The
    QP's own weight bound is a soft target; this gate is the hard
    "never add to a name already at cap" enforcement.

Wired in InferencePipeline AFTER the buy-candidate scoring (Phase 2b) and
BEFORE Phase 3 (PanelScoringJob → RankingJob → ...). Disabled by setting
``risk_gates.{realized_vol,position_concentration}.enabled = false`` —
default true.

Invariant: no candidate buys a ticker whose realized vol exceeds the cap
or which already owns ≥ cap% of portfolio.
"""
from __future__ import annotations

import logging
import math

from .pipeline import Task

log = logging.getLogger("kernel.pipeline.risk_gates")


class RealizedVolGateTask(Task):
    """Drop buy candidates with trailing realized vol over cap.

    Reads:
      ctx.config['risk_gates']['realized_vol'] = {enabled, max_annualized, window_days}
      ctx.candidates (list[CandidateResult])
      ctx.ohlcv (dict[ticker, DataFrame with 'close'])
    Writes:
      ctx.candidates — drops violators in place
      ctx.counters['risk_gate_vol_dropped'] — count of drops
    """

    def run(self, ctx) -> bool:
        cfg = (ctx.config or {}).get("risk_gates", {}).get("realized_vol", {})
        if cfg.get("enabled", True) is False:
            return True
        cap = float(cfg.get("max_annualized", 0.60))
        window = int(cfg.get("window_days", 60))

        candidates = list(getattr(ctx, "candidates", []) or [])
        if not candidates:
            return True

        kept, dropped = [], []
        blocked = getattr(ctx, "_blocked_by_ticker", None)
        if blocked is None:
            blocked = {}
            ctx._blocked_by_ticker = blocked
        for cand in candidates:
            tkr = getattr(cand, "ticker", None)
            df = (getattr(ctx, "ohlcv", None) or {}).get(tkr)
            vol = self._realized_vol_annualized(df, window)
            if vol is None:
                # Insufficient history → permissive (don't drop unknown)
                kept.append(cand)
                continue
            if vol > cap:
                dropped.append((tkr, vol))
                if tkr:
                    blocked.setdefault(tkr, "risk_gate_vol")
            else:
                kept.append(cand)

        if dropped:
            sample = ", ".join(f"{t}({v:.0%})" for t, v in dropped[:5])
            extra = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
            log.info(
                "RealizedVolGateTask: dropped %d/%d candidates over %.0f%% "
                "annualized vol cap (window=%dd): %s%s",
                len(dropped), len(candidates), cap * 100, window, sample, extra,
            )
            ctx.counters["risk_gate_vol_dropped"] = (
                ctx.counters.get("risk_gate_vol_dropped", 0) + len(dropped)
            )

        ctx.candidates = kept
        return True

    @staticmethod
    def _realized_vol_annualized(df, window: int) -> float | None:
        if df is None:
            return None
        try:
            close = df["close"]
        except (KeyError, TypeError):
            return None
        if len(close) < max(window, 5):
            return None
        rets = close.pct_change().tail(window).dropna()
        if len(rets) < max(window // 2, 5):
            return None
        std = float(rets.std())
        if not math.isfinite(std):
            return None
        return std * math.sqrt(252.0)


class PositionConcentrationGateTask(Task):
    """Drop buy candidates whose existing weight already ≥ cap.

    The QP's per-asset weight bound is a *soft* target driven by μ/σ
    optimization. This gate is the *hard* refusal — once a position's
    current portfolio weight meets or exceeds ``max_pct``, no additional
    buy will be emitted for it this bar. Sells / trims still flow through
    unrelated paths.

    Reads:
      ctx.config['risk_gates']['position_concentration'] = {enabled, max_pct}
      ctx.candidates (list[CandidateResult])
      ctx.holdings (dict[ticker, HoldingState])
      ctx.prices (dict[ticker, float])
      ctx.portfolio_value (float; falls back to derived sum if zero)
    Writes:
      ctx.candidates — drops violators in place
      ctx.counters['risk_gate_concentration_dropped'] — count
    """

    def run(self, ctx) -> bool:
        cfg = (ctx.config or {}).get("risk_gates", {}).get("position_concentration", {})
        if cfg.get("enabled", True) is False:
            return True
        cap_pct = float(cfg.get("max_pct", 0.15))

        candidates = list(getattr(ctx, "candidates", []) or [])
        if not candidates:
            return True

        equity = self._portfolio_equity(ctx)
        # 2026-05-04 audit Issue 41 fix: NaN equity slipped past `<= 0`
        # (NaN comparisons all False) → division by NaN → NaN concentration
        # → `pct >= cap_pct` is False → ALL candidates silently kept,
        # gate disabled. Fail-SAFE: skip with explicit isfinite check.
        if not math.isfinite(equity) or equity <= 0:
            log.warning(
                "PositionConcentrationGateTask: portfolio_value=%s "
                "(non-finite or zero) — skipping concentration check.",
                equity,
            )
            return True

        holdings = getattr(ctx, "holdings", None) or {}
        prices = getattr(ctx, "prices", None) or {}

        kept, dropped = [], []
        blocked = getattr(ctx, "_blocked_by_ticker", None)
        if blocked is None:
            blocked = {}
            ctx._blocked_by_ticker = blocked
        for cand in candidates:
            tkr = getattr(cand, "ticker", None)
            held = holdings.get(tkr) if tkr else None
            if held is None:
                kept.append(cand)
                continue
            shares = float(getattr(held, "shares", 0.0) or 0.0)
            px = float(prices.get(tkr, 0.0) or getattr(held, "prev_close", 0.0) or 0.0)
            value = shares * px
            pct = value / equity if equity > 0 else 0.0
            if pct >= cap_pct:
                dropped.append((tkr, pct))
                if tkr:
                    blocked.setdefault(tkr, "risk_gate_concentration")
            else:
                kept.append(cand)

        if dropped:
            sample = ", ".join(f"{t}({p:.1%})" for t, p in dropped[:5])
            extra = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
            log.info(
                "PositionConcentrationGateTask: dropped %d/%d candidates "
                "already ≥ %.1f%% portfolio weight: %s%s",
                len(dropped), len(candidates), cap_pct * 100, sample, extra,
            )
            ctx.counters["risk_gate_concentration_dropped"] = (
                ctx.counters.get("risk_gate_concentration_dropped", 0) + len(dropped)
            )

        ctx.candidates = kept
        return True

    @staticmethod
    def _portfolio_equity(ctx) -> float:
        pv = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
        if pv > 0:
            return pv
        # Fallback: cash + sum(shares*price) over holdings
        cash = float(getattr(ctx, "cash", 0.0) or 0.0)
        prices = getattr(ctx, "prices", None) or {}
        holdings = getattr(ctx, "holdings", None) or {}
        held_value = 0.0
        for t, h in holdings.items():
            s = float(getattr(h, "shares", 0.0) or 0.0)
            p = float(prices.get(t, 0.0) or getattr(h, "prev_close", 0.0) or 0.0)
            held_value += s * p
        return cash + held_value
