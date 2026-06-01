"""SnapshotHoldingsTask — records per-day per-position features for meta-label training.

Reads ctx (post-pipeline state with exits + candidates resolved) and
writes one row per held ticker to ``ctx.snapshot_logger``. No-op when
the logger is absent / None.

Architectural placement: end of :class:`MetaLabelLoggingJob`, run as
the very last job in InferencePipeline (after Selection / JointActions
so ctx.exits is finalized).

This task is single-responsibility per CLAUDE.md §1c (one of:
``{extract, validate, compute, transform, persist, emit}``). It
EMITS to the logger; the logger persists later (on adapter teardown).
"""
from __future__ import annotations

import datetime
import math
from typing import Any

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task
from renquant_pipeline.kernel.exit_types import META_LABEL_VETO_ELIGIBLE

# Regime → integer code (snapshot column `regime_code`). Pinned here
# so the meta-label classifier sees a stable encoding.
_REGIME_CODE: dict[str, int] = {
    "BULL_CALM":     0,
    "BULL_VOLATILE": 1,
    "CHOPPY":        2,
    "BEAR":          3,
}


def _cum_ret_over_window(rets: list[float], window: int) -> float:
    """Cumulative return over the last `window` elements of `rets`.

    Returns NaN if not enough data (early bars in sim). Uses geometric
    compounding for fidelity to actual portfolio return.
    """
    if not rets or len(rets) < window:
        return float("nan")
    prod = 1.0
    for r in rets[-window:]:
        try:
            f = float(r)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(f):
            return float("nan")
        prod *= (1.0 + f)
    return prod - 1.0


def _realized_vol(rets: list[float], window: int) -> float:
    """Annualised realised vol of the last `window` returns. NaN if insufficient."""
    if not rets or len(rets) < window:
        return float("nan")
    seg = [float(r) for r in rets[-window:]
           if r is not None and math.isfinite(float(r))]
    if len(seg) < 2:
        return float("nan")
    mean = sum(seg) / len(seg)
    var  = sum((r - mean) ** 2 for r in seg) / (len(seg) - 1)
    if var <= 0 or not math.isfinite(var):
        return float("nan")
    return math.sqrt(var) * math.sqrt(252.0)


def _safe_div(a: Any, b: Any) -> float:
    """a/b → NaN on non-finite or zero divisor."""
    try:
        af = float(a); bf = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(af) and math.isfinite(bf)) or bf == 0:
        return float("nan")
    return af / bf


class SnapshotHoldingsTask(Task):
    """Emit one feature-row per held ticker to ctx.snapshot_logger.

    Guards:
      * ctx.snapshot_logger absent / None → silent no-op
      * empty ctx.holdings → no-op
      * NaN inputs → row recorded with NaN values (don't drop the row;
        the meta-label trainer's PurgedKFold + NaN-tolerant XGB will
        handle them, OR a separate clean-up pass can filter)
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        logger = getattr(ctx, "snapshot_logger", None)
        if logger is None:
            return None
        if not ctx.holdings:
            return None

        today    = ctx.today
        date_iso = today.isoformat() if hasattr(today, "isoformat") else str(today)

        # ── Market features (shared across all positions this bar) ─────
        spy_returns = list(ctx.spy_returns) if getattr(ctx, "spy_returns", None) else []
        spy_5d   = _cum_ret_over_window(spy_returns, 5)
        spy_20d  = _cum_ret_over_window(spy_returns, 20)
        spy_60d  = _cum_ret_over_window(spy_returns, 60)
        spy_vol  = _realized_vol(spy_returns, 20)
        regime_code = _REGIME_CODE.get(ctx.regime, -1)
        confidence  = float(ctx.confidence) if hasattr(ctx, "confidence") else float("nan")

        # Trigger lookup — only path-rule exits are eligible for the
        # meta-label veto. Model/QP exits must not leak into training labels.
        triggers_by_ticker: dict[str, str] = {}
        for ticker, sig in (getattr(ctx, "exits", []) or []):
            if sig is not None and getattr(sig, "should_exit", False):
                exit_type = getattr(sig, "exit_type", "")
                if exit_type in META_LABEL_VETO_ELIGIBLE:
                    triggers_by_ticker[ticker] = exit_type
        n_triggers = len(triggers_by_ticker)

        # Portfolio context
        pv  = float(ctx.portfolio_value) if hasattr(ctx, "portfolio_value") else float("nan")
        hwm = float(ctx.hwm)              if hasattr(ctx, "hwm")              else float("nan")
        port_dd = 0.0
        if math.isfinite(pv) and math.isfinite(hwm) and hwm > 0:
            port_dd = max(0.0, (hwm - pv) / hwm)

        # Holdings ranked by panel_score (for rank-among-holdings feature)
        scored = [
            (tkr, hs.panel_score if hs.panel_score is not None else float("-inf"))
            for tkr, hs in ctx.holdings.items()
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        rank_by_ticker = {tkr: i for i, (tkr, _) in enumerate(scored)}

        # ── Per-holding rows ───────────────────────────────────────────
        for ticker, hs in ctx.holdings.items():
            price = ctx.prices.get(ticker) if hasattr(ctx, "prices") else None

            # Position state
            cum_pnl = _safe_div(price - hs.entry_price if price is not None else float("nan"),
                                  hs.entry_price)
            peak_gain = _safe_div(hs.high_watermark - hs.entry_price, hs.entry_price)
            dd_from_peak = _safe_div(hs.high_watermark - price
                                       if price is not None else float("nan"),
                                       hs.high_watermark)
            days_held = 0
            if hs.entry_date is not None and isinstance(today, datetime.date):
                days_held = max(0, (today - hs.entry_date).days)

            # Trigger flags
            trig = triggers_by_ticker.get(ticker, "")
            is_sl    = 1 if trig == "stop_loss"        else 0
            is_trail = 1 if trig == "trailing_stop"    else 0
            is_sdl   = 1 if trig == "single_day_loss"  else 0
            is_mh    = 1 if trig == "max_hold"         else 0
            any_trig = 1 if trig                       else 0

            # Position weight
            pos_weight = float("nan")
            if hasattr(hs, "shares") and hs.shares and price is not None and pv > 0:
                pos_weight = (float(hs.shares) * float(price)) / pv

            row = {
                "date":   date_iso,
                "ticker": ticker,

                "cum_pnl_pct":             cum_pnl,
                "peak_gain_pct":           peak_gain,
                "drawdown_from_peak_pct":  dd_from_peak,
                "days_held":               days_held,
                "consec_underwater_days":  0,    # TODO: track in HoldingState
                "prev_day_return":         float("nan"),
                "gap_open_pct":            float("nan"),
                "realized_vol_20d":        hs.realized_sigma_daily * math.sqrt(252.0)
                                            if hs.realized_sigma_daily is not None
                                            else float("nan"),

                "spy_5d_ret":              spy_5d,
                "spy_20d_ret":             spy_20d,
                "spy_60d_ret":             spy_60d,
                "spy_realized_vol_20d":    spy_vol,
                "regime_code":             regime_code,
                "regime_just_switched":    0,   # TODO: track regime history
                "regime_confidence":       confidence,

                "panel_score_current":             hs.panel_score
                                                     if hs.panel_score is not None
                                                     else float("nan"),
                "panel_score_at_entry":            hs.entry_panel_score
                                                     if hs.entry_panel_score is not None
                                                     else float("nan"),
                "panel_score_delta":               (hs.panel_score - hs.entry_panel_score)
                                                     if hs.panel_score is not None
                                                     and hs.entry_panel_score is not None
                                                     else float("nan"),
                "panel_score_rank_among_holdings": rank_by_ticker.get(ticker, -1),
                "mu_current":              hs.mu    if hs.mu    is not None else float("nan"),
                "sigma_current":           hs.sigma if hs.sigma is not None else float("nan"),

                "position_weight":               pos_weight,
                "sector_concentration":          float("nan"),  # TODO: needs sector map at task
                "portfolio_drawdown_now":        port_dd,
                "n_concurrent_exits_this_bar":   n_triggers,

                "trigger_stop_loss":         is_sl,
                "trigger_trailing_stop":     is_trail,
                "trigger_single_day_loss":   is_sdl,
                "trigger_max_hold":          is_mh,
                "any_trigger":               any_trig,

                # Outcomes filled by triple-barrier labeler (P4.2)
                "fwd_5d_ret":  float("nan"),
                "fwd_20d_ret": float("nan"),
            }
            logger.record(row)

        return None
