"""PostStopCooldownFilterTask — block re-entry within N bars of a stop.

User-driven 2026-05-04 (the APP killer): trailing_stop fired @ $587 on
2025-10-06; same day a 1-share rebuy executed at $587, then a
12-share top-up at $631. Within a week single_day_loss fired
at -5%. Same pattern repeated 2026-01-13: bought 15 shares at $668,
single_day_loss next day -7.6%.

Industry standard (cvxportfolio holding cost + practitioner protocol):
extend wash-sale concept to a post-stop cooldown that blocks RE-ENTRY
in the just-stopped name regardless of P&L sign. Wash-sale only kicks
in when there's a loss (IRS rule); post-stop blackout is a
risk-management protocol — the model is still bullish but the price
action invalidated entry timing.

Pre-conditions
--------------
* `risk.post_stop_cooldown.enabled = true` (default false; opt-in)
* `risk.post_stop_cooldown.bars = N` (default 5 trading days)
* `risk.post_stop_cooldown.exit_types = [...]` (default
  trailing_stop, stop_loss, single_day_loss)
* `ctx.last_stop_exit_dates` populated by the adapter when these
  exit_types fire (sim.py / runner.py / lean.py — adapter side)

Filter scope
------------
This task drops candidates whose ticker has a recent stop event from
`ctx.candidates` BEFORE PanelScoringJob runs. Downstream tasks
(SizeAndEmit, EmitRotations, JointAction, JointPortfolioQP, TopUp)
can additionally consult `ctx.last_stop_exit_dates` for defense in
depth, but the centralized filter here is the primary gate.
"""
from __future__ import annotations

import datetime
import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.post_stop_cooldown")


# Default exit types that trigger the cooldown.
#
# 2026-05-04 audit fix: max_hold / max_hold_days REMOVED. Those are
# TIME exits — a position ages out after N days regardless of P&L. The
# blackout is meant for *price-action stops* where the timing signal
# was bad. Time exits don't carry that signal, so blocking re-entry
# after a max_hold creates spurious churn — the model may legitimately
# want back in immediately if the score has refreshed. Audit P1-1.
DEFAULT_STOP_EXIT_TYPES: frozenset[str] = frozenset({
    "trailing_stop",
    "trailing_stop_loss",
    "stop_loss",
    "single_day_loss",
    "sdl",
    "gap_down",
})


def is_post_stop_blocked(
    ticker: str,
    today: datetime.date,
    last_stop_exit_dates: dict | None,
    cooldown_bars: int,
) -> bool:
    """Return True iff `ticker` had a stop-class exit within the last
    `cooldown_bars` trading days before `today`.

    The check counts CALENDAR days (not trading days) for simplicity.
    cooldown_bars=5 calendar days ≈ ~3-4 trading days; if precise
    trading-day counting matters, use kernel.exits._is_nyse_trading_day.

    Returns False when:
      * cooldown_bars <= 0 (disabled)
      * last_stop_exit_dates is None / empty
      * ticker not in dict (never stopped)
      * stop date is older than cooldown_bars
    """
    if cooldown_bars <= 0 or not last_stop_exit_dates:
        return False
    last = last_stop_exit_dates.get(ticker)
    if last is None:
        return False
    # Coerce string ISO dates if a JSON-loaded dict slipped in
    if isinstance(last, str):
        try:
            last = datetime.date.fromisoformat(last[:10])
        except ValueError:
            return False
    if not isinstance(last, datetime.date):
        return False
    delta = (today - last).days
    return 0 <= delta < cooldown_bars


class PostStopCooldownFilterTask(Task):
    """Drop candidates whose ticker had a stop-class exit recently.

    Reads:
      ctx.today, ctx.candidates, ctx.last_stop_exit_dates,
      ctx.config["risk"]["post_stop_cooldown"]
    Writes:
      ctx.candidates (in place — drops violators)
      ctx.counters["post_stop_blocked"]
    """

    name = "PostStopCooldownFilterTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config or {}).get("risk", {}).get("post_stop_cooldown", {}) or {}
        if not cfg.get("enabled", False):
            return True
        cooldown_bars = int(cfg.get("bars", 5))
        if cooldown_bars <= 0:
            return True

        candidates = list(getattr(ctx, "candidates", []) or [])
        if not candidates:
            return True

        last_stops = getattr(ctx, "last_stop_exit_dates", None) or {}
        if not last_stops:
            return True   # nothing to block

        today = getattr(ctx, "today", None)
        if today is None:
            return True
        if isinstance(today, datetime.datetime):
            today = today.date()

        kept, dropped = [], []
        for cand in candidates:
            tkr = getattr(cand, "ticker", None)
            if tkr and is_post_stop_blocked(tkr, today, last_stops, cooldown_bars):
                last = last_stops.get(tkr)
                dropped.append((tkr, last))
            else:
                kept.append(cand)

        if dropped:
            sample = ", ".join(f"{t}@{d}" for t, d in dropped[:5])
            extra = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
            log.info(
                "PostStopCooldownFilterTask: dropped %d/%d candidate(s) "
                "within %d-bar post-stop blackout: %s%s",
                len(dropped), len(candidates), cooldown_bars, sample, extra,
            )
            ctx.counters["post_stop_blocked"] = (
                ctx.counters.get("post_stop_blocked", 0) + len(dropped)
            )
        ctx.candidates = kept
        return True


__all__ = [
    "DEFAULT_STOP_EXIT_TYPES",
    "is_post_stop_blocked",
    "PostStopCooldownFilterTask",
]
