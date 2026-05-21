"""MonitorIdleStreakTask — detect persistent no-trade periods.

Rationale: the user explicitly requires that a systematic no-trade period
(many consecutive days with zero orders placed AND zero exits) must be
treated as a failure mode, not a silent state. Silent no-trade spans mean
some upstream gate is broken (stale calibrator, mis-configured tier
thresholds, panel feature-frame missing, universe emptied, …) and the
strategy is just sitting in cash while the market moves.

This Task is a pipeline-level guard that runs AFTER SelectionJob / Rotation.
It reads the previous streak counter from a config-provided dict (injected
by the adapter / sim loop) and updates it based on today's activity:

  * `no_trade_streak`       — consecutive days with 0 orders + 0 exits
  * `no_candidate_streak`   — consecutive days with 0 candidates surviving
                              CandidateJob (even before ranking)
  * `first_trade_date`      — the first date an order or exit landed
  * `last_activity_date`    — the most recent date with any order/exit

Monitor-only: does not block trades. Surfaces counters on `ctx.counters`
for the adapter to persist or ntfy alert.
"""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.monitor")


class MonitorIdleStreakTask(Task):
    """Update and surface the no-trade / no-candidate streak counters.

    Reads `ctx.monitor_state` (dict), writes back updated counters plus
    emits a WARNING log when any streak exceeds the configured threshold
    so the scheduled-run ntfy captures it.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config.get("monitoring", {}) or {})
        max_idle = int(cfg.get("max_no_trade_days",      15))
        max_blnk = int(cfg.get("max_no_candidate_days",  15))

        # Prior streak state lives on ctx.monitor_state (persisted by adapter).
        state: dict = getattr(ctx, "monitor_state", None) or {}

        had_activity   = bool(ctx.orders) or bool(ctx.exits)
        had_candidates = bool(ctx.candidates)

        # 2026-05-20 fix: counter is PER-TRADING-DAY, not per-invocation.
        # Pre-fix, intraday SellOnlyPipeline ran every ~12 min during market
        # hours (~33 runs/day); each no-activity tick incremented the streak
        # by 1, inflating "consecutive days zero orders" by ~34× per actual
        # idle day. Observed 2026-05-20: streak=32 falsely alerted while
        # LIVE Alpaca account had 47 fills over 16 trading days in the prior
        # 40 calendar days (incl. TXN sell at 06:54:58 the same morning,
        # which DID reset streak to 0 — but ~33 intraday cron firings after
        # the reset then brought it back to 32 by the 14:17 daily run).
        # Track last_check_date; only increment once per trading day.
        today_str = str(ctx.today)
        last_check = state.get("last_check_date")

        prev_no_trade = int(state.get("no_trade_streak",     0))
        prev_no_cand  = int(state.get("no_candidate_streak", 0))

        if had_activity:
            new_no_trade = 0
        elif last_check == today_str:
            new_no_trade = prev_no_trade   # already counted this day
        else:
            new_no_trade = prev_no_trade + 1

        if had_candidates:
            new_no_cand = 0
        elif last_check == today_str:
            new_no_cand = prev_no_cand
        else:
            new_no_cand = prev_no_cand + 1

        state["last_check_date"] = today_str

        first_trade = state.get("first_trade_date")
        last_activity = state.get("last_activity_date")
        if had_activity:
            last_activity = str(ctx.today)
            if first_trade is None:
                first_trade = str(ctx.today)

        state["no_trade_streak"]     = new_no_trade
        state["no_candidate_streak"] = new_no_cand
        state["last_activity_date"]  = last_activity
        state["first_trade_date"]    = first_trade

        # Per-ticker filter streak (used by FilterAutoDropTask).
        # Watchlist tickers that don't appear in ctx.candidates this bar
        # had their candidate filtered (volume / earnings / etc). Increment
        # streak; reset to 0 when ticker reappears.
        # Only tracked when auto_drop enabled to keep state file slim.
        auto_drop = int(cfg.get("auto_drop_filter_days", 0) or 0)
        if auto_drop > 0:
            cand_set = {getattr(c, "ticker", None) for c in ctx.candidates}
            cand_set.discard(None)
            streaks: dict[str, int] = state.get("filter_streaks", {}) or {}
            watchlist = ctx.config.get("watchlist", []) or []
            for t in watchlist:
                if t in cand_set:
                    streaks[t] = 0
                else:
                    streaks[t] = int(streaks.get(t, 0)) + 1
            state["filter_streaks"] = streaks

        ctx.monitor_state = state
        ctx.counters["no_trade_streak"]     = new_no_trade
        ctx.counters["no_candidate_streak"] = new_no_cand

        # Promote to WARN when either streak exceeds the threshold — the
        # adapter / scheduled run picks this up through its log scraper.
        if new_no_trade > max_idle:
            log.warning(
                "NoTradeAlert: %d consecutive days with zero orders (limit=%d) — "
                "some upstream gate is blocking. Regime=%s  candidates=%d  holdings=%d",
                new_no_trade, max_idle, ctx.regime,
                len(ctx.candidates), len(ctx.holdings),
            )
        if new_no_cand > max_blnk:
            log.warning(
                "NoCandidateAlert: %d consecutive days with zero candidates "
                "surviving CandidateJob (limit=%d) — ScoreBuyTask rejecting all. "
                "Regime=%s  buy_blocked=%s",
                new_no_cand, max_blnk, ctx.regime, ctx.buy_blocked,
            )
