"""P-BROKER-FILL-FRESHNESS — alert when broker hasn't filled a runner order recently.

2026-06-02 daily decision-tree audit Finding 9: ``monitor_state.last_fill_date``
was ``2026-05-27`` — broker hadn't filled a runner-initiated order in 5
trading days. The strategy is in a stationary fixed-point loop:

  * ``regime_admission`` blocks every BULL_CALM candidate (artifact is
    ``promotion_status=gated_buys``)
  * QP infeasible because one holding (ORCL) is over per-asset cap and
    no buy slack exists to redistribute
  * No external trigger has changed; tomorrow's run produces the same
    counters as today's

That equilibrium is **not surfaced** anywhere in the daily preflight or
ntfy. The operator only finds out by reading ``monitor_state`` from the
state file. This task lifts that signal into the preflight surface so a
multi-day no-fill streak triggers a soft warn (or hard fail when the
streak crosses the operator-set red line).

Default thresholds (per CLAUDE.md §1.4 every knob per-regime; default
sits under ``monitoring.fill_freshness_*``):

  * ``warn_after_trading_days = 5``  — soft yellow alert
  * ``hard_after_trading_days = 20`` — hard red (something is broken)

These align with the existing ``monitoring.max_no_trade_days = 15`` knob
on the streak counter. The hard cap is set conservatively (20) so a
disciplined Kelly-driven no-trade period (legitimate quiet market) does
not fail-close production by accident.
"""
from __future__ import annotations

import datetime as _dt

from kernel.preflight import PreflightCheck  # noqa: PLC0415 (legacy bridge)

from ..base import PreflightTask
from ..ctx import PreflightContext


class BrokerFillFreshnessTask(PreflightTask):
    """P-BROKER-FILL-FRESHNESS — runner-initiated broker fill within N
    trading days.

    Behaviour:

    * ctx.broker is None → soft pass ("dry-run; skip")
    * broker lacks ``get_filled_orders`` → soft pass ("not surfaceable")
    * 0 fills in the configured lookback window AND streak ≥ hard cap
      → HARD FAIL. Operator must intervene (retrain / config flip /
      cap-compliance opt-in / etc.).
    * 0 fills AND streak ≥ warn cap → SOFT WARN.
    * Otherwise → HARD PASS.
    """

    check_name = "P-BROKER-FILL-FRESHNESS"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if ctx.broker is None:
            return PreflightCheck(
                self.check_name, "soft", True,
                "no broker (dry-run); skip",
            )
        if not hasattr(ctx.broker, "get_filled_orders"):
            # Sim broker, paper sandboxes without fill history etc. Skip
            # rather than fail-close because this check is purely about
            # observability, not artifact correctness.
            return PreflightCheck(
                self.check_name, "soft", True,
                "broker does not expose get_filled_orders; skip",
            )
        cfg = (ctx.config.get("monitoring", {}) or {})
        warn_after = int(cfg.get("fill_freshness_warn_after_trading_days", 5))
        hard_after = int(cfg.get("fill_freshness_hard_after_trading_days", 20))
        # 120-calendar-day lookback matches the runner's stateful streak
        # override window (adapters/runner.py::_override_no_trade_streak_from_broker)
        # so the two surfaces agree on what counts as a fill.
        today = _dt.date.today()
        after = (today - _dt.timedelta(days=120)).isoformat()
        try:
            fills = ctx.broker.get_filled_orders(after=after) or []
        except Exception as exc:  # noqa: BLE001
            # Don't fail-close on transient broker API errors; this check
            # is about observability and the daily run should continue
            # so the operator sees the rest of the preflight verdict.
            return PreflightCheck(
                self.check_name, "soft", True,
                f"broker.get_filled_orders failed ({exc}); skipping freshness check",
            )
        fill_dates: set[_dt.date] = set()
        for f in fills:
            iso = f.get("filled_at")
            if not iso:
                continue
            try:
                if iso.endswith("Z"):
                    iso = iso[:-1] + "+00:00"
                fill_dates.add(_dt.datetime.fromisoformat(iso).date())
            except Exception:
                continue
        if not fill_dates:
            streak_td = "≥120"
            streak_int = 120
            most_recent_str = "none in 120d"
        else:
            most_recent = max(fill_dates)
            # Trading-day count strictly between most_recent and today.
            from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415
            streak_int = 0
            d = most_recent + _dt.timedelta(days=1)
            while d <= today:
                if _is_nyse_trading_day(d):
                    streak_int += 1
                d += _dt.timedelta(days=1)
            streak_td = str(streak_int)
            most_recent_str = most_recent.isoformat()
        # Decision tree (hard cap takes precedence over warn cap).
        if streak_int >= hard_after:
            return PreflightCheck(
                self.check_name, "hard", False,
                f"no runner-driven broker fill in {streak_td} trading days "
                f"(hard cap {hard_after}); last_fill={most_recent_str}. "
                f"Strategy is dormant — investigate gates (regime_admission, "
                f"gated_buys, QP cap-compliance) before next live cycle.",
            )
        if streak_int >= warn_after:
            return PreflightCheck(
                self.check_name, "soft", True,
                f"no runner-driven broker fill in {streak_td} trading days "
                f"(warn cap {warn_after}); last_fill={most_recent_str}. "
                f"Strategy may be stuck in a no-trade equilibrium.",
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"last runner fill {most_recent_str} ({streak_td} trading days ago)",
        )
