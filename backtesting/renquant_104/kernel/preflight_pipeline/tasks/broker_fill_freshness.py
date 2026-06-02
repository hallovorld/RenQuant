"""P-BROKER-FILL-FRESHNESS — alert when runner-driven activity is stale.

2026-06-02 daily decision-tree audit Finding 9: the strategy had not
emitted a runner-driven decision in many trading days. The "0 trades
today" result was a stable equilibrium — `regime_admission` blocked
every candidate (artifact is `promotion_status=gated_buys`), QP was
infeasible because one holding was over per-asset cap (ORCL at 22%),
no external trigger had changed. Nothing in the preflight surface
flagged it; the operator had to read ``monitor_state`` from the state
file.

This task surfaces that streak as a preflight check.

**Codex PR #84 review fix (2026-06-02 evening):**

The first iteration of this task counted any row from
``broker.get_filled_orders()`` and reset the streak on the most recent
``filled_at``. That conflated three distinct sources of broker activity:

  * **runner-driven** — fills triggered by ``broker.place_order(...)``
    inside ``RunnerAdapter`` (Phase 2a sells, Kelly/QP buys). These
    ARE alpha decisions.
  * **broker-side stop (Z9)** — Alpaca-managed protective stops the
    runner installs once, then the broker fills automatically when
    price hits. NOT a fresh alpha decision.
  * **external / manual** — operator close at Alpaca's UI, corporate
    action, etc. Surfaced via ``STATE-EXT-SELL`` in the runner log.
    NOT a runner decision.

A manual close yesterday or a Z9 stop firing two days ago would have
mis-reset the streak to "0 trading days, HARD PASS", defeating the
audit's intent.

This task now reads the persisted runner-emission field
``monitor_state.last_activity_date`` (set by
``MonitorIdleStreakTask`` from ``ctx.orders`` / ``ctx.exits`` — i.e.,
orders the runner actually emitted this bar). Companion runner.py
change stops the previous override from clobbering this field with
broker-truth fills.

**Decision tree**

* HARD PASS — runner activity within ``warn_after_trading_days`` (default 5)
* SOFT WARN — runner activity between warn and hard caps
* HARD FAIL — runner activity older than ``hard_after_trading_days``
  (default 20) → "Strategy is dormant — investigate gates
  (regime_admission, gated_buys, QP cap-compliance) before next
  live cycle."
* SOFT PASS (skip) — dry-run / no broker_name / state file absent /
  unparseable. Soft-pass paths exist so this preflight doesn't fail
  closed on developer/sim contexts.

Tuning via ``monitoring.fill_freshness_warn_after_trading_days`` and
``…_hard_after_trading_days``. Conservative defaults so a disciplined
Kelly no-trade period (legitimate quiet market) does NOT fail-close
production by accident.
"""
from __future__ import annotations

import datetime as _dt
import json

from kernel.preflight import PreflightCheck  # noqa: PLC0415 (legacy bridge)

from ..base import PreflightTask
from ..ctx import PreflightContext


class BrokerFillFreshnessTask(PreflightTask):
    """P-BROKER-FILL-FRESHNESS — runner-driven activity within N trading days."""

    check_name = "P-BROKER-FILL-FRESHNESS"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if not ctx.broker_name:
            return PreflightCheck(
                self.check_name, "soft", True,
                "no broker_name (dry-run); skip",
            )
        try:
            from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"state_paths unavailable: {exc}; skip",
            )
        try:
            state_path, _ = resolve_live_state_read(
                ctx.strategy_dir, ctx.broker_name,
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"resolve_live_state_read failed: {exc}; skip",
            )
        if not state_path.exists():
            return PreflightCheck(
                self.check_name, "soft", True,
                f"state file absent at {state_path} (first run?); skip",
            )
        try:
            state = json.loads(state_path.read_text())
        except Exception as exc:  # noqa: BLE001
            return PreflightCheck(
                self.check_name, "soft", True,
                f"state file unparseable: {exc}; skip",
            )
        mon = state.get("monitor_state") or {}
        # codex PR #84 fix: read RUNNER-EMISSION truth, NOT broker-fill truth.
        #
        # ``last_activity_date`` is set by MonitorIdleStreakTask from
        # ``ctx.orders`` / ``ctx.exits`` — runner-issued orders/exits this
        # bar. Companion runner.py change stops the previous override from
        # clobbering it with broker-truth fills.
        #
        # If ``last_activity_date`` is absent (state file from a code
        # version that didn't write it, OR runner has never emitted an
        # order on this broker), fall back to ``first_trade_date`` — same
        # runner-emission field, set on the FIRST emission. If both are
        # absent, the runner has never traded → streak == ∞ → HARD FAIL.
        last_runner_str = (
            mon.get("last_activity_date") or mon.get("first_trade_date") or ""
        )
        cfg = (ctx.config.get("monitoring", {}) or {})
        warn_after = int(cfg.get("fill_freshness_warn_after_trading_days", 5))
        hard_after = int(cfg.get("fill_freshness_hard_after_trading_days", 20))
        today = _dt.date.today()
        if not last_runner_str:
            return PreflightCheck(
                self.check_name, "hard", False,
                f"no runner-driven activity recorded in monitor_state "
                f"(last_activity_date / first_trade_date both absent); "
                f"hard cap {hard_after} trading days. Strategy has never "
                f"emitted a runner order on this broker — investigate "
                f"gates (regime_admission, gated_buys, QP cap-compliance) "
                f"before next live cycle.",
            )
        try:
            last_runner = _dt.date.fromisoformat(last_runner_str.split("T")[0])
        except Exception:
            return PreflightCheck(
                self.check_name, "soft", True,
                f"monitor_state.last_activity_date unparseable "
                f"({last_runner_str!r}); skip",
            )
        try:
            from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415
            streak_int = 0
            d = last_runner + _dt.timedelta(days=1)
            while d <= today:
                if _is_nyse_trading_day(d):
                    streak_int += 1
                d += _dt.timedelta(days=1)
        except Exception:
            # Fallback: calendar-day count (still meaningful — slightly
            # over-counts because weekends count as 0 trading days).
            streak_int = max((today - last_runner).days, 0)
        if streak_int >= hard_after:
            return PreflightCheck(
                self.check_name, "hard", False,
                f"no runner-driven activity in {streak_int} trading days "
                f"(hard cap {hard_after}); last_activity={last_runner_str}. "
                f"Strategy is dormant — investigate gates (regime_admission, "
                f"gated_buys, QP cap-compliance) before next live cycle.",
            )
        if streak_int >= warn_after:
            return PreflightCheck(
                self.check_name, "soft", True,
                f"no runner-driven activity in {streak_int} trading days "
                f"(warn cap {warn_after}); last_activity={last_runner_str}. "
                f"Strategy may be stuck in a no-trade equilibrium.",
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"last runner-driven activity {last_runner_str} "
            f"({streak_int} trading days ago)",
        )
