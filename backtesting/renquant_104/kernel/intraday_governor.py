"""Intraday protective-action governor — rate-limit / cooldown / session caps.

Roadmap task #26 ("Protection: intraday cadence + governors + validation").

WHY
---
`scripts/intraday_sell_104.sh` runs the `SellOnlyPipeline` every ~30 min during
market hours (launchd `com.renquant.intraday104`). Each pass can fire stop-loss /
trailing-stop / SDL / max-hold / model-sell exits against the freshest 5-min bar.
There is currently **no throttle** on how often a protective action may fire:

  * a transient intraday wick that recovers can trip a stop and churn a name;
  * the same symbol can be acted on on back-to-back passes;
  * there is no ceiling on how many protective actions a single session emits.

This module is the **governing primitive** for that gap. It answers one question
deterministically — "is this protective action permitted right now, given what
has already fired this session?" — via four independent governors:

  * ``per_symbol_cooldown_seconds`` — min seconds between two actions on the SAME
    symbol (debounces a flapping intraday print).
  * ``global_cooldown_seconds``     — min seconds between ANY two actions (throttles
    the whole loop so one bad bar can't flatten the book in one pass).
  * ``per_symbol_session_cap``      — max actions per symbol per session.
  * ``global_session_cap``          — max total actions across all symbols / session.

SCOPE / SAFETY (roadmap #26 constraint — "behind a flag, default off")
----------------------------------------------------------------------
* **Flag-gated, default OFF.** With ``enabled=False`` (the default), :meth:`decide`
  always permits and :meth:`record` is a no-op — i.e. importing/constructing this
  has **zero effect** on live behaviour until an operator sets policy values and
  flips the flag. This file does NOT wire itself into the sell path; wiring +
  the actual cooldown/cap numbers are a separate, operator-signed-off change.
* **Pure / deterministic.** No wall-clock, no I/O, no network. ``now_epoch`` and
  ``session_date`` are injected by the caller, so every decision is unit-testable
  and reproducible. :meth:`decide` never mutates; only :meth:`record` does, and
  only after an action is actually taken.
* **Loop cadence itself** (every ~30 min) is owned by launchd, not this module —
  the governor throttles *actions within* a pass, which is the safety-critical
  surface. The cooldown governors compose with that fixed cadence.

STATE / PERSISTENCE
-------------------
State is a plain JSON-serialisable dict, intended to live under the
``intraday_governor`` key of ``live_state.{broker}.json`` alongside the other
cross-run protection counters (``protection_breaches``, ``sell_streaks``,
``regime_state``) that `adapters/runner.py` already round-trips. The governor
auto-resets its per-session counters when ``session_date`` advances, so a stale
snapshot from a previous day starts clean without an explicit reset call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GovernorDecision", "IntradayGovernor"]


@dataclass(frozen=True)
class GovernorDecision:
    """Outcome of a single permission check.

    ``governor`` names which rule blocked (``""`` when allowed) so callers can
    log/alert with attribution rather than a bare boolean.
    """

    allowed: bool
    reason: str
    governor: str = ""


@dataclass
class IntradayGovernor:
    """Deterministic policy over intraday protective actions.

    Construct via :meth:`from_config` (reads an ``intraday_governor`` config
    block) and rehydrate per-run state via :meth:`load_state`. A value of ``0``
    for any limit disables *that* governor while leaving the others active.
    """

    enabled: bool = False
    per_symbol_cooldown_seconds: float = 0.0
    global_cooldown_seconds: float = 0.0
    per_symbol_session_cap: int = 0
    global_session_cap: int = 0

    # ── mutable per-session state (persisted under live_state.intraday_governor)
    session_date: str = ""
    last_action_epoch: dict[str, float] = field(default_factory=dict)
    symbol_counts: dict[str, int] = field(default_factory=dict)
    global_count: int = 0
    last_global_action_epoch: float | None = None

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "IntradayGovernor":
        """Build from an ``intraday_governor`` config block (``None`` → disabled)."""
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            per_symbol_cooldown_seconds=_non_negative_float(
                cfg.get("per_symbol_cooldown_seconds", 0) or 0,
                "per_symbol_cooldown_seconds",
            ),
            global_cooldown_seconds=_non_negative_float(
                cfg.get("global_cooldown_seconds", 0) or 0,
                "global_cooldown_seconds",
            ),
            per_symbol_session_cap=_non_negative_int(
                cfg.get("per_symbol_session_cap", 0) or 0,
                "per_symbol_session_cap",
            ),
            global_session_cap=_non_negative_int(
                cfg.get("global_session_cap", 0) or 0,
                "global_session_cap",
            ),
        )

    def load_state(self, state: dict[str, Any] | None) -> "IntradayGovernor":
        """Rehydrate persisted counters in place (returns self for chaining)."""
        state = state or {}
        self.session_date = str(state.get("session_date", "") or "")
        self.last_action_epoch = {
            str(k): _non_negative_float(v, f"last_action_epoch[{k!r}]")
            for k, v in (state.get("last_action_epoch") or {}).items()
        }
        self.symbol_counts = {
            str(k): _non_negative_int(v, f"symbol_counts[{k!r}]")
            for k, v in (state.get("symbol_counts") or {}).items()
        }
        self.global_count = _non_negative_int(state.get("global_count", 0) or 0, "global_count")
        lge = state.get("last_global_action_epoch", None)
        self.last_global_action_epoch = (
            _non_negative_float(lge, "last_global_action_epoch")
            if lge is not None else None
        )
        return self

    def to_state(self) -> dict[str, Any]:
        """Serialise counters for persistence under ``live_state.intraday_governor``."""
        return {
            "session_date": self.session_date,
            "last_action_epoch": dict(self.last_action_epoch),
            "symbol_counts": dict(self.symbol_counts),
            "global_count": self.global_count,
            "last_global_action_epoch": self.last_global_action_epoch,
        }

    # ── session rollover ──────────────────────────────────────────────────────
    def _roll_session(self, session_date: str) -> None:
        """Clear per-session counters when the trading session advances.

        Idempotent within a session; a fresh ``session_date`` zeroes counts and
        cooldowns so yesterday's snapshot never throttles today.
        """
        if session_date != self.session_date:
            self.session_date = session_date
            self.last_action_epoch = {}
            self.symbol_counts = {}
            self.global_count = 0
            self.last_global_action_epoch = None

    # ── decision (pure: no mutation) ──────────────────────────────────────────
    def decide(self, symbol: str, now_epoch: float, session_date: str) -> GovernorDecision:
        """Return whether a protective action on ``symbol`` is permitted now.

        Pure read of current state — does NOT mutate and does NOT record the
        action. Callers must call :meth:`record` after an action actually fires.
        Disabled governor (the default) always permits.
        """
        if not self.enabled:
            return GovernorDecision(True, "governor disabled", "")

        # Evaluate against a fresh session WITHOUT mutating self (decide is pure):
        # a snapshot from an earlier session contributes no counts/cooldowns.
        if session_date != self.session_date:
            return GovernorDecision(True, "ok (new session)", "")

        sym = symbol.upper()

        if self.global_cooldown_seconds > 0 and self.last_global_action_epoch is not None:
            elapsed = now_epoch - self.last_global_action_epoch
            if elapsed < self.global_cooldown_seconds:
                return GovernorDecision(
                    False,
                    f"global cooldown: {elapsed:.0f}s < {self.global_cooldown_seconds:.0f}s",
                    "global_cooldown",
                )

        if self.per_symbol_cooldown_seconds > 0:
            last = self.last_action_epoch.get(sym)
            if last is not None:
                elapsed = now_epoch - last
                if elapsed < self.per_symbol_cooldown_seconds:
                    return GovernorDecision(
                        False,
                        f"{sym} cooldown: {elapsed:.0f}s < {self.per_symbol_cooldown_seconds:.0f}s",
                        "per_symbol_cooldown",
                    )

        if self.global_session_cap > 0 and self.global_count >= self.global_session_cap:
            return GovernorDecision(
                False,
                f"global session cap reached: {self.global_count}/{self.global_session_cap}",
                "global_session_cap",
            )

        if self.per_symbol_session_cap > 0:
            count = self.symbol_counts.get(sym, 0)
            if count >= self.per_symbol_session_cap:
                return GovernorDecision(
                    False,
                    f"{sym} session cap reached: {count}/{self.per_symbol_session_cap}",
                    "per_symbol_session_cap",
                )

        return GovernorDecision(True, "ok", "")

    # ── record (mutates state) ────────────────────────────────────────────────
    def record(self, symbol: str, now_epoch: float, session_date: str) -> None:
        """Register that a protective action on ``symbol`` fired at ``now_epoch``.

        Rolls the session if needed, then advances cooldown timestamps and
        counters. No-op when the governor is disabled (keeps state empty so the
        off→on transition starts from a clean slate).
        """
        if not self.enabled:
            return
        self._roll_session(session_date)
        sym = symbol.upper()
        self.last_action_epoch[sym] = now_epoch
        self.last_global_action_epoch = now_epoch
        self.symbol_counts[sym] = self.symbol_counts.get(sym, 0) + 1
        self.global_count += 1


def _non_negative_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    as_float = float(value)
    if not math.isfinite(as_float) or as_float < 0 or not as_float.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(as_float)
