"""Canonical exit-type taxonomy — single source of truth.

Refactor 2026-05-11 (§5.13.5 — one business decision, one function).
Previously 5+ overlapping frozensets across task_*.py defined ad-hoc
membership for "which exits bypass this gate". The risk: adding a new
exit type (e.g. a future "drawdown_flatten") required hunting down all
5 callers and remembering which set to extend; a missed update silently
let the new type fall through wrong gates.

This module owns the taxonomy. Every task imports from here.

Base sets (atomic groups):
  * ``PATH_RULE_CORE``       — mechanical price-rule exits, primary names
  * ``PATH_RULE_SYNONYMS``   — historical / variant names for the same
  * ``PORTFOLIO_RISK``       — portfolio-level (rotation, kelly_trim,
                               joint_sell, joint_rotation)
  * ``MODEL_DRIVEN``         — exits emitted by signal models
                               (model_sell, panel_conviction)

Derived sets (named for caller, composed from bases):
  * ``META_LABEL_VETO_ELIGIBLE``  — only PATH_RULE_CORE (no synonyms; the
                                    meta-label classifier was trained on
                                    canonical names)
  * ``PANEL_VETO_BYPASS``         — path + portfolio + synonyms (
                                    task_panel_veto.RISK_EXIT_TYPES)
  * ``PER_BAR_CAP_EXEMPT``        — path + portfolio + synonyms (
                                    task_limit_sells._RISK_EXIT_TYPES)
  * ``PER_BAR_CAP_SUBJECT``       — model-driven (task_limit_sells
                                    ._SOFT_SELL_TYPES)
  * ``PATH_DRIVEN_LEGACY``        — task_sell.PATH_DRIVEN_EXIT_TYPES
                                    (core + kelly_trim + rotation)
  * ``POST_STOP_COOLDOWN_TRIGGERS`` — price-stop variants only (no
                                     max_hold; that's a time exit)

References
----------
* CLAUDE.md §5.13.5 — One business decision = one function
* Audit 2026-05-11 — `task_*.py` had 5 overlapping frozensets
"""
from __future__ import annotations


# ── Base sets ──────────────────────────────────────────────────────────

PATH_RULE_CORE: frozenset[str] = frozenset({
    "stop_loss",
    "trailing_stop",
    "single_day_loss",
    "max_hold",
})

PATH_RULE_SYNONYMS: frozenset[str] = frozenset({
    "trailing_stop_loss",   # alias for trailing_stop
    "sdl",                  # alias for single_day_loss
    "gap_down",             # variant single-day fall
    "max_hold_days",        # alias for max_hold
})

PORTFOLIO_RISK: frozenset[str] = frozenset({
    "rotation",
    "kelly_trim",
    "joint_sell",
    "joint_rotation",
})

MODEL_DRIVEN: frozenset[str] = frozenset({
    "model_sell",
    "panel_conviction",
})


# ── Derived sets (named for caller) ────────────────────────────────────

META_LABEL_VETO_ELIGIBLE: frozenset[str] = PATH_RULE_CORE

PANEL_VETO_BYPASS: frozenset[str] = (
    PATH_RULE_CORE | PORTFOLIO_RISK | PATH_RULE_SYNONYMS
)

PER_BAR_CAP_EXEMPT: frozenset[str] = (
    PATH_RULE_CORE | PORTFOLIO_RISK | PATH_RULE_SYNONYMS
)

PER_BAR_CAP_SUBJECT: frozenset[str] = MODEL_DRIVEN

PATH_DRIVEN_LEGACY: frozenset[str] = (
    PATH_RULE_CORE | frozenset({"kelly_trim", "rotation"})
)

POST_STOP_COOLDOWN_TRIGGERS: frozenset[str] = frozenset({
    "trailing_stop", "trailing_stop_loss",
    "stop_loss",
    "single_day_loss", "sdl",
    "gap_down",
})


__all__ = [
    # Base sets
    "PATH_RULE_CORE",
    "PATH_RULE_SYNONYMS",
    "PORTFOLIO_RISK",
    "MODEL_DRIVEN",
    # Derived sets
    "META_LABEL_VETO_ELIGIBLE",
    "PANEL_VETO_BYPASS",
    "PER_BAR_CAP_EXEMPT",
    "PER_BAR_CAP_SUBJECT",
    "PATH_DRIVEN_LEGACY",
    "POST_STOP_COOLDOWN_TRIGGERS",
]
