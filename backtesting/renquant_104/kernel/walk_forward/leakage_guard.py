"""Single-source-of-truth leakage guard for sim model loading.

Defends against the 2026-05-10 audit class: prod model trained 2026-05-09
used in a sim covering 2024-01 → 2026-03 (i.e. the model has seen
~26 months of forward labels relative to every bar in the backtest).

Per CLAUDE.md §5.13.5 (single source of truth): both legacy static-model
path AND walk-forward path in `adapters/sim.py` MUST call this function.
Adding a parallel implementation requires deleting this one first.

Per CLAUDE.md §5.13.3: the regression invariant lives in
`tests/test_leakage_guard.py::TestLeakageGuardRegression` — pin it.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def _to_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    """Coerce date / datetime / str / Timestamp to pd.Timestamp.

    Raises TypeError with a useful label when coercion fails — the leakage
    guard should never silently swallow a malformed input (silent swallow
    is exactly how the original class of bug shipped to prod).
    """
    if value is None:
        raise TypeError(f"{label} is None — cannot evaluate leakage")
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date, str)):
        try:
            return pd.Timestamp(value)
        except Exception as exc:  # pragma: no cover — defensive
            raise TypeError(f"{label}={value!r} not coercible to Timestamp: {exc}")
    raise TypeError(f"{label}={value!r} of type {type(value).__name__} not supported")


def assert_no_leakage(
    model_trained_date: Any,
    sim_today: Any,
    context: str = "",
) -> None:
    """Raise ValueError if model_trained_date >= sim_today.

    Both legacy static-model load AND walk-forward per-bar lookup route
    through this. The leakage check that should have been there from
    day one (see CLAUDE.md §5.13.5).

    Args:
        model_trained_date: date the model was trained (date / datetime /
            ISO-string / pd.Timestamp). Must be strictly less than
            sim_today.
        sim_today: the sim bar's "today" — typically the last bar of the
            sim window when checking the legacy path, or the per-bar
            today when checking the walk-forward path.
        context: optional string included in the error message (e.g.
            "legacy SimAdapter load", "WalkForwardModelLoader.model_as_of",
            "ticker=AAPL bar=2024-06-03").

    Raises:
        ValueError: when model_trained_date >= sim_today.
        TypeError: when either argument cannot be coerced to a Timestamp.
    """
    trained = _to_timestamp(model_trained_date, label="model_trained_date")
    today = _to_timestamp(sim_today, label="sim_today")
    if trained >= today:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"Look-ahead leakage detected{ctx}: model trained_date "
            f"{trained.date().isoformat()} is not strictly before sim "
            f"today {today.date().isoformat()}. The model has been "
            f"trained on labels that include the sim's evaluation window "
            f"— results would be inflated by data leakage. Use a "
            f"walk-forward manifest with cutoff_date < every sim bar."
        )
