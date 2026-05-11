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
    *,
    lookahead_days: int = 0,
) -> None:
    """Raise ValueError if model could have seen training labels at/after sim_today.

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
    # 2026-05-11 audit Round 3 (G2 strengthening): defensive input validation.
    # Pre-fix: NaN/inf/float/string lookahead_days silently bypassed (NaN > 0
    # is False; float(10.7) truncated; "60" raised opaque TypeError). Now:
    # explicit type + finite + sign check with labeled errors so a bad
    # manifest entry surfaces loudly instead of degrading to "no lookahead".
    import math as _math  # noqa: PLC0415
    if lookahead_days is None:
        lookahead_days = 0
    if isinstance(lookahead_days, bool):
        raise TypeError(
            f"lookahead_days must be int, got bool {lookahead_days!r}"
        )
    if not isinstance(lookahead_days, int):
        raise TypeError(
            f"lookahead_days must be int (got {type(lookahead_days).__name__} "
            f"{lookahead_days!r}) — coerce upstream so the guard's check "
            f"can't silently drift on float-vs-int comparisons."
        )
    if lookahead_days < 0:
        raise ValueError(
            f"lookahead_days must be ≥ 0, got {lookahead_days}"
        )
    # 2026-05-11 G1: when lookahead_days > 0, training labels reach
    # `trained + lookahead_days` (calendar days, conservative upper bound).
    # E.g. fwd_60d_excess at cutoff 2024-01-01 saw prices through ~2024-03-01.
    if lookahead_days > 0:
        last_label_seen = trained + pd.tseries.offsets.BDay(int(lookahead_days))
    else:
        last_label_seen = trained
    if last_label_seen >= today:
        ctx = f" [{context}]" if context else ""
        la_note = (
            f" + lookahead_days={lookahead_days} → last label-seen "
            f"{last_label_seen.date().isoformat()}"
            if lookahead_days else ""
        )
        raise ValueError(
            f"Look-ahead leakage detected{ctx}: model trained_date "
            f"{trained.date().isoformat()}{la_note} is not strictly "
            f"before sim today {today.date().isoformat()}. The model "
            f"was trained on labels that may include forward returns "
            f"reaching into the sim's evaluation window — results would "
            f"be inflated by data leakage. Use a walk-forward manifest "
            f"with cutoff_date + lookahead_days < every sim bar."
        )
