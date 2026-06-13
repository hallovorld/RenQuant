"""Runner prep helpers — runner.py decomposition slice 7.

EXTRACTED 2026-06-13 from adapters/runner.py (eng plan S2 item 5). Pure
pre-context helpers: ISO datetime parsing, high-water-mark staleness
resolution (the hwm/equity ratio snap), the held-mark OHLCV frame
builder, and the persisted skip_buys reader. No RunnerAdapter state;
re-exported from runner for back-compat.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

import pandas as pd

log = logging.getLogger("adapters.runner")  # same logger — log contract unchanged


# Ratio above which a stored high_water_mark is treated as "stale" relative to
# current account value and snapped down. Chosen so that a real 33% drawdown
# (hwm/equity ratio = 1.49) is preserved but the typical stale-seed case
# (hwm=$100k, equity=$10k → ratio 10×) trips the snap.
_HWM_STALE_RATIO = 1.5


def _parse_iso_dt(s: Any) -> "datetime.datetime | None":
    """Parse an ISO-formatted datetime string; return None on any failure.

    Used to restore RegimeState.cooldown_start from live_state.json across
    invocations (CUSUM-v2 Design C).
    """
    if s is None or s == "":
        return None
    try:
        return datetime.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def resolve_hwm(stored_hwm: float, account_value: float,
                stale_ratio: float = _HWM_STALE_RATIO) -> tuple[float, bool]:
    """Resolve the effective high_water_mark for the current bar.

    The live-trading DrawdownCircuitTask divides `(hwm - equity) / hwm` and
    compares to `halt_pct`. When `hwm` is stale (e.g. initial-seed $100k
    from a fresh install, actual Alpaca equity a fraction of that), this
    ratio blows up and the drawdown halt latches on every bar — exactly
    the 2026-04-23 "zero orders despite healthy models" bug.

    Rule: if stored_hwm > stale_ratio * account_value, snap to account_value.
    Otherwise ratchet up to max(stored_hwm, account_value) as before.

    Returns (resolved_hwm, was_snapped).

    Audit fix RU-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
    account_value (broker outage / Alpaca returns NaN equity) slipped
    past `account_value > 0` (NaN comparisons False), then `max(hwm,
    NaN) = NaN` → resolved_hwm = NaN → DrawdownCircuitTask's
    `(NaN - equity) / NaN` = NaN → `NaN >= halt_pct` False → drawdown
    gate silently disabled in LIVE TRADING for the rest of the run.
    Post-fix: explicit isfinite check; on bad account_value, fall back
    to stored_hwm unchanged (fail-SAFE behaviour — keeps the gate
    armed against the LAST-known good HWM).
    """
    import math
    if not math.isfinite(account_value):
        # Bad broker data → preserve stored HWM intact, no snap.
        if math.isfinite(stored_hwm):
            return float(stored_hwm), False
        return 0.0, False
    if not math.isfinite(stored_hwm):
        # Stored HWM is corrupted but account_value is good → reset to
        # account_value so future drawdown calc is meaningful.
        return float(account_value), True
    if account_value > 0 and stored_hwm > stale_ratio * account_value:
        return float(account_value), True
    return float(max(stored_hwm, account_value)), False


def _held_mark_ohlcv_frame(
    symbol: str,
    today: datetime.date,
    price: float,
    base_df: Any | None = None,
) -> Any | None:
    """Build a risk-only OHLCV frame from a broker mark for held sell checks."""
    import math as _math  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    try:
        px = float(price)
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(px) or px <= 0:
        return None

    idx = pd.DatetimeIndex([pd.Timestamp(today)])
    mark_bar = pd.DataFrame(
        {
            "open": [px],
            "high": [px],
            "low": [px],
            "close": [px],
            "volume": [0.0],
        },
        index=idx,
    )

    if base_df is None or getattr(base_df, "empty", True):
        return mark_bar

    try:
        df = base_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index.date != today]
        return pd.concat([df, mark_bar]).sort_index()
    except Exception:
        return mark_bar


def persisted_skip_buys(state: dict | None) -> bool:
    """Read persisted drawdown-halt state with legacy-safe coercion.

    SimAdapter carries ``_skip_buys`` across bars in-process. RunnerAdapter is
    relaunched by scheduled jobs, so the same hysteresis state must round-trip
    through live_state; otherwise live exits the drawdown recovery band earlier
    than sim.
    """
    if not isinstance(state, dict):
        return False
    value = state.get("skip_buys", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False
