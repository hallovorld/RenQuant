"""Leakage guard for static SPY GMM regime artifacts.

The regime detector controls every downstream regime-conditional policy.
Using a GMM fitted on data after a simulation start gives the sim future
volatility/cluster structure, even if the alpha model itself is clean.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from .leakage_guard import _to_timestamp

log = logging.getLogger("kernel.walk_forward.gmm_guard")


def gmm_artifact_as_of(raw: Mapping[str, Any] | None) -> str | None:
    """Return the latest data date used by a regime artifact, if stamped.

    ``trained_date`` is a wall-clock build timestamp for HMM artifacts. Leakage
    is about the data window available to the detector, so prefer explicit
    as-of / window-end metadata before falling back to wall-clock time.
    """
    if not isinstance(raw, Mapping):
        return None
    training_window = raw.get("training_window")
    window_end = None
    if isinstance(training_window, (list, tuple)) and len(training_window) >= 2:
        window_end = training_window[-1]
    as_of = (
        raw.get("as_of_date")
        or raw.get("data_window_end")
        or window_end
        or raw.get("trained_date")
    )
    return str(as_of) if as_of is not None else None


def _to_comparable_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    ts = _to_timestamp(value, label=label)
    if ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def assert_gmm_no_leakage(
    artifact: Mapping[str, Any] | None,
    backtest_start: Any,
    *,
    is_live_mode: bool = False,
    context: str = "",
) -> None:
    """Raise if a static GMM artifact was fit after a sim's first bar."""
    if is_live_mode or backtest_start is None:
        return
    as_of = gmm_artifact_as_of(artifact)
    if as_of is None:
        ctx = f" [{context}]" if context else ""
        log.warning(
            "GMM artifact has no as_of_date%s — legacy schema. Accepting "
            "for backward compatibility, but regime labels in backtests "
            "cannot be proven leakage-free until this artifact is regenerated.",
            ctx,
        )
        return

    as_of_ts = _to_comparable_timestamp(as_of, label="gmm_as_of_date")
    start_ts = _to_comparable_timestamp(backtest_start, label="backtest_start")
    if as_of_ts > start_ts:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"Look-ahead leakage detected{ctx}: GMM regime artifact "
            f"as_of_date {as_of_ts.date().isoformat()} is after backtest "
            f"start {start_ts.date().isoformat()}. Regime labels would "
            f"reflect volatility/cluster structure unavailable at the "
            f"first simulated bar. Use a sim artifact with data window "
            f"ending on or before the backtest start."
        )
