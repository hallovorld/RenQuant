"""Sim point-in-time leakage guards — sim.py decomposition slice 4.

EXTRACTED 2026-06-13 from adapters/sim.py (eng plan S2 item 5). Pure
predicates that detect whether a correlation / GMM artifact's stamped
as_of_date is strictly after the backtest start — used to decide whether
to substitute a historical sim artifact BEFORE the hard no-leakage guard
fires. No SimAdapter state. Re-exported from sim for back-compat.
"""
from __future__ import annotations

import pandas as pd


def corr_leakage_present(as_of_date, backtest_start) -> bool:
    """True iff a parsed correlation ``as_of_date`` is strictly after start.

    Mirrors ``assert_correlation_no_leakage`` semantics without raising,
    used to decide whether to fall back to a historical sim correlation
    artifact before invoking the guard. Returns False for any ``as_of_date``
    that can't be parsed or for unstamped (legacy) artifacts — the guard
    accepts those with a warning when ``allow_legacy_without_as_of=True``,
    so substitution would only mask the operator's intent.
    """
    if as_of_date is None or backtest_start is None:
        return False
    try:
        as_of_ts = pd.Timestamp(as_of_date)
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
        start_ts = pd.Timestamp(backtest_start)
        if start_ts.tz is not None:
            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    except (TypeError, ValueError):
        return False
    return as_of_ts > start_ts

def gmm_leakage_present(artifact, backtest_start, as_of_extractor) -> bool:
    """True iff artifact has a stamped as_of_date strictly after start.

    Mirrors ``assert_gmm_no_leakage`` semantics without raising — used to
    decide whether to fall back to a historical sim GMM before invoking
    the guard. Legacy (unstamped) artifacts return False (the guard
    accepts them with a warning), so we never substitute on those.
    """
    if artifact is None or backtest_start is None:
        return False
    as_of = as_of_extractor(artifact)
    if as_of is None:
        return False
    try:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)
        start_ts = pd.Timestamp(backtest_start)
        if start_ts.tz is not None:
            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    except (TypeError, ValueError):
        return False
    return as_of_ts > start_ts
