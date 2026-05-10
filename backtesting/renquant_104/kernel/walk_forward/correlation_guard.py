"""Single-source-of-truth leakage guard for the watchlist correlation artifact.

Defends against the 2026-05-10 audit class: `watchlist-correlation.json`
is a static correlation matrix loaded by both sim and LEAN. The legacy
artifact had NO `as_of_date` metadata — when a backtest runs in 2024-01,
the correlation matrix (computed in 2026 from late-2025/2026 returns)
reflects forward regime structure → forward-looking leakage at every bar.

Per CLAUDE.md §5.13.5 (single source of truth): every consumer of
`watchlist-correlation.json` (SimAdapter / RunnerAdapter / main.py /
ComputeFullSigmaTask) routes through `assert_correlation_no_leakage`.
Adding a parallel implementation requires deleting this one first.

Per CLAUDE.md §5.13.3, the regression invariant lives in
`tests/test_correlation_guard.py::TestCorrelationGuardRegression`.

Per CLAUDE.md §5.13.10: legacy artifacts without `as_of_date` parse
to None, the guard warns once, and the sim continues (so the migration
is gradual). New artifacts written via `CorrelationJob` MUST include
`as_of_date` (the generation site is updated to always stamp it).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from .leakage_guard import _to_timestamp

log = logging.getLogger("kernel.walk_forward.correlation_guard")


def parse_correlation_artifact(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, float]], str | None]:
    """Split an on-disk corr artifact dict into (matrix, as_of_date).

    Supports BOTH schemas:
      - Legacy (v1): flat ticker → ticker → float dict (no metadata).
        Returns (raw, None).
      - Wrapped (v2): {"schema_version": 2, "as_of_date": "YYYY-MM-DD",
        "data_window_start": ..., "data_window_end": ..., "matrix": {...}}.
        Returns (matrix, as_of_date).

    Detection rule: a v2 artifact has the literal key "matrix" mapping
    to a dict. Anything else is treated as legacy flat-dict.
    """
    if raw is None:
        return {}, None
    if isinstance(raw, Mapping) and isinstance(raw.get("matrix"), Mapping):
        return dict(raw["matrix"]), raw.get("as_of_date")
    # Legacy flat format.
    return dict(raw), None


def assert_correlation_no_leakage(
    as_of_date: Any,
    backtest_start: Any,
    *,
    is_live_mode: bool = False,
    context: str = "",
) -> None:
    """Raise ValueError if as_of_date > backtest_start (in backtest mode).

    Args:
        as_of_date: ISO-string / date / Timestamp — the LATEST date used
            in the correlation computation. None ⇒ legacy artifact;
            warn once and skip (backward compat per §5.13.10).
        backtest_start: ISO-string / date / Timestamp — the first bar of
            the sim window. None ⇒ unknown; skip silently (caller didn't
            pass it through, older API surface).
        is_live_mode: when True, the guard skips silently. Live runs use
            the freshest correlation, which by construction reflects
            data up to "now".
        context: optional string included in the error message
            (e.g. "SimAdapter", "LEAN main.py", "ComputeFullSigmaTask").

    Raises:
        ValueError: when as_of_date > backtest_start in backtest mode.
        TypeError: when either argument cannot be coerced to a Timestamp.
    """
    if is_live_mode:
        return
    if backtest_start is None:
        return
    if as_of_date is None:
        ctx = f" [{context}]" if context else ""
        log.warning(
            "Correlation artifact has no as_of_date%s — legacy v1 schema. "
            "Accepting (backward compat) but the next regeneration will "
            "stamp as_of_date. Forward leakage cannot be verified for "
            "this artifact; treat backtest results with caution.",
            ctx,
        )
        return

    as_of_ts = _to_timestamp(as_of_date, label="as_of_date")
    start_ts = _to_timestamp(backtest_start, label="backtest_start")
    if as_of_ts > start_ts:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"Look-ahead leakage detected{ctx}: correlation artifact "
            f"as_of_date {as_of_ts.date().isoformat()} is after backtest "
            f"start {start_ts.date().isoformat()}. The correlation matrix "
            f"reflects regime structure from data the sim has not yet "
            f"observed — pairwise blocks and Σ_full would be biased by "
            f"forward information. Regenerate the artifact with data "
            f"window ≤ backtest_start, or use a walk-forward correlation "
            f"manifest."
        )
