"""Calibrator curve quality helpers (extracted 2026-05-18 user audit).

Single source of truth for _largest_flat_fraction and related quality
metrics. Previously duplicated 3× across preflight.py / fit_calibrator
script / test helper — now imported from here to prevent silent drift.
"""
from __future__ import annotations
from typing import Sequence


def flat_region_stats(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Return largest flat-region diagnostics for a calibrator curve.

    The returned ``fraction`` is the value consumed by acceptance gates.
    ``longest_span`` and ``x_total`` are included so preflight failure
    messages can be auditable without reimplementing the scan locally.
    """
    stats = {"fraction": 0.0, "longest_span": 0.0, "x_total": 0.0}
    if x is None or y is None:
        return stats
    x = list(x)
    y = list(y)
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return stats
    x_total = float(x[-1] - x[0])
    stats["x_total"] = x_total
    if x_total <= 0:
        return stats
    longest = 0.0
    cur_start = 0
    cur_y = y[0]
    for i in range(1, len(y)):
        if y[i] != cur_y:
            if i - cur_start >= 2:
                span = float(x[i - 1] - x[cur_start])
                if span > longest:
                    longest = span
            cur_start = i
            cur_y = y[i]
    if len(y) - cur_start >= 2:
        span = float(x[-1] - x[cur_start])
        if span > longest:
            longest = span
    stats["longest_span"] = longest
    stats["fraction"] = longest / x_total
    return stats


def largest_flat_fraction(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the largest flat region as fraction of total x-domain.

    A "flat region" is a maximal run of >=2 consecutive same-y points.
    Single-point segments (every value differs from previous) don't
    count - they're not flat in any meaningful sense.

    Args:
        x: monotone-non-decreasing x-values (length N)
        y: y-values aligned to x (length N)

    Returns:
        Fraction in [0, 1]. 0 means no flat region; 1 means entire
        x-domain is one flat y.
    """
    return flat_region_stats(x, y)["fraction"]
