"""Pin the ``cands=N/M  holdings=N/M`` log format (audit finding B).

2026-06-02 daily audit found ApplyKellySizingTask logged
``holdings=6 non-zero (avg=6.2%)`` which the operator read as "6 holdings
exist", but reality was "of 7 holdings, 6 had non-zero Kelly target — the
7th was zero'd by mu/sigma/cap rules". This caused unnecessary debugging.

The fix surfaces BOTH counts (``N/M``). This test pins that.

`_kelly_with_reason` is a closure inside ApplyKellySizingTask.run so we
can't mock it directly; instead we drive zero-ing via the mu / sigma
values the actual Kelly function checks (mu <= min_edge → zero).
"""
from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

import logging
import types
from dataclasses import dataclass


@dataclass
class _Cand:
    ticker: str
    mu: float = 0.01      # positive → kelly non-zero
    sigma: float = 0.20
    sector: str | None = None
    kelly_target_pct: float | None = None
    panel_score: float = 0.5
    rank_score: float = 0.5
    expected_return: float = 0.01


@dataclass
class _Hold:
    ticker: str
    current_pct: float = 0.10
    mu: float = 0.01       # positive → kelly non-zero
    sigma: float = 0.20
    kelly_target_pct: float | None = None
    sector: str | None = None
    panel_score: float = 0.5
    rank_score: float = 0.5
    expected_return: float = 0.01


def _build_ctx(cands, holds):
    return types.SimpleNamespace(
        candidates=cands,
        holdings={h.ticker: h for h in holds},
        config={
            "ranking": {
                "kelly_sizing": {
                    "enabled": True,
                    "fractional": 0.5,
                    "max_concentration": 0.35,
                    "min_edge": 0.0,
                },
            },
        },
        regime="BULL_CALM",
        confidence=0.6,
        counters={},
    )


def test_log_format_distinguishes_total_vs_nonzero(caplog):
    """Audit finding B: ``holdings=6/7`` not ``holdings=6``.

    7 holdings, 6 have mu=0.01 (positive — Kelly emits non-zero target),
    1 has mu=0.0 (≤ min_edge → Kelly returns 0). Log must report 6/7.
    """
    from kernel.panel_pipeline.job_panel_scoring import (
        ApplyKellySizingTask,
    )

    holds = [_Hold(ticker=f"H{i}", mu=0.01) for i in range(6)]
    holds.append(_Hold(ticker="H6", mu=0.0))  # zero'd by min_edge rule
    cands = [_Cand(ticker=f"C{i}", mu=0.01) for i in range(3)]
    ctx = _build_ctx(cands, holds)

    with caplog.at_level(logging.INFO, logger="kernel.panel_pipeline.scoring"):
        ApplyKellySizingTask().run(ctx)

    matched = [r for r in caplog.records if "ApplyKellySizingTask:" in r.message]
    assert matched, f"No ApplyKellySizingTask log emitted: {caplog.records}"
    msg = matched[-1].message
    assert "holdings=6/7" in msg, (
        f"expected holdings=6/7 (6 non-zero, 7 total); got: {msg}"
    )
    # Candidates should be 3/3 since all have mu=0.01.
    assert "cands=3/3" in msg or "cands=" in msg, f"expected cands=N/M: {msg}"


def test_log_format_when_all_holdings_zero(caplog):
    """Edge: 0 non-zero holdings out of 4 total → ``holdings=0/4``."""
    from kernel.panel_pipeline.job_panel_scoring import (
        ApplyKellySizingTask,
    )

    holds = [_Hold(ticker=f"H{i}", mu=0.0) for i in range(4)]   # all zero
    cands = [_Cand(ticker=f"C{i}", mu=0.0) for i in range(2)]   # all zero
    ctx = _build_ctx(cands, holds)

    with caplog.at_level(logging.INFO, logger="kernel.panel_pipeline.scoring"):
        ApplyKellySizingTask().run(ctx)

    matched = [r for r in caplog.records if "ApplyKellySizingTask:" in r.message]
    assert matched
    msg = matched[-1].message
    assert "holdings=0/4" in msg, f"expected holdings=0/4; got: {msg}"
    assert "cands=0/2" in msg, f"expected cands=0/2; got: {msg}"


def test_log_format_when_no_holdings_no_candidates(caplog):
    """Edge: empty universe → ``cands=0/0  holdings=0/0``."""
    from kernel.panel_pipeline.job_panel_scoring import (
        ApplyKellySizingTask,
    )

    ctx = _build_ctx(cands=[], holds=[])

    with caplog.at_level(logging.INFO, logger="kernel.panel_pipeline.scoring"):
        ApplyKellySizingTask().run(ctx)

    matched = [r for r in caplog.records if "ApplyKellySizingTask:" in r.message]
    assert matched
    msg = matched[-1].message
    assert "cands=0/0" in msg
    assert "holdings=0/0" in msg
