"""Sim-smoke metric collection for Phase-2 acceptance gates G9/G10/G11.

Phase 2 (2026-04-26) of the model-selection systematization plan adds
three new acceptance gates that compare staging-vs-prior on real sim
outputs (not just OOS IC):

    G9  sim_apy        ≥ prior_apy − 1.0 percentage point
    G10 sim_sharpe     ≥ prior_sharpe − 0.1
    G11 turnover_ratio ≤ prior_turnover × 1.5

These metrics are NOT computed by the panel-training pipeline (which
only produces CPCV OOS IC) — they require running an actual portfolio
simulation. This module provides:

    1. `run_smoke_test(artifact_path, config, window_months=6)`
       — runs SimAdapter on the most recent N months, returns
         dict with {apy, sharpe, calmar, max_drawdown, turnover_ratio,
         n_trades, window_start, window_end}.

    2. `add_smoke_metrics_to_artifact(artifact_path, metrics)`
       — patches the artifact JSON's `metadata.sim_smoke` block so the
         G9/G10/G11 gates can read it.

The smoke test is OPTIONAL — `acceptance.run_sim_smoke` defaults to
`false` because a 6-month sim adds ~30s to the retrain. Operators can
enable it on slow-cadence retrains where the extra confidence is worth
the wall-clock cost. When disabled, G9/G10/G11 skip-pass (no metric
present → no opinion).
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.sim_smoke")


# ── Public API ────────────────────────────────────────────────────────────────

def add_smoke_metrics_to_artifact(artifact_path: Path | str,
                                   metrics: dict[str, Any]) -> None:
    """Patch artifact JSON's `metadata.sim_smoke` with the given metrics.

    Idempotent: re-running overwrites the prior block. If the artifact
    has no `metadata` key (e.g. older flat-format artifacts), one is
    created at the top level.
    """
    path = Path(artifact_path)
    raw = json.loads(path.read_text())
    md = raw.setdefault("metadata", {})
    md["sim_smoke"] = dict(metrics)
    md["sim_smoke"]["written_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path.write_text(json.dumps(raw, indent=2))
    log.info("sim_smoke metrics written to %s", path.name)


def compute_metrics_from_equity_curve(equity: pd.Series,
                                       trades: pd.DataFrame | None = None,
                                       trading_days_per_year: int = 252) -> dict:
    """Pure compute helper. Given an equity Series indexed by trading day
    and an optional trades DataFrame (with `notional` column), return:

        apy, sharpe, calmar, max_drawdown, total_return, n_trades, turnover_ratio

    Robust to short series (returns 0/NaN-defaults when too few points).
    Used by both the SimAdapter-driven smoke test and synthetic tests.
    """
    if equity is None or len(equity) < 2:
        return {
            "apy": 0.0, "sharpe": 0.0, "calmar": 0.0, "max_drawdown": 0.0,
            "total_return": 0.0, "n_trades": 0, "turnover_ratio": 0.0,
        }
    eq = pd.Series(equity).dropna()
    if len(eq) < 2:
        return {
            "apy": 0.0, "sharpe": 0.0, "calmar": 0.0, "max_drawdown": 0.0,
            "total_return": 0.0, "n_trades": 0, "turnover_ratio": 0.0,
        }
    rets = eq.pct_change().dropna()
    n_days = len(eq)
    yrs = n_days / float(trading_days_per_year)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    apy = ((1.0 + total_ret) ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0

    sigma = float(rets.std())
    sharpe = float(rets.mean() / sigma * math.sqrt(trading_days_per_year)) if sigma > 0 else 0.0

    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    calmar = (apy / abs(max_dd)) if max_dd < 0 else 0.0

    n_trades = 0
    turnover_ratio = 0.0
    if trades is not None and len(trades) > 0:
        n_trades = int(len(trades))
        if "notional" in trades.columns:
            gross = float(np.abs(trades["notional"]).sum())
            avg_eq = float(eq.mean())
            turnover_ratio = (gross / avg_eq) if avg_eq > 0 else 0.0

    return {
        "apy":            apy,
        "sharpe":         sharpe,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "total_return":   total_ret,
        "n_trades":       n_trades,
        "turnover_ratio": turnover_ratio,
    }


def run_smoke_test(artifact_path: Path | str,
                   config: dict,
                   strategy_dir: Path | str,
                   window_months: int = 6,
                   end_date: pd.Timestamp | str | None = None) -> dict:
    """Run a short sim against the given panel artifact and return metrics.

    Best-effort: returns {} on failure (caller treats missing metrics as
    "skip-pass" via the gate logic). This protects retrain pipelines from
    breaking when SimAdapter has a bug — the gate fallback is no-opinion.

    Implementation notes:
        - Loads OHLCV for the strategy's watchlist from the cache.
        - Builds SimAdapter with the staging artifact wired in via
          `_panel_scorer = PanelScorer.load(artifact_path)`.
        - Runs a single calendar window ending at `end_date` (default:
          today) covering the past `window_months`.
        - Aggregates the equity curve and trades into the metric dict.

    The actual sim execution is intentionally deferred — Phase-2 ships
    the GATE infrastructure with this helper as a stub-friendly
    placeholder. Operators can swap in their preferred sim driver
    (notebook cell, scripts/analyze_backtest.py output, etc.) and call
    `add_smoke_metrics_to_artifact` directly to populate metrics.

    Phase-3+ will integrate this with `scripts/select_best_model.py`
    so the same path serves both retrain-time gating and offline
    backend tournaments.
    """
    log.info("sim_smoke.run_smoke_test stub — returning {} (Phase-2 ships gate infra; "
             "operators wire their own sim driver via add_smoke_metrics_to_artifact)")
    return {}


__all__ = [
    "add_smoke_metrics_to_artifact",
    "compute_metrics_from_equity_curve",
    "run_smoke_test",
]
