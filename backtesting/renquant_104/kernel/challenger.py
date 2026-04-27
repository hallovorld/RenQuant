"""Challenger / shadow-mode infrastructure (Phase 4 of model-selection plan).

Phase 4 (2026-04-26) of the model-selection systematization. Goal:
before a new artifact takes over production, run it in SHADOW mode for
N sessions — score the same universe, log "what would challenger have
done", but DO NOT trade. Operator compares challenger vs live decisions
post-window to validate before promotion.

This file ships the PLATFORM (data structures + APIs + DB schema).
Wiring into the live runner / sim is intentionally NOT done today —
that's a Phase 4b extension that requires its own audit + e2e test.

Why platform-only first:
- Challenger eval doubles inference cost per bar (load second model,
  score, compare). We need bench numbers before turning it on in the
  live path.
- The decision log table needs to exist first so a Phase 4b commit can
  start writing to it without schema migration.
- Tests / smoke runs of ChallengerEvaluator are easier in isolation.

Public API
==========

`ChallengerConfig` — config dataclass (artifact path, name, enabled,
shadow_period_days). Loaded from `strategy_config.json`'s
`acceptance.challenger` block.

`ChallengerEvaluator(config, scorer)` — lightweight wrapper around a
loaded PanelScorer that exposes `score(X) -> Series` and `decide(...)`.
Doesn't bind to live runner — caller invokes per bar.

`log_decision(db, run_id, ...)` — writes one row to the
`challenger_decisions` table.

`compare_window(db, name, start, end)` — aggregates challenger vs
actual decisions over a window; returns a verdict dict the operator
can use as the de-facto Phase-4 "promote yes/no" signal.

DB schema lives in `kernel.persistence` (this module just inserts
into the table; persistence.py owns the CREATE).
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("kernel.challenger")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ChallengerConfig:
    """Parsed `acceptance.challenger` block from strategy_config.json.

    Default state (all-OFF) shipped today; operator opts in by setting
    `enabled: true` and providing an artifact name + path.
    """
    enabled:             bool
    artifact_path:       str | None       # e.g. 'artifacts/panel-ltr.macro-enabled.bak.json'
    name:                str | None       # e.g. 'macro-enabled' (used as challenger_name)
    shadow_period_days:  int              # how long to run in shadow before promote eligible

    @classmethod
    def from_strategy_config(cls, config: dict) -> "ChallengerConfig":
        acc = config.get("acceptance") or {}
        ch  = acc.get("challenger") or {}
        return cls(
            enabled            = bool(ch.get("enabled", False)),
            artifact_path      = ch.get("artifact_path"),
            name               = ch.get("name"),
            shadow_period_days = int(ch.get("shadow_period_days", 0)),
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ChallengerEvaluator:
    """Wraps a loaded PanelScorer + the challenger's config so callers
    have a uniform `score(X) -> Series` interface regardless of which
    backend the artifact uses.

    Phase 4a (today): the only consumer is the test suite + offline
    operator scripts. Phase 4b will add per-bar invocation from
    `pp_inference.py` after a shadow_period_days window starts.
    """

    def __init__(self, config: ChallengerConfig, scorer: Any | None = None):
        self.config = config
        self.scorer = scorer

    @classmethod
    def maybe_load(cls, config: dict, strategy_dir: Path) -> "ChallengerEvaluator | None":
        """Build an Evaluator from config + the artifact on disk, OR
        return None if the challenger is disabled / artifact missing.

        Returns None silently — challenger-not-enabled is the default
        state and shouldn't fire warnings.
        """
        cc = ChallengerConfig.from_strategy_config(config)
        if not cc.enabled:
            return None
        if not cc.artifact_path:
            log.warning("challenger.enabled=true but no artifact_path; skipping")
            return None
        from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
        path = strategy_dir / cc.artifact_path
        if not path.exists():
            log.warning("challenger artifact missing: %s; skipping", path)
            return None
        try:
            scorer = PanelScorer.load(path)
        except Exception as exc:
            log.warning("challenger artifact load failed (%s): %s", path, exc)
            return None
        return cls(cc, scorer)

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score the inference matrix X using the challenger artifact.

        Returns an empty Series if the evaluator is in degraded state
        (no scorer loaded). This keeps callers from needing a None check
        on every invocation — they get an empty join instead.
        """
        if self.scorer is None or X is None or X.empty:
            return pd.Series(dtype=float, name="challenger_score")
        return self.scorer.score(X).rename("challenger_score")


# ── DB persistence ────────────────────────────────────────────────────────────

def log_decision(conn: sqlite3.Connection, *,
                 run_id: str,
                 decision_date: pd.Timestamp | datetime.date,
                 ticker: str,
                 challenger_name: str,
                 challenger_score: float | None,
                 challenger_rank_score: float | None,
                 challenger_action: str | None,
                 actual_score: float | None = None,
                 actual_action: str | None = None) -> None:
    """Append one row to the challenger_decisions table.

    Used per-ticker per-bar when the live runner is in shadow mode.
    Idempotent at the (run_id, ticker, decision_date) level — re-runs
    of the same bar will INSERT new rows; deduplication is the caller's
    responsibility (rare, since runs are bar-aligned by run_id).
    """
    conn.execute(
        """
        INSERT INTO challenger_decisions (
            run_id, decision_date, ticker, challenger_name,
            challenger_score, challenger_rank_score, challenger_action,
            actual_score, actual_action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            pd.Timestamp(decision_date).isoformat(),
            ticker,
            challenger_name,
            float(challenger_score) if challenger_score is not None else None,
            float(challenger_rank_score) if challenger_rank_score is not None else None,
            challenger_action,
            float(actual_score) if actual_score is not None else None,
            actual_action,
        ),
    )


def compare_window(conn: sqlite3.Connection, *,
                   challenger_name: str,
                   start_date: pd.Timestamp | datetime.date,
                   end_date: pd.Timestamp | datetime.date) -> dict:
    """Aggregate challenger vs actual decisions over [start_date, end_date].

    Returns a dict suitable as the operator-facing verdict for "should
    I promote this challenger?". Phase 4b will wire this into a CLI
    (`scripts/compare_challenger.py`) that prints the dict + recommendation.

    Output keys:
        n_decisions:           total rows in window
        agreement_rate:        fraction where challenger_action == actual_action
        challenger_only_buy:   actions challenger took that live didn't
        live_only_buy:         actions live took that challenger didn't
        score_corr:            Pearson corr between challenger_score and actual_score
        score_rank_corr:       Spearman corr between challenger_rank_score and actual_rank
    """
    df = pd.read_sql_query(
        """
        SELECT decision_date, ticker, challenger_action, actual_action,
               challenger_score, actual_score, challenger_rank_score
        FROM challenger_decisions
        WHERE challenger_name = ?
          AND decision_date >= ?
          AND decision_date <= ?
        """,
        conn,
        params=(
            challenger_name,
            pd.Timestamp(start_date).isoformat(),
            pd.Timestamp(end_date).isoformat(),
        ),
    )
    if df.empty:
        return {
            "n_decisions": 0, "agreement_rate": 0.0,
            "challenger_only_buy": 0, "live_only_buy": 0,
            "score_corr": None, "score_rank_corr": None,
        }
    agree = (df["challenger_action"] == df["actual_action"]).mean()
    ch_only = ((df["challenger_action"] == "BUY") & (df["actual_action"] != "BUY")).sum()
    li_only = ((df["actual_action"]    == "BUY") & (df["challenger_action"] != "BUY")).sum()
    score_corr = None
    rank_corr  = None
    pair = df[["challenger_score", "actual_score"]].dropna()
    if len(pair) >= 3 and pair["challenger_score"].std() > 0 and pair["actual_score"].std() > 0:
        score_corr = float(pair["challenger_score"].corr(pair["actual_score"]))
    # Audit fix #5 (2026-04-26): pre-fix, rank_corr was always returned
    # as None — function docstring promised Spearman corr but never
    # computed it. Now we compute Spearman between challenger_rank_score
    # and the rank of actual_score (best proxy for live rank, since live
    # rank isn't recorded as a separate column today).
    rank_pair = df[["challenger_rank_score", "actual_score"]].dropna()
    if len(rank_pair) >= 3:
        ch_rank   = rank_pair["challenger_rank_score"]
        live_rank = rank_pair["actual_score"].rank()
        if ch_rank.std() > 0 and live_rank.std() > 0:
            rank_corr = float(ch_rank.corr(live_rank, method="spearman"))
    return {
        "n_decisions":          int(len(df)),
        "agreement_rate":       float(agree),
        "challenger_only_buy":  int(ch_only),
        "live_only_buy":        int(li_only),
        "score_corr":           score_corr,
        "score_rank_corr":      rank_corr,
    }


__all__ = [
    "ChallengerConfig",
    "ChallengerEvaluator",
    "log_decision",
    "compare_window",
]
