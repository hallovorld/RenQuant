"""Shadow model scoring — record what alternative models WOULD have done,
without affecting primary trading decisions.

Per 2026-05-18 user request: "inference 能不能接受一个 shadow 模型, 只
跟着走流程算参数但是不做最后一步下单, 所有数据进数据库为以后 review 做
备份".

Industry standard pattern:
  - Primary model (config.ranking.panel_scoring.kind) makes REAL trading
    decisions (orders to broker)
  - Shadow models (config.ranking.panel_scoring.shadow_models[]) run
    in parallel: same candidates, same Tasks, but ZERO order submission
  - Shadow scores recorded to SQLite (data/shadow_scores.db) for later
    review / A/B analysis / promotion-readiness check

Workflow:
  1. ApplyScoresTask runs primary model → writes scores to candidates
  2. ApplyShadowScoringTask runs (this Task) → for each shadow model:
     - Load via registry
     - Score same candidates (handles sequence vs snapshot dispatch)
     - Record (date, shadow_name, ticker, shadow_score, primary_score,
       primary_minus_shadow_diff) to shadow_scores table
  3. Primary decisions flow downstream unchanged (calibrator, Kelly, QP,
     broker submission). Shadow NEVER touches order flow.

DB schema (SQLite at data/shadow_scores.db):
  CREATE TABLE shadow_scores (
    as_of_date    DATE,
    ticker        TEXT,
    shadow_name   TEXT,    -- e.g. "patchtst_seed42"
    shadow_kind   TEXT,    -- e.g. "patchtst"
    primary_score REAL,
    shadow_score  REAL,
    diff          REAL,    -- shadow - primary (positive = shadow more bullish)
    primary_rank  INT,
    shadow_rank   INT,
    rank_diff     INT,     -- shadow_rank - primary_rank
    inserted_at   TIMESTAMP
  )

Tests in tests/test_shadow_scoring.py.
"""
from __future__ import annotations
import datetime
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

log = logging.getLogger("kernel.panel_pipeline.shadow_scoring")

# OMP fix per [[concurrency_resource_budget]]: ensure single-thread BEFORE
# any torch model construction in shadow scorers (PatchTST etc).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_DB_PATH_DEFAULT = "data/shadow_scores.db"


def _init_shadow_db(db_path: Path) -> None:
    """Create the shadow_scores table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_scores (
                as_of_date    DATE,
                ticker        TEXT,
                shadow_name   TEXT,
                shadow_kind   TEXT,
                primary_score REAL,
                shadow_score  REAL,
                diff          REAL,
                primary_rank  INT,
                shadow_rank   INT,
                rank_diff     INT,
                inserted_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (as_of_date, ticker, shadow_name)
            )
        """)
        # Index for review queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_date_name
            ON shadow_scores(as_of_date, shadow_name)
        """)
        conn.commit()
    finally:
        conn.close()


def _persist_shadow_rows(db_path: Path, rows: list[dict]) -> None:
    """Insert rows into shadow_scores. INSERT OR REPLACE on PK conflict."""
    if not rows:
        return
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany("""
            INSERT OR REPLACE INTO shadow_scores
              (as_of_date, ticker, shadow_name, shadow_kind,
               primary_score, shadow_score, diff,
               primary_rank, shadow_rank, rank_diff, inserted_at)
            VALUES (:as_of_date, :ticker, :shadow_name, :shadow_kind,
                    :primary_score, :shadow_score, :diff,
                    :primary_rank, :shadow_rank, :rank_diff,
                    CURRENT_TIMESTAMP)
        """, rows)
        conn.commit()
    finally:
        conn.close()


class ApplyShadowScoringTask(Task):
    """Run each configured shadow model on the SAME candidates as primary,
    record scores to DB. Read-only — no mutations to ctx.candidates.

    Reads:
      - ctx.candidates (with .ticker and .panel_score already set by main)
      - ctx.config["ranking"]["panel_scoring"]["shadow_models"] (list)
      - ctx.config["ranking"]["panel_scoring"]["shadow_db_path"] (optional)
      - ctx._panel_history (set by adapter or lazy-loaded for PatchTST shadows)

    Writes:
      - data/shadow_scores.db (or config override)

    Soft-fail: shadow scoring errors are logged but don't stop the pipeline.
    """

    name = "ApplyShadowScoringTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        shadow_models = panel_cfg.get("shadow_models", []) or []
        if not shadow_models:
            return None

        # Resolve DB path
        db_rel = panel_cfg.get("shadow_db_path", _DB_PATH_DEFAULT)
        repo = Path(__file__).resolve().parents[4]
        db_path = Path(db_rel) if Path(db_rel).is_absolute() else (repo / db_rel)
        _init_shadow_db(db_path)

        # Primary scores (from main ApplyScoresTask)
        cands = list(ctx.candidates) if ctx.candidates else []
        if not cands:
            log.info("ApplyShadowScoringTask: 0 candidates — skip")
            return None
        primary_scores = {c.ticker: float(c.panel_score)
                          for c in cands if c.panel_score is not None}
        if not primary_scores:
            log.info("ApplyShadowScoringTask: no primary scores set — skip")
            return None

        # Primary ranks (descending)
        primary_ranks = (
            pd.Series(primary_scores)
            .sort_values(ascending=False)
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]  # original ticker → rank
        )
        # Wait — actually simpler:
        sorted_primary = sorted(primary_scores.items(), key=lambda x: -x[1])
        primary_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_primary)}

        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415

        for sm in shadow_models:
            name = sm.get("name", "unnamed_shadow")
            kind = sm.get("kind")
            artifact_path = sm.get("artifact_path")
            if not kind or not artifact_path:
                log.warning("ApplyShadowScoringTask: shadow %s missing "
                             "kind/artifact_path — skip", name)
                continue
            p = Path(artifact_path)
            if not p.is_absolute():
                p = repo / p
            try:
                handler = registry.get(kind)
            except ValueError as exc:
                log.warning("ApplyShadowScoringTask: %s — skip", exc)
                continue

            # Inject shadow's feature_cols + seq_len into the config copy
            shadow_cfg = dict(ctx.config)
            shadow_panel_cfg = dict(panel_cfg)
            if "feature_cols" in sm:
                shadow_panel_cfg["feature_cols"] = sm["feature_cols"]
            if "seq_len" in sm:
                shadow_panel_cfg["seq_len"] = sm["seq_len"]
            shadow_cfg.setdefault("ranking", {})["panel_scoring"] = shadow_panel_cfg

            try:
                scorer = handler.scorer_loader(p, shadow_cfg)
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: shadow %s (%s) failed to "
                             "load — %s", name, kind, exc)
                continue

            # Score the same candidates
            target_tickers = list(primary_scores.keys())
            try:
                if getattr(scorer, "requires_history", False):
                    panel_history = getattr(ctx, "_panel_history", None)
                    if panel_history is None:
                        # Lazy load like main path
                        panel_parquet = (repo / "data"
                                          / "alpha158_291_fundamental_dataset.parquet")
                        full = pd.read_parquet(panel_parquet)
                        full["date"] = pd.to_datetime(full["date"])
                        today_ts = pd.Timestamp(getattr(ctx, "today",
                                                          datetime.date.today()))
                        past = full[full["date"] < today_ts]
                        dates = sorted(past["date"].unique())[-scorer.seq_len:]
                        panel_history = past[
                            past["ticker"].isin(target_tickers) &
                            past["date"].isin(dates)]
                    shadow_scores_series = scorer.score_with_history(
                        panel_history, target_tickers)
                else:
                    # Snapshot — need feature matrix. For shadow, just use
                    # ctx._panel_matrix if available (built by main path).
                    X = getattr(ctx, "_panel_matrix", None)
                    if X is None or X.empty:
                        log.warning("ApplyShadowScoringTask: shadow %s needs "
                                     "feature matrix but ctx._panel_matrix is empty",
                                     name)
                        continue
                    # Re-align to shadow's feature_cols (may differ)
                    fc = scorer.feature_cols
                    missing = [c for c in fc if c not in X.columns]
                    if missing:
                        log.warning("ApplyShadowScoringTask: shadow %s missing "
                                     "cols in matrix: %s", name, missing[:5])
                        continue
                    shadow_scores_series = scorer.score(X[fc].fillna(0))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: shadow %s scoring failed: %s",
                             name, exc)
                continue

            # Build shadow ranks
            shadow_dict = shadow_scores_series.to_dict()
            sorted_shadow = sorted(shadow_dict.items(), key=lambda x: -x[1])
            shadow_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_shadow)}

            # Build rows for DB
            rows = []
            today = getattr(ctx, "today", datetime.date.today())
            for t in target_tickers:
                ps = primary_scores.get(t)
                ss = shadow_dict.get(t)
                if ps is None or ss is None:
                    continue
                rows.append({
                    "as_of_date":    str(today),
                    "ticker":        t,
                    "shadow_name":   name,
                    "shadow_kind":   kind,
                    "primary_score": float(ps),
                    "shadow_score":  float(ss),
                    "diff":          float(ss - ps),
                    "primary_rank":  int(primary_ranks.get(t, 0)),
                    "shadow_rank":   int(shadow_ranks.get(t, 0)),
                    "rank_diff":     int(shadow_ranks.get(t, 0)
                                          - primary_ranks.get(t, 0)),
                })
            _persist_shadow_rows(db_path, rows)
            log.info("ApplyShadowScoringTask: shadow %s (%s) recorded %d rows "
                     "to %s  mean_diff=%+.4f", name, kind, len(rows),
                     db_path.name, float(np.mean([r["diff"] for r in rows])))

        return None


__all__ = ["ApplyShadowScoringTask"]
