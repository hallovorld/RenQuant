"""RecordScoreDistributionTask — persist daily score distribution + percentiles.

Per user spec 2026-04-26 round-5: "建立 calibrate 数据库, 知道什么 score
value 是 top 5%". Phase 1: collect-only (no decision impact).

Runs at the END of Phase 3 (after PanelScoringJob populates rank_score
on candidates AND holdings, after RankingJob/JointActionJob consume them).
Writes:
  * score_distribution rows (one per ticker/date)
  * score_percentiles_daily aggregated row

Decisions don't yet read from these tables — Phase 2 will add a config
`panel_buy_pctile` that JointActionTask consults via percentile lookup.

Default OFF — opt-in via `score_db.enabled` config flag.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from kernel.decision_trace import candidate_trace_pool, model_types_from_models

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.score_db")


class RecordScoreDistributionTask(Task):
    """Persist this bar's panel-LTR score distribution to runs.db.

    Reads:
      ctx.candidates  (panel_score, rank_score on each)
      ctx.holdings    (panel_score, rank_score on each — may have None)
      ctx._db         (sqlite3 connection injected by adapters)

    Writes:
      score_distribution    INSERT OR REPLACE per (run_id, ticker)
      score_percentiles_daily  INSERT OR REPLACE one row for this run
    """

    PERCENTILES = [1, 5, 10, 25, 50, 75, 85, 90, 95, 99]

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = ctx.config.get("score_db") or {}
        if not cfg.get("enabled", False):
            return False
        db = getattr(ctx, "_db", None)
        if db is None:
            return False
        if not ctx.candidates and not ctx.holdings:
            return False

        date_iso = ctx.today.isoformat()
        run_id = (
            getattr(ctx, "run_id", None)
            or getattr(ctx, "_run_id", None)
            or f"{date_iso}-unscoped"
        )
        run_type = _ctx_run_type(ctx)
        regime = str(ctx.regime or "")
        cand_pool = candidate_trace_pool(ctx)
        blocked_map = getattr(ctx, "_blocked_by_ticker", None) or {}
        sector_map = (ctx.config or {}).get("sector_map", {}) or {}
        model_types = model_types_from_models(getattr(ctx, "models", None) or {})
        candidate_tickers = {getattr(c, "ticker", None) for c in cand_pool}

        rows: list[tuple] = []
        for c in cand_pool:
            ticker = getattr(c, "ticker", None)
            rows.append((
                run_id, date_iso, run_type, ticker,
                getattr(c, "panel_score", None),
                getattr(c, "rank_score", None),
                getattr(c, "expected_return_horizon_days", None),
                getattr(c, "mu", None),
                getattr(c, "mu_horizon_days", None),
                getattr(c, "sigma", None),
                regime,
                0,  # is_holding=False
                model_types.get(ticker),
                _sector_for(ticker, sector_map),
                blocked_map.get(ticker),
            ))
        for ticker, hs in ctx.holdings.items():
            if ticker in candidate_tickers:
                continue
            rows.append((
                run_id, date_iso, run_type, ticker,
                getattr(hs, "panel_score", None),
                getattr(hs, "rank_score", None),
                getattr(hs, "expected_return_horizon_days", None),
                getattr(hs, "mu", None),
                getattr(hs, "mu_horizon_days", None),
                getattr(hs, "sigma", None),
                regime,
                1,  # is_holding=True
                model_types.get(ticker),
                _sector_for(ticker, sector_map),
                blocked_map.get(ticker),
            ))

        try:
            cur = db.cursor()
            cur.executemany(
                """INSERT OR REPLACE INTO score_distribution
                   (run_id, date, run_type, ticker, raw_panel, rank_score,
                    expected_return_horizon_days, mu, mu_horizon_days, sigma,
                    regime, is_holding, model_type, sector, blocked_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

            # Aggregate percentiles from CANDIDATE scores (not holdings —
            # holdings already in the portfolio aren't comparable to fresh
            # cands for "top X% buy threshold" purposes).
            cand_scores = [
                float(getattr(c, "rank_score", None))
                for c in cand_pool
                if getattr(c, "rank_score", None) is not None
                and np.isfinite(float(getattr(c, "rank_score", None)))
            ]
            if cand_scores:
                arr = np.asarray(cand_scores, dtype=float)
                p_vals = np.percentile(arr, self.PERCENTILES)
                cur.execute(
                    """INSERT OR REPLACE INTO score_percentiles_daily
                       (run_id, date, run_type, n_cands, p01, p05, p10, p25, p50, p75, p85,
                        p90, p95, p99, score_min, score_max, score_mean,
                        score_std, regime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        date_iso,
                        run_type,
                        len(cand_scores),
                        float(p_vals[0]), float(p_vals[1]), float(p_vals[2]),
                        float(p_vals[3]), float(p_vals[4]), float(p_vals[5]),
                        float(p_vals[6]), float(p_vals[7]), float(p_vals[8]),
                        float(p_vals[9]),
                        float(arr.min()), float(arr.max()),
                        float(arr.mean()), float(arr.std(ddof=0)),
                        regime,
                    ),
                )
            db.commit()
            log.info(
                "RecordScoreDistributionTask: saved %d ticker rows + percentiles "
                "(n_cands=%d) for run_id=%s date=%s",
                len(rows), len(cand_scores), run_id, date_iso,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("RecordScoreDistributionTask: skip — %s", exc)
            return False


# ── Helpers (Phase 2 will use these from JointActionTask) ──────────────────────

def get_score_percentile_threshold(
    db: Any, today_iso: str, percentile: int = 85,
    lookback_days: int = 5,
    run_type: str | None = None,
    include_today: bool = True,
) -> float | None:
    """Return the score-percentile threshold averaged across the last
    `lookback_days` of trading days, or None if no rows yet.

    Example: percentile=85 lookback_days=5 → mean of p85 values across
    last 5 daily rows. Useful as buy_floor surrogate.

    ``include_today=False`` is for live decision gates. It prevents a
    same-date rerun from reading a percentile row written by an earlier run
    on the same market date.
    """
    col = f"p{percentile:02d}"
    if col not in {"p01", "p05", "p10", "p25", "p50", "p75",
                    "p85", "p90", "p95", "p99"}:
        raise ValueError(f"Unsupported percentile {percentile}")
    cur = db.cursor()
    date_op = "<=" if include_today else "<"
    run_filter = "AND run_type = ?" if run_type else ""
    params: tuple[Any, ...]
    if run_type:
        params = (today_iso, run_type, lookback_days)
    else:
        params = (today_iso, lookback_days)
    cur.execute(
        f"""SELECT {col}
              FROM (
                    SELECT date, {col},
                           ROW_NUMBER() OVER (
                               PARTITION BY date
                               ORDER BY created_at DESC, run_id DESC
                           ) AS rn
                      FROM score_percentiles_daily
                     WHERE date {date_op} ?
                       {run_filter}
                   )
             WHERE rn = 1
             ORDER BY date DESC
             LIMIT ?""",
        params,
    )
    rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    if not rows:
        return None
    return float(np.mean(rows))


def _ctx_run_type(ctx: Any) -> str | None:
    value = getattr(ctx, "_run_type", None) or getattr(ctx, "run_type", None)
    if isinstance(value, str) and value:
        return value
    run_id = str(getattr(ctx, "run_id", "") or getattr(ctx, "_run_id", ""))
    for token in ("live", "sim", "lean"):
        if f"-{token}-" in run_id or run_id.endswith(f"-{token}"):
            return token
    return None


def _sector_for(ticker: Any, sector_map: dict[str, str]) -> str | None:
    if ticker is None:
        return None
    value = sector_map.get(str(ticker))
    if isinstance(value, str) and value:
        return value
    value = sector_map.get(str(ticker).upper())
    return value if isinstance(value, str) and value else None
