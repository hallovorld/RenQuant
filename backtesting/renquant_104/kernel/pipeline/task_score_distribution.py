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
      score_distribution    INSERT OR REPLACE per (date, ticker)
      score_percentiles_daily  INSERT OR REPLACE one row for today
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
        regime = str(ctx.regime or "")

        rows: list[tuple] = []
        for c in ctx.candidates:
            rows.append((
                date_iso, c.ticker,
                getattr(c, "panel_score", None),
                getattr(c, "rank_score", None),
                getattr(c, "mu", None),
                getattr(c, "sigma", None),
                regime,
                0,  # is_holding=False
            ))
        for ticker, hs in ctx.holdings.items():
            rows.append((
                date_iso, ticker,
                getattr(hs, "panel_score", None),
                getattr(hs, "rank_score", None),
                getattr(hs, "mu", None),
                getattr(hs, "sigma", None),
                regime,
                1,  # is_holding=True
            ))

        try:
            cur = db.cursor()
            cur.executemany(
                """INSERT OR REPLACE INTO score_distribution
                   (date, ticker, raw_panel, rank_score, mu, sigma,
                    regime, is_holding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

            # Aggregate percentiles from CANDIDATE scores (not holdings —
            # holdings already in the portfolio aren't comparable to fresh
            # cands for "top X% buy threshold" purposes).
            cand_scores = [
                float(getattr(c, "rank_score", None))
                for c in ctx.candidates
                if getattr(c, "rank_score", None) is not None
                and np.isfinite(float(getattr(c, "rank_score", None)))
            ]
            if cand_scores:
                arr = np.asarray(cand_scores, dtype=float)
                p_vals = np.percentile(arr, self.PERCENTILES)
                cur.execute(
                    """INSERT OR REPLACE INTO score_percentiles_daily
                       (date, n_cands, p01, p05, p10, p25, p50, p75, p85,
                        p90, p95, p99, score_min, score_max, score_mean,
                        score_std, regime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        date_iso,
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
                "(n_cands=%d) for date=%s",
                len(rows), len(cand_scores), date_iso,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("RecordScoreDistributionTask: skip — %s", exc)
            return False


# ── Helpers (Phase 2 will use these from JointActionTask) ──────────────────────

def get_score_percentile_threshold(
    db: Any, today_iso: str, percentile: int = 85,
    lookback_days: int = 5,
) -> float | None:
    """Return the score-percentile threshold averaged across the last
    `lookback_days` of trading days, or None if no rows yet.

    Example: percentile=85 lookback_days=5 → mean of p85 values across
    last 5 daily rows. Useful as buy_floor surrogate.
    """
    col = f"p{percentile:02d}"
    if col not in {"p01", "p05", "p10", "p25", "p50", "p75",
                    "p85", "p90", "p95", "p99"}:
        raise ValueError(f"Unsupported percentile {percentile}")
    cur = db.cursor()
    cur.execute(
        f"""SELECT {col} FROM score_percentiles_daily
            WHERE date <= ?
            ORDER BY date DESC
            LIMIT ?""",
        (today_iso, lookback_days),
    )
    rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    if not rows:
        return None
    return float(np.mean(rows))
