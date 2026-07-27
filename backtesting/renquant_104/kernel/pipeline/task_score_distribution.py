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

import datetime as dt
import logging
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from kernel.decision_trace import candidate_trace_pool, model_types_from_models

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.score_db")

# Simulated decision instant fallback (pipeline#215 design §2.2): the
# official US-equity session close in the session timezone — the
# ``decision_schedule.run_bundle_timestamp`` convention. KEEP IN SYNC with
# ``renquant_pipeline.kernel.pipeline.task_score_distribution`` (the
# pipeline copy of this task); the record construction + PIT check are
# imported from the pipeline provenance module, never re-implemented.
_SESSION_TZ = "America/New_York"
_SESSION_CLOSE = dt.time(16, 0)


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
        rows = collect_score_rows(
            ctx, run_id=run_id, date_iso=date_iso, run_type=run_type,
            regime=regime, cand_pool=cand_pool,
        )

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

        # WF sim-time provenance (pipeline#215 §2.3): score_committed is
        # emitted immediately AFTER the successful INSERT, binding the
        # provenance to the exact observation Phase-A will read. The sink
        # and the fold echo ride on ctx (stamped by the sim adapter when it
        # binds the fold's scorer); absent sink attr = no-op, so the
        # default daily/live path is byte-identical. Emit failures
        # propagate — a sim that persisted a score but cannot persist its
        # evidence chain must fail loudly, not degrade into the post-hoc
        # reconstruction this contract exists to kill.
        emit_wf_score_committed(
            ctx, rows, run_id=run_id, date_iso=date_iso, run_type=run_type,
            persisted=True,
        )


# ── WF provenance emit (pipeline#215 §2.1/§2.2) ────────────────────────────────

# Insert-tuple coordinates of the canonical score-payload fields. MUST
# mirror the column order of the score_distribution INSERT above:
# (run_id, date, run_type, ticker, raw_panel, rank_score,
#  expected_return_horizon_days, mu, mu_horizon_days, sigma, ...).
_ROW_TICKER, _ROW_RAW_PANEL, _ROW_RANK_SCORE = 3, 4, 5
_ROW_MU, _ROW_SIGMA = 7, 9


def collect_score_rows(
    ctx: Any,
    *,
    run_id: str,
    date_iso: str,
    run_type: "str | None",
    regime: str,
    cand_pool: "list | None" = None,
) -> list[tuple]:
    """Build the score_distribution INSERT tuples for this bar.

    Extracted from ``RecordScoreDistributionTask.run`` so the persistence-
    off provenance leg (``emit_unpersisted_wf_score_committed``, called by
    the sim adapter) digests the SAME payload the task would have
    persisted — one row-building implementation, never two.
    """
    if cand_pool is None:
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
    return rows


def emit_wf_score_committed(
    ctx: Any,
    rows: list[tuple],
    *,
    run_id: str,
    date_iso: str,
    run_type: "str | None",
    persisted: bool,
) -> bool:
    """Emit ``score_committed`` for this bar (no-op without a ctx sink).

    Record construction, digest grammar, and the PIT invariant
    (``input_watermark <= score_timestamp``) are IMPORTS ONLY from
    ``renquant_pipeline.kernel.walk_forward.provenance`` — the lazy import
    only runs when a sink is stamped, which itself requires a post-#216
    pipeline pin. Returns True iff a record was emitted; also stamps
    ``ctx._wf_score_committed`` so the sim adapter's persistence-off leg
    knows the persisted emit already happened.
    """
    sink = getattr(ctx, "_wf_provenance_sink", None)
    if sink is None:
        return False
    from renquant_pipeline.kernel.walk_forward.provenance import (  # noqa: PLC0415
        build_score_committed_record,
        score_payload_digest,
    )
    payload_rows = [
        {
            "ticker": r[_ROW_TICKER],
            "raw_panel": r[_ROW_RAW_PANEL],
            "mu": r[_ROW_MU],
            "rank_score": r[_ROW_RANK_SCORE],
            "sigma": r[_ROW_SIGMA],
        }
        for r in rows
    ]
    fold = getattr(ctx, "_wf_active_fold", None)
    record = build_score_committed_record(
        prediction_date=date_iso,
        score_observation_key=[run_id, date_iso, run_type],
        score_payload_digest=score_payload_digest(payload_rows),
        n_rows=len(rows),
        artifact_digest=_fold_field(fold, "artifact_digest"),
        score_timestamp=_score_timestamp(ctx),
        input_watermark=getattr(ctx, "_wf_input_watermark", None),
        persisted=persisted,
    )
    sink.emit(record)
    ctx._wf_score_committed = True  # noqa: SLF001 - adapter handshake
    return True


def emit_unpersisted_wf_score_committed(ctx: Any) -> bool:
    """Persistence-off leg (design §2.1 durability): ``persisted: false``.

    Called by the sim adapter post-scoring when the task did NOT commit
    rows this bar (``--no-persist`` / ``persistence.enabled=False`` /
    ``score_db`` disabled). Digests the in-memory payload the task WOULD
    have persisted. Such rows are Phase-A-INADMISSIBLE by construction —
    emitted for append-only honesty, never as evidence. Bars with an empty
    payload emit nothing (there is no observation to bind; the orphaned
    ``fold_resolved`` correctly marks the date incomplete).
    """
    if getattr(ctx, "_wf_provenance_sink", None) is None:
        return False
    if getattr(ctx, "_wf_score_committed", False):
        return False  # the persisted emit already covered this bar
    date_iso = ctx.today.isoformat()
    run_id = (
        getattr(ctx, "run_id", None)
        or getattr(ctx, "_run_id", None)
        or f"{date_iso}-unscoped"
    )
    run_type = _ctx_run_type(ctx)
    rows = collect_score_rows(
        ctx, run_id=run_id, date_iso=date_iso, run_type=run_type,
        regime=str(ctx.regime or ""),
    )
    if not rows:
        return False
    return emit_wf_score_committed(
        ctx, rows, run_id=run_id, date_iso=date_iso, run_type=run_type,
        persisted=False,
    )


def _fold_field(fold: Any, name: str) -> Any:
    """Read a field off ``ctx._wf_active_fold`` (mapping or object)."""
    if fold is None:
        return None
    if isinstance(fold, dict):
        return fold.get(name)
    return getattr(fold, name, None)


def _score_timestamp(ctx: Any) -> str:
    """The simulated session's decision instant for this bar (§2.2).

    Primary source: ``ctx.run_timestamp`` — the one decision timestamp the
    InferenceContext carries. A naive value is interpreted in the session
    timezone (America/New_York). Fallback (sim/LEAN deliberately leave
    ``run_timestamp=None`` for bar-date-only semantics): the official
    US-equity session close — 16:00 America/New_York on the bar date.
    ISO-8601 with offset either way. Mirrors the pipeline task's helper.
    """
    tz = ZoneInfo(_SESSION_TZ)
    ts = getattr(ctx, "run_timestamp", None)
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=tz)
        return ts.isoformat()
    return dt.datetime.combine(ctx.today, _SESSION_CLOSE, tzinfo=tz).isoformat()


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
