"""PanelScoringJob — swap in cross-sectional panel scores during inference.

Slots between CandidateJob (Phase 2) and RankingJob (Phase 3) of the
standard InferencePipeline. When the config flag
`ranking.panel_scoring.enabled` is true and a panel-LTR artifact is
configured, this Job loads the scorer, builds today's inference matrix
for every candidate ticker, and overwrites each CandidateResult's
`rank_score` in place. The existing RankingJob then blends that panel
score with rs_score using the same `ranking.blend_weights`.

Task chain::

    LoadScorerTask           read artifact path from config, cache scorer
    BuildFeatureMatrixTask   pick today's rows per candidate ticker
    ApplyScoresTask          write panel_score into CandidateResult.rank_score

The Job is a no-op when:
  • the config flag is off, OR
  • no candidates or holdings require panel scores.

When panel scoring is enabled, scorer/feature/score failures are fail-closed
for buys: candidates are cleared and tagged in `_blocked_by_ticker`. This keeps
the decision tree from silently falling back to weaker per-ticker scores.

Kept isolated from the Stage-1 training pipeline so revert is purely
additive: remove this file + the one-line import wiring.
"""
from __future__ import annotations

import datetime
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .panel_scorer import PanelScorer
from .feature_matrix import build_inference_matrix

log = logging.getLogger("kernel.panel_pipeline.scoring")


def _runtime_cache(ctx: Any) -> dict | None:
    cache = getattr(ctx, "_panel_runtime_cache", None)
    return cache if isinstance(cache, dict) else None


def _cached_parquet(ctx: Any, key: tuple, path: Path) -> pd.DataFrame | None:
    cache = _runtime_cache(ctx)
    if cache is None:
        return pd.read_parquet(path)
    if key not in cache:
        cache[key] = pd.read_parquet(path)
    return cache[key]


def _cached_earnings_surprise(ctx: Any, path: Path) -> pd.DataFrame | None:
    cache_key = ("earnings_surprise", str(path))
    cache = _runtime_cache(ctx)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    earn = pd.read_parquet(path).reset_index()
    earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
    earn = earn.sort_values("earnings_date").reset_index(drop=True)
    if cache is not None:
        cache[cache_key] = earn
    return earn


def _cached_sentiment(ctx: Any, path: Path) -> pd.DataFrame | None:
    cache_key = ("news_sentiment", str(path))
    cache = _runtime_cache(ctx)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    sdf = pd.read_parquet(path)
    sdf["date"] = pd.to_datetime(sdf["date"])
    if cache is not None:
        cache[cache_key] = sdf
    return sdf


def _alpha158_cached_rows(
    ctx: Any,
    tickers: list[str],
    today: Any,
) -> dict[str, dict[str, float]]:
    cache = getattr(ctx, "_alpha158_feature_cache", None)
    if not isinstance(cache, dict) or not cache:
        return {}
    today_ts = pd.Timestamp(today)
    rows: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        frame = cache.get(ticker)
        if frame is None or frame.empty:
            continue
        sub = frame.loc[:today_ts]
        if sub.empty:
            continue
        rows[ticker] = sub.iloc[-1].to_dict()
    return rows


def _stable_feature_context_tickers(
    ctx: Any,
    target_tickers: list[str],
    scorer: Any | None = None,
) -> list[str]:
    """Return the stable cross-section used for extra-feature fill/rank.

    Training fills fundamentals/sentiment and ranks PEAD over the full date
    cross-section. Runtime must therefore not compute medians/ranks over the
    post-filter candidate subset; that makes a ticker's feature value depend on
    which other tickers survived gates on the same bar.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add_many(values: Any) -> None:
        if isinstance(values, dict):
            values = values.keys()
        if not isinstance(values, (list, tuple, set)):
            return
        for value in values:
            if value is None:
                continue
            ticker = str(value)
            if ticker and ticker not in seen:
                seen.add(ticker)
                out.append(ticker)

    panel_cfg = (getattr(ctx, "config", {}) or {}).get("ranking", {}) \
        .get("panel_scoring", {})
    for key in (
        "feature_context_tickers",
        "training_universe",
        "train_tickers",
        "tickers",
    ):
        add_many(panel_cfg.get(key))
    metadata = getattr(scorer, "metadata", {}) or {}
    for key in (
        "feature_context_tickers",
        "training_universe",
        "train_tickers",
        "tickers",
        "watchlist",
    ):
        add_many(metadata.get(key))
    add_many((getattr(ctx, "config", {}) or {}).get("watchlist", []))
    add_many(getattr(ctx, "models", {}) or {})
    add_many(getattr(ctx, "holdings", {}) or {})
    add_many(target_tickers)
    return out


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median_fill_rows(
    raw_by_ticker: dict[str, dict[str, float | None]],
    target_tickers: list[str],
    context_tickers: list[str],
    cols: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    medians: dict[str, float] = {}
    filled: dict[str, dict[str, float]] = {}
    for col in cols:
        vals = [
            float(raw_by_ticker.get(t, {}).get(col))
            for t in context_tickers
            if _finite_or_none(raw_by_ticker.get(t, {}).get(col)) is not None
        ]
        medians[col] = float(np.median(vals)) if vals else 0.0
    for ticker in target_tickers:
        row: dict[str, float] = {}
        raw = raw_by_ticker.get(ticker, {})
        for col in cols:
            value = _finite_or_none(raw.get(col))
            row[col] = value if value is not None else medians[col]
        filled[ticker] = row
    return filled, medians


def _apply_fund_features(
    rows: dict[str, dict[str, float]],
    fund_panel: pd.DataFrame,
    today: Any,
    context_tickers: list[str],
    fund_cols: list[str],
) -> tuple[int, int, dict[str, float]]:
    today_ts = pd.Timestamp(today)
    panel = fund_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    snap = panel[panel["date"] <= today_ts] \
        .sort_values("date").groupby("ticker").tail(1)
    by_ticker = {
        str(t): g.iloc[-1]
        for t, g in snap.groupby("ticker", sort=False)
    }
    raw: dict[str, dict[str, float | None]] = {}
    for ticker in context_tickers:
        src = by_ticker.get(str(ticker))
        raw[ticker] = {
            col: (_finite_or_none(src[col])
                  if src is not None and col in src.index else None)
            for col in fund_cols
        }
    target_tickers = list(rows.keys())
    filled, medians = _median_fill_rows(raw, target_tickers, context_tickers, fund_cols)
    n_real = 0
    n_imputed = 0
    for ticker in target_tickers:
        for col in fund_cols:
            if _finite_or_none(raw.get(ticker, {}).get(col)) is None:
                n_imputed += 1
            else:
                n_real += 1
            rows[ticker][col] = filled[ticker][col]
    return n_real, n_imputed, medians


def _earnings_raw_row(
    ctx: Any,
    earn_dir: Path,
    ticker: str,
    today_ts: pd.Timestamp,
) -> tuple[dict[str, float | None], bool, bool, bool]:
    ep = earn_dir / f"{ticker}.parquet"
    if not ep.exists():
        return {}, True, False, False
    earn = _cached_earnings_surprise(ctx, ep)
    if earn is None:
        return {}, True, False, False
    prior = earn[earn["earnings_date"] <= today_ts]
    if len(prior) == 0:
        return {}, False, True, False
    last = prior.iloc[-1]
    days_since = int((today_ts - last["earnings_date"]).days)
    if days_since > 60 or days_since < 0:
        return {}, False, False, True
    decay = max(0.0, 1.0 - days_since / 60)
    surprise = _finite_or_none(last.get("surprise_pct")) or 0.0
    return {
        "days_since_earnings": float(days_since),
        "pead_signal": surprise * decay,
        "pead_surprise": surprise,
    }, False, False, False


def _apply_pead_features(
    ctx: Any,
    rows: dict[str, dict[str, float]],
    earn_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    pead_cols: list[str],
) -> tuple[int, int, int, int]:
    raw: dict[str, dict[str, float | None]] = {}
    n_no_data = n_no_prior = n_out_of_window = 0
    for ticker in context_tickers:
        row, no_data, no_prior, oow = _earnings_raw_row(ctx, earn_dir, ticker, today_ts)
        n_no_data += int(no_data)
        n_no_prior += int(no_prior)
        n_out_of_window += int(oow)
        raw[ticker] = {
            "days_since_earnings": row.get("days_since_earnings"),
            "pead_signal": row.get("pead_signal"),
            "pead_quintile_rank": None,
        }
        if row.get("pead_surprise") is not None:
            raw[ticker]["pead_surprise"] = row["pead_surprise"]
    surprises = {
        ticker: raw[ticker]["pead_surprise"]
        for ticker in context_tickers
        if _finite_or_none(raw.get(ticker, {}).get("pead_surprise")) is not None
    }
    if surprises:
        ranks = pd.Series(surprises, dtype=float).rank(pct=True)
        for ticker, rank in ranks.items():
            raw[ticker]["pead_quintile_rank"] = float(rank)
    filled, _medians = _median_fill_rows(
        raw, list(rows.keys()), context_tickers, pead_cols,
    )
    for ticker, vals in filled.items():
        for col in pead_cols:
            rows[ticker][col] = vals[col]
    return len(surprises), n_no_data, n_no_prior, n_out_of_window


def _apply_sue_features(
    ctx: Any,
    rows: dict[str, dict[str, float]],
    earn_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    sue_cols: list[str],
) -> tuple[int, int, int]:
    raw: dict[str, dict[str, float | None]] = {}
    n_active = n_no_data = n_oow = 0
    for ticker in context_tickers:
        ep = earn_dir / f"{ticker}.parquet"
        if not ep.exists():
            n_no_data += 1
            raw[ticker] = {col: None for col in sue_cols}
            continue
        earn = _cached_earnings_surprise(ctx, ep)
        if earn is None:
            n_no_data += 1
            raw[ticker] = {col: None for col in sue_cols}
            continue
        prior = earn[earn["earnings_date"] <= today_ts]
        if len(prior) == 0:
            raw[ticker] = {col: 0.0 for col in sue_cols}
            continue
        last = prior.iloc[-1]
        days_since = int((today_ts - last["earnings_date"]).days)
        if days_since > 60 or days_since < 0:
            n_oow += 1
            raw[ticker] = {col: 0.0 for col in sue_cols}
            continue
        decay = max(0.0, 1.0 - days_since / 60)
        s = prior["surprise_pct"].astype(float)
        if len(s) >= 2:
            denom_window = s.iloc[max(0, len(s) - 1 - 4):len(s) - 1]
            denom = float(denom_window.std()) if len(denom_window) >= 2 else 0.0
            sue = float(s.iloc[-1]) / max(denom, 1e-6)
            sue = max(min(sue, 5.0), -5.0)
        else:
            sue = 0.0
        mom = float(s.iloc[-1] - s.iloc[-2]) if len(s) >= 2 else 0.0
        streak = 0
        cur_sign = 0
        for v in s:
            sign = 1 if v > 0 else (-1 if v < 0 else 0)
            if sign == 0 or sign != cur_sign:
                streak = sign
                cur_sign = sign
            else:
                streak += sign
        raw[ticker] = {
            "sue_signal": sue * decay,
            "surprise_momentum": mom * decay,
            "surprise_streak": float(streak) * decay,
        }
        n_active += 1
    filled, _medians = _median_fill_rows(raw, list(rows.keys()), context_tickers, sue_cols)
    for ticker, vals in filled.items():
        for col in sue_cols:
            rows[ticker][col] = vals[col]
    return n_active, n_no_data, n_oow


def _sentiment_runtime_gate_declared(scorer: Any) -> bool:
    metadata = getattr(scorer, "metadata", {}) or {}
    contract = (
        metadata.get("sentiment_runtime_gate_contract")
        or metadata.get("sentiment_gate_contract")
    )
    return contract in {"trained_zeroing", "runtime_zeroing"} or bool(
        metadata.get("sentiment_runtime_gate_trained", False)
    )


def _apply_sentiment_features(
    ctx: Any,
    scorer: Any,
    rows: dict[str, dict[str, float]],
    sent_dir: Path,
    today_ts: pd.Timestamp,
    context_tickers: list[str],
    sent_cols: list[str],
) -> tuple[int, int, bool]:
    raw: dict[str, dict[str, float | None]] = {}
    n_hit = n_miss = 0
    for ticker in context_tickers:
        sp = sent_dir / f"{ticker}.parquet"
        raw[ticker] = {col: None for col in sent_cols}
        if not sp.exists():
            n_miss += 1
            continue
        try:
            sdf = _cached_sentiment(ctx, sp)
        except Exception:
            n_miss += 1
            continue
        exact = sdf[pd.to_datetime(sdf["date"]) == today_ts]
        if len(exact) == 0:
            n_miss += 1
            continue
        last = exact.iloc[-1]
        if "sentiment_pos_share" in sent_cols:
            raw[ticker]["sentiment_pos_share"] = _finite_or_none(
                last.get("sentiment_pos_share")
            )
        if "mean_sentiment" in sent_cols:
            raw[ticker]["mean_sentiment"] = _finite_or_none(last.get("mean_sentiment"))
        if "n_articles_log" in sent_cols:
            if "n_articles_log" in last.index:
                raw[ticker]["n_articles_log"] = _finite_or_none(last.get("n_articles_log"))
            else:
                n_articles = _finite_or_none(last.get("n_articles")) or 0.0
                raw[ticker]["n_articles_log"] = float(np.log1p(n_articles))
        n_hit += 1
    filled, _medians = _median_fill_rows(
        raw, list(rows.keys()), context_tickers, sent_cols,
    )
    for ticker, vals in filled.items():
        for col in sent_cols:
            rows[ticker][col] = vals[col]
    sent_enabled = bool(_sentiment_cfg(ctx).get("enabled", True))
    gate_applied = False
    if not sent_enabled:
        if _sentiment_runtime_gate_declared(scorer):
            for ticker in rows:
                for col in sent_cols:
                    rows[ticker][col] = 0.0
            gate_applied = True
        else:
            log.warning(
                "ApplyScoresTask[panel_ltr_xgboost]: sentiment gate OFF for "
                "regime=%s, but artifact lacks trained runtime-zeroing "
                "contract; leaving exact-date sentiment features unchanged.",
                getattr(ctx, "regime", "?"),
            )
    return n_hit, n_miss, gate_applied


def _candidate_ticker(candidate: Any) -> str | None:
    ticker = getattr(candidate, "ticker", None)
    return str(ticker) if ticker else None


def _ensure_blocked_map(ctx: Any) -> dict[str, str]:
    blocked = getattr(ctx, "_blocked_by_ticker", None)
    if blocked is None:
        blocked = {}
        ctx._blocked_by_ticker = blocked  # noqa: SLF001
    return blocked


def _snapshot_buy_candidates(ctx: Any) -> list[Any]:
    """Preserve the candidate pool so decision-trace persistence can explain drops."""
    existing = list(getattr(ctx, "_full_candidate_snapshot", None) or [])
    seen = {_candidate_ticker(c) for c in existing}
    for cand in list(getattr(ctx, "candidates", None) or []):
        ticker = _candidate_ticker(cand)
        if ticker and ticker not in seen:
            existing.append(cand)
            seen.add(ticker)
    ctx._full_candidate_snapshot = existing  # noqa: SLF001
    return existing


def _fail_closed_panel_scoring(ctx: Any, reason: str) -> None:
    """Block buy/QP when enabled panel scoring cannot provide the alpha surface."""
    candidates = list(getattr(ctx, "candidates", None) or [])
    _snapshot_buy_candidates(ctx)
    blocked = _ensure_blocked_map(ctx)
    for cand in candidates:
        ticker = _candidate_ticker(cand)
        if ticker:
            blocked[ticker] = reason
    ctx.candidates = []
    ctx.buy_blocked = True
    ctx.skip_buys = True
    ctx._panel_scoring_contract_failed = True  # noqa: SLF001
    ctx._panel_scoring_fail_reason = reason  # noqa: SLF001
    counters = getattr(ctx, "counters", None)
    if isinstance(counters, dict):
        counters["panel_scoring_fail_closed"] = (
            counters.get("panel_scoring_fail_closed", 0) + len(candidates)
        )
    log.error(
        "Panel scoring contract failed (%s). Cleared %d buy candidate(s); "
        "buy/QP path is fail-closed for this run.",
        reason,
        len(candidates),
    )


def _drop_unscored_panel_candidates(
    ctx: Any,
    scored_tickers: set[str],
    reason: str,
) -> int:
    """Drop candidates that did not receive a finite panel score."""
    candidates = list(getattr(ctx, "candidates", None) or [])
    if not candidates:
        return 0
    _snapshot_buy_candidates(ctx)
    blocked = _ensure_blocked_map(ctx)
    kept = []
    dropped = 0
    for cand in candidates:
        ticker = _candidate_ticker(cand)
        if ticker and ticker in scored_tickers:
            kept.append(cand)
            continue
        if ticker:
            blocked[ticker] = reason
        dropped += 1
    if dropped:
        ctx.candidates = kept
        counters = getattr(ctx, "counters", None)
        if isinstance(counters, dict):
            counters["panel_score_missing"] = (
                counters.get("panel_score_missing", 0) + dropped
            )
        log.error(
            "ApplyScoresTask: dropped %d/%d candidate(s) without finite panel score "
            "(%s). Refusing per-ticker-score fallback.",
            dropped,
            len(candidates),
            reason,
        )
    return dropped


# ── Task chain ────────────────────────────────────────────────────────────────

class LoadScorerTask(Task):
    """Load the PanelScorer artifact from config. Cache on ctx for reuse."""

    @staticmethod
    def _resolve_artifact_path(
        ctx: InferenceContext,
        panel_cfg: dict,
        scorer: Any | None = None,
    ) -> Path | None:
        metadata = getattr(scorer, "metadata", {}) or {}
        artifact_path = metadata.get("artifact_path") or panel_cfg.get("artifact_path")
        if not artifact_path:
            return None
        p = Path(artifact_path)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p
        return p

    @staticmethod
    def _assert_config_consistency(
        ctx: InferenceContext,
        panel_cfg: dict,
        scorer: Any,
        path: Path | None,
    ) -> bool:
        strict = bool(panel_cfg.get("strict_config_consistency", True))
        try:
            from kernel.config_consistency import (  # noqa: PLC0415
                assert_consistent, ConfigModelMismatch,
            )
            import json as _j  # noqa: PLC0415

            metadata = getattr(scorer, "metadata", {}) or {}
            artifact_meta = dict(metadata)
            if path is not None and path.suffix.lower() == ".json":
                artifact_meta = _j.loads(path.read_text())
            try:
                assert_consistent(
                    ctx.config,
                    artifact_meta,
                    artifact_label=str(path.name if path is not None else "<preloaded>"),
                    strict=strict,
                )
            except ConfigModelMismatch as e:
                log.error("LoadScorerTask: %s", e)
                _fail_closed_panel_scoring(ctx, "panel_scorer_config_mismatch")
                return False
        except Exception as exc:  # noqa: BLE001
            if strict:
                log.error("LoadScorerTask: consistency check failed: %s", exc)
                _fail_closed_panel_scoring(ctx, "panel_scorer_consistency_check_failed")
                return False
            log.warning("LoadScorerTask: consistency check failed: %s", exc)
        return True

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("enabled", False):
            log.debug("LoadScorerTask: panel scoring disabled — skipping chain")
            return False

        # Scorer may have been pre-loaded by the adapter (live runner / LEAN)
        scorer = getattr(ctx, "_panel_scorer", None)
        if scorer is not None:
            p = self._resolve_artifact_path(ctx, panel_cfg, scorer)
            if not self._assert_config_consistency(ctx, panel_cfg, scorer, p):
                return False
            return

        p = self._resolve_artifact_path(ctx, panel_cfg)
        if p is None:
            log.error("LoadScorerTask: panel_scoring.enabled but no artifact_path")
            _fail_closed_panel_scoring(ctx, "panel_scorer_missing_artifact_path")
            return False
        # 2026-05-18 Model registry dispatch — supports XGB/PatchTST/future kinds
        # via single config knob `ranking.panel_scoring.kind`. Default xgb
        # for back-compat. Each kind's handler in kernel/panel_pipeline/
        # model_registry.py decides how to load its scorer.
        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415
        kind = panel_cfg.get("kind", "xgb")
        try:
            handler = registry.get(kind)
        except ValueError as exc:
            log.error("LoadScorerTask: %s", exc)
            _fail_closed_panel_scoring(ctx, "panel_scorer_invalid_kind")
            return False
        try:
            ctx._panel_scorer = handler.scorer_loader(p, ctx.config)  # noqa: SLF001
        except Exception as exc:
            log.error("LoadScorerTask: failed to load %s artifact %s — %s",
                      kind, p, exc)
            _fail_closed_panel_scoring(ctx, "panel_scorer_load_failed")
            return False
        log.info("LoadScorerTask: loaded %s artifact (features=%d, "
                 "requires_history=%s)", kind,
                 len(ctx._panel_scorer.feature_cols),
                 getattr(ctx._panel_scorer, "requires_history", False))

        # 2026-04-28 self-audit: config / model consistency check.
        # Invariant: a fingerprint mismatch must — by default — prevent
        # panel scoring from running, because the alternative is silent
        # miscalibrated trades. Three incidents in 24h proved log-only
        # isn't enough (operators don't tail logs every bar).
        # Set ranking.panel_scoring.strict_config_consistency=false to
        # downgrade to log-only (only for staged migrations).
        # Artifacts without a stored fingerprint fail closed when strict is
        # enabled; only explicit staged migrations may opt into log-only mode.
        if not self._assert_config_consistency(ctx, panel_cfg, ctx._panel_scorer, p):
            return False


class BuildFeatureMatrixTask(Task):
    """Back-compat shim. The 165-line monolith was split per CLAUDE.md
    §1c (2026-05-04) into `BuildFeatureMatrixJob` with 4 Tasks:

        ResolveInferenceFramesTask    — subset frames, macro v1/v2
        AssembleInferenceMatrixTask   — call build_inference_matrix
        RowCoverageGateTask           — drop low-coverage rows
        DriftGuardTask                — structural vs transient NaN

    See `kernel/panel_pipeline/tasks_feature_matrix.py`. Existing
    callers (PanelScoringJob.tasks list) keep working unchanged.
    """

    _job = None   # lazy-init to avoid circular import at module load

    def run(self, ctx: InferenceContext) -> bool | None:
        if BuildFeatureMatrixTask._job is None:
            from .tasks_feature_matrix import BuildFeatureMatrixJob
            BuildFeatureMatrixTask._job = BuildFeatureMatrixJob()
        BuildFeatureMatrixTask._job.run(ctx)


def _scorer_requires_history(scorer: object) -> bool:
    """Return True only when a scorer explicitly opts into sequence history."""
    return getattr(scorer, "requires_history", False) is True


class ApplyScoresTask(Task):
    """Score the matrix and write panel_score onto candidates AND holdings.

    For candidates the panel score also overwrites `rank_score` so the
    downstream RankingJob/SelectionJob path is unchanged. For holdings we
    only populate the new `panel_score` field — per-ticker `rank_score`
    (set by ScoreModelTask) stays intact for exit logic.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        scorer: PanelScorer = getattr(ctx, "_panel_scorer", None)
        X = getattr(ctx, "_panel_matrix", None)
        if scorer is None or X is None or X.empty:
            if getattr(ctx, "candidates", None):
                reason = (
                    "panel_scorer_missing"
                    if scorer is None else "panel_score_matrix_missing"
                )
                _fail_closed_panel_scoring(ctx, reason)
            # Audit P-21: previously `return False` short-circuited the
            # rest of the chain (VetoWeak, LoadNGBoost, ApplyNGBoost,
            # LoadGlobalCal, ApplyGlobalCal, ApplyKellySizing). That
            # meant Kelly target stayed stale on empty-matrix bars and
            # downstream sizing used last-bar Kelly numbers. Each of
            # those tasks already has its own None/empty guard, so we
            # return None (continue) and let them no-op individually.
            return None

        # 2026-05-19 (full-e2e shadow): when a sequence-input scorer
        # (hf_patchtst, future PatchTST kinds) is the PRIMARY panel scorer,
        # bypass the snapshot-X path entirely. The scorer builds its own
        # per-ticker sequences from a panel_history DataFrame and applies
        # its own preprocessing (CSRankNorm per day for HF PatchTST). The
        # legacy `if scorer_kind in (panel_linear, panel_ltr_xgboost)`
        # block below ALSO has a requires_history dispatch, but only for
        # the alpha158-feature-path which expects scorer_kind to be
        # panel_ltr_xgboost. For hf_patchtst (scorer_kind=hf_patchtst),
        # we never enter that block, so we'd fall through to the bare
        # snapshot scorer.score(X) which raises NotImplementedError. Caught
        # in first shadow-as-primary smoke 2026-05-19 19:43.
        scorer_kind_early = (scorer.metadata.get("kind")
                             if hasattr(scorer, "metadata") else None)
        if (scorer_kind_early not in ("panel_linear", "panel_ltr_xgboost")
                and _scorer_requires_history(scorer)):
            today = getattr(ctx, "today", None)
            target_tickers = list(X.index)
            panel_history = getattr(ctx, "_panel_history", None)
            if panel_history is None:
                from pathlib import Path as _P  # noqa: PLC0415
                repo = _P(__file__).resolve().parents[4]
                panel_path = repo / "data" / "alpha158_291_fundamental_dataset.parquet"
                try:
                    full_panel = pd.read_parquet(panel_path)
                    full_panel["date"] = pd.to_datetime(full_panel["date"])
                except Exception as exc:
                    log.error("ApplyScoresTask[%s]: failed to load panel history: %s",
                              scorer_kind_early, exc)
                    _fail_closed_panel_scoring(ctx, "panel_history_load_failed")
                    return None
                today_ts = pd.Timestamp(today)
                past = full_panel[full_panel["date"] < today_ts]
                recent_dates = sorted(past["date"].unique())[-scorer.seq_len:]
                panel_history = past[past["date"].isin(recent_dates)]
                log.info("ApplyScoresTask[%s]: lazy-loaded panel history "
                         "(%d rows × %d tickers × %d dates) for %d candidates",
                         scorer_kind_early, len(panel_history),
                         panel_history["ticker"].nunique(),
                         len(recent_dates), len(target_tickers))
            try:
                scores = scorer.score_with_history(panel_history, target_tickers)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "ApplyScoresTask[%s]: scorer.score_with_history failed: %s",
                    scorer_kind_early, exc, exc_info=True,
                )
                _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                return None
            log.info("ApplyScoresTask[%s]: scored %d via score_with_history "
                     "(seq_len=%d)", scorer_kind_early, len(scores), scorer.seq_len)
            ctx._panel_scores_all = scores  # noqa: SLF001
            n_cand_scored = 0
            scored_tickers: set[str] = set()
            for cand in ctx.candidates:
                v = scores.get(cand.ticker)
                if v is None or pd.isna(v):
                    continue
                cand.rank_score = float(v)
                cand.panel_score = float(v)
                n_cand_scored += 1
                scored_tickers.add(str(cand.ticker))
            _drop_unscored_panel_candidates(
                ctx,
                scored_tickers,
                "panel_score_missing",
            )
            n_held_scored = 0
            for ticker, hs in ctx.holdings.items():
                v = scores.get(ticker)
                if v is None or pd.isna(v):
                    continue
                hs.panel_score = float(v)
                n_held_scored += 1
            log.info(
                "ApplyScoresTask[%s]: assigned panel_score to %d/%d candidates, "
                "%d/%d holdings",
                scorer_kind_early, n_cand_scored, len(ctx.candidates),
                n_held_scored, len(ctx.holdings),
            )
            return None

        # Phase 3 (2026-05-06): alpha158 models need different features than
        # the production XGB pipeline produces. `BuildFeatureMatrixJob` builds
        # the 21-feature matrix; alpha158 models expect 158 features computed
        # from raw OHLCV. Rebuild X here for both panel_linear and
        # panel_ltr_xgboost alpha158 artifacts.
        scorer_kind = scorer.metadata.get("kind") if hasattr(scorer, "metadata") else None
        if scorer_kind in ("panel_linear", "panel_ltr_xgboost"):
            from kernel.panel_pipeline.alpha158_features import compute_alpha158_at  # noqa: PLC0415
            today = getattr(ctx, "today", None)
            ohlcv_dict = getattr(ctx, "ohlcv", None) or getattr(ctx, "ohlcv_all", None)
            if ohlcv_dict is None:
                log.warning("ApplyScoresTask[alpha158]: ctx.ohlcv unavailable")
                _fail_closed_panel_scoring(ctx, "panel_alpha158_ohlcv_missing")
                return None
            tickers = list(X.index)   # candidates + holdings already de-duped
            rows = _alpha158_cached_rows(ctx, tickers, today)
            cache_hits = len(rows)
            for t in tickers:
                if t in rows:
                    continue
                ohlcv_t = ohlcv_dict.get(t)
                if ohlcv_t is None or len(ohlcv_t) < 70:
                    continue
                feats = compute_alpha158_at(ohlcv_t, today)
                if feats:
                    rows[t] = feats
            if not rows:
                log.warning("ApplyScoresTask[alpha158]: 0/%d tickers had "
                             "sufficient history for alpha158", len(tickers))
                _fail_closed_panel_scoring(ctx, "panel_alpha158_rows_missing")
                return None
            if cache_hits:
                log.info(
                    "ApplyScoresTask[alpha158]: cache hits %d/%d tickers",
                    cache_hits, len(tickers),
                )
            X = pd.DataFrame.from_dict(rows, orient="index")
            if scorer_kind == "panel_linear":
                # PanelLinearScorer.score_raw applies stored ZScoreNorm + Fillna + Clip
                try:
                    scores: pd.Series = scorer.score_raw(X)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "ApplyScoresTask[panel_linear]: scorer.score_raw failed: %s",
                        exc, exc_info=True,
                    )
                    _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                    return None
                log.info("ApplyScoresTask[panel_linear]: scored %d tickers via "
                         "alpha158 + score_raw", len(rows))
            else:
                # XGBoost panel_ltr_xgboost: artifact may have additional fund features
                # (earnings_yield, book_to_price, etc.) beyond alpha158. If so, look them up
                # from the daily SEC fundamentals panel (point-in-time).
                fund_cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
                needs_fund = any(fc in scorer.feature_cols for fc in fund_cols)
                if needs_fund:
                    from pathlib import Path                                         # noqa: PLC0415
                    repo = Path(__file__).resolve().parents[4]
                    fp = repo / "data" / "sec_fundamentals_daily.parquet"
                    if not fp.exists():
                        _fail_closed_panel_scoring(ctx, "panel_fundamentals_missing")
                        return None
                    fund_panel = _cached_parquet(ctx, ("sec_fundamentals_daily", str(fp)), fp)
                    if fund_panel is None or fund_panel.empty:
                        _fail_closed_panel_scoring(ctx, "panel_fundamentals_empty")
                        return None
                    context_tickers = _stable_feature_context_tickers(
                        ctx, list(rows.keys()), scorer,
                    )
                    n_real, n_imputed, _medians = _apply_fund_features(
                        rows, fund_panel, today, context_tickers, fund_cols,
                    )
                    log.info(
                        "ApplyScoresTask[panel_ltr_xgboost]: merged 5 fund features "
                        "from %s over context=%d (real=%d imputed_xs_median=%d)",
                        fp.name, len(context_tickers), n_real, n_imputed,
                    )

                # PEAD features (E47 promotion 2026-05-08): if the artifact
                # has days_since_earnings / pead_signal / pead_quintile_rank,
                # compute them online from data/earnings_surprise/{tkr}.parquet.
                # Bernard-Thomas 1989 60d decay window; missing tickers get
                # cross-sectional zero (consistent with build-time fallback).
                # Shared earnings-data resources used by both PEAD and SUE blocks.
                # Hoisted so SUE block can run independently when PEAD-only
                # cols aren't in feature_cols, and vice versa.
                pead_cols = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
                sue_cols  = ["sue_signal", "surprise_momentum", "surprise_streak"]
                needs_pead = any(pc in scorer.feature_cols for pc in pead_cols)
                needs_sue  = any(sc in scorer.feature_cols for sc in sue_cols)
                if needs_pead or needs_sue:
                    from pathlib import Path  # noqa: PLC0415
                    repo = Path(__file__).resolve().parents[4]
                    earn_dir = repo / "data" / "earnings_surprise"
                    today_ts = pd.Timestamp(today)
                    context_tickers = _stable_feature_context_tickers(
                        ctx, list(rows.keys()), scorer,
                    )

                if needs_pead:
                    n_active, n_no_data, n_no_prior, n_out_of_window = (
                        _apply_pead_features(
                            ctx, rows, earn_dir, today_ts, context_tickers, pead_cols,
                        )
                    )
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: computed 3 PEAD features "
                             "today=%s (%d/%d tickers active in context 60d window; "
                             "no_data=%d no_prior=%d out_of_window=%d)",
                             today_ts.date().isoformat(),
                             n_active, len(context_tickers),
                             n_no_data, n_no_prior, n_out_of_window)

                # ── SUE features (E49 promotion 2026-05-09): SUE +
                # surprise_momentum + surprise_streak. Same earnings_surprise
                # data source as PEAD; computed independently because they
                # use multiple historical events (4Q std denominator for SUE,
                # prior-event diff for momentum, run-length for streak)
                # whereas PEAD only uses the most-recent event.
                # Foster-Olsen-Shevlin 1984 + Bernard-Thomas 60d decay.
                if needs_sue:
                    n_sue_active, n_sue_no_data, n_sue_oow = _apply_sue_features(
                        ctx, rows, earn_dir, today_ts, context_tickers, sue_cols,
                    )
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: computed 3 SUE features "
                             "today=%s (%d/%d context tickers active; no_data=%d out_of_window=%d)",
                             today_ts.date().isoformat(), n_sue_active, len(context_tickers),
                             n_sue_no_data, n_sue_oow)

                # ── Sentiment features (2026-05-18 regime-conditional ─────────
                # promotion): if the artifact's feature_cols include
                # sentiment_* columns, load per-ticker scored news from
                # data/news_sentiment_alpaca/ for today and apply the
                # regime gate per _sentiment_cfg(ctx).
                sent_cols = list(SENTIMENT_FEATURE_COLS)
                needs_sent = any(sc in scorer.feature_cols for sc in sent_cols)
                if needs_sent:
                    from pathlib import Path as _P  # noqa: PLC0415
                    repo_root = _P(__file__).resolve().parents[4]
                    sent_dir = repo_root / "data" / "news_sentiment_alpaca"
                    today_ts_sent = pd.Timestamp(today)
                    context_tickers = _stable_feature_context_tickers(
                        ctx, list(rows.keys()), scorer,
                    )
                    n_sent_hit, n_sent_miss, gate_applied = _apply_sentiment_features(
                        ctx, scorer, rows, sent_dir, today_ts_sent,
                        context_tickers, sent_cols,
                    )
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: sentiment "
                             "features (regime=%s gate=%s context=%d) hit=%d miss=%d",
                             getattr(ctx, "regime", "?"),
                             "APPLIED" if gate_applied else "TRAIN_PARITY",
                             len(context_tickers), n_sent_hit, n_sent_miss)

                # ── Feature-health check (2026-05-08 path-bug regression guard) ─
                # Catches the silent-zero failure mode that hid the parents[3]
                # path bug: if EVERY ticker reports value 0.0 for a feature
                # we just supposedly populated, the data lookup is dead.
                # Both fund and PEAD blocks use rows[t].setdefault(col, 0.0)
                # as their fallback, so an all-zero column is a strong
                # signal of a runtime data outage (path wrong, file missing,
                # API throttle).
                if rows:
                    health_warnings = []
                    expected_nonzero_cols = []
                    if needs_fund:
                        expected_nonzero_cols.extend(c for c in fund_cols if c in scorer.feature_cols)
                    if needs_pead:
                        expected_nonzero_cols.extend(c for c in pead_cols if c in scorer.feature_cols)
                    if needs_sue:
                        expected_nonzero_cols.extend(c for c in sue_cols if c in scorer.feature_cols)
                    for c in expected_nonzero_cols:
                        vals = [float(rows[t].get(c, 0.0)) for t in rows]
                        if vals and max(abs(v) for v in vals) < 1e-12:
                            health_warnings.append(c)
                    fund_dead = bool(needs_fund) and all(
                        c in health_warnings for c in fund_cols if c in scorer.feature_cols
                    )
                    pead_dead = bool(needs_pead) and all(
                        c in health_warnings for c in pead_cols if c in scorer.feature_cols
                    )
                    sue_dead = bool(needs_sue) and all(
                        c in health_warnings for c in sue_cols if c in scorer.feature_cols
                    )
                    if fund_dead:
                        log.warning(
                            "ApplyScoresTask FEATURE-HEALTH: ALL %d fund features "
                            "are 0 across %d tickers — runtime data lookup likely "
                            "broken (sec_fundamentals_daily.parquet path / read). "
                            "Production XGB will rank as if these features did not "
                            "exist. Affected: %s",
                            len([c for c in fund_cols if c in scorer.feature_cols]),
                            len(rows),
                            [c for c in health_warnings if c in fund_cols],
                        )
                    if pead_dead:
                        log.warning(
                            "ApplyScoresTask FEATURE-HEALTH: ALL %d PEAD features "
                            "are 0 across %d tickers — possible if no ticker has "
                            "earnings in the 60d window today (e.g. between cycles), "
                            "but ALSO the failure mode of the parents[3] path bug "
                            "fixed 2026-05-08. Cross-reference n_no_data above: "
                            "if n_no_data == n_total, path is broken. "
                            "Affected: %s",
                            len([c for c in pead_cols if c in scorer.feature_cols]),
                            len(rows),
                            [c for c in health_warnings if c in pead_cols],
                        )
                    if sue_dead:
                        log.warning(
                            "ApplyScoresTask FEATURE-HEALTH: ALL %d SUE features "
                            "are 0 across %d tickers — same diagnostics as PEAD: "
                            "either no ticker has earnings in the 60d window OR "
                            "earnings_surprise/ data lookup is broken. Affected: %s",
                            len([c for c in sue_cols if c in scorer.feature_cols]),
                            len(rows),
                            [c for c in health_warnings if c in sue_cols],
                        )
                # Rebuild X with fund + PEAD cols included
                X = pd.DataFrame.from_dict(rows, orient="index")
                X_aligned = X.reindex(columns=scorer.feature_cols, fill_value=float("nan"))

                # 2026-05-09 BUG #6 fix: ApplyNGBoostTask reads ctx._panel_matrix
                # downstream and uses it to feed QuantileHead.predict_distribution.
                # Pre-fix, ctx._panel_matrix held the LEGACY pre-alpha158 matrix
                # built by AssembleInferenceMatrixTask, which lacks alpha158/fund/
                # PEAD/SUE columns. QuantileHead's median imputation then filled
                # ALL of them with feature_medians_ → identical input vector for
                # every ticker → identical μ̂ across the entire candidate set.
                # Diagnostic showed n=49 mean=-0.0026 std=0.0000 (constant).
                # Fix: stamp the freshly-built RAW matrix (before normalization)
                # to ctx._panel_matrix so downstream NGB head sees per-ticker
                # alpha158 features. Normalization is XGB-rank-only and does NOT
                # propagate (X_aligned local variable below).
                ctx._panel_matrix = X_aligned.copy()  # noqa: SLF001

                # Raw inference rows must be transformed through the artifact
                # feature contract before XGB scoring.
                from kernel.panel_pipeline.feature_transform import (  # noqa: PLC0415
                    transform_feature_frame,
                )
                # Apply artifact-stored normalization. transform_feature_frame
                # reads feature_means / feature_stds from scorer.metadata.
                X_aligned = transform_feature_frame(
                    X_aligned,
                    scorer.feature_cols,
                    getattr(scorer, "metadata", {}) or {},
                    source_space="raw",
                )
                log.info(
                    "ApplyScoresTask[panel_ltr_xgboost]: applied raw→model "
                    "feature transform for %d features",
                    len(scorer.feature_cols),
                )

                # 2026-05-18 PatchTST dispatch: if scorer requires history
                # (PatchTST sequence model), call score_with_history instead
                # of legacy snapshot score().
                if _scorer_requires_history(scorer):
                    panel_history = getattr(ctx, "_panel_history", None)
                    if panel_history is None:
                        # 2026-05-18 FIRST-WIRE-IN: lazy-load from training
                        # panel parquet. TODO: replace with rolling fresh-
                        # compute via compute_alpha158_at for live inference
                        # past panel-max-date. For SIM tests on dates ≤
                        # 2026-02-10 this is correct.
                        from pathlib import Path as _P  # noqa: PLC0415
                        repo = _P(__file__).resolve().parents[4]
                        panel_path = repo / "data" / "alpha158_291_fundamental_dataset.parquet"
                        try:
                            full_panel = pd.read_parquet(panel_path)
                            full_panel["date"] = pd.to_datetime(full_panel["date"])
                        except Exception as exc:
                            log.error("PatchTST: failed to load panel parquet: %s", exc)
                            _fail_closed_panel_scoring(ctx, "panel_history_load_failed")
                            return None
                        else:
                            target_tickers = list(rows.keys())
                            today_ts = pd.Timestamp(today)
                            past = full_panel[full_panel["date"] < today_ts]
                            # Use last seq_len dates × candidate tickers
                            recent_dates = sorted(past["date"].unique())[-scorer.seq_len:]
                            history = past[past["date"].isin(recent_dates)]
                            log.info("PatchTST: lazy-loaded panel history "
                                     "(%d rows × %d tickers × %d dates) for %d candidates",
                                     len(history), history["ticker"].nunique(),
                                     len(recent_dates), len(target_tickers))
                            try:
                                scores = scorer.score_with_history(history, target_tickers)
                            except Exception as exc:  # noqa: BLE001
                                log.error(
                                    "ApplyScoresTask[patchtst]: "
                                    "scorer.score_with_history failed: %s",
                                    exc, exc_info=True,
                                )
                                _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                                return None
                    else:
                        target_tickers = list(rows.keys())
                        try:
                            scores = scorer.score_with_history(panel_history,
                                                                target_tickers)
                        except Exception as exc:  # noqa: BLE001
                            log.error(
                                "ApplyScoresTask[patchtst]: "
                                "scorer.score_with_history failed: %s",
                                exc, exc_info=True,
                            )
                            _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                            return None
                    log.info("ApplyScoresTask[patchtst]: scored %d via "
                             "PatchTST (seq_len=%d)",
                             len(scores), scorer.seq_len)
                else:
                    try:
                        scores: pd.Series = scorer.score(X_aligned)
                    except Exception as exc:  # noqa: BLE001
                        log.error(
                            "ApplyScoresTask[panel_ltr_xgboost]: scorer.score failed: %s",
                            exc, exc_info=True,
                        )
                        _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                        return None
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: scored %d tickers via alpha158%s",
                             len(rows), "+fund" if needs_fund else "")
        else:
            try:
                scores: pd.Series = scorer.score(X)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "ApplyScoresTask: scorer.score failed: %s",
                    exc, exc_info=True,
                )
                _fail_closed_panel_scoring(ctx, "panel_score_runtime_error")
                return None

        # 2026-05-14 Phase 2B: stash the full-universe score series for the
        # short-candidate selection task. Only kept; not consumed unless
        # long_short.enabled=true. ApplyScoresTask's only mutation here.
        ctx._panel_scores_all = scores  # noqa: SLF001

        n_cand_scored = 0
        scored_tickers: set[str] = set()
        for cand in ctx.candidates:
            v = scores.get(cand.ticker)
            if v is None or pd.isna(v):
                continue
            cand.rank_score  = float(v)
            cand.panel_score = float(v)
            n_cand_scored += 1
            scored_tickers.add(str(cand.ticker))

        # 2026-05-05 wl183 0-trade diagnostic. Only fires on the failure
        # path where every candidate lookup missed. Surfaces the dtype +
        # sample mismatch that would otherwise need a code edit + re-sim
        # to debug. Cheap (one log line on failure, none on the happy path).
        if ctx.candidates and n_cand_scored == 0:
            cand_sample = [c.ticker for c in ctx.candidates[:5]]
            log.error(
                "ApplyScoresTask 0/N LOOKUP MISS: scores.shape=%s "
                "scores.dtype=%s n_finite=%d scores.index[:5]=%s "
                "cand_ticker[:5]=%s first_lookup=%r X.shape=%s "
                "X.index.dtype=%s",
                scores.shape, scores.dtype, scores.notna().sum(),
                list(scores.index[:5]), cand_sample,
                scores.get(cand_sample[0]) if cand_sample else None,
                X.shape, X.index.dtype,
            )
        _drop_unscored_panel_candidates(
            ctx,
            scored_tickers,
            "panel_score_missing",
        )

        n_held_scored = 0
        for ticker, hs in ctx.holdings.items():
            v = scores.get(ticker)
            if v is None or pd.isna(v):
                continue
            hs.panel_score = float(v)
            n_held_scored += 1

        log.info("ApplyScoresTask: panel scored %d/%d candidates, %d/%d holdings",
                 n_cand_scored, len(ctx.candidates),
                 n_held_scored, len(ctx.holdings))


class VetoWeakBuysTask(Task):
    """Drop candidates whose CALIBRATED rank_score is below `buy_floor`.

    Invariant (P0 fix 2026-05-03): the buy_floor compares against the SAME
    scale that downstream tier thresholds (rotation, QualityFloor) use —
    calibrated rank_score in [0, 1]. Pre-fix this task read raw
    ``cand.panel_score`` (XGBoost rank:pairwise margin, range ~ [0, 0.05])
    while running BEFORE ``ApplyGlobalCalibrationTask``, so the 0.30 floor
    set on 2026-04-29 (commit 410758b "buy_floor null→0.30") could never
    be crossed by any candidate. Production cron silently dropped 55/55
    candidates daily for 5 days — no fresh entries opened, only TopUps on
    existing holdings. Audit log:

        2026-04-30 16:05  Phase 2b: 55 candidates from 78 tickers
        2026-04-30 16:05  VetoWeakBuysTask: dropped 55 below panel_score=0.300

    Fix: this task is reordered to run AFTER ``ApplyGlobalCalibrationTask``
    so ``cand.rank_score`` is the calibrated probability, not raw margin.
    Configs that set ``buy_floor: 0.30`` now express "drop bottom 30% by
    calibrator" as intended.

    No-op when buy_floor is unset. Candidates without a rank_score (e.g.
    missing features) are kept — RankingJob blends rs_score in.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        # Audit fix VETO-EMPTY-CANDS (Round 2 deep audit, 2026-04-25):
        # pre-fix returned False when ctx.candidates was empty, which
        # short-circuits the rest of PanelScoringJob's chain. Empty
        # candidates is now a continue (None), not a stop.
        if not ctx.candidates:
            return None

        # 2026-05-04 user mandate ("rank_score need to be collected
        # properly for future fine tune"). Snapshot the full pre-veto
        # candidate list (references, not deep copies) onto ctx so the
        # adapter's record_candidate_scores can persist BOTH kept and
        # vetoed rows — the offline analysis needs the FULL rank_score
        # distribution per bar, not just the survivors. The cands'
        # rank_score / mu / sigma are already populated by
        # ApplyGlobalCalibration + ApplyNGBoost at this point in the
        # chain. Vetoed cands are tagged via ctx._blocked_by_ticker
        # ("veto:rank_score_below_floor" / "veto:rank_score_nan").
        # ALWAYS captured, regardless of whether the veto fires —
        # offline analysis needs the data either way.
        ctx._full_candidate_snapshot = list(ctx.candidates)    # noqa: SLF001

        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        raw_floor = panel_cfg.get("buy_floor")
        if raw_floor is None:
            return

        # 2026-05-30 — escape hatch for distribution-fair model comparison.
        # When RQ_SIM_BYPASS_BUY_FLOOR=1, skip the floor entirely. Used by
        # WF gate sims to evaluate models whose calibrated score distribution
        # is narrower than the adaptive_mean_std rule expects (per-cut
        # PatchTST calibrators output prob ranges as tight as 0.07, vs daily
        # shadow's 0.49 — same model, same scoring, but buy_floor rejects
        # all WF cut candidates and admits daily-shadow candidates).
        # See memory: project_wf_sim_unfair_to_compressed_models_2026-05-30.
        # Prod live / cron NEVER set this — they keep the floor strict.
        import os  # noqa: PLC0415
        if os.environ.get("RQ_SIM_BYPASS_BUY_FLOOR") == "1":
            log.info(
                "VetoWeakBuysTask: RQ_SIM_BYPASS_BUY_FLOOR=1 — skipping floor "
                "(distribution-fair sim mode); raw_floor=%r ignored",
                raw_floor,
            )
            return

        # 2026-05-04 user spec (final form):
        #   floor = min(max(buy_floor_min, mean+std), buy_floor_adaptive_cap)
        # i.e. clamp `mean+std` to the interval [min, cap].
        #   defaults: min=0.20, cap=0.30
        #
        # Three rules in one formula:
        #   - if mean+std < min:    use min        (don't go below absolute floor)
        #   - if mean+std in range: use mean+std   (per-bar adaptive)
        #   - if mean+std > cap:    use cap        (don't go above legacy ceiling)
        #
        # The min bound is a fail-safe: even when the distribution is
        # extremely degenerate (e.g. all cands clustered far below
        # base_rate), we still require rank_score ≥ 0.20 for entry.
        # Prevents accidentally accepting tiny rank_scores when the
        # mean+std happens to land low.
        floor: float
        floor_label: str
        if isinstance(raw_floor, str) and raw_floor in {"adaptive_mean_std_cap", "adaptive_mean_std"}:
            cap     = float(panel_cfg.get("buy_floor_adaptive_cap", 0.30))
            min_fl  = float(panel_cfg.get("buy_floor_min",          0.20))
            std_mult = float(panel_cfg.get("buy_floor_std_mult",     1.0))
            raw_scores = [getattr(c, "rank_score", None) for c in ctx.candidates]
            scores = []
            for s in raw_scores:
                try:
                    f = float(s)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    scores.append(f)
            if len(scores) >= 2:
                import statistics as _stats  # noqa: PLC0415
                mean_s = _stats.fmean(scores)
                std_s  = _stats.stdev(scores)
                adaptive = mean_s + std_mult * std_s
                if raw_floor == "adaptive_mean_std":
                    # New production mode (2026-05-21): keep the
                    # cross-sectional mean+σ threshold on the calibrated
                    # probability scale, but do not cap it at 0.30. The old
                    # cap became a no-op once scores clustered around
                    # 0.55-0.65; floor=0.30 admitted everything and let the
                    # QP sort weak signals by tiny μ differences.
                    floor = max(min_fl, adaptive)
                    floor_label = (
                        f"max(min={min_fl:.2f}, mean+{std_mult:.2f}*std="
                        f"{adaptive:.3f}) = {floor:.3f}  (n={len(scores)})"
                    )
                else:
                    # Back-compat experiment mode: clamp mean+std to [min, cap].
                    floor = min(max(min_fl, adaptive), cap)
                    floor_label = (
                        f"min(max(min={min_fl:.2f}, mean+std={adaptive:.3f}), "
                        f"cap={cap:.2f}) = {floor:.3f}  (n={len(scores)})"
                    )
            else:
                # Insufficient cross-section — use the absolute minimum for
                # uncapped mode, legacy cap for capped mode.
                floor = min_fl if raw_floor == "adaptive_mean_std" else cap
                floor_label = f"{floor:.3f} (fallback; n<2 for stats)"
        else:
            floor = float(raw_floor)
            floor_label = f"{floor:.3f} (absolute)"

        kept: list = []
        dropped = 0
        blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
        for cand in ctx.candidates:
            # 2026-05-03 fix: read CALIBRATED rank_score (post-calibration).
            # Pre-fix this read cand.panel_score (raw XGB margin) — see
            # docstring for the production incident this caused.
            score = getattr(cand, "rank_score", None)
            # Audit P-22: differentiate three states:
            #   score is None      → no score available; KEEP — rs_score still
            #                        ranks it (matches original behavior).
            #   score is NaN       → scoring ran but produced NaN → DROP.
            #                        Pre-fix this slipped through because
            #                        NaN < float is False.
            #   score < floor      → DROP (the documented veto).
            if score is None:
                kept.append(cand)
                continue
            if pd.isna(score):
                dropped += 1
                blocked[cand.ticker] = "veto:rank_score_nan"
                continue
            if score < floor:
                dropped += 1
                blocked[cand.ticker] = "veto:rank_score_below_floor"
                continue
            kept.append(cand)
        ctx._blocked_by_ticker = blocked                       # noqa: SLF001

        # Audit #43: keep counter present even when nothing dropped.
        ctx.counters["panel_vetoed"] = ctx.counters.get("panel_vetoed", 0) + dropped
        if dropped:
            ctx.candidates = kept
            log.info("VetoWeakBuysTask: dropped %d candidate(s) below "
                     "rank_score floor=%s", dropped, floor_label)


def _regime_stats_map(raw: object) -> dict[str, dict]:
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        return {
            str(v.get("regime")): v
            for v in raw
            if isinstance(v, dict) and v.get("regime")
        }
    return {}


def _trade_monotonicity_admission(metadata: dict, regime: str) -> tuple[bool, str, dict]:
    wf = metadata.get("wf_gate_metadata") if isinstance(metadata, dict) else {}
    tm = wf.get("trade_monotonicity") if isinstance(wf, dict) else {}
    if not isinstance(tm, dict) or not tm:
        return False, "regime_admission:no_trade_monotonicity", {}
    stats = _regime_stats_map(tm.get("regimes")).get(str(regime))
    if not stats:
        return False, f"regime_admission:no_trade_stats:{regime}", {"trade_monotonicity": tm}
    if not bool(stats.get("eligible", False)):
        return False, f"regime_admission:ineligible:{regime}", {"stats": stats}
    if not bool(stats.get("passed", False)):
        return False, f"regime_admission:failed:{regime}", {"stats": stats}
    return True, "ok", {"stats": stats}


def _sanity_regime_admission(
    metadata: dict,
    regime: str,
    *,
    min_ic: float,
    max_placebo_ratio: float,
) -> tuple[bool, str, dict]:
    wf = metadata.get("wf_gate_metadata") if isinstance(metadata, dict) else {}
    sanity = wf.get("sanity_regime_ic") if isinstance(wf, dict) else {}
    if not isinstance(sanity, dict) or not sanity:
        return False, "regime_admission:no_sanity_regime_ic", {}
    stats = _regime_stats_map(sanity.get("regimes")).get(str(regime))
    if not stats:
        return False, f"regime_admission:no_sanity_stats:{regime}", {"sanity": sanity}
    if stats.get("eligible") is False:
        return False, f"regime_admission:ineligible_sanity:{regime}", {"stats": stats}
    mean_ic = stats.get("mean_ic")
    try:
        mean_ic_f = float(mean_ic)
    except (TypeError, ValueError):
        return False, f"regime_admission:bad_sanity_ic:{regime}", {"stats": stats}
    if not math.isfinite(mean_ic_f) or mean_ic_f < float(min_ic):
        return False, f"regime_admission:weak_sanity_ic:{regime}", {"stats": stats}
    placebo_60_ic = stats.get("placebo_60_ic")
    if placebo_60_ic is not None:
        try:
            placebo_60_ic_f = float(placebo_60_ic)
        except (TypeError, ValueError):
            return False, f"regime_admission:bad_placebo_sanity:{regime}", {"stats": stats}
        placebo_ref = mean_ic_f
        aligned_real_ic = stats.get("placebo_60_aligned_real_ic")
        if aligned_real_ic is not None:
            try:
                aligned_real_ic_f = float(aligned_real_ic)
                if math.isfinite(aligned_real_ic_f):
                    placebo_ref = aligned_real_ic_f
            except (TypeError, ValueError):
                return False, f"regime_admission:bad_aligned_placebo_sanity:{regime}", {"stats": stats}
        if math.isfinite(placebo_60_ic_f) and abs(placebo_60_ic_f) > max(
            0.005,
            float(max_placebo_ratio) * abs(placebo_ref),
        ):
            return False, f"regime_admission:placebo_sanity:{regime}", {"stats": stats}
    if stats.get("passed") is False:
        return False, f"regime_admission:failed_sanity:{regime}", {"stats": stats}
    return True, "ok", {"stats": stats}


class RegimeModelAdmissionTask(Task):
    """Block buy candidates when the current regime lacks model evidence.

    This is the model/QP separation guard: model evidence decides whether
    names are eligible to buy in the current regime; QP may only size the
    surviving candidates.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        candidates = list(getattr(ctx, "candidates", []) or [])
        holdings = getattr(ctx, "holdings", {}) or {}
        if not candidates and not holdings:
            return None
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        cfg = panel_cfg.get("regime_admission", {}) or {}
        if cfg.get("enabled", True) is False:
            return None
        scorer = getattr(ctx, "_panel_scorer", None)
        metadata = getattr(scorer, "metadata", {}) or {}
        regime = str(getattr(ctx, "regime", "") or "UNKNOWN")

        ok, reason, details = _trade_monotonicity_admission(metadata, regime)
        if ok and bool(cfg.get("require_sanity_regime_ic", True)):
            ok, reason, details = _sanity_regime_admission(
                metadata,
                regime,
                min_ic=float(cfg.get("min_sanity_regime_ic", 0.02)),
                max_placebo_ratio=float(cfg.get("max_placebo_ratio", 0.5)),
            )
        ctx._regime_model_admission = {  # noqa: SLF001
            "ok": bool(ok), "reason": reason, "regime": regime, **details,
        }
        if ok:
            return None

        ctx._full_candidate_snapshot = list(getattr(ctx, "_full_candidate_snapshot", None)
                                            or candidates)  # noqa: SLF001
        blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
        for cand in candidates:
            blocked[cand.ticker] = reason
        if holdings:
            exit_only = set(getattr(ctx, "_qp_exit_only_tickers", set()) or set())
            exit_only_reasons = dict(
                getattr(ctx, "_qp_exit_only_reasons", {}) or {}
            )
            for ticker in holdings:
                exit_only.add(ticker)
                exit_only_reasons.setdefault(ticker, reason)
                blocked.setdefault(ticker, reason)
            ctx._qp_exit_only_tickers = exit_only  # noqa: SLF001
            ctx._qp_exit_only_reasons = exit_only_reasons  # noqa: SLF001
        ctx._blocked_by_ticker = blocked  # noqa: SLF001
        n = len(candidates)
        ctx.candidates = []
        ctx.counters["regime_admission_blocked"] = (
            ctx.counters.get("regime_admission_blocked", 0) + n
        )
        log.warning("RegimeModelAdmissionTask: blocked %d candidates: %s", n, reason)


# ── Global calibration (Item #2 — optional) ───────────────────────────────────

def _fingerprint_values(metadata: dict | None) -> list[str]:
    """Return scorer identities, never shared strategy config fingerprints.

    New artifacts bind calibrators by ``model_content_fingerprint`` because
    acceptance metadata is mutable. Legacy artifacts used full-file hashes, so
    keep those as fallback identities until the old folds are re-stamped.
    """
    if not metadata:
        return []
    out: list[str] = []
    for key in (
        "model_content_fingerprint",
        "scorer_model_content_fingerprint",
        "artifact_fingerprint",
        "scorer_artifact_fingerprint",
        "model_fingerprint",
        "artifact_sha256",
        "scorer_artifact_sha256",
        "fingerprint",
    ):
        value = metadata.get(key)
        if value:
            out.append(str(value))
    return out


def _normalize_fingerprint(value: str | None) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _fingerprints_match(expected: str | None, actual: str | None) -> bool:
    """Accept exact matches and historical short-sha prefixes."""
    exp = _normalize_fingerprint(expected)
    act = _normalize_fingerprint(actual)
    if not exp or not act:
        return False
    if exp == act:
        return True
    min_prefix = 12
    return (
        len(exp) >= min_prefix
        and len(act) >= min_prefix
        and (exp.startswith(act) or act.startswith(exp))
    )


def _any_fingerprints_match(expected: list[str], actual: list[str]) -> bool:
    return any(
        _fingerprints_match(exp, act)
        for exp in expected
        for act in actual
    )


def _active_scorer_metadata(ctx: InferenceContext) -> dict:
    scorer = getattr(ctx, "_panel_scorer", None)
    return dict(getattr(scorer, "metadata", {}) or {})


def _assert_calibrator_matches_scorer(
    ctx: InferenceContext,
    calibrator: Any,
    artifact_path: Path,
    *,
    strict: bool,
) -> None:
    """Fail fast when a calibrator was fit to a different panel scorer.

    Invariant: calibrated rank_score / expected_return may only be produced by
    the scorer distribution the calibrator was fitted on. Otherwise Kelly/QP
    sees a shifted μ surface and a sim can report plausible but invalid APY.
    """
    if not strict:
        return
    scorer_meta = _active_scorer_metadata(ctx)
    if not scorer_meta:
        log.info(
            "LoadGlobalCalibrationTask: no active scorer metadata present; "
            "skipping scorer/calibrator contract for %s",
            artifact_path,
        )
        return

    active_fps = _fingerprint_values(scorer_meta)
    cal_fps = _fingerprint_values(getattr(calibrator, "metadata", {}) or {})
    if not active_fps or not cal_fps:
        raise ValueError(
            "LoadGlobalCalibrationTask contract fail: missing scorer/calibrator "
            f"fingerprint for {artifact_path}. active={active_fps!r} "
            f"calibrator={cal_fps!r}. Refit the calibrator with "
            "scorer_model_content_fingerprint stamped."
        )
    if not _any_fingerprints_match(cal_fps, active_fps):
        raise ValueError(
            "LoadGlobalCalibrationTask contract fail: calibrator/scorer "
            f"fingerprint mismatch for {artifact_path}. calibrator={cal_fps} "
            f"active_scorer={active_fps}. Refusing to map panel_score to "
            "rank_score/mu with a foreign calibration surface."
        )


def _fail_closed_missing_calibrator(ctx: InferenceContext, reason: str) -> None:
    """Block buy/QP when an enabled calibrator cannot be used.

    Preflight should catch this before a daily/full run, but runtime must still
    fail closed so a missing calibrator never silently reverts to raw panel
    scores. Exits already emitted earlier in the pipeline are left intact.
    """
    ctx._calibrator_contract_failed = True  # noqa: SLF001
    ctx.buy_blocked = True
    ctx.skip_buys = True
    blocked_map = getattr(ctx, "_blocked_by_ticker", None)
    if blocked_map is None:
        blocked_map = {}
        ctx._blocked_by_ticker = blocked_map  # noqa: SLF001
    pool = list(getattr(ctx, "_full_candidate_snapshot", None) or ctx.candidates or [])
    if pool and not getattr(ctx, "_full_candidate_snapshot", None):
        ctx._full_candidate_snapshot = list(pool)  # noqa: SLF001
    for c in pool:
        ticker = getattr(c, "ticker", None)
        if ticker:
            blocked_map.setdefault(ticker, reason)
    ctx.candidates = []
    log.error(
        "Global calibration contract failed (%s). Buy candidates cleared; "
        "buy/QP path is fail-closed for this run.",
        reason,
    )


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _calibrator_native_horizon_days(cal: Any, ctx: InferenceContext) -> int | None:
    meta = getattr(cal, "metadata", {}) or {}
    for key in ("lookahead_days_used", "lookahead_days", "er_lookahead"):
        days = _positive_int(meta.get(key))
        if days is not None:
            return days
    return _positive_int((ctx.config.get("panel_ltr", {}) or {}).get("lookahead_days"))


def _rotation_er_horizon_days(ctx: InferenceContext, cal: Any) -> int | None:
    return (
        _positive_int((ctx.config.get("rotation", {}) or {}).get("target_horizon_days"))
        or _calibrator_native_horizon_days(cal, ctx)
    )


def _qp_mu_horizon_days(ctx: InferenceContext, cal: Any) -> int | None:
    joint_cfg = (
        ((ctx.config.get("rotation", {}) or {}).get("joint_actions", {}) or {})
    )
    return (
        _positive_int(joint_cfg.get("qp_mu_horizon_days"))
        or _positive_int((ctx.config.get("panel_ltr", {}) or {}).get("lookahead_days"))
        or _calibrator_native_horizon_days(cal, ctx)
    )


def _calibrator_expected_return_at_horizon(
    cal: Any,
    raw_score: float,
    horizon_days: int | None,
    native_horizon_days: int | None,
) -> float:
    try:
        return float(cal.expected_return(raw_score, horizon_days=horizon_days))
    except TypeError:
        base = float(cal.expected_return(raw_score))
    if (
        horizon_days is None
        or native_horizon_days is None
        or native_horizon_days <= 0
        or int(horizon_days) == int(native_horizon_days)
    ):
        return base
    return base * (float(horizon_days) / float(native_horizon_days))


class LoadGlobalCalibrationTask(Task):
    """Load the global panel calibrator artifact(s) if enabled.

    Default: loads the pooled calibrator at
    `artifact_path` into `ctx._global_calibrator`.

    When `regime_conditional.enabled=true` also loads per-regime
    calibrators from `regime_conditional.artifact_pattern` (with
    `{regime}` placeholder) into `ctx._regime_calibrators: dict[str,
    GlobalPanelCalibration]`. Any regime whose file is missing or
    fails to load falls back to the pooled calibrator at apply time.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        gc_cfg = (ctx.config.get("ranking", {})
                           .get("panel_scoring", {})
                           .get("global_calibration", {}))
        if not gc_cfg.get("enabled", False):
            return
        strict_match = bool(gc_cfg.get("strict_scorer_match", True))

        strategy_dir = ctx.config.get("_strategy_dir")

        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() or not strategy_dir else Path(strategy_dir) / p

        from training_panel.global_calibrator import GlobalPanelCalibration  # noqa: PLC0415

        # Pooled calibrator — always attempted (acts as fallback).
        # §5.13.14: require explicit artifact_path. Pre-fix this defaulted
        # to artifacts/prod/panel-rank-calibration.json, so a sim that
        # forgot to override would silently load the prod calibrator and
        # report misleading sim results (no corruption, just confusion).
        preloaded = getattr(ctx, "_global_calibrator", None)
        if preloaded is not None:
            _assert_calibrator_matches_scorer(
                ctx,
                preloaded,
                Path("<preloaded_global_calibrator>"),
                strict=strict_match,
            )
        if getattr(ctx, "_global_calibrator", None) is None:
            pooled_rel = gc_cfg.get("artifact_path")
            if not pooled_rel:
                log.error(
                    "LoadGlobalCalibrationTask: global_calibration.enabled=true "
                    "but artifact_path is not set in cfg.ranking.panel_scoring."
                    "global_calibration. Refusing to default to any prod path — "
                    "buy path will fail closed."
                )
                ctx._global_calibrator = None  # noqa: SLF001
                ctx._global_calibrator_missing_reason = "calibrator_missing_path"  # noqa: SLF001
            else:
                pooled_path = _resolve(Path(pooled_rel))
                try:
                    loaded = GlobalPanelCalibration.load(pooled_path)
                    _assert_calibrator_matches_scorer(
                        ctx, loaded, pooled_path, strict=strict_match,
                    )
                    ctx._global_calibrator = loaded  # noqa: SLF001
                    log.info("LoadGlobalCalibrationTask: loaded pooled (pool_IC=%s)",
                             ctx._global_calibrator.metadata.get("pool_ic"))
                except ValueError:
                    raise
                except Exception as exc:
                    log.warning("LoadGlobalCalibrationTask: pooled load %s failed — %s",
                                pooled_path, exc)
                    ctx._global_calibrator = None  # noqa: SLF001
                    ctx._global_calibrator_missing_reason = "calibrator_load_failed"  # noqa: SLF001

        # Regime-conditional (Plan F) — opt-in.
        rc_cfg = gc_cfg.get("regime_conditional", {})
        if not rc_cfg.get("enabled", False):
            return
        if getattr(ctx, "_regime_calibrators", None):
            return

        pattern = rc_cfg.get(
            "artifact_pattern", "artifacts/panel-calibration-{regime}.json",
        )
        regimes = rc_cfg.get(
            "regimes", ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"],
        )
        loaded: dict[str, GlobalPanelCalibration] = {}
        for regime in regimes:
            p = _resolve(Path(pattern.format(regime=regime)))
            try:
                cal = GlobalPanelCalibration.load(p)
                _assert_calibrator_matches_scorer(
                    ctx, cal, p, strict=strict_match,
                )
                loaded[regime] = cal
            except ValueError:
                raise
            except Exception as exc:
                log.info("LoadGlobalCalibrationTask: regime=%s artifact %s "
                         "unavailable — pooled fallback (%s)",
                         regime, p, exc)
        ctx._regime_calibrators = loaded  # noqa: SLF001
        log.info("LoadGlobalCalibrationTask: %d/%d regime calibrators loaded",
                 len(loaded), len(regimes))


class ApplyGlobalCalibrationTask(Task):
    """Transform panel_score → calibrated P(outperform) + E[R - SPY].

    Per 2026-04-23 task #2 refactor: now always runs, regardless of NGBoost
    mode. Runs AFTER ApplyNGBoostTask in the PanelScoringJob chain, so:

      - score_mode="additive": NGBoost leaves panel_score untouched →
        calibrator maps raw panel_score → probability (same behavior as
        pre-refactor additive mode).
      - score_mode="mu_minus_lambda_sigma": NGBoost overwrites panel_score
        with μ−λσ first → calibrator then maps μ−λσ → probability. The
        isotonic calibrator was fit on raw panel_score, but μ−λσ is the
        same scale, so the map is directionally correct (not strictly
        metric-calibrated; acceptable for ranking).

    Previously this task short-circuited when score_mode was
    "mu_minus_lambda_sigma", which left rank_score as raw μ−λσ ∈
    [~-0.06, +0.04] — always below the 0.10 tier threshold → zero trades
    in that mode. Reordering + removing the short-circuit unlocks
    σ-aware ranking as a live-testable option.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("global_calibration", {}).get("enabled", False):
            return
        # Note (audit P-37 reconsidered 2026-04-24): the calibrator was
        # fit on Gaussianized LTR panel_score (range ~ ±3) but in
        # `score_mode=mu_minus_lambda_sigma` mode panel_score has been
        # overwritten with `μ−λσ` (range ~ ±0.05). Mapping μ−λσ through
        # the isotonic compresses output near the central probability,
        # which is *not* metric-calibrated — but the isotonic is still
        # MONOTONIC, so the cross-sectional ranking order is preserved.
        # Without calibration, raw μ−λσ would be entirely below the
        # 0.10 tier threshold → zero trades. So calibrator wins on
        # ranking even when it loses on metric meaning. Documented here
        # so future readers don't try to "fix" this again. R2 audit
        # task #2 reordered the chain to make this work; that decision
        # is reaffirmed.

        # Plan F: prefer per-regime calibrator when one is loaded for the
        # current regime; pooled calibrator is the universal fallback.
        regime_map = getattr(ctx, "_regime_calibrators", None) or {}
        pooled     = getattr(ctx, "_global_calibrator", None)
        cal = regime_map.get(getattr(ctx, "regime", None)) or pooled
        if cal is None:
            reason = getattr(
                ctx, "_global_calibrator_missing_reason",
                "calibrator_missing",
            )
            _fail_closed_missing_calibrator(ctx, str(reason))
            return False

        # 2026-05-15 Phase 3: opt-in c.mu wiring. When
        # ranking.kelly_sizing.use_calibrator_mu=true, the calibrator's
        # expected_return head is wired into c.mu so Kelly sizing has a
        # real μ value when NGBoost is OFF. Disabled by default so prod
        # behavior is unchanged; flip to A/B test against current
        # uniform-fallback QP path. See doc/AUDIT_2026-05-12_dead_paths.md
        # and tests/test_calibrator_saturation_guards.py.
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        use_cal_mu = bool(kelly_cfg.get("use_calibrator_mu", False))
        native_horizon = _calibrator_native_horizon_days(cal, ctx)
        rotation_horizon = _rotation_er_horizon_days(ctx, cal)
        qp_mu_horizon = _qp_mu_horizon_days(ctx, cal)
        if use_cal_mu:
            meta = getattr(cal, "metadata", {}) or {}
            er_contract = meta.get("expected_return_label_contract")
            if er_contract != "raw_return_units_required":
                log.error(
                    "ApplyGlobalCalibrationTask: use_calibrator_mu=true but "
                    "calibrator expected_return_label_contract=%r. QP/Kelly "
                    "requires raw return units for μ; refusing buy path.",
                    er_contract,
                )
                _fail_closed_missing_calibrator(
                    ctx,
                    "calibrator_er_contract_invalid",
                )
                return False

        n_cand = 0
        for c in ctx.candidates:
            if c.panel_score is None or c.panel_score != c.panel_score:
                continue
            prob = cal.calibrate_probability(c.panel_score)
            er = _calibrator_expected_return_at_horizon(
                cal,
                c.panel_score,
                rotation_horizon,
                native_horizon,
            )
            c.rank_score      = float(prob)
            c.expected_return = float(er)
            c.expected_return_horizon_days = rotation_horizon
            if use_cal_mu and math.isfinite(er):
                mu = _calibrator_expected_return_at_horizon(
                    cal,
                    c.panel_score,
                    qp_mu_horizon,
                    native_horizon,
                )
                # c.expected_return is clipped to [-0.20, +0.20] at load time
                # (GlobalPanelCalibration.load). Kelly numerator is therefore
                # bounded; Kelly denominator (σ²) still needs σ via NGBoost
                # OR the realized-vol fallback (see ApplyRealizedVolFallbackTask).
                c.mu = float(mu)
                c.mu_horizon_days = qp_mu_horizon
            n_cand += 1

        n_held = 0
        for ticker, hs in ctx.holdings.items():
            ps = getattr(hs, "panel_score", None)
            if ps is None or ps != ps:
                continue
            hs.rank_score      = cal.calibrate_probability(ps)
            hs.expected_return = _calibrator_expected_return_at_horizon(
                cal,
                ps,
                rotation_horizon,
                native_horizon,
            )
            hs.expected_return_horizon_days = rotation_horizon
            if use_cal_mu and math.isfinite(hs.expected_return):
                hs.mu = float(_calibrator_expected_return_at_horizon(
                    cal,
                    ps,
                    qp_mu_horizon,
                    native_horizon,
                ))
                hs.mu_horizon_days = qp_mu_horizon
            n_held += 1

        log.info(
            "ApplyGlobalCalibrationTask: calibrated %d/%d candidates, %d/%d "
            "holdings (er_horizon=%s, mu_horizon=%s, native_horizon=%s)",
            n_cand, len(ctx.candidates), n_held, len(ctx.holdings),
            rotation_horizon, qp_mu_horizon if use_cal_mu else None,
            native_horizon,
        )
        # 2026-05-09 BUG #6 GUARD CLASS: post-calibrate diversity check.
        # If the calibrator collapses to constant output across candidates,
        # the panel becomes un-rankable. Symptom of (a) all panel_score
        # values identical (upstream collapse) or (b) calibrator artifact
        # truncated to a single bucket. Pre-fix: candidates would all get
        # identical rank_score → top-K selects deterministically by ticker
        # alphabetic order, no signal-driven trading.
        if n_cand >= 2:
            from training_panel.model_contract import soft_check_score_series  # noqa: PLC0415
            ranks = pd.Series(
                [c.rank_score for c in ctx.candidates if c.rank_score is not None],
                dtype=float,
            )
            if len(ranks) >= 2:
                soft_check_score_series(
                    ranks, model_name="ApplyGlobalCalibrationTask",
                    expected_min=0.0, expected_max=1.0,
                )
                # 2026-05-15 BUG #7 GUARD: upper-tail saturation detection.
                # User-observed silent failure since 2026-05-12: calibrator
                # mapped >50% of candidates to rank_score >= 0.95 because the
                # isotonic curve has no clip at +1.0 and the training-x
                # range was narrower than live-x range. soft_check_score_series
                # only catches CONSTANT output (std<1e-8); a saturated
                # upper-tail has high std but is still un-rankable.
                #
                # 2026-05-21 correction: low probability IQR alone is not a
                # trade-stop condition for a smooth Platt calibrator. A
                # sigmoid may compress probabilities while still preserving a
                # fully usable monotone ordering. Abstain only when the
                # cross-section is actually un-rankable: too few unique scores,
                # a dominant exact-tie bucket, or saturated upper tail.
                iqr = float(ranks.quantile(0.75) - ranks.quantile(0.25))
                sat_top = float((ranks >= 0.95).mean())
                rounded = ranks.round(6)
                n_unique = int(rounded.nunique())
                dominant_tie_frac = (
                    float(rounded.value_counts(normalize=True).iloc[0])
                    if len(rounded) else 0.0
                )
                sat_cfg = (
                    (ctx.config or {}).get("ranking", {})
                                    .get("panel_scoring", {})
                                    .get("calibrator_saturation", {})
                )
                iqr_warn_floor = float(sat_cfg.get("iqr_warn_floor", 0.05))
                min_unique = int(sat_cfg.get("min_unique_scores", 5))
                max_tie_frac = float(sat_cfg.get("max_tie_fraction", 0.50))
                low_iqr = iqr < iqr_warn_floor
                score_collapse = n_unique < min_unique or dominant_tie_frac >= max_tie_frac
                upper_tail_saturation = sat_top >= 0.50
                if low_iqr or score_collapse or upper_tail_saturation:
                    log.warning(
                        "CALIBRATOR-SATURATED: rank_score IQR=%.3f "
                        "(warn_floor=%.3f), fraction>=0.95=%.0f%%, "
                        "n_unique=%d, dominant_tie=%.0f%%. Abstain requires "
                        "upper-tail saturation or true score collapse; low "
                        "IQR alone is diagnostic for Platt-style compression.",
                        iqr, iqr_warn_floor, sat_top * 100,
                        n_unique, dominant_tie_frac * 100,
                    )
                    # 2026-05-18 NEW-BUY GATE: when calibrator is degenerate,
                    # the model has effectively NO conviction for today.
                    # Tie-broken buys = strategy noise (MCD rebuy incident).
                    # Mark ctx so downstream QP can refuse new positions.
                    # Existing holdings can still be exited (sell logic doesn't
                    # require calibrator conviction); only NEW buys gated.
                    # Default ON unless config disables.
                    abstain_on_sat = bool(
                        (ctx.config or {}).get("ranking", {})
                                            .get("panel_scoring", {})
                                            .get("abstain_on_calibrator_saturation", True)
                    )
                    if abstain_on_sat:
                        if score_collapse or upper_tail_saturation:
                            ctx._calibrator_saturated = True  # noqa: SLF001
                            log.warning(
                                "CALIBRATOR-SATURATED → ABSTAIN-NEW-BUYS "
                                "(reason=%s%s). QP will skip new BUY actions "
                                "today; existing holdings may still SELL. To "
                                "disable: ranking.panel_scoring."
                                "abstain_on_calibrator_saturation=false",
                                "score_collapse" if score_collapse else "",
                                "+upper_tail" if upper_tail_saturation else "",
                            )
                        else:
                            log.warning(
                                "CALIBRATOR-SATURATED diagnostic only: low "
                                "rank_score IQR without score collapse; new "
                                "buys remain enabled."
                            )
                # 2026-05-15 BUG #8 GUARD: expected_return out-of-range
                # detection. Live prod calibrator's expected_return.y has
                # values up to +1.0 (= +100% expected return) — clearly
                # broken. Any candidate hitting that knot would get a
                # Kelly target of "full position regardless of σ". Fire
                # warning if any |expected_return| > 0.20 (20% over
                # 20-day horizon is the highest plausibly real bound).
                ers = [c.expected_return for c in ctx.candidates
                       if c.expected_return is not None
                       and c.expected_return == c.expected_return]
                if ers:
                    max_abs_er = max(abs(x) for x in ers)
                    if max_abs_er > 0.20:
                        log.warning(
                            "CALIBRATOR-ER-OUT-OF-RANGE: max|expected_return|"
                            "=%.3f over %d candidates exceeds 0.20 sanity "
                            "bound. Calibrator's expected_return head was "
                            "not clipped at train site (CLAUDE.md §5.13.12 "
                            "violation). Kelly sizing on this signal would "
                            "over-leverage these positions. [P0 detected 2026-05-15]",
                            max_abs_er, len(ers),
                        )


# ── NGBoost tasks (Stage 2 — optional) ────────────────────────────────────────

def _fail_closed_ngboost(ctx: InferenceContext, reason: str, *, detail: str = "") -> bool:
    """Block new buys when an enabled NGBoost scoring path is unusable."""
    _nan = float("nan")
    blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
    for cand in list(getattr(ctx, "candidates", []) or []):
        ticker = getattr(cand, "ticker", None)
        if ticker:
            blocked[ticker] = reason
        if hasattr(cand, "mu"):
            cand.mu = _nan
        if hasattr(cand, "sigma"):
            cand.sigma = _nan
    ctx._blocked_by_ticker = blocked  # noqa: SLF001
    ctx._ngboost_head = None  # noqa: SLF001
    ctx._ngboost_fail_closed_reason = reason  # noqa: SLF001
    if detail:
        ctx._ngboost_fail_closed_detail = detail  # noqa: SLF001
    ctx.buy_blocked = True
    ctx.skip_buys = True
    ctx.candidates = []
    if hasattr(ctx, "counters"):
        ctx.counters["ngb_fail_closed"] = (
            ctx.counters.get("ngb_fail_closed", 0) + 1
        )
    log.error("NGBoost fail-closed: %s%s", reason, f" ({detail})" if detail else "")
    return False


class LoadNGBoostTask(Task):
    """Load the NGBoostHead artifact when enabled.

    No-op when the effective NGBoost flag is false. When it is true, failure
    is fail-closed for new buys; otherwise live/full silently trades a weaker
    panel-only score while the operator believes μ/σ is active.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        # 2026-05-17 BUG FIX: use _ngb_cfg (per-regime + hysteresis aware)
        # rather than raw config. Without this, the per-regime overlay
        # never loads the head because the global enabled=false short-
        # circuits, so ApplyNGBoostTask sees head=None and never fires.
        ngb_cfg = _ngb_cfg(ctx)
        if not ngb_cfg.get("enabled", False):
            return

        head = getattr(ctx, "_ngboost_head", None)
        if head is not None:
            return

        # §5.13.14: never default to a hardcoded artifact filename. The path
        # MUST come from config — otherwise a sim that enables NGBoost
        # without overriding artifact_path would silently load the
        # production model and breach sim/prod isolation.
        artifact = ngb_cfg.get("artifact_path")
        if not artifact:
            return _fail_closed_ngboost(
                ctx,
                "ngb_artifact_path_missing",
                detail="ranking.panel_scoring.ngboost.artifact_path missing",
            )
        p = Path(artifact)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p
        if not p.exists():
            return _fail_closed_ngboost(
                ctx,
                "ngb_artifact_missing",
                detail=str(p),
            )

        try:
            # Polymorphic loader: dispatches on artifact `kind` field.
            # - ngboost_head → training_panel.ngboost_head.NGBoostHead
            # - quantile_head → training_panel.quantile_head.QuantileHead
            #   (XGBoost-quantile triplet, replaces single-thread NGBoost
            #    on 166-feat panels — see commit 5aad137)
            # Both classes expose identical predict_distribution() so this
            # task and downstream ApplyNGBoostTask are agnostic.
            from training_panel.quantile_head import load_head_by_kind  # noqa: PLC0415
            ctx._ngboost_head = load_head_by_kind(p)  # noqa: SLF001
        except Exception as exc:
            ctx._ngboost_head = None  # noqa: SLF001
            return _fail_closed_ngboost(
                ctx,
                "ngb_load_failed",
                detail=f"{p}: {type(exc).__name__}: {exc}",
            )
        if not getattr(ctx._ngboost_head, "feature_cols", None):
            ctx._ngboost_head = None  # noqa: SLF001
            return _fail_closed_ngboost(
                ctx,
                "ngb_feature_cols_missing",
                detail=str(p),
            )
        head_kind = type(ctx._ngboost_head).__name__
        log.info("LoadNGBoostTask: loaded %s (features=%d)",
                 head_kind, len(ctx._ngboost_head.feature_cols))


# 2026-05-17 σ-wire per-regime override layer (mirrors B-track _qp_cfg).
# Reading order (per CLAUDE.md PRIME DIRECTIVE: regime-conditional strategy):
#   regime_params.<ctx.regime>.ngboost.<KEY>  →
#     ranking.panel_scoring.ngboost.<KEY>
# Test pin: tests/test_per_regime_sigma_wire.py.
# Rationale (2026-05-17 σ-wire A/B): global σ-on lost pooled mean but
# WON +14pp on 4 BEAR/crisis windows, LOST -14pp on 2 BULL windows.
# Per-regime activation lets us capture the BEAR wins without paying
# the BULL drag — same regime-conditional pattern that B-track per-regime
# CVaR was built for.
_NGB_PER_REGIME_KEYS = (
    "enabled",
    "score_mode",
    "lambda_sigma",
)


def _ngb_cfg(ctx) -> dict:
    """Read ngboost config with per-regime overlay + hysteresis (2026-05-17).

    Resolution order (highest priority first):
      1) Live per-regime overlay — `regime_params.<ctx.regime>.ngboost.<KEY>`
         (when current regime has an entry with enabled=True).
      2) Hysteresis memo — `regime_state.sigma_wire_overlay_memo`
         (when sigma_wire_hysteresis_remaining > 0; carries the last
         live overlay for N bars so brief regime-flicker doesn't churn
         the strategy).
      3) Global default — `ranking.panel_scoring.ngboost.<KEY>`.

    Pure read; state updates happen in
    kernel.pipeline.task_regime.RegimeFinalizeTask (once per bar).
    """
    base = dict((ctx.config.get("ranking", {})
                            .get("panel_scoring", {})
                            .get("ngboost", {})) or {})
    regime = getattr(ctx, "regime", None)
    state = getattr(ctx, "regime_state", None)

    # (1) live per-regime overlay
    live_overlay = {}
    if regime:
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(regime, {}) or {}
        regime_ngb = (regime_p.get("ngboost") or {}) if isinstance(regime_p, dict) else {}
        for key in _NGB_PER_REGIME_KEYS:
            if key in regime_ngb:
                live_overlay[key] = regime_ngb[key]

    if live_overlay.get("enabled") is True:
        # Live trigger — apply overlay directly.
        base.update(live_overlay)
    elif state is not None and getattr(state, "sigma_wire_hysteresis_remaining", 0) > 0:
        # (2) Hysteresis — use memo overlay so σ-wire stays sticky.
        memo = getattr(state, "sigma_wire_overlay_memo", {}) or {}
        base.update(memo)
    # else: cold — global defaults only.

    return base


# ── Sentiment per-regime gate (added 2026-05-18) ─────────────────────────────
# Per CLAUDE.md PRIME DIRECTIVE: every feature regime-conditional.
# 2026-05-18 regime-stratified IC verdict:
#   HIGH_SPIKED  IC +0.054 / +0.045 / +0.046 — DEPLOY
#   HIGH_NORMAL  IC +0.041 (mean_sentiment × fwd_20d) — DEPLOY
#   MED_CALM     IC +0.042 (sentiment_pos_share × fwd_20d) — DEPLOY
#   MED_SPIKED   IC +0.030 (noise) — keep ON (positive direction, safe)
#   LOW_*        mostly noise or slightly negative — gate OFF
#   MED_NORMAL   net NEGATIVE — gate OFF
#   LOW_NORMAL   net NEGATIVE — gate OFF
#
# Default policy: enable in regimes where the IC eval showed positive
# net signal; disable where ts-30-placebo-adjusted net IC was negative.
# Operator can override via regime_params.<R>.sentiment.enabled.

from kernel.artifact_contract import (  # noqa: E402
    SENTIMENT_DEFAULT_REGIME_POLICY as _SENTIMENT_DEFAULT_REGIME_POLICY,
    SENTIMENT_FEATURE_COLS,
)


def _sentiment_cfg(ctx) -> dict:
    """Read sentiment-gate config with per-regime overlay.

    Resolution order (highest first):
      1) regime_params.<ctx.regime>.sentiment.enabled (live override)
      2) ranking.panel_scoring.sentiment.regime_policy.<REGIME> (config policy)
      3) _SENTIMENT_DEFAULT_REGIME_POLICY[REGIME] (hardcoded default per
         2026-05-18 regime-stratified IC eval)
      4) ranking.panel_scoring.sentiment.enabled (global on/off)
      5) True (failsafe — don't zero out, let model decide)

    Returns dict with key 'enabled': bool.
    """
    base_global = bool((ctx.config.get("ranking", {})
                                  .get("panel_scoring", {})
                                  .get("sentiment", {})
                                  .get("enabled", True)))
    regime = getattr(ctx, "regime", None)
    if not regime:
        return {"enabled": base_global}

    # (1) live per-regime overlay
    regime_p = (ctx.config.get("regime_params", {}) or {}).get(regime, {}) or {}
    regime_sent = regime_p.get("sentiment") if isinstance(regime_p, dict) else None
    if isinstance(regime_sent, dict) and "enabled" in regime_sent:
        return {"enabled": bool(regime_sent["enabled"])}

    # (2) config-level regime policy table
    policy = (ctx.config.get("ranking", {}).get("panel_scoring", {})
                        .get("sentiment", {}).get("regime_policy") or {})
    if regime in policy:
        return {"enabled": bool(policy[regime])}

    # (3) hardcoded default policy
    if regime in _SENTIMENT_DEFAULT_REGIME_POLICY:
        return {"enabled": _SENTIMENT_DEFAULT_REGIME_POLICY[regime]}

    # (4)/(5) fallthrough
    return {"enabled": base_global}


class ApplySentimentGateTask(Task):
    """Zero out sentiment feature columns when regime gate is OFF.

    Per CLAUDE.md PRIME DIRECTIVE: sentiment IC is regime-conditional.
    HIGH_SPIKED IC +0.054, but LOW_NORMAL net NEGATIVE — same model
    weights, opposite effective contribution. Zeroing the inputs in
    OFF-regimes makes the sentiment terms drop out of the booster's
    cumulative score, leaving the 169-feat backbone to act alone.

    Runs after AssembleInferenceMatrixTask (X is built) and BEFORE
    panel scoring (ApplyScoresTask consumes X to compute panel_score).

    The zeroing is in-place on ctx._panel_matrix. Reads:
      ctx._panel_matrix  (the feature DataFrame)
      ctx.regime         (current regime label)
      ctx.config         (regime_params overlay + sentiment.regime_policy)
    """

    name = "ApplySentimentGateTask"

    def run(self, ctx) -> bool | None:
        X = getattr(ctx, "_panel_matrix", None)
        if X is None or X.empty:
            return None
        cfg = _sentiment_cfg(ctx)
        if cfg.get("enabled", True):
            # Sentiment ON for this regime — leave untouched
            return None
        # Sentiment OFF — zero the columns present in X
        zeroed = []
        for col in SENTIMENT_FEATURE_COLS:
            if col in X.columns:
                X[col] = 0.0
                zeroed.append(col)
        if zeroed:
            log.info("ApplySentimentGateTask: regime=%s sentiment OFF — "
                     "zeroed cols=%s", getattr(ctx, "regime", "?"), zeroed)
        return None


class ApplyNGBoostTask(Task):
    """Apply NGBoost μ,σ predictions on top of the LTR panel scoring.

    - Writes `mu` + `sigma` onto every candidate / holding for which a
      prediction is available.
    - When `ngboost.score_mode == "mu_minus_lambda_sigma"` (the default
      when ngboost is enabled), overwrites `rank_score` AND `panel_score`
      with `μ − λ·σ` so downstream ranking + rotation use the combined
      signal. Set score_mode = "additive" to keep the LTR rank_score
      unchanged and only populate mu/sigma for sizing.

    2026-05-17 per-regime override: `regime_params.<REGIME>.ngboost.<KEY>`
    overrides the global `ranking.panel_scoring.ngboost.<KEY>` for any of
    {enabled, score_mode, lambda_sigma}. Lets σ-wire fire conditional on
    regime (e.g. ON in BEAR/CHOPPY, OFF in BULL_CALM/BULL_STRONG).
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        ngb_cfg = _ngb_cfg(ctx)
        if not ngb_cfg.get("enabled", False):
            return
        head = getattr(ctx, "_ngboost_head", None)
        X    = getattr(ctx, "_panel_matrix", None)
        if head is None:
            return _fail_closed_ngboost(ctx, "ngb_head_missing")
        if X is None or X.empty:
            return _fail_closed_ngboost(ctx, "ngb_matrix_missing")

        # Audit N-25 (2026-04-25): pre-fix this returned early if ANY
        # head.feature_cols was missing from X — one missing column killed
        # the entire bar's NGBoost output. Post-fix, fill missing columns
        # with 0.0 (z-scored "neutral") and warn loudly so the operator
        # knows the prediction is using a partial feature set.
        #
        # 2026-04-27 incident: NGBoost head was trained with 140+ macro
        # cols (vxx/hyg/dgs10/cpiaucsl/...) but inference panel no longer
        # produces them after macro was disabled. 140/167 cols zero-filled
        # → σ corrupted → all live edge_sharpe scores compressed below
        # Gate B threshold → 0 buy candidates all day. The warning fired
        # but was buried under 100 PerformanceWarnings and missed.
        # Hard-fail when too many cols missing so the operator can't
        # silently keep trading on a degraded NGBoost head.
        missing = [c for c in head.feature_cols if c not in X.columns]
        if missing:
            n_total   = len(head.feature_cols)
            n_missing = len(missing)
            pct_miss  = n_missing / max(1, n_total)
            drift_thr = float(ngb_cfg.get("max_feature_drift_pct", 0.05))
            allow_partial = bool(ngb_cfg.get("allow_partial_feature_fill", False))
            if not allow_partial or pct_miss > drift_thr:
                reason = (
                    "ngb_missing_features"
                    if not allow_partial else
                    "ngb_feature_drift"
                )
                return _fail_closed_ngboost(
                    ctx,
                    reason,
                    detail=(
                        f"{n_missing}/{n_total} missing "
                        f"({pct_miss:.1%}); first={missing[:10]}"
                    ),
                )
            log.warning(
                "ApplyNGBoostTask: feature matrix missing %d/%d cols (%.1f%%, "
                "below %.0f%% hard-fail threshold) — filling with 0.0 (z-scored "
                "neutral). Predictions partial. First 10 missing: %s",
                n_missing, n_total, pct_miss * 100, drift_thr * 100,
                missing[:10],
            )
            X = X.copy()
            for c in missing:
                X[c] = 0.0

        # 2026-05-09 BUG #6 GUARD: pre-predict input variance check.
        # Invariant: ≥80% of feature columns must have non-zero per-row
        # variance (i.e., not all rows identical) when n_rows ≥ 2. If too
        # many columns are constant, downstream model will produce constant
        # predictions (the BUG #6 failure mode). Constant columns also signal
        # upstream feature corruption (BUG #1 fund-zero, BUG #2 SEC date drift).
        try:
            import numpy as _np  # noqa: PLC0415
            X_head = X[head.feature_cols] if all(c in X.columns for c in head.feature_cols) else X
            if len(X_head) >= 2:
                col_stds = X_head.std(axis=0, skipna=True).fillna(0.0).values
                n_zero_var = int((_np.abs(col_stds) < 1e-12).sum())
                n_total_cols = len(col_stds)
                pct_zero = n_zero_var / max(1, n_total_cols)
                INPUT_ZERO_VAR_FLOOR = 0.20  # > 20% constant columns = bad
                if pct_zero > INPUT_ZERO_VAR_FLOOR:
                    log.error(
                        "ApplyNGBoostTask INPUT-VARIANCE GUARD FAILED: %d/%d "
                        "(%.1f%%) feature columns have zero per-row variance "
                        "across %d candidates (threshold %.0f%%). Constant "
                        "input columns → constant predictions. Likely causes: "
                        "(a) ctx._panel_matrix carries legacy schema with all-"
                        "NaN cols median-imputed to constants (BUG #6), (b) "
                        "fund features all 0 (BUG #1), (c) panel build SEC-date "
                        "misalignment (BUG #2). FAIL-SAFE: clearing candidates.",
                        n_zero_var, n_total_cols, pct_zero * 100,
                        len(X_head), INPUT_ZERO_VAR_FLOOR * 100,
                    )
                    _nan = float("nan")
                    for cand in ctx.candidates:
                        cand.mu = _nan
                        cand.sigma = _nan
                    ctx.candidates = []
                    if hasattr(ctx, "counters"):
                        ctx.counters["ngb_input_variance_fail"] = (
                            ctx.counters.get("ngb_input_variance_fail", 0) + 1
                        )
                    return False
                if pct_zero > 0.10:
                    log.warning(
                        "ApplyNGBoostTask: %d/%d (%.1f%%) feature columns have "
                        "zero per-row variance — partial constant inputs. "
                        "Predictions may be degraded. Below %.0f%% hard-fail.",
                        n_zero_var, n_total_cols, pct_zero * 100,
                        INPUT_ZERO_VAR_FLOOR * 100,
                    )
        except Exception as _exc:
            log.warning("ApplyNGBoostTask input-variance check failed: %s", _exc)

        try:
            dist = head.predict_distribution(X)
        except Exception as exc:
            return _fail_closed_ngboost(
                ctx,
                "ngb_predict_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )

        lambda_sigma = float(ngb_cfg.get("lambda_sigma", 1.0))
        score_mode   = str(ngb_cfg.get("score_mode", "mu_minus_lambda_sigma"))
        override     = (score_mode == "mu_minus_lambda_sigma")

        try:
            mu    = dist["mu"]
            sigma = dist["sigma"]
        except Exception as exc:
            return _fail_closed_ngboost(
                ctx,
                "ngb_predict_contract_failed",
                detail=f"missing mu/sigma: {type(exc).__name__}: {exc}",
            )
        combined = mu - lambda_sigma * sigma

        missing_or_bad: list[str] = []
        for cand in ctx.candidates:
            ticker = getattr(cand, "ticker", None)
            if not ticker or ticker not in mu.index or ticker not in sigma.index:
                missing_or_bad.append(str(ticker))
                continue
            if pd.isna(mu.loc[ticker]) or pd.isna(sigma.loc[ticker]):
                missing_or_bad.append(str(ticker))
        coverage_floor = float(
            ngb_cfg.get(
                "min_prediction_coverage",
                1.0 if override else 0.0,
            )
        )
        coverage = (
            (len(ctx.candidates) - len(missing_or_bad)) / max(1, len(ctx.candidates))
        )
        strict_coverage = bool(
            ngb_cfg.get("strict_prediction_coverage", override)
        )
        if missing_or_bad and (strict_coverage or coverage < coverage_floor):
            return _fail_closed_ngboost(
                ctx,
                "ngb_prediction_incomplete",
                detail=(
                    f"coverage={coverage:.1%} floor={coverage_floor:.1%}; "
                    f"bad={missing_or_bad[:10]}"
                ),
            )

        # Audit N-5 / N-25 (2026-04-25): after the NGBoost head's NaN
        # passthrough, predict_distribution returns NaN at rows it couldn't
        # score (NaN/inf input features). Skip those tickers cleanly so
        # downstream sizers / rotators don't compute Kelly = μ/σ² on NaN.
        # 2026-05-04 instrumentation: per-candidate skip-reason counters
        # so the funnel is explainable end-to-end (the user mandate that
        # spawned this audit). Without these, the log says n_cands=48
        # then n_kelly=0 with no way to tell if the leak is in
        # NaN-passthrough, predict_distribution missing rows, or μ
        # values landing exactly at zero.
        n_set = n_not_in_idx = n_mu_nan = n_sigma_nan = 0
        blocked = getattr(ctx, "_blocked_by_ticker", None) or {}
        for cand in ctx.candidates:
            if cand.ticker not in mu.index:
                n_not_in_idx += 1
                blocked[cand.ticker] = "ngb_skipped:not_in_predict_index"
                continue
            mu_val    = mu.loc[cand.ticker]
            sigma_val = sigma.loc[cand.ticker]
            if pd.isna(mu_val):
                n_mu_nan += 1
                blocked[cand.ticker] = "ngb_skipped:mu_nan"
                continue
            if pd.isna(sigma_val):
                n_sigma_nan += 1
                blocked[cand.ticker] = "ngb_skipped:sigma_nan"
                continue
            cand.mu    = float(mu_val)
            cand.sigma = float(sigma_val)
            n_set += 1
            if override:
                v = float(combined.loc[cand.ticker])
                cand.rank_score  = v
                cand.panel_score = v
        ctx._blocked_by_ticker = blocked  # noqa: SLF001

        for ticker, hs in ctx.holdings.items():
            if ticker not in mu.index:
                continue
            mu_val    = mu.loc[ticker]
            sigma_val = sigma.loc[ticker]
            if pd.isna(mu_val) or pd.isna(sigma_val):
                continue
            hs.mu    = float(mu_val)
            hs.sigma = float(sigma_val)
            if override:
                # Audit #40: hold-side rank_score must mirror cand-side.
                # Without this, rotation comparisons (which use rank_score
                # on both sides) saw mu-minus-lambda-sigma on cands but
                # stale per-ticker scores on holds. The downstream
                # ApplyGlobalCalibrationTask will then map rank_score
                # through the isotonic head consistently.
                v = float(combined.loc[ticker])
                hs.panel_score = v
                hs.rank_score  = v

        log.info("ApplyNGBoostTask: mode=%s  λ=%.2f  n_cands=%d  n_holdings=%d  "
                 "(set_μσ=%d  not_in_idx=%d  mu_nan=%d  sigma_nan=%d)",
                 score_mode, lambda_sigma, len(ctx.candidates), len(ctx.holdings),
                 n_set, n_not_in_idx, n_mu_nan, n_sigma_nan)
        # 2026-05-09 BUG #6 GUARD: post-predict diversity check.
        # Invariant: cross-sectional std of μ̂ across candidates must be > ε
        # (typically training-time x-sec std is ~0.02 — anything below 1e-4
        # signals collapse). Pre-fix, BUG #6 produced n=49 std=0.00000 silently
        # (every ticker got the same feature_medians-imputed input vector).
        # Kelly downstream rejected all 49 with mu_le_min_edge but no log
        # surfaced WHY. Now: hard-fail with ERROR + clear candidates so the
        # operator sees the prediction collapse immediately.
        import numpy as _np  # noqa: PLC0415
        mu_arr = _np.asarray(mu.values, dtype=float)
        sd_arr = _np.asarray(sigma.values, dtype=float)
        mu_finite = mu_arr[_np.isfinite(mu_arr)]
        sd_finite = sd_arr[_np.isfinite(sd_arr)]
        if len(mu_finite) >= 2:
            mu_xs_std = float(mu_finite.std())
            sd_xs_std = float(sd_finite.std()) if len(sd_finite) >= 2 else 0.0
            n_unique_mu = int(len(_np.unique(mu_finite.round(8))))
            log.info(
                "ApplyNGBoostTask μ̂ stats: n=%d mean=%+.4f std=%.4f "
                "n_unique=%d  σ̂ mean=%.4f std=%.4f",
                len(mu_finite), float(mu_finite.mean()), mu_xs_std, n_unique_mu,
                float(sd_finite.mean()) if len(sd_finite) else float("nan"),
                sd_xs_std,
            )
            # Hard-fail thresholds. Training x-sec std ≈ 0.02; a healthy run
            # is at least 1e-3. Below that, predictions have collapsed —
            # either feature input is constant OR model is degenerate.
            DIVERSITY_FLOOR = 1e-4
            if mu_xs_std < DIVERSITY_FLOOR or n_unique_mu < 2:
                log.error(
                    "ApplyNGBoostTask DIVERSITY GUARD FAILED: μ̂ x-sec "
                    "std=%.6f (< %.0e floor) AND n_unique_mu=%d. Predictions "
                    "have collapsed to a constant — typically caused by (a) "
                    "ctx._panel_matrix carrying legacy schema (BUG #6), (b) "
                    "all features all-NaN at the candidate rows triggering "
                    "median imputation everywhere, or (c) head-input feature "
                    "subset disjoint from training. FAIL-SAFE: clearing "
                    "ctx.candidates so QP/Kelly do not trade on collapsed μ̂.",
                    mu_xs_std, DIVERSITY_FLOOR, n_unique_mu,
                )
                # Stamp NaN so anything downstream that reads cand.mu / cand.sigma
                # also fails-safe rather than silently treating constant as truth.
                _nan = float("nan")
                for cand in ctx.candidates:
                    cand.mu = _nan
                    cand.sigma = _nan
                ctx.candidates = []
                if hasattr(ctx, "counters"):
                    ctx.counters["ngb_diversity_fail"] = (
                        ctx.counters.get("ngb_diversity_fail", 0) + 1
                    )
                return False


# ── σ fallback when NGBoost off (Phase 3 of 2026-05-15 P0) ──────────────────

class ApplyRealizedVolFallbackTask(Task):
    """Fill c.sigma with trailing realized vol when NGBoost OFF.

    Background: NGBoost is the only task that writes `c.sigma` today.
    When NGBoost is disabled (current prod since 2026-05-09), every
    candidate's sigma is None → Kelly skips with `kelly_zero:sigma_none`.
    This task provides a fallback: annualized stdev of trailing 60-day
    daily returns from ctx.ohlcv[ticker]['close'].

    OPT-IN via `ranking.kelly_sizing.use_realized_vol_fallback=true`.
    Disabled by default so prod behavior is unchanged. Pairs with the
    Phase-3 `use_calibrator_mu` flag — both must be on to re-enable
    Kelly sizing with proper μ/σ via the calibrator + realized-vol path.

    Runs AFTER ApplyGlobalCalibrationTask (so c.mu is set) and BEFORE
    ApplyKellySizingTask (so Kelly sees the populated sigma).

    Reuses the same helper logic as RealizedVolGateTask, kept local
    here to avoid a kernel.pipeline import cycle.
    """

    def run(self, ctx: "InferenceContext") -> "bool | None":
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not bool(kelly_cfg.get("use_realized_vol_fallback", False)):
            return
        window = int(kelly_cfg.get("realized_vol_window_days", 60))
        floor = float(kelly_cfg.get("realized_vol_floor", 0.05))     # 5% σ floor
        ceiling = float(kelly_cfg.get("realized_vol_ceiling", 1.50)) # 150% σ cap

        ohlcv = getattr(ctx, "ohlcv", None) or {}
        n_filled = 0
        for c in ctx.candidates:
            if getattr(c, "sigma", None) is not None and math.isfinite(c.sigma):
                continue  # already populated by NGBoost
            sig = _realized_vol_annualized(ohlcv.get(c.ticker), window)
            if sig is not None:
                c.sigma = float(np.clip(sig, floor, ceiling))
                n_filled += 1

        for ticker, hs in ctx.holdings.items():
            if getattr(hs, "sigma", None) is not None and math.isfinite(hs.sigma):
                continue
            sig = _realized_vol_annualized(ohlcv.get(ticker), window)
            if sig is not None:
                hs.sigma = float(np.clip(sig, floor, ceiling))

        if n_filled:
            log.info(
                "ApplyRealizedVolFallbackTask: filled c.sigma from realized "
                "vol (window=%dd, clip=[%.2f, %.2f]) for %d/%d candidates",
                window, floor, ceiling, n_filled, len(ctx.candidates),
            )


def _realized_vol_annualized(df, window: int):
    """Return annualized stdev of daily returns over last `window` bars,
    or None if df is missing / has insufficient history.

    Pure function — mirrors RealizedVolGateTask._realized_vol_annualized
    so we don't create a kernel.pipeline → kernel.panel_pipeline cycle.
    """
    if df is None:
        return None
    try:
        close = df["close"]
    except (KeyError, TypeError):
        return None
    if len(close) < max(window, 5):
        return None
    rets = close.pct_change().tail(window).dropna()
    if len(rets) < max(window // 2, 5):
        return None
    std = float(rets.std())
    if not math.isfinite(std):
        return None
    return std * math.sqrt(252.0)


# ── Kelly sizing (Plan C — the smart part) ───────────────────────────────────

class ApplyKellySizingTask(Task):
    """Populate `kelly_target_pct` on every candidate AND holding using
    the classical continuous-returns Kelly: f* = μ/σ².

    Runs LAST in PanelScoringJob — after ApplyNGBoostTask writes μ,σ
    and ApplyGlobalCalibrationTask settles rank_score. The Kelly
    target is then consumed by three downstream layers:

      SizeAndEmitTask  — caps new-buy size at `kelly_target_pct`.
      TopUpHeldTask    — emits a BUY if held.kelly_target exceeds
                         current weight by `top_up_threshold`.
      RotationJob      — (future) rotation advantage test in Kelly
                         units rather than raw rank_score.

    One math, one place, one field. See `kernel/kelly.py` for the
    full formula + safety discussion.
    """

    def run(self, ctx: "InferenceContext") -> "bool | None":
        kelly_cfg = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        if not kelly_cfg.get("enabled", False):
            return   # no-op — golden behaviour preserved

        from kernel.kelly import kelly_target_pct      # noqa: PLC0415

        fractional        = float(kelly_cfg.get("fractional",        0.25))
        min_edge          = float(kelly_cfg.get("min_edge",          0.0))
        max_concentration = float(kelly_cfg.get("max_concentration", 0.35))

        # Audit fix CONF-MULT (2026-04-25): floored confidence multiplier.
        from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
        _conf_mult = confidence_to_size_multiplier(ctx.confidence)
        regime_p = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        max_pct  = float(regime_p.get("max_position_pct", 0.15)) * _conf_mult

        # 2026-05-15 P0 cleanup: vol-target + DD-Kelly scaling REMOVED
        # from this local-variable path. They previously modified `max_pct`
        # (a function-scope variable that QP never reads) — see
        # doc/AUDIT_2026-05-12_dead_paths.md. The live implementation
        # lives in kernel.portfolio_qp.tasks.ApplyExposureScalingTask
        # which writes ctx._vol_target_scale / ctx._dd_kelly_scale and
        # multiplies them into ctx._qp_w_upper inside the QP job. That
        # is the architecturally correct location: all exposure-cap
        # modifiers compose at the QP bound, not inside a Kelly local
        # that may be unused when mu is None.

        # 2026-05-04 instrumentation (user mandate: explainable funnel,
        # decision-tree DB persistence). Per-candidate skip-reason
        # counters + write to ctx._blocked_by_ticker so SQL queries on
        # candidate_scores.blocked_by show exactly why each ticker was
        # filtered. Without this, the funnel stage "n_cands=48 →
        # kelly=0 non-zero" was opaque.
        import math   # noqa: PLC0415
        skip_counts = {
            "kelly_zero:mu_none":        0,
            "kelly_zero:mu_nonfinite":   0,
            "kelly_zero:sigma_none":     0,
            "kelly_zero:sigma_nonfinite":0,
            "kelly_zero:sigma_nonpos":   0,
            "kelly_zero:mu_le_min_edge": 0,
            "kelly_zero:capped_zero":    0,
        }
        blocked = getattr(ctx, "_blocked_by_ticker", None) or {}

        def _kelly_with_reason(obj):
            mu_v = getattr(obj, "mu",    None)
            sg_v = getattr(obj, "sigma", None)
            if mu_v is None:    return 0.0, "kelly_zero:mu_none"
            if sg_v is None:    return 0.0, "kelly_zero:sigma_none"
            try:
                mu_f = float(mu_v); sg_f = float(sg_v)
            except (TypeError, ValueError):
                return 0.0, "kelly_zero:mu_nonfinite"
            if not math.isfinite(mu_f):  return 0.0, "kelly_zero:mu_nonfinite"
            if not math.isfinite(sg_f):  return 0.0, "kelly_zero:sigma_nonfinite"
            if sg_f <= 0:                return 0.0, "kelly_zero:sigma_nonpos"
            if mu_f <= min_edge:         return 0.0, "kelly_zero:mu_le_min_edge"
            target = kelly_target_pct(
                mu_f, sg_f,
                max_pct           = max_pct,
                max_concentration = max_concentration,
                fractional        = fractional,
                min_edge          = min_edge,
            )
            if target <= 0:              return 0.0, "kelly_zero:capped_zero"
            return target, None

        for cand in ctx.candidates:
            target, reason = _kelly_with_reason(cand)
            cand.kelly_target_pct = target
            if reason is not None:
                skip_counts[reason] += 1
                # Don't clobber a more upstream block (e.g. ngb_skipped)
                blocked.setdefault(cand.ticker, reason)

        for hs in ctx.holdings.values():
            target, _ = _kelly_with_reason(hs)
            hs.kelly_target_pct = target

        ctx._blocked_by_ticker = blocked  # noqa: SLF001

        # Audit summary — most informative when live.
        cand_targets = [c.kelly_target_pct for c in ctx.candidates
                         if c.kelly_target_pct]
        held_targets = [h.kelly_target_pct for h in ctx.holdings.values()
                         if h.kelly_target_pct]
        # Compact skip-reason summary: only emit non-zero counts.
        skip_str = " ".join(f"{r.split(':',1)[1]}={c}"
                              for r, c in skip_counts.items() if c > 0)
        log.info(
            "ApplyKellySizingTask: fractional=%.2f max_conc=%.2f  "
            "cands=%d non-zero (avg=%.1f%%)  holdings=%d non-zero (avg=%.1f%%)"
            "%s",
            fractional, max_concentration,
            len(cand_targets),
            (sum(cand_targets) / len(cand_targets) * 100) if cand_targets else 0,
            len(held_targets),
            (sum(held_targets) / len(held_targets) * 100) if held_targets else 0,
            f"  zero_reasons[{skip_str}]" if skip_str else "",
        )


# ── Job ──────────────────────────────────────────────────────────────────────

class PanelScoringJob(Job):
    """Overwrite rank_score on surviving candidates with cross-sectional panel scores.

    Task chain:
      LoadScorer → BuildFeatureMatrix → ApplyScores → ApplyShadowScoring
        → LoadNGBoost → ApplyNGBoost                 (no-op if ngboost.enabled is false)
        → LoadGlobalCalibration → ApplyGlobalCalibration (always-runs; see below)
        → VetoWeakBuys → ApplyRealizedVolFallback → ApplyKellySizing
        → QualityFloor

    Ordering rationale (task #2, 2026-04-23):
      NGBoost runs BEFORE global calibration so that when NGBoost's
      score_mode == "mu_minus_lambda_sigma" it overwrites panel_score
      with μ−λσ, and the calibrator then maps μ−λσ → probability via its
      isotonic head. Previously calibration ran first and short-circuited
      in mu_minus_lambda_sigma mode, leaving rank_score as raw μ−λσ
      (always < 0.10 tier threshold → zero trades). With this ordering,
      both additive and mu_minus_lambda_sigma modes produce calibrated
      rank_score and the tier logic works in either.
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        # Run even with no candidates so holdings can still be panel-scored
        # for rotation decisions later in the pipeline.
        if not ctx.candidates and not ctx.holdings:
            return True
        return not ctx.config.get("ranking", {}).get("panel_scoring", {}).get("enabled", False)

    @property
    def tasks(self) -> list[Task]:
        # Lazy import — avoids a circular import that fires when
        # job_panel_scoring is imported by InferencePipeline init.
        from kernel.panel_pipeline.task_quality_floor import (  # noqa: PLC0415
            QualityFloorTask,
        )
        # 2026-05-18 SHADOW SCORING — register here so it runs AFTER
        # ApplyScoresTask (which writes primary scores). Lazy-imported to
        # avoid forcing import cost on configs that don't use shadow.
        from kernel.panel_pipeline.shadow_scoring import ApplyShadowScoringTask  # noqa: PLC0415
        return [
            LoadScorerTask(),
            BuildFeatureMatrixTask(),
            ApplyScoresTask(),
            ApplyShadowScoringTask(),   # NEW: no-op if no shadow_models configured
            LoadNGBoostTask(),
            ApplyNGBoostTask(),
            LoadGlobalCalibrationTask(),
            ApplyGlobalCalibrationTask(),
            RegimeModelAdmissionTask(),
            # 2026-05-03 P0 fix: VetoWeakBuysTask MOVED to here (was right
            # after ApplyScoresTask). Veto must compare against calibrated
            # rank_score, not raw XGB margin. See VetoWeakBuysTask
            # docstring for the production incident this resolves.
            VetoWeakBuysTask(),
            # 2026-05-15 Phase 3: σ fallback to realized 60d vol when
            # NGBoost OFF. No-op unless `kelly_sizing.use_realized_vol_
            # fallback=true`. Pairs with `use_calibrator_mu` flag in
            # ApplyGlobalCalibrationTask — both ON re-enables Kelly.
            ApplyRealizedVolFallbackTask(),
            ApplyKellySizingTask(),   # Plan C — f*=μ/σ² (no-op unless kelly_sizing.enabled)
            # Buy-logic redesign Stage 0 (2026-04-26): quality gates
            # filter weak-signal candidates AFTER all scoring + sizing.
            # All gates default OFF — bit-for-bit parity preserved.
            # See doc/components/buy-logic-design.md for theory.
            QualityFloorTask(),
        ]
