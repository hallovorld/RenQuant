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

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .panel_scorer import PanelScorer

log = logging.getLogger("kernel.panel_pipeline.scoring")

#: Unit domain of `cand.rank_score`. Scoring writes the scorer's RAW output;
#: calibration overwrites it with a probability in [0, 1]. The buy floor is a
#: probability-domain threshold, so comparing it against a raw score is a unit
#: error. Mirror of renquant-pipeline#219.
RANK_SCORE_DOMAIN_RAW = "raw"
RANK_SCORE_DOMAIN_PROBABILITY = "probability"


# ── Runtime feature-assembly cluster — EXTRACTED to runtime_features.py ─
# (eng plan S2 item 5 decomposition slice 3, 2026-06-12; DRPH-gated.)
from kernel.panel_pipeline.runtime_features import (  # noqa: F401,E402
    _alpha158_cached_rows,
    _apply_fund_features,
    _apply_pead_features,
    _apply_sentiment_features,
    _apply_sue_features,
    _cached_earnings_surprise,
    _cached_parquet,
    _cached_sentiment,
    _earnings_raw_row,
    _finite_or_none,
    _median_fill_rows,
    _runtime_cache,
    _sentiment_runtime_gate_declared,
    _stable_feature_context_tickers,
)


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


def _bootstrap_gate_registry_import() -> None:
    """Best-effort sibling checkout discovery for GateRegistry telemetry."""
    repo_root = Path(__file__).resolve().parents[4]
    candidates: list[Path] = []
    subrepo_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if subrepo_root:
        candidates.append(Path(subrepo_root) / "renquant-pipeline" / "src")
    candidates.append(repo_root.parent / "renquant-pipeline" / "src")
    for src in candidates:
        src_str = str(src)
        if src.is_dir() and src_str not in sys.path:
            sys.path.insert(0, src_str)


def _submit_gate_verdict(ctx: Any, *, gate: str, reason: str, inputs: dict) -> None:
    """Dual-write a block verdict to the GateRegistry (eng plan S2-PR4).

    Lazy import + loud degrade: the registry lives in renquant-pipeline
    (sibling checkout); during a merged-not-deployed window the sibling
    may predate kernel.gate_registry. Telemetry must never block trading,
    so a missing module skips the ledger row WITH a warning, never silently.
    """
    _bootstrap_gate_registry_import()
    # Degrade-safe block latch (retirement prerequisite): the choke point
    # applies the flag from EITHER the registry aggregate OR this plain
    # attribute — a pin regression that breaks the import can therefore
    # never silently disable the gates.
    ctx._gate_block_pending = True  # noqa: SLF001
    try:
        from renquant_pipeline.kernel.gate_registry import ctx_registry  # noqa: PLC0415
    except ImportError as exc:
        log.warning("gate_registry unavailable (%s) — ledger row for gate=%s "
                    "skipped (block still latched); sync the "
                    "renquant-pipeline checkout", exc, gate)
        return
    ctx_registry(ctx).submit(gate=gate, scope="book", verdict="block",
                             reason=reason, inputs=inputs)


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
    ctx.skip_buys = True
    _submit_gate_verdict(ctx, gate="panel_scoring_fail_closed", reason=reason,
                         inputs={"candidates_blocked": len(candidates)})
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
                # Single resolution authority (eng plan §III.5): adds the
                # existence-checked repo-root fallback — the #114 incident
                # class (primary vs shadow resolving the same ref against
                # different roots) is dead on the umbrella side too.
                from kernel.artifact_resolver import locate_artifact  # noqa: PLC0415

                p = locate_artifact(p, strategy_dir=Path(strategy_dir))
        return p

    @staticmethod
    def _blend_component0_path(ctx: InferenceContext, panel_cfg: dict) -> Path | None:
        """Anchor the strict-consistency gate on the FIRST blend component
        (the frozen construction fixes it as the production scorer) when
        ``kind="blend"`` carries no top-level ``artifact_path``.

        Umbrella mirror of pipeline#218 ``LoadScorerTask._blend_component0_path``
        — shared by both the preloaded (adapter/LEAN) and fresh-load branches
        of ``run()`` so the preloaded path fails closed on the same artifact
        as the fresh-load path instead of on the composite fingerprint.
        """
        comps = panel_cfg.get("components") or []
        first = comps[0] if comps and isinstance(comps[0], dict) else {}
        if not first.get("artifact_path"):
            return None
        p = Path(str(first["artifact_path"]))
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                from kernel.artifact_resolver import locate_artifact  # noqa: PLC0415

                p = locate_artifact(p, strategy_dir=Path(strategy_dir))
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
            if p is None and panel_cfg.get("kind") == "blend":
                p = self._blend_component0_path(ctx, panel_cfg)
            if not self._assert_config_consistency(ctx, panel_cfg, scorer, p):
                return False
            return

        p = self._resolve_artifact_path(ctx, panel_cfg)
        if p is None and panel_cfg.get("kind") == "blend":
            p = self._blend_component0_path(ctx, panel_cfg)
        if p is None:
            log.error("LoadScorerTask: panel_scoring.enabled but no artifact_path")
            _fail_closed_panel_scoring(ctx, "panel_scorer_missing_artifact_path")
            return False
        # 2026-06-02 Track C wire-in: when `ranking.panel_scoring.specialists`
        # is configured, route through the regime-specialist ensemble loader.
        # The ensemble wraps the global PanelScorer (loaded from `artifact_path`)
        # with up to 4 per-regime PanelScorer artifacts; `ApplyScoresTask` then
        # calls `scorer.score(X, ctx=ctx)` so the ensemble can dispatch by
        # `ctx.final_regime` + `ctx.regime_confidence` + `ctx.regime_posterior`.
        # Back-compat: when `specialists` is absent/empty the model_registry
        # path below runs unchanged. Only applies to the xgb kind (panel-LTR)
        # because the ensemble loader builds PanelScorer specialists; routing
        # PatchTST sequence specialists is a future extension.
        kind = panel_cfg.get("kind", "xgb")
        specialists_cfg = panel_cfg.get("specialists") or {}
        if specialists_cfg and kind == "xgb":
            from kernel.panel_pipeline.regime_ensemble_scorer import (  # noqa: PLC0415
                load_panel_scorer_with_ensemble,
                RegimeEnsemblePanelScorer,
                StaleSpecialistArtifact,
            )
            try:
                ctx._panel_scorer = load_panel_scorer_with_ensemble(  # noqa: SLF001
                    panel_cfg,
                    strategy_dir=ctx.config.get("_strategy_dir"),
                )
            except StaleSpecialistArtifact as exc:
                log.error("LoadScorerTask: stale specialist artifact — %s", exc)
                _fail_closed_panel_scoring(ctx, "panel_specialist_recipe_mismatch")
                return False
            except Exception as exc:
                log.error("LoadScorerTask: failed to load specialist ensemble — %s", exc)
                _fail_closed_panel_scoring(ctx, "panel_specialist_load_failed")
                return False
            if isinstance(ctx._panel_scorer, RegimeEnsemblePanelScorer):
                log.info(
                    "LoadScorerTask: loaded regime-specialist ensemble "
                    "(global features=%d, specialists=%s, confidence_threshold=%.2f)",
                    len(ctx._panel_scorer.global_scorer.feature_cols),
                    sorted(ctx._panel_scorer.specialists.keys()),
                    ctx._panel_scorer.confidence_threshold,
                )
            else:
                # No specialists actually loaded (all paths missing) — loader
                # returned the legacy global PanelScorer. Same observability as
                # the legacy load path below.
                log.warning(
                    "LoadScorerTask: specialists configured but none loadable; "
                    "fell back to global panel scorer (features=%d)",
                    len(ctx._panel_scorer.feature_cols),
                )
            if not self._assert_config_consistency(
                ctx, panel_cfg, ctx._panel_scorer, p,
            ):
                return False
            return

        # 2026-05-18 Model registry dispatch — supports XGB/PatchTST/future kinds
        # via single config knob `ranking.panel_scoring.kind`. Default xgb
        # for back-compat. Each kind's handler in kernel/panel_pipeline/
        # model_registry.py decides how to load its scorer.
        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415
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
            # rank_score below is the RAW scorer output; calibration
            # overwrites it with a probability when it runs. Record the domain
            # so the probability-domain buy floor refuses a unit-mismatched
            # comparison instead of vetoing the whole cross-section.
            ctx._rank_score_domain = RANK_SCORE_DOMAIN_RAW  # noqa: SLF001
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
        # panel_ltr_xgboost alpha158 artifacts. 2026-07-28: kind "blend"
        # (BlendPanelScorer — both components are alpha158+fund snapshot
        # artifacts; pipeline#218, umbrella mirror) takes the SAME rebuild
        # path; its union feature_cols drives the fund/PEAD/SUE/sentiment
        # lookups below.
        scorer_kind = scorer.metadata.get("kind") if hasattr(scorer, "metadata") else None
        if scorer_kind in ("panel_linear", "panel_ltr_xgboost", "blend"):
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

                if scorer_kind == "blend":
                    # 2026-07-28 blend composite (pipeline#218, umbrella
                    # mirror): BlendPanelScorer consumes the RAW union
                    # matrix and applies EACH component's stored raw→model
                    # transform internally — the two components carry
                    # different feature_means/feature_stds, so one outer
                    # transform (which would read the composite's metadata,
                    # deliberately stat-less) cannot be correct for both
                    # legs. Skip the outer transform here.
                    log.info(
                        "ApplyScoresTask[blend]: passing RAW union matrix "
                        "(%d features) — per-component transforms applied "
                        "inside BlendPanelScorer",
                        len(scorer.feature_cols),
                    )
                else:
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
                        # 2026-06-02 Track C: pass ctx so a configured
                        # RegimeEnsemblePanelScorer can dispatch by
                        # ctx.final_regime / ctx.regime_confidence /
                        # ctx.regime_posterior. PanelScorer ignores ctx
                        # (regime-blind) — signature is uniform per §7.5.
                        scores: pd.Series = scorer.score(X_aligned, ctx=ctx)
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
                # 2026-06-02 Track C: ctx forwarded for ensemble dispatch.
                scores: pd.Series = scorer.score(X, ctx=ctx)
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


# ── Global calibration cluster — EXTRACTED to calibration.py ────────────
# (eng plan S2 item 5 decomposition, 2026-06-12; DRPH-gated move.)
# Symbols re-exported for back-compat.
from kernel.panel_pipeline.calibration import (  # noqa: F401,E402
    ApplyGlobalCalibrationTask,
    LoadGlobalCalibrationTask,
    _assert_calibrator_matches_scorer,
    _calibrator_expected_return_at_horizon,
    _calibrator_native_horizon_days,
    _fail_closed_missing_calibrator,
    _fail_closed_ngboost,
)


# ── NGBoost + sentiment cluster — EXTRACTED to ngboost_tasks.py ─────────
# (eng plan S2 item 5 decomposition slice 2, 2026-06-12; DRPH-gated.)
from kernel.artifact_contract import SENTIMENT_FEATURE_COLS  # noqa: E402
from kernel.panel_pipeline.ngboost_tasks import (  # noqa: F401,E402
    ApplyNGBoostTask,
    ApplySentimentGateTask,
    LoadNGBoostTask,
    _ngb_cfg,
    _sentiment_cfg,
)

# ── Buy-admission cluster — EXTRACTED to admission_tasks.py ─────────────
# (eng plan S2 item 5 decomposition slice 4, 2026-06-12; DRPH-gated.)
from kernel.panel_pipeline.admission_tasks import (  # noqa: F401,E402
    RegimeModelAdmissionTask,
    VetoWeakBuysTask,
    _regime_stats_map,
    _sanity_regime_admission,
    _trade_monotonicity_admission,
)


# ── Sizing cluster — EXTRACTED to sizing_tasks.py ───────────────────────
# (eng plan S2 item 5 decomposition slice 5, 2026-06-13; DRPH-gated.)
from kernel.panel_pipeline.sizing_tasks import (  # noqa: F401,E402
    ApplyKellySizingTask,
    ApplyRealizedVolFallbackTask,
    _kelly_sigma_horizon_days,
    _realized_vol_annualized,
    _rescale_annualized_sigma_for_kelly,
)


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


    def run(self, ctx) -> None:
        """Apply the gate aggregate ONCE at the job boundary (errata C).

        Degrade-safe: the flag lands from EITHER the registry max-join
        aggregate OR the plain _gate_block_pending latch — so a broken
        registry import (pin regression) can never silently disable the
        gates. Mirrors pipeline #123/#125/#128.
        """
        super().run(ctx)
        registry = getattr(ctx, "gate_registry", None)
        if (registry is not None and registry.blocked("book")) or \
                getattr(ctx, "_gate_block_pending", False):
            ctx.buy_blocked = True

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
