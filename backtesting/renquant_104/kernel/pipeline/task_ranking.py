"""Ranking tasks: blend scores then sort."""
from __future__ import annotations

import logging

from .context import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.ranking")


def _parse_weight_pair(raw: object) -> tuple[float, float] | None:
    """Parse a ``(rank, rs)`` weight pair and normalize it."""
    try:
        if isinstance(raw, dict):
            w_rank = float(raw.get("rank", raw.get("rank_score", 0.0)))
            w_rs = float(raw.get("rs", raw.get("rs_score", 0.0)))
        else:
            seq = list(raw)  # type: ignore[arg-type]
            w_rank = float(seq[0])
            w_rs = float(seq[1])
    except (IndexError, TypeError, ValueError):
        return None
    if w_rank < 0.0 or w_rs < 0.0:
        return None
    total = w_rank + w_rs
    if total <= 0.0:
        return None
    return (w_rank / total, w_rs / total)


def _resolve_blend_weights(ctx: InferenceContext) -> tuple[float, float, str]:
    """Return regime-aware ranking weights.

    Default remains panel-rank only. ``ranking.regime_blend_weights`` is the
    explicit opt-in surface for regime-conditional research, for example:
    ``{"BULL_CALM": [0.4, 0.6]}``.
    """
    ranking_cfg = ctx.config.get("ranking", {}) or {}
    regime_weights = ranking_cfg.get("regime_blend_weights") or {}
    if isinstance(regime_weights, dict):
        raw = regime_weights.get(ctx.regime)
        source = f"regime_blend_weights.{ctx.regime}"
        if raw is None:
            raw = regime_weights.get("_default")
            source = "regime_blend_weights._default"
        parsed = _parse_weight_pair(raw) if raw is not None else None
        if parsed is not None:
            return parsed[0], parsed[1], source
        if raw is not None:
            log.warning(
                "BlendScoresTask: invalid %s=%r; using panel-rank only",
                source,
                raw,
            )

    legacy_bw = ranking_cfg.get("blend_weights")
    if legacy_bw is not None:
        parsed = _parse_weight_pair(legacy_bw)
        if parsed is not None and parsed[1] > 0.0:
            log.warning(
                "BlendScoresTask: ignoring legacy ranking.blend_weights=%s "
                "(use ranking.regime_blend_weights for explicit regime-aware "
                "rs_score blending)",
                legacy_bw,
            )
    return 1.0, 0.0, "default_panel_rank_only"


class BlendScoresTask(Task):
    """Rank candidates by a regime-aware blend.

    Tier/QP admission continues to read each candidate's calibrated
    ``rank_score``. The blend only controls ordering, so an rs-driven order
    cannot smuggle a weak panel candidate through later quality gates.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import score_candidates  # noqa: PLC0415
        w_rank, w_rs, source = _resolve_blend_weights(ctx)
        ctx._blended = score_candidates(ctx.candidates, w_rank, w_rs)  # noqa: SLF001
        ctx._blend_w = (w_rank, w_rs)                                   # noqa: SLF001
        ctx._blend_source = source                                       # noqa: SLF001
        log.debug(
            "BlendScoresTask: %d candidates  w_rank=%.2f  w_rs=%.2f source=%s",
            len(ctx.candidates), w_rank, w_rs, source,
        )


class SortCandidatesTask(Task):
    def run(self, ctx: InferenceContext) -> bool | None:
        blended = getattr(ctx, "_blended", None)
        if blended is not None:
            ctx.ranked = list(blended)
            w_rank, w_rs = getattr(ctx, "_blend_w", (1.0, 0.0))
            log.info(
                "SortCandidatesTask: %d ranked (w_rank=%.2f w_rs=%.2f source=%s)",
                len(ctx.ranked),
                w_rank,
                w_rs,
                getattr(ctx, "_blend_source", "unknown"),
            )
            return None

        # Fallback for direct unit callers that invoke SortCandidatesTask
        # without BlendScoresTask. Keep NaN rank_score deterministic.
        import math
        def _key(c):
            s = getattr(c, "rank_score", None)
            if s is None or not math.isfinite(s):
                return float("-inf")
            return s
        ctx.ranked = sorted(ctx.candidates, key=_key, reverse=True)
        w_rank, w_rs = getattr(ctx, "_blend_w", (1.0, 0.0))
        log.info("SortCandidatesTask: %d ranked (w_rank=%.2f w_rs=%.2f)",
                 len(ctx.ranked), w_rank, w_rs)
