"""Buy-admission task cluster — weak-buy veto + regime model admission.

EXTRACTED 2026-06-12 from job_panel_scoring.py (eng plan S2 item 5,
decomposition slice 4; behavior-identical move, DRPH-gated with
pre-change baselines). Symbols re-exported from job_panel_scoring.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("kernel.panel_pipeline.scoring")


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
        # When RQ_SIM_BYPASS_BUY_FLOOR=1 on a ctx tagged as sim, skip the floor
        # entirely. Used by WF gate sims to evaluate models whose calibrated
        # score distribution is narrower than the adaptive_mean_std rule expects
        # (per-cut PatchTST calibrators output prob ranges as tight as 0.07, vs
        # daily shadow's 0.49 — same model, same scoring, but buy_floor rejects
        # all WF cut candidates and admits daily-shadow candidates).
        # See memory: project_wf_sim_unfair_to_compressed_models_2026-05-30.
        # Prod live / cron keep the floor strict even if the env leaks.
        import os  # noqa: PLC0415
        if os.environ.get("RQ_SIM_BYPASS_BUY_FLOOR") == "1":
            run_type = str(
                getattr(ctx, "_run_type", None)
                or getattr(ctx, "run_type", None)
                or ctx.config.get("_run_type", "")
            ).strip().lower()
            if run_type == "sim":
                log.info(
                    "VetoWeakBuysTask: RQ_SIM_BYPASS_BUY_FLOOR=1 — skipping floor "
                    "(distribution-fair sim mode); raw_floor=%r ignored",
                    raw_floor,
                )
                return
            log.warning(
                "VetoWeakBuysTask: RQ_SIM_BYPASS_BUY_FLOOR=1 ignored outside "
                "sim run_type=%r; buy_floor remains active",
                run_type or None,
            )

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
        n_candidates = len(candidates)
        n_holdings_exit_only = len(holdings) if holdings else 0
        ctx.candidates = []
        ctx.counters["regime_admission_blocked"] = (
            ctx.counters.get("regime_admission_blocked", 0) + n_candidates
        )
        ctx.counters["regime_admission_holdings_exit_only"] = (
            ctx.counters.get("regime_admission_holdings_exit_only", 0)
            + n_holdings_exit_only
        )
        # Fix 2026-06-01 (decision-tree audit): old log printed only
        # "blocked %d candidates" with n derived from ctx.candidates at task
        # entry. When an upstream task already clears candidates (e.g. NGB
        # variance-fail, panel_scoring fail-closed), this logs "blocked 0"
        # while still marking holdings as exit-only and recording the regime
        # decision — confusing operator output. New format always shows:
        #   - the decision regime
        #   - explicit candidate + holdings-exit-only counts
        #   - the structured reason
        log.warning(
            "RegimeModelAdmissionTask: regime=%s decision=BLOCK reason=%s "
            "candidates_blocked=%d holdings_exit_only=%d",
            regime, reason, n_candidates, n_holdings_exit_only,
        )


# ── Global calibration (Item #2 — optional) ───────────────────────────────────
