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
  • no candidates survived Phase 2 (ctx.candidates empty), OR
  • the artifact can't be loaded (logged, Job short-circuits).

Kept isolated from the Stage-1 training pipeline so revert is purely
additive: remove this file + the one-line import wiring.
"""
from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

from .panel_scorer import PanelScorer
from .feature_matrix import build_inference_matrix

log = logging.getLogger("kernel.panel_pipeline.scoring")


# ── Task chain ────────────────────────────────────────────────────────────────

class LoadScorerTask(Task):
    """Load the PanelScorer artifact from config. Cache on ctx for reuse."""

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("enabled", False):
            log.debug("LoadScorerTask: panel scoring disabled — skipping chain")
            return False

        # Scorer may have been pre-loaded by the adapter (live runner / LEAN)
        scorer = getattr(ctx, "_panel_scorer", None)
        if scorer is not None:
            return

        artifact_path = panel_cfg.get("artifact_path")
        if not artifact_path:
            log.warning("LoadScorerTask: panel_scoring.enabled but no artifact_path — skipping")
            return False
        p = Path(artifact_path)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p
        try:
            ctx._panel_scorer = PanelScorer.load(p)  # noqa: SLF001
        except Exception as exc:
            log.error("LoadScorerTask: failed to load %s — %s", p, exc)
            return False
        log.info("LoadScorerTask: loaded artifact (features=%d)",
                 len(ctx._panel_scorer.feature_cols))


class BuildFeatureMatrixTask(Task):
    """Pick today's row per candidate + held ticker into a single feature matrix.

    Held positions are scored alongside candidates so rotation can compare
    them on the same cross-sectional panel scale.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        if not ctx.candidates and not ctx.holdings:
            # No work to do — but DON'T short-circuit the chain so
            # downstream calibration / ngboost loaders can still
            # initialize for next bar's use.
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        scorer: PanelScorer = getattr(ctx, "_panel_scorer", None)
        if scorer is None:
            ctx._panel_matrix = None  # noqa: SLF001
            return None

        feature_frames = getattr(ctx, "_panel_feature_frames", None)
        factor_frames  = getattr(ctx, "_panel_factor_frames", None)
        if feature_frames is None:
            log.warning("BuildFeatureMatrixTask: ctx has no _panel_feature_frames "
                        "(adapter must populate) — leaving matrix unset; "
                        "downstream tasks will no-op individually")
            # Audit #39: don't kill the chain — let LoadGlobalCalibration /
            # LoadNGBoost still initialize. ApplyScoresTask checks the matrix
            # itself and no-ops cleanly when None/empty.
            ctx._panel_matrix = None  # noqa: SLF001
            return None

        today = ctx.today
        today_ts = pd.Timestamp(today if isinstance(today, datetime.date) else today)
        target_tickers = {c.ticker for c in ctx.candidates} | set(ctx.holdings.keys())
        ff_subset = {t: feature_frames[t] for t in target_tickers if t in feature_frames}
        fac_subset = None
        if factor_frames is not None:
            fac_subset = {t: factor_frames[t] for t in target_tickers if t in factor_frames}

        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        nan_prone = list(panel_cfg.get("nan_prone_cols", []))

        X = build_inference_matrix(
            ff_subset, fac_subset, today_ts,
            feature_cols=scorer.feature_cols,
            nan_prone_cols=nan_prone,
        )
        if X.empty:
            log.warning("BuildFeatureMatrixTask: empty inference matrix")
            ctx._panel_matrix = None  # noqa: SLF001
            return None
        ctx._panel_matrix = X  # noqa: SLF001
        log.debug("BuildFeatureMatrixTask: matrix %s", X.shape)


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
            # Audit P-21: previously `return False` short-circuited the
            # rest of the chain (VetoWeak, LoadNGBoost, ApplyNGBoost,
            # LoadGlobalCal, ApplyGlobalCal, ApplyKellySizing). That
            # meant Kelly target stayed stale on empty-matrix bars and
            # downstream sizing used last-bar Kelly numbers. Each of
            # those tasks already has its own None/empty guard, so we
            # return None (continue) and let them no-op individually.
            return None
        scores: pd.Series = scorer.score(X)

        n_cand_scored = 0
        for cand in ctx.candidates:
            v = scores.get(cand.ticker)
            if v is None or pd.isna(v):
                continue
            cand.rank_score  = float(v)
            cand.panel_score = float(v)
            n_cand_scored += 1

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
    """Drop candidates whose panel_score is below `buy_floor`.

    No-op when buy_floor is unset or <= -inf. Candidates without a panel
    score (e.g. missing features) are kept — RankingJob blends rs_score in.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        # Audit fix VETO-EMPTY-CANDS (Round 2 deep audit, 2026-04-25):
        # pre-fix returned False when ctx.candidates was empty, which
        # short-circuits the rest of PanelScoringJob's chain
        # (LoadNGBoost → ApplyNGBoost → ApplyGlobalCalibration →
        # ApplyKellySizing). On a "holdings-only bar" (zero candidates,
        # non-empty ctx.holdings), holdings ALSO need their panel scores
        # calibrated and Kelly-sized for downstream rotation/sell logic.
        # Pre-fix the chain stopped, leaving holding rank_score / mu /
        # sigma / kelly_target_pct unset. Now: return None (continue)
        # so the holding-side branches of the next tasks still fire.
        if not ctx.candidates:
            return None
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        floor     = panel_cfg.get("buy_floor")
        if floor is None:
            return
        floor = float(floor)

        kept: list = []
        dropped = 0
        for cand in ctx.candidates:
            ps = cand.panel_score
            # Audit P-22: differentiate three states:
            #   ps is None         → no panel score available (e.g. ticker
            #                        not in matrix); KEEP — rs_score still
            #                        ranks it (matches original behavior).
            #   ps is NaN          → panel scoring ran but produced NaN
            #                        (missing features, model crash) →
            #                        DROP. Pre-fix this slipped through
            #                        because NaN < float is False.
            #   ps < floor         → DROP (the documented veto).
            if ps is None:
                kept.append(cand)
                continue
            if pd.isna(ps) or ps < floor:
                dropped += 1
                continue
            kept.append(cand)

        # Audit #43: keep the counter present even when nothing was dropped
        # so downstream readers don't see KeyError on ctx.counters["panel_vetoed"].
        ctx.counters["panel_vetoed"] = ctx.counters.get("panel_vetoed", 0) + dropped
        if dropped:
            ctx.candidates = kept
            log.info("VetoWeakBuysTask: dropped %d candidate(s) below panel_score=%.3f",
                     dropped, floor)


# ── Global calibration (Item #2 — optional) ───────────────────────────────────

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

        strategy_dir = ctx.config.get("_strategy_dir")

        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() or not strategy_dir else Path(strategy_dir) / p

        from training_panel.global_calibrator import GlobalPanelCalibration  # noqa: PLC0415

        # Pooled calibrator — always attempted (acts as fallback).
        if getattr(ctx, "_global_calibrator", None) is None:
            pooled_path = _resolve(Path(gc_cfg.get(
                "artifact_path", "artifacts/panel-rank-calibration.json",
            )))
            try:
                ctx._global_calibrator = GlobalPanelCalibration.load(pooled_path)  # noqa: SLF001
                log.info("LoadGlobalCalibrationTask: loaded pooled (pool_IC=%s)",
                         ctx._global_calibrator.metadata.get("pool_ic"))
            except Exception as exc:
                log.warning("LoadGlobalCalibrationTask: pooled load %s failed — %s",
                            pooled_path, exc)
                ctx._global_calibrator = None  # noqa: SLF001

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
                loaded[regime] = GlobalPanelCalibration.load(p)
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
            return

        n_cand = 0
        for c in ctx.candidates:
            if c.panel_score is None or c.panel_score != c.panel_score:
                continue
            prob = cal.calibrate_probability(c.panel_score)
            er   = cal.expected_return(c.panel_score)
            c.rank_score      = float(prob)
            c.expected_return = float(er)
            n_cand += 1

        n_held = 0
        for ticker, hs in ctx.holdings.items():
            ps = getattr(hs, "panel_score", None)
            if ps is None or ps != ps:
                continue
            hs.rank_score      = cal.calibrate_probability(ps)
            hs.expected_return = cal.expected_return(ps)
            n_held += 1

        log.info(
            "ApplyGlobalCalibrationTask: calibrated %d/%d candidates, %d/%d holdings",
            n_cand, len(ctx.candidates), n_held, len(ctx.holdings),
        )


# ── NGBoost tasks (Stage 2 — optional) ────────────────────────────────────────

class LoadNGBoostTask(Task):
    """Load the NGBoostHead artifact when enabled.

    No-op when the `ngboost.enabled` sub-flag is false. Failure to load is
    logged and downstream NGBoost tasks short-circuit — the LTR-only path
    keeps working.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        ngb_cfg   = panel_cfg.get("ngboost", {})
        if not ngb_cfg.get("enabled", False):
            return

        head = getattr(ctx, "_ngboost_head", None)
        if head is not None:
            return

        artifact = ngb_cfg.get("artifact_path", "artifacts/ngboost-head.json")
        p = Path(artifact)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p

        try:
            from training_panel.ngboost_head import NGBoostHead  # noqa: PLC0415
            ctx._ngboost_head = NGBoostHead.load(p)  # noqa: SLF001
        except Exception as exc:
            log.warning("LoadNGBoostTask: failed to load %s — %s", p, exc)
            ctx._ngboost_head = None  # noqa: SLF001
            return
        log.info("LoadNGBoostTask: loaded ngboost head (features=%d)",
                 len(ctx._ngboost_head.feature_cols))


class ApplyNGBoostTask(Task):
    """Apply NGBoost μ,σ predictions on top of the LTR panel scoring.

    - Writes `mu` + `sigma` onto every candidate / holding for which a
      prediction is available.
    - When `ngboost.score_mode == "mu_minus_lambda_sigma"` (the default
      when ngboost is enabled), overwrites `rank_score` AND `panel_score`
      with `μ − λ·σ` so downstream ranking + rotation use the combined
      signal. Set score_mode = "additive" to keep the LTR rank_score
      unchanged and only populate mu/sigma for sizing.
    """

    def run(self, ctx: InferenceContext) -> bool | None:
        ngb_cfg = (ctx.config.get("ranking", {})
                             .get("panel_scoring", {})
                             .get("ngboost", {}))
        if not ngb_cfg.get("enabled", False):
            return
        head = getattr(ctx, "_ngboost_head", None)
        X    = getattr(ctx, "_panel_matrix", None)
        if head is None or X is None or X.empty:
            return

        # Audit N-25 (2026-04-25): pre-fix this returned early if ANY
        # head.feature_cols was missing from X — one missing column killed
        # the entire bar's NGBoost output. Post-fix, fill missing columns
        # with 0.0 (z-scored "neutral") and warn loudly so the operator
        # knows the prediction is using a partial feature set.
        missing = [c for c in head.feature_cols if c not in X.columns]
        if missing:
            log.warning(
                "ApplyNGBoostTask: feature matrix missing cols %s — filling "
                "with 0.0 (z-scored neutral). Predictions partial.",
                missing,
            )
            X = X.copy()
            for c in missing:
                X[c] = 0.0

        try:
            dist = head.predict_distribution(X)
        except Exception as exc:
            log.warning("ApplyNGBoostTask: predict failed — %s", exc)
            return

        lambda_sigma = float(ngb_cfg.get("lambda_sigma", 1.0))
        score_mode   = str(ngb_cfg.get("score_mode", "mu_minus_lambda_sigma"))
        override     = (score_mode == "mu_minus_lambda_sigma")

        mu    = dist["mu"]
        sigma = dist["sigma"]
        combined = mu - lambda_sigma * sigma

        # Audit N-5 / N-25 (2026-04-25): after the NGBoost head's NaN
        # passthrough, predict_distribution returns NaN at rows it couldn't
        # score (NaN/inf input features). Skip those tickers cleanly so
        # downstream sizers / rotators don't compute Kelly = μ/σ² on NaN.
        for cand in ctx.candidates:
            if cand.ticker not in mu.index:
                continue
            mu_val    = mu.loc[cand.ticker]
            sigma_val = sigma.loc[cand.ticker]
            if pd.isna(mu_val) or pd.isna(sigma_val):
                continue
            cand.mu    = float(mu_val)
            cand.sigma = float(sigma_val)
            if override:
                v = float(combined.loc[cand.ticker])
                cand.rank_score  = v
                cand.panel_score = v

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

        log.info("ApplyNGBoostTask: mode=%s  λ=%.2f  n_cands=%d  n_holdings=%d",
                 score_mode, lambda_sigma, len(ctx.candidates), len(ctx.holdings))


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

        def _kelly(obj):
            return kelly_target_pct(
                getattr(obj, "mu",    None),
                getattr(obj, "sigma", None),
                max_pct           = max_pct,
                max_concentration = max_concentration,
                fractional        = fractional,
                min_edge          = min_edge,
            )

        for cand in ctx.candidates:
            cand.kelly_target_pct = _kelly(cand)
        for hs in ctx.holdings.values():
            hs.kelly_target_pct = _kelly(hs)

        # Audit summary — most informative when live.
        cand_targets = [c.kelly_target_pct for c in ctx.candidates
                         if c.kelly_target_pct]
        held_targets = [h.kelly_target_pct for h in ctx.holdings.values()
                         if h.kelly_target_pct]
        log.info(
            "ApplyKellySizingTask: fractional=%.2f max_conc=%.2f  "
            "cands=%d non-zero (avg=%.1f%%)  holdings=%d non-zero (avg=%.1f%%)",
            fractional, max_concentration,
            len(cand_targets),
            (sum(cand_targets) / len(cand_targets) * 100) if cand_targets else 0,
            len(held_targets),
            (sum(held_targets) / len(held_targets) * 100) if held_targets else 0,
        )


# ── Job ──────────────────────────────────────────────────────────────────────

class PanelScoringJob(Job):
    """Overwrite rank_score on surviving candidates with cross-sectional panel scores.

    Task chain:
      LoadScorer → BuildFeatureMatrix → ApplyScores → VetoWeakBuys
        → LoadNGBoost → ApplyNGBoost                 (no-op if ngboost.enabled is false)
        → LoadGlobalCalibration → ApplyGlobalCalibration (always-runs; see below)

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
        return [
            LoadScorerTask(),
            BuildFeatureMatrixTask(),
            ApplyScoresTask(),
            VetoWeakBuysTask(),
            LoadNGBoostTask(),
            ApplyNGBoostTask(),
            LoadGlobalCalibrationTask(),
            ApplyGlobalCalibrationTask(),
            ApplyKellySizingTask(),   # Plan C — f*=μ/σ² (no-op unless kelly_sizing.enabled)
        ]
