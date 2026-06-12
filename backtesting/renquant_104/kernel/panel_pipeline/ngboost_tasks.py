"""NGBoost head + sentiment-gate task cluster.

EXTRACTED 2026-06-12 from job_panel_scoring.py (eng plan S2 item 5,
god-file decomposition slice 2; behavior-identical move gated by the
DRPH replay corpus with pre-change baselines). Symbols re-exported from
job_panel_scoring for back-compat. _fail_closed_ngboost stays in
calibration.py (moved there by slice 1, #319); imported lazily below.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.panel_pipeline.calibration import _fail_closed_ngboost
from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

log = logging.getLogger("panel_pipeline.ngboost")


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

