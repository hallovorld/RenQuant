"""Global panel-rank calibration task cluster.

EXTRACTED 2026-06-12 from job_panel_scoring.py (eng plan S2 item 5,
god-file decomposition; behavior-identical move gated by the DRPH
replay corpus — sim_2026-06-10 + sim_2026-06-11). No logic change:
every symbol is re-exported from job_panel_scoring for back-compat.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task
log = logging.getLogger("panel_pipeline.calibration")


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


def _submit_gate_verdict(ctx: Any, *, gate: str, reason: str, inputs: dict) -> None:
    # Lazy bridge to the module-of-origin helper (job_panel_scoring) —
    # avoids a circular import; call-time the module is fully loaded.
    from kernel.panel_pipeline.job_panel_scoring import (  # noqa: PLC0415
        _submit_gate_verdict as _impl,
    )

    _impl(ctx, gate=gate, reason=reason, inputs=inputs)


def _fail_closed_missing_calibrator(ctx: InferenceContext, reason: str) -> None:
    """Block buy/QP when an enabled calibrator cannot be used.

    Preflight should catch this before a daily/full run, but runtime must still
    fail closed so a missing calibrator never silently reverts to raw panel
    scores. Exits already emitted earlier in the pipeline are left intact.
    """
    ctx._calibrator_contract_failed = True  # noqa: SLF001
    ctx.skip_buys = True
    _submit_gate_verdict(ctx, gate="calibrator_fail_closed", reason=reason,
                         inputs={})
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
            if p.is_absolute() or not strategy_dir:
                return p
            from kernel.artifact_resolver import locate_artifact  # noqa: PLC0415

            return locate_artifact(p, strategy_dir=Path(strategy_dir))

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

        # Track A (2026-06-02): explicit per-regime calibrator dict at
        # ranking.panel_scoring.calibrator_per_regime: {regime: path}. Unlike
        # the regime_conditional pattern (which discovers files via a glob
        # template and silently falls back to pooled on miss), this is an
        # opt-in explicit map — every regime listed MUST resolve to a loadable
        # artifact or LoadScorerTask fails closed. Regimes NOT listed fall
        # back to the pooled calibrator at apply time via the existing
        # ApplyGlobalCalibrationTask `regime_map.get(ctx.regime) or pooled`
        # path. Back-compat: when the key is absent, behavior is unchanged.
        # See doc/research/2026-06-02-bull-calm-signal-recovery-plan.md.
        panel_cfg = (ctx.config.get("ranking", {})
                              .get("panel_scoring", {}))
        per_regime_cfg = panel_cfg.get("calibrator_per_regime")
        if per_regime_cfg:
            if not isinstance(per_regime_cfg, dict):
                raise ValueError(
                    "LoadGlobalCalibrationTask: calibrator_per_regime must be "
                    f"a dict[regime, path], got {type(per_regime_cfg).__name__}"
                )
            valid_regimes = {"BULL_CALM", "BULL_VOLATILE", "BEAR", "CHOPPY"}
            bad = set(per_regime_cfg) - valid_regimes
            if bad:
                raise ValueError(
                    "LoadGlobalCalibrationTask: calibrator_per_regime has "
                    f"invalid regime keys {sorted(bad)}; valid keys are "
                    f"{sorted(valid_regimes)}"
                )
            loaded_pr: dict[str, GlobalPanelCalibration] = dict(
                getattr(ctx, "_regime_calibrators", None) or {}
            )
            for regime, raw_path in per_regime_cfg.items():
                if not raw_path:
                    raise ValueError(
                        f"LoadGlobalCalibrationTask: calibrator_per_regime[{regime}] "
                        "is empty; either remove the key or set a path."
                    )
                # Per-regime path cache: avoid reloading identical artifact
                # on every bar in long-running adapters (SimAdapter, RunnerAdapter).
                # ctx._regime_calibrator_paths records the resolved path that
                # produced each entry; we reload only when path changes.
                p = _resolve(Path(raw_path))
                path_cache = getattr(ctx, "_regime_calibrator_paths", None)
                if path_cache is None:
                    path_cache = {}
                    ctx._regime_calibrator_paths = path_cache  # noqa: SLF001
                if path_cache.get(regime) == str(p) and regime in loaded_pr:
                    continue
                if not p.exists():
                    raise FileNotFoundError(
                        f"LoadGlobalCalibrationTask: calibrator_per_regime[{regime}] "
                        f"artifact not found: {p}. Per-regime calibrators are "
                        "opt-in and must exist when configured (no silent fallback)."
                    )
                cal = GlobalPanelCalibration.load(p)
                _assert_calibrator_matches_scorer(
                    ctx, cal, p, strict=strict_match,
                )
                loaded_pr[regime] = cal
                path_cache[regime] = str(p)
                log.info(
                    "LoadGlobalCalibrationTask: regime=%s loaded explicit "
                    "calibrator from %s (pool_ic=%s)",
                    regime, p, cal.metadata.get("pool_ic"),
                )
            ctx._regime_calibrators = loaded_pr  # noqa: SLF001

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
    ctx.skip_buys = True
    ctx.candidates = []
    _submit_gate_verdict(ctx, gate="ngboost_fail_closed", reason=reason,
                         inputs={"detail": str(detail)[:120]})
    if hasattr(ctx, "counters"):
        ctx.counters["ngb_fail_closed"] = (
            ctx.counters.get("ngb_fail_closed", 0) + 1
        )
    log.error("NGBoost fail-closed: %s%s", reason, f" ({detail})" if detail else "")
    return False


