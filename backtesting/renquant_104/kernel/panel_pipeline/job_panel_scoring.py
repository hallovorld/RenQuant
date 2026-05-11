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

        # 2026-04-28 self-audit: config / model consistency check.
        # Invariant: a fingerprint mismatch must — by default — prevent
        # panel scoring from running, because the alternative is silent
        # miscalibrated trades. Three incidents in 24h proved log-only
        # isn't enough (operators don't tail logs every bar).
        # Set ranking.panel_scoring.strict_config_consistency=false to
        # downgrade to log-only (only for staged migrations).
        # Backwards-compat: artifacts without a stored fingerprint pass
        # with WARNING (stamped on next retrain).
        strict = bool(panel_cfg.get("strict_config_consistency", True))
        try:
            from kernel.config_consistency import (  # noqa: PLC0415
                assert_consistent, ConfigModelMismatch,
            )
            import json as _j  # noqa: PLC0415
            artifact_meta = _j.loads(p.read_text())
            try:
                assert_consistent(
                    ctx.config, artifact_meta,
                    artifact_label=str(p.name),
                    strict=strict,
                )
            except ConfigModelMismatch as e:
                log.error("LoadScorerTask: %s", e)
                # strict=True ⇒ skip panel scoring this bar. Selection
                # loop will fall back to per-ticker scores or no-op.
                return False
        except ConfigModelMismatch:
            raise   # bubble unhandled (defensive — shouldn't reach here)
        except Exception as exc:
            log.warning("LoadScorerTask: consistency check failed: %s", exc)


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
                return None
            tickers = list(X.index)   # candidates + holdings already de-duped
            rows = {}
            for t in tickers:
                ohlcv_t = ohlcv_dict.get(t)
                if ohlcv_t is None or len(ohlcv_t) < 70:
                    continue
                feats = compute_alpha158_at(ohlcv_t, today)
                if feats:
                    rows[t] = feats
            if not rows:
                log.warning("ApplyScoresTask[alpha158]: 0/%d tickers had "
                             "sufficient history for alpha158", len(tickers))
                return None
            X = pd.DataFrame.from_dict(rows, orient="index")
            if scorer_kind == "panel_linear":
                # PanelLinearScorer.score_raw applies stored ZScoreNorm + Fillna + Clip
                scores: pd.Series = scorer.score_raw(X)
                log.info("ApplyScoresTask[panel_linear]: scored %d tickers via "
                         "alpha158 + score_raw", len(rows))
            else:
                # XGBoost panel_ltr_xgboost: artifact may have additional fund features
                # (earnings_yield, book_to_price, etc.) beyond alpha158. If so, look them up
                # from the daily SEC fundamentals panel (point-in-time).
                fund_cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
                needs_fund = any(fc in scorer.feature_cols for fc in fund_cols)
                if needs_fund:
                    import os                                                       # noqa: PLC0415
                    from pathlib import Path                                         # noqa: PLC0415
                    import numpy as np                                              # noqa: PLC0415
                    repo = Path(__file__).resolve().parents[4]
                    fp = repo / "data" / "sec_fundamentals_daily.parquet"
                    if fp.exists():
                        fund_panel = pd.read_parquet(fp)
                        fund_panel["date"] = pd.to_datetime(fund_panel["date"])
                        snap = fund_panel[fund_panel["date"] <= pd.Timestamp(today)] \
                            .sort_values("date").groupby("ticker").tail(1)
                        # ── 2026-05-09 BUG #1 fix: match training-time imputation chain ──
                        # Training (build_alpha158_fund_panel.py):
                        #     1) per-date cross-sectional median fillna
                        #     2) final fillna(0) only if median was also NaN
                        # Pre-fix runtime: NaN→0 directly (skipped step 1).
                        # Result: 33% of (ticker,date) cells with NaN raw fund
                        # values got DIFFERENT imputed values at train vs runtime
                        # → SHAP shows fund features give ~constant contribution
                        # at runtime (all map to same z-score branch).
                        # Fix: replicate the median-then-zero chain.
                        ticker_raw_fund = {}
                        for t in rows.keys():
                            row = snap[snap["ticker"] == t]
                            if len(row):
                                ticker_raw_fund[t] = {
                                    fc: (float(row[fc].iloc[0])
                                          if (fc in row.columns and pd.notna(row[fc].iloc[0]))
                                          else None)
                                    for fc in fund_cols
                                }
                            else:
                                ticker_raw_fund[t] = {fc: None for fc in fund_cols}
                        # Step 1: cross-sectional median of CURRENT day's
                        # candidates (mirror training's per-date median)
                        cs_median = {}
                        for fc in fund_cols:
                            vals = [v for v in (ticker_raw_fund[t][fc] for t in rows) if v is not None]
                            cs_median[fc] = float(np.median(vals)) if vals else 0.0
                        # Apply median where ticker's value is missing; else use real value
                        n_real, n_imputed = 0, 0
                        for t in rows:
                            for fc in fund_cols:
                                v = ticker_raw_fund[t][fc]
                                if v is None:
                                    rows[t][fc] = cs_median[fc]
                                    n_imputed += 1
                                else:
                                    rows[t][fc] = v
                                    n_real += 1
                        log.info(
                            "ApplyScoresTask[panel_ltr_xgboost]: merged 5 fund features "
                            "from %s (real=%d imputed_xs_median=%d)",
                            fp.name, n_real, n_imputed,
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
                    import numpy as np         # noqa: PLC0415
                    repo = Path(__file__).resolve().parents[4]
                    earn_dir = repo / "data" / "earnings_surprise"
                    today_ts = pd.Timestamp(today)
                    DECAY = 60   # Bernard-Thomas 1989 drift window

                if needs_pead:
                    # Per-ticker: pull most-recent earnings ≤ today
                    surprises_today = {}  # ticker → surprise_pct (for x-sec rank)
                    n_no_data = 0; n_no_prior = 0; n_out_of_window = 0
                    for t in list(rows.keys()):
                        ep = earn_dir / f"{t}.parquet"
                        if not ep.exists():
                            n_no_data += 1
                            for pc in pead_cols: rows[t].setdefault(pc, 0.0)
                            continue
                        earn = pd.read_parquet(ep).reset_index()
                        earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
                        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
                        # Defensive: sort by earnings_date ASC. Parquet doesn't
                        # guarantee row-order preservation across re-reads, and
                        # we rely on iloc[-1] being most-recent below.
                        earn = earn.sort_values("earnings_date").reset_index(drop=True)
                        prior = earn[earn["earnings_date"] <= today_ts]
                        if len(prior) == 0:
                            n_no_prior += 1
                            for pc in pead_cols: rows[t].setdefault(pc, 0.0)
                            continue
                        last = prior.iloc[-1]
                        days_since = int((today_ts - last["earnings_date"]).days)
                        if days_since > DECAY or days_since < 0:
                            n_out_of_window += 1
                            for pc in pead_cols: rows[t].setdefault(pc, 0.0)
                            continue
                        decay = max(0.0, 1.0 - days_since / DECAY)
                        surprise = float(last["surprise_pct"]) if pd.notna(last["surprise_pct"]) else 0.0
                        rows[t]["days_since_earnings"] = float(days_since)
                        rows[t]["pead_signal"]         = surprise * decay
                        surprises_today[t] = surprise
                    # Cross-sectional quintile rank of today's surprise across all
                    # tickers in the snap (matches build-time per-date rank logic).
                    if surprises_today:
                        ranks = pd.Series(surprises_today).rank(pct=True)
                        for t, r in ranks.items():
                            rows[t]["pead_quintile_rank"] = float(r)
                    # Tickers without active surprise → zero quintile rank
                    for t in rows:
                        rows[t].setdefault("pead_quintile_rank", 0.0)
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: computed 3 PEAD features "
                             "today=%s (%d/%d tickers active in 60d window; "
                             "no_data=%d no_prior=%d out_of_window=%d)",
                             today_ts.date().isoformat(),
                             len(surprises_today), len(rows),
                             n_no_data, n_no_prior, n_out_of_window)

                # ── SUE features (E49 promotion 2026-05-09): SUE +
                # surprise_momentum + surprise_streak. Same earnings_surprise
                # data source as PEAD; computed independently because they
                # use multiple historical events (4Q std denominator for SUE,
                # prior-event diff for momentum, run-length for streak)
                # whereas PEAD only uses the most-recent event.
                # Foster-Olsen-Shevlin 1984 + Bernard-Thomas 60d decay.
                if needs_sue:
                    SUE_WINDOW = 4
                    n_sue_active = 0; n_sue_no_data = 0; n_sue_oow = 0
                    for t in list(rows.keys()):
                        ep = earn_dir / f"{t}.parquet"
                        if not ep.exists():
                            n_sue_no_data += 1
                            for sc in sue_cols: rows[t].setdefault(sc, 0.0)
                            continue
                        earn = pd.read_parquet(ep).reset_index()
                        earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
                        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
                        earn = earn.sort_values("earnings_date").reset_index(drop=True)
                        prior = earn[earn["earnings_date"] <= today_ts]
                        if len(prior) == 0:
                            for sc in sue_cols: rows[t].setdefault(sc, 0.0)
                            continue
                        last = prior.iloc[-1]
                        days_since = int((today_ts - last["earnings_date"]).days)
                        if days_since > DECAY or days_since < 0:
                            n_sue_oow += 1
                            for sc in sue_cols: rows[t].setdefault(sc, 0.0)
                            continue
                        decay = max(0.0, 1.0 - days_since / DECAY)
                        s = prior["surprise_pct"].astype(float)
                        # SUE: most-recent surprise / std(prior 4 quarters)
                        if len(s) >= 2:
                            denom_window = s.iloc[max(0, len(s)-1-SUE_WINDOW):len(s)-1]
                            denom = float(denom_window.std()) if len(denom_window) >= 2 else 0.0
                            sue = float(s.iloc[-1]) / max(denom, 1e-6)
                            sue = max(min(sue, 5.0), -5.0)   # clip
                        else:
                            sue = 0.0
                        # Momentum: surprise_t - surprise_(t-1)
                        mom = float(s.iloc[-1] - s.iloc[-2]) if len(s) >= 2 else 0.0
                        # Streak: signed consecutive same-direction count
                        streak = 0
                        cur_sign = 0
                        for v in s:
                            sgn = 1 if v > 0 else (-1 if v < 0 else 0)
                            if sgn == 0 or sgn != cur_sign:
                                streak = sgn; cur_sign = sgn
                            else:
                                streak += sgn
                        rows[t]["sue_signal"]        = sue * decay
                        rows[t]["surprise_momentum"] = mom * decay
                        rows[t]["surprise_streak"]   = float(streak) * decay
                        n_sue_active += 1
                    for t in rows:
                        for sc in sue_cols: rows[t].setdefault(sc, 0.0)
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: computed 3 SUE features "
                             "today=%s (%d/%d tickers active; no_data=%d out_of_window=%d)",
                             today_ts.date().isoformat(), n_sue_active, len(rows),
                             n_sue_no_data, n_sue_oow)

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

                # Apply artifact-stored normalization chain if available (new artifacts
                # store per-feature mean/std for raw→normalized inference parity)
                meta = getattr(scorer, "metadata", {}) or {}
                fmeans = meta.get("feature_means")
                fstds  = meta.get("feature_stds")
                if fmeans is not None and fstds is not None and len(fmeans) == len(scorer.feature_cols):
                    import numpy as np                                               # noqa: PLC0415
                    mu = np.asarray(fmeans); sd = np.asarray(fstds) + 1e-9
                    Xv = X_aligned.fillna(0).values.astype(float)
                    Xn = ((Xv - mu) / sd).clip(-5, 5)
                    X_aligned = pd.DataFrame(Xn, index=X_aligned.index, columns=X_aligned.columns)
                    log.info("ApplyScoresTask[panel_ltr_xgboost]: applied artifact normalization "
                             "(mean/std for %d features)", len(fmeans))

                scores: pd.Series = scorer.score(X_aligned)
                log.info("ApplyScoresTask[panel_ltr_xgboost]: scored %d tickers via alpha158%s",
                         len(rows), "+fund" if needs_fund else "")
        else:
            scores: pd.Series = scorer.score(X)

        n_cand_scored = 0
        for cand in ctx.candidates:
            v = scores.get(cand.ticker)
            if v is None or pd.isna(v):
                continue
            cand.rank_score  = float(v)
            cand.panel_score = float(v)
            n_cand_scored += 1

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
        if isinstance(raw_floor, str) and raw_floor == "adaptive_mean_std_cap":
            cap     = float(panel_cfg.get("buy_floor_adaptive_cap", 0.30))
            min_fl  = float(panel_cfg.get("buy_floor_min",          0.20))
            scores = [getattr(c, "rank_score", None) for c in ctx.candidates]
            scores = [float(s) for s in scores
                       if s is not None and not pd.isna(s)]
            if len(scores) >= 2:
                import statistics as _stats  # noqa: PLC0415
                mean_s = _stats.fmean(scores)
                std_s  = _stats.stdev(scores)
                adaptive = mean_s + std_s
                # Clamp mean+std to [min_fl, cap].
                floor = min(max(min_fl, adaptive), cap)
                floor_label = (
                    f"min(max(min={min_fl:.2f}, mean+std={adaptive:.3f}), "
                    f"cap={cap:.2f}) = {floor:.3f}  (n={len(scores)})"
                )
            else:
                # Insufficient sample — collapse to the safe upper end.
                floor = cap
                floor_label = f"{cap:.3f} (cap; n<2 for stats)"
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
                "artifact_path", "artifacts/prod/panel-rank-calibration.json",
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

        artifact = ngb_cfg.get("artifact_path", "artifacts/prod/ngboost-head.alpha158_fund.json")
        p = Path(artifact)
        if not p.is_absolute():
            strategy_dir = ctx.config.get("_strategy_dir")
            if strategy_dir:
                p = Path(strategy_dir) / p

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
            log.warning("LoadNGBoostTask: failed to load %s — %s", p, exc)
            ctx._ngboost_head = None  # noqa: SLF001
            return
        head_kind = type(ctx._ngboost_head).__name__
        log.info("LoadNGBoostTask: loaded %s (features=%d)",
                 head_kind, len(ctx._ngboost_head.feature_cols))


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
            if pct_miss > drift_thr:
                log.error(
                    "ApplyNGBoostTask: %d/%d (%.1f%%) feature cols MISSING from "
                    "inference panel — exceeds max_feature_drift_pct=%.2f. "
                    "NGBoost head was likely trained with features that the "
                    "current panel pipeline no longer produces (e.g. macro "
                    "block disabled after head was trained). FAIL-SAFE: "
                    "writing NaN μ/σ on every candidate + holding so Gate B "
                    "rejects them all + clearing ctx.candidates to block "
                    "buys outright. RETRAIN: `python scripts/train_104.py "
                    "--skip-baseline --skip-recalibrate --force`. First 10 "
                    "missing: %s",
                    n_missing, n_total, pct_miss * 100, drift_thr,
                    missing[:10],
                )
                # CRIT-1 fix (2026-04-28 self-audit): pre-fix, returning here
                # left every cand.mu / cand.sigma as None. Gate B's
                # `_gate_b_edge_sharpe` PASSES None-μ/σ ("no NGBoost → no
                # signal to gate; pass") so drift hard-fail silently
                # promoted ALL candidates through the quality floor — the
                # opposite of fail-safe. Now: stamp NaN so Gate B rejects
                # ("mu_nan" reason) AND clear candidate list to block buys.
                # Holdings keep their None μ/σ so SellGateB also no-ops
                # (path rules continue to govern exits).
                _nan = float("nan")
                for cand in ctx.candidates:
                    cand.mu    = _nan
                    cand.sigma = _nan
                ctx.candidates = []   # block all buys this bar
                if hasattr(ctx, "counters"):
                    ctx.counters["ngb_drift_fail"] = (
                        ctx.counters.get("ngb_drift_fail", 0) + 1
                    )
                return
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

        # R-02 (2026-05-11): Moskowitz-Ooi-Pedersen 2012 volatility-
        # targeting. Optional — disabled by default. Scales the global
        # gross-exposure cap by `clip(target_vol / spy_realized_vol_60d,
        # floor, ceiling)` so the portfolio de-leverages in vol-spike
        # regimes and re-leverages in calm ones. Composes multiplicatively
        # with the R-04 DD scaling — both shrink `max_pct` independently.
        vt_cfg = kelly_cfg.get("vol_target") or {}
        if vt_cfg.get("enabled", False):
            from kernel.vol_target import compute_vol_target_scale  # noqa: PLC0415
            _vt_scale = compute_vol_target_scale(
                ctx.spy_returns,
                target_vol  = float(vt_cfg.get("target_vol",  0.15)),
                window_days = int  (vt_cfg.get("window_days", 60)),
                floor       = float(vt_cfg.get("floor",       0.30)),
                ceiling     = float(vt_cfg.get("ceiling",     1.50)),
            )
            if _vt_scale != 1.0:
                log.info(
                    "ApplyKellySizing: vol-target max_pct %.4f -> %.4f "
                    "(scale=%.3f target_vol=%.3f window=%dd)",
                    max_pct, max_pct * _vt_scale, _vt_scale,
                    float(vt_cfg.get("target_vol", 0.15)),
                    int(vt_cfg.get("window_days", 60)),
                )
            max_pct *= _vt_scale

        # R-04 (2026-05-11): Grossman-Zhou 1993 drawdown-conditioned Kelly
        # scaling. Optional config block — disabled by default so golden
        # behaviour is preserved. Tightens the SINGLE-NAME cap as the
        # portfolio drawdown deepens; the existing DrawdownCircuitTask
        # halts buys hard, and DrawdownRebalanceTask trims at the basket
        # level. This scaling sits between them: rather than wait for
        # halt_pct (binary), shrink new bet sizes continuously so the
        # portfolio de-risks gracefully.
        dd_cfg = kelly_cfg.get("drawdown_scaling") or {}
        if dd_cfg.get("enabled", False):
            from kernel.kelly import compute_kelly_dd_scale  # noqa: PLC0415
            from kernel.pipeline.task_drawdown_rebalance import (  # noqa: PLC0415
                compute_portfolio_drawdown,
            )
            _dd = compute_portfolio_drawdown(ctx.hwm, ctx.portfolio_value)
            _dd_max   = float(dd_cfg.get("dd_max", 0.30))
            _dd_pow   = float(dd_cfg.get("exponent", 1.0))
            _dd_scale = compute_kelly_dd_scale(
                _dd, dd_max=_dd_max, exponent=_dd_pow,
            )
            if _dd_scale < 1.0:
                log.info(
                    "ApplyKellySizing: DD-scaling max_pct %.4f -> %.4f "
                    "(drawdown=%.1f%% dd_max=%.1f%% exp=%.1f scale=%.3f)",
                    max_pct, max_pct * _dd_scale,
                    _dd * 100, _dd_max * 100, _dd_pow, _dd_scale,
                )
            max_pct *= _dd_scale

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
        # Lazy import — avoids a circular import that fires when
        # job_panel_scoring is imported by InferencePipeline init.
        from kernel.panel_pipeline.task_quality_floor import (  # noqa: PLC0415
            QualityFloorTask,
        )
        return [
            LoadScorerTask(),
            BuildFeatureMatrixTask(),
            ApplyScoresTask(),
            LoadNGBoostTask(),
            ApplyNGBoostTask(),
            LoadGlobalCalibrationTask(),
            ApplyGlobalCalibrationTask(),
            # 2026-05-03 P0 fix: VetoWeakBuysTask MOVED to here (was right
            # after ApplyScoresTask). Veto must compare against calibrated
            # rank_score, not raw XGB margin. See VetoWeakBuysTask
            # docstring for the production incident this resolves.
            VetoWeakBuysTask(),
            ApplyKellySizingTask(),   # Plan C — f*=μ/σ² (no-op unless kelly_sizing.enabled)
            # Buy-logic redesign Stage 0 (2026-04-26): quality gates
            # filter weak-signal candidates AFTER all scoring + sizing.
            # All gates default OFF — bit-for-bit parity preserved.
            # See doc/components/buy-logic-design.md for theory.
            QualityFloorTask(),
        ]
