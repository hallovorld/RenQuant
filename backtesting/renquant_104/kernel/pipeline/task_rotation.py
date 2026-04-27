"""Rotation tasks — swap held positions for stronger candidates.

Three tasks compose RotationJob:

  BuildPairsTask     gather scores + expected returns, run kernel.rotation
                     and emit a structured decision-tree log per pair
                     considered (whether or not it survives)
  ValidatePairsTask  re-check wash-sale + sector + correlation guards on the
                     virtual post-swap holdings set
  EmitRotationsTask  convert each surviving pair into exit + buy order;
                     prune the rotated-in ticker from ctx.ranked so
                     SelectionJob does not double-buy
"""
from __future__ import annotations

import datetime
import logging
import math

from .context  import InferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.rotation")


def _log_decision_tree(
    *,
    cand_ticker: str,
    cand_er: float,
    cand_score: float,
    held_table: list[dict],   # one row per eligible held with fields below
    threshold: float,
    txn_cost: float,
    horizon: int,
    chosen: str | None,
) -> None:
    """Emit a structured per-candidate decision log.

    held_table rows: {ticker, score, er, unreal_pct, hold_days, tax_drag,
                      raw_adv, net_adv, decision}
    decision ∈ {"swap", "below_threshold", "lt_protected", "min_hold",
                "no_score", "no_er", "used"}
    """
    log.info(
        "ROTATION_TREE  cand=%s  cand_er=%+.4f  cand_rank=%.3f  "
        "horizon=%dd  threshold=%+.4f  cost=%.4f  chosen=%s",
        cand_ticker, cand_er, cand_score, horizon, threshold, txn_cost,
        chosen or "NONE",
    )
    for row in held_table:
        log.info(
            "  ↳ held=%-5s  er=%+.4f  rank=%.3f  hold=%dd  pnl=%+.3f  "
            "tax=%.4f  raw_adv=%+.4f  net_adv=%+.4f  → %s",
            row["ticker"], row["er"], row["score"], row["hold_days"],
            row["unreal_pct"], row["tax_drag"], row["raw_adv"], row["net_adv"],
            row["decision"],
        )


class BuildPairsTask(Task):
    """Score holdings, call kernel rotation primitive, log decision tree."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.rotation import (  # noqa: PLC0415
            find_rotation_pairs, is_lt_protected, tax_drag,
        )

        rotation_cfg = ctx.config.get("rotation", {})
        if not rotation_cfg.get("enabled", False):
            return False
        if not ctx.ranked or not ctx.holdings:
            return False
        if ctx.bear_only:
            return False

        # V3 (2026-04-24): regime gate. If `rotation.enabled_regimes` is set
        # to a list, rotation fires ONLY in those regimes. Default None =
        # no regime filter (current behaviour). Use case: disable rotation
        # in BULL_VOLATILE where whipsaw dominates; keep it in
        # BULL_CALM / CHOPPY where trends give rotation a chance.
        allowed_regimes = rotation_cfg.get("enabled_regimes")
        if allowed_regimes is not None and ctx.regime not in allowed_regimes:
            log.info("RotationJob: skipped — regime=%s not in enabled_regimes=%s",
                     ctx.regime, allowed_regimes)
            return False

        threshold   = float(rotation_cfg.get("min_expected_advantage_pct", 0.03))
        horizon     = int(rotation_cfg.get("target_horizon_days", 20))
        txn_cost    = float(rotation_cfg.get("transaction_cost_pct", 0.0))
        min_hold    = int(rotation_cfg.get("min_rotation_hold_days", 30))
        lt_protect  = int(rotation_cfg.get("lt_protection_days", 30))

        # Phase 1 (2026-04-25) — score-threshold double-gate. Calibrated
        # rank_score floors:
        #   panel_buy_floor  → candidate must have rank_score >= this
        #   panel_sell_floor → held must have today rank_score <= this
        # Default None on both = disabled (current behaviour preserved).
        # Spec: "被替换的 portfolio 里的 stock 的 score 要低于一个值，
        # 进到 portfolio 的 stock 的 score 要高于一个值" — both bounds
        # apply to the calibrated probability that ApplyGlobalCalibrationTask
        # writes onto holdings + candidates.
        _bf_raw = rotation_cfg.get("panel_buy_floor")
        _sf_raw = rotation_cfg.get("panel_sell_floor")
        panel_buy_floor  = float(_bf_raw) if _bf_raw is not None else None
        panel_sell_floor = float(_sf_raw) if _sf_raw is not None else None

        # Cross-sectional panel gate — candidate panel_score must beat held
        # panel_score by this fraction. 0.0 disables the gate (default).
        panel_cfg           = ctx.config.get("ranking", {}).get("panel_scoring", {})
        panel_rot_advantage = float(panel_cfg.get("rotation_advantage", 0.0))

        # BC: Kelly-delta rotation gate. Candidate's kelly_target_pct must
        # beat held's by this fraction. Unifies swap math with the Kelly
        # decision surfaces (SelectionJob Kelly sizing, TopUpHeldTask,
        # TrimHeldTask). 0.0 disables the gate (default).
        kelly_cfg           = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        kelly_rot_advantage = float(kelly_cfg.get("rotation_advantage", 0.0))

        # Approach A — thesis-degradation rotation gate. Compares today's
        # candidate to the held's FIXED ENTRY score (not today's held
        # score, which is noisy). Swap fires only when:
        #   (1) held has degraded:     held.entry_score - held.today_score >= degradation_pct
        #   (2) cand beats the baseline: cand.today_score - held.entry_score >= uplift_pct
        # When either threshold is 0.0 that check is effectively disabled.
        # When held.entry_rank_score is None (legacy positions without
        # stamped baseline), the gate falls back to KEEP the pair.
        thesis_cfg          = ctx.config.get("ranking", {}).get("thesis_rotation", {})
        thesis_enabled      = bool(thesis_cfg.get("enabled", False))
        thesis_degradation  = float(thesis_cfg.get("degradation_pct", 0.30))
        thesis_uplift       = float(thesis_cfg.get("uplift_pct", 0.10))

        tax_cfg     = ctx.config.get("tax", {})
        st_rate     = float(tax_cfg.get("short_term_rate", 0.50))
        lt_rate     = float(tax_cfg.get("long_term_rate", 0.32))
        lt_threshold = int(tax_cfg.get("long_term_threshold_days", 365))

        # Holdings already exiting today are not eligible to rotate.
        exit_tickers = {t for t, _ in ctx.exits}

        # V2 (2026-04-24) — when `rotation.scoring_mode == "mu_minus_lambda_sigma"`
        # replace the isotonic-calibrated ER with direct NGBoost μ − λσ as the
        # decision driver. Threshold semantics stay the same (fraction units).
        # Falls back to ER on any ticker missing μ/σ so mixed panels still
        # work. λ defaults to 1.0 (balanced risk), overridable via
        # `rotation.lambda_` or the panel-wide `ranking.panel_scoring.ngboost.lambda_`.
        scoring_mode   = str(rotation_cfg.get("scoring_mode", "er"))
        lam            = float(rotation_cfg.get(
            "lambda_",
            ctx.config.get("ranking", {}).get("panel_scoring", {})
                     .get("ngboost", {}).get("lambda_", 1.0),
        ))

        # Sharpe driver floor — avoid divide-by-zero when σ ~ 0
        sharpe_sigma_floor = float(rotation_cfg.get("sharpe_sigma_floor", 1e-4))

        def _drive_score(obj) -> "float | None":
            """Pick the rotation driver score.

            - "er": calibrated expected_return (default, backward compat).
            - "mu_minus_lambda_sigma": NGBoost μ − λσ.
            - "sharpe": μ / max(σ, floor) (Barroso-Santa-Clara 2015).

            2026-04-24 unit-mismatch guard: when scoring_mode is one of the
            σ-aware modes but a row is missing μ/σ, return None instead of
            silently falling back to `expected_return`. Mixing μ−λσ for one
            side of the comparison and ER for the other made `raw_advantage`
            meaningless. Callers (the eligible-held loop) treat None as
            "skip this row" — same effect as `decision == "no_er"`.
            """
            if scoring_mode == "mu_minus_lambda_sigma":
                mu = getattr(obj, "mu", None)
                sg = getattr(obj, "sigma", None)
                if mu is None or sg is None:
                    return None
                try:
                    return float(mu) - lam * float(sg)
                except (TypeError, ValueError):
                    return None
            if scoring_mode == "sharpe":
                mu = getattr(obj, "mu", None)
                sg = getattr(obj, "sigma", None)
                if mu is None or sg is None:
                    return None
                try:
                    return float(mu) / max(float(sg), sharpe_sigma_floor)
                except (TypeError, ValueError):
                    return None
            return getattr(obj, "expected_return", None)

        held_scores: dict = {}
        held_er:     dict = {}
        held_meta:   dict = {}
        # For decision-tree log: track per-held context independent of eligibility
        held_diag:   dict = {}

        for ticker, hs in ctx.holdings.items():
            if ticker in exit_tickers:
                continue
            score   = getattr(hs, "rank_score", None)
            er      = _drive_score(hs)
            entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
            cur_p   = ctx.prices.get(ticker, entry_p)
            entry_d = getattr(hs, "entry_date", None)

            unreal_pct = ((cur_p - entry_p) / entry_p) if entry_p > 0 else 0.0
            hold_days  = (ctx.today - entry_d).days if entry_d is not None else 0
            drag       = tax_drag(unreal_pct, hold_days,
                                  st_rate, lt_rate, lt_threshold)

            decision = None
            if score is None:
                decision = "no_score"
            elif er is None or not math.isfinite(float(er)):
                decision = "no_er"
            elif entry_d is None or entry_p <= 0:
                decision = "no_meta"
            elif hold_days < min_hold:
                decision = "min_hold"
            elif is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
                decision = "lt_protected"

            held_diag[ticker] = {
                "ticker":     ticker,
                "score":      float(score) if score is not None else float("nan"),
                "er":         float(er) if er is not None else float("nan"),
                "unreal_pct": unreal_pct,
                "hold_days":  hold_days,
                "tax_drag":   drag,
                "raw_adv":    float("nan"),     # filled per-candidate below
                "net_adv":    float("nan"),
                "decision":   decision,         # None means eligible
            }

            if decision is None:
                held_scores[ticker] = float(score)
                held_er[ticker]     = float(er)
                held_meta[ticker]   = {
                    "entry_date":    entry_d,
                    "entry_price":   entry_p,
                    "current_price": cur_p,
                }

        held_set = set(ctx.holdings.keys())
        eligible_candidates = [c for c in ctx.ranked if c.ticker not in held_set]

        # Route B — rotation_mode "thesis_primary" bypasses ER-based pair
        # discovery and uses thesis-degradation + uplift as PRIMARY gate.
        # Useful when ER magnitudes are systematically smaller than
        # `min_expected_advantage_pct` (as in current v4.1 golden data
        # where 0 rotations fire because ER delta never reaches 3%).
        rotation_mode = str(rotation_cfg.get("mode", "er"))
        if rotation_mode == "thesis_primary":
            from kernel.rotation import find_thesis_primary_pairs  # noqa: PLC0415
            held_entry_rs = {t: getattr(hs, "entry_rank_score", None)
                             for t, hs in ctx.holdings.items()}
            held_today_rs = {t: getattr(hs, "rank_score", None)
                             for t, hs in ctx.holdings.items()}
            # Build held_meta for anyone past min_hold (thesis_primary
            # decides eligibility internally — pass everyone through).
            held_meta_all: dict = {}
            for t, hs in ctx.holdings.items():
                entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
                cur_p   = ctx.prices.get(t, entry_p)
                held_meta_all[t] = {
                    "entry_date":    getattr(hs, "entry_date", None),
                    "entry_price":   entry_p,
                    "current_price": cur_p,
                }
            merged_rot_cfg = {**rotation_cfg}
            merged_rot_cfg.setdefault("thesis", {}).setdefault(
                "degradation_pct",
                ctx.config.get("ranking", {}).get("thesis_rotation", {})
                                   .get("degradation_pct", 0.30))
            merged_rot_cfg["thesis"].setdefault(
                "uplift_pct",
                ctx.config.get("ranking", {}).get("thesis_rotation", {})
                                   .get("uplift_pct", 0.10))
            pairs = find_thesis_primary_pairs(
                held_entry_scores = held_entry_rs,
                held_today_scores = held_today_rs,
                held_meta         = held_meta_all,
                candidates        = eligible_candidates,
                today             = ctx.today,
                rotation_cfg      = merged_rot_cfg,
                tax_cfg           = tax_cfg,
                panel_buy_floor   = panel_buy_floor,
                panel_sell_floor  = panel_sell_floor,
            )
            log.info("RotationJob: thesis_primary mode — %d pair(s)", len(pairs))
            ctx.rotations = pairs
            return  # skip ER-based discovery + gates

        # Route C — rotation_mode "thesis_symmetric" (V4, 2026-04-24).
        # Full 4-point comparison (A_entry, A_today, B_entry, B_today) via
        # DB lookup of B's rank on A's entry date. Literature basis
        # (Avellaneda-Lee pair-trading + Gu-Kelly-Xiu ML ranking) in
        # doc/research/rotation-research.md.
        if rotation_mode == "thesis_symmetric":
            from kernel.rotation import find_thesis_symmetric_pairs  # noqa: PLC0415
            from kernel.persistence import lookup_candidate_scores_on_date  # noqa: PLC0415

            held_entry_rs = {t: getattr(hs, "entry_rank_score", None)
                             for t, hs in ctx.holdings.items()}
            held_today_rs = {t: getattr(hs, "rank_score", None)
                             for t, hs in ctx.holdings.items()}
            held_meta_all: dict = {}
            for t, hs in ctx.holdings.items():
                entry_p = float(getattr(hs, "entry_price", 0.0) or 0.0)
                cur_p   = ctx.prices.get(t, entry_p)
                held_meta_all[t] = {
                    "entry_date":    getattr(hs, "entry_date", None),
                    "entry_price":   entry_p,
                    "current_price": cur_p,
                }

            # Build entry_day_lookup: (cand_ticker, A_entry_date) → cand's rank
            # on that date. Query candidate_scores × pipeline_runs for each
            # unique A_entry_date in the held set.
            entry_day_lookup: dict = {}
            db = getattr(ctx, "_db", None)
            if db is None:
                # Surface this loudly — V4 mode without DB will produce
                # zero rotations regardless of signal quality. LEAN runs
                # without our SQLite (intentional) so this is the expected
                # no-op path; live + sim should always have DB.
                log.warning(
                    "RotationJob[thesis_symmetric]: ctx._db is None — "
                    "B_entry_score lookup unavailable; rotation will produce "
                    "0 pairs this bar (LEAN-style no-op).")
            if db is not None:
                cand_tickers = [c.ticker for c in eligible_candidates]
                unique_entry_dates = {
                    meta.get("entry_date")
                    for meta in held_meta_all.values()
                    if meta.get("entry_date") is not None
                }
                for entry_date in unique_entry_dates:
                    rows = lookup_candidate_scores_on_date(
                        db, cand_tickers, entry_date,
                    )
                    for ticker, scores in rows.items():
                        entry_day_lookup[(ticker, entry_date)] = scores.get("rank_score")

            # Optional own-momentum dict (Proposal 1, Moskowitz 2012).
            # 63d return per ticker; computed from OHLCV close series.
            own_mom: dict = {}
            thesis_sym_cfg = rotation_cfg.get("thesis_symmetric", {})
            if thesis_sym_cfg.get("own_momentum_enabled", False):
                tickers_to_score = set(held_meta_all.keys()) | {
                    c.ticker for c in eligible_candidates
                }
                ohlcv = getattr(ctx, "ohlcv", {}) or {}
                for t in tickers_to_score:
                    df = ohlcv.get(t)
                    if df is None or len(df) < 64:
                        continue
                    try:
                        today_close = float(df["close"].iloc[-1])
                        past_close  = float(df["close"].iloc[-64])  # 63 bars back
                        if past_close > 0:
                            own_mom[t] = (today_close - past_close) / past_close
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue

            pairs = find_thesis_symmetric_pairs(
                held_entry_scores = held_entry_rs,
                held_today_scores = held_today_rs,
                held_meta         = held_meta_all,
                candidates        = eligible_candidates,
                entry_day_lookup  = entry_day_lookup,
                today             = ctx.today,
                rotation_cfg      = rotation_cfg,
                tax_cfg           = tax_cfg,
                own_momentum      = own_mom or None,
                panel_buy_floor   = panel_buy_floor,
                panel_sell_floor  = panel_sell_floor,
            )
            log.info(
                "RotationJob: thesis_symmetric mode — %d pair(s), "
                "entry_lookup_size=%d own_mom_size=%d",
                len(pairs), len(entry_day_lookup), len(own_mom),
            )
            ctx.rotations = pairs
            return  # skip ER-based discovery + gates

        # V1 persistence gate: pass the context's prior-bar proposals to
        # the primitive via a private config key so the kernel stays
        # stateless.
        persistence = int(rotation_cfg.get("persistence_bars", 0))
        if persistence > 0:
            merged_cfg = dict(rotation_cfg)
            merged_cfg["_prior_proposals"] = list(
                getattr(ctx, "prior_rotation_proposals", []) or []
            )
        else:
            merged_cfg = rotation_cfg

        # V2 (2026-04-24): when μ−λσ OR sharpe scoring mode is on,
        # transiently override c.expected_return with the chosen driver
        # BEFORE passing into the kernel primitive. Shallow-copy
        # candidates so we don't permanently mutate their cached state.
        if scoring_mode in ("mu_minus_lambda_sigma", "sharpe"):
            import copy as _copy  # noqa: PLC0415
            v2_candidates = []
            for c in eligible_candidates:
                d = _drive_score(c)
                if d is None:
                    v2_candidates.append(c)
                    continue
                cc = _copy.copy(c)
                cc.expected_return = float(d)
                v2_candidates.append(cc)
            candidates_for_pairing = v2_candidates
        else:
            candidates_for_pairing = eligible_candidates

        pairs = find_rotation_pairs(
            held_scores      = held_scores,
            held_er          = held_er,
            held_meta        = held_meta,
            candidates       = candidates_for_pairing,
            today            = ctx.today,
            rotation_cfg     = merged_cfg,
            tax_cfg          = tax_cfg,
            panel_buy_floor  = panel_buy_floor,
            panel_sell_floor = panel_sell_floor,
        )

        # Cross-sectional panel gate: require cand.panel_score to beat
        # held.panel_score by panel_rot_advantage (both populated by
        # PanelScoringJob.ApplyScoresTask). Pairs with missing panel scores
        # on either side skip the gate (fall back to ER-only rule).
        if panel_rot_advantage > 0.0 and pairs:
            cand_ps = {c.ticker: getattr(c, "panel_score", None)
                       for c in eligible_candidates}
            held_ps = {t: getattr(hs, "panel_score", None)
                       for t, hs in ctx.holdings.items()}
            kept: list = []
            rejected = 0
            for p in pairs:
                c_ps = cand_ps.get(p.buy_ticker)
                h_ps = held_ps.get(p.sell_ticker)
                # Audit fix RG-NaN (Round 2 deep audit, 2026-04-25):
                # treat non-finite scores the same as None — fall back to
                # KEEP. Pre-fix, NaN is not None and `(NaN - X) >= thresh`
                # is False, so a NaN panel_score silently REJECTED the
                # pair with a "reason=panel_advantage cand_ps=nan ..."
                # log line, when the documented intent of the missing-
                # score branch is "skip gate, preserve pair".
                if (c_ps is None or h_ps is None
                        or not math.isfinite(c_ps)
                        or not math.isfinite(h_ps)
                        or (c_ps - h_ps) >= panel_rot_advantage):
                    kept.append(p)
                else:
                    rejected += 1
                    log.info("ROTATION_REJECT  swap=%s→%s  reason=panel_advantage "
                             "cand_ps=%.3f  held_ps=%.3f  need=%+.3f",
                             p.sell_ticker, p.buy_ticker, c_ps, h_ps, panel_rot_advantage)
            if rejected:
                ctx.counters["panel_rotation_rejects"] = (
                    ctx.counters.get("panel_rotation_rejects", 0) + rejected
                )
            pairs = kept

        # Approach A — thesis-degradation gate BEFORE the Kelly-delta
        # gate, since Approach A uses fixed baselines (more robust) and
        # should filter first.
        if thesis_enabled and pairs:
            cand_rs = {c.ticker: getattr(c, "rank_score", None)
                       for c in eligible_candidates}
            held_entry_rs  = {t: getattr(hs, "entry_rank_score", None)
                              for t, hs in ctx.holdings.items()}
            held_today_rs  = {t: getattr(hs, "rank_score", None)
                              for t, hs in ctx.holdings.items()}
            kept = []
            rejected = 0
            for p in pairs:
                cand_score  = cand_rs.get(p.buy_ticker)
                held_entry  = held_entry_rs.get(p.sell_ticker)
                held_today  = held_today_rs.get(p.sell_ticker)
                # Fallback: if baseline missing or invalid, preserve the
                # pair (legacy rule). Audit fix RG-NaN: also fall back on
                # NaN/inf — pre-fix, a NaN entry_rank_score (stamped during
                # a corrupted-score bar) bypassed `<= 0` (NaN<=0 is False)
                # and propagated through the degradation/uplift calc as
                # NaN, then the final `degradation >= thresh` was False
                # → pair silently REJECTED. The intent of the missing-
                # baseline branch is "skip gate, preserve pair".
                if (held_entry is None or cand_score is None or held_today is None
                        or not math.isfinite(held_entry)
                        or not math.isfinite(held_today)
                        or not math.isfinite(cand_score)
                        or held_entry <= 0):
                    kept.append(p)
                    continue
                degradation = (held_entry - held_today) / held_entry  # + = worse
                uplift      = cand_score - held_entry                 # + = cand beats baseline
                if degradation >= thesis_degradation and uplift >= thesis_uplift:
                    kept.append(p)
                else:
                    rejected += 1
                    log.info("ROTATION_REJECT  swap=%s→%s  reason=thesis  "
                             "held_entry=%.3f held_today=%.3f deg=%.1f%%  "
                             "cand_today=%.3f uplift=%+.3f  need deg≥%.1f%% uplift≥%+.3f",
                             p.sell_ticker, p.buy_ticker,
                             held_entry, held_today, degradation * 100,
                             cand_score, uplift,
                             thesis_degradation * 100, thesis_uplift)
            if rejected:
                ctx.counters["thesis_rotation_rejects"] = (
                    ctx.counters.get("thesis_rotation_rejects", 0) + rejected
                )
            pairs = kept

        # BC: Kelly-delta rotation gate — require cand.kelly_target_pct
        # to beat held.kelly_target_pct by kelly_rot_advantage. Pairs
        # with missing Kelly target on either side skip the gate (fall
        # back to prior decision).
        #
        # Preventive guards (ported from AB-trim audit 2026-04-24,
        # CLAUDE.md §2b): kelly_target is noisy bar-to-bar. Don't filter
        # a pair based on NOISE when:
        #   * held.kelly_target < floor (too small to drive a swap
        #     decision — let ER-based rule handle it)
        #   * held.mu <= 0 (model turned bearish; swapping is fine,
        #     don't block it with a stale Kelly comparison)
        # Default kelly_target_floor = 0.05.
        kelly_target_floor = float(kelly_cfg.get("rotation_target_floor", 0.05))
        if kelly_rot_advantage > 0.0 and pairs:
            cand_kt = {c.ticker: getattr(c, "kelly_target_pct", None)
                       for c in eligible_candidates}
            held_kt = {t: getattr(hs, "kelly_target_pct", None)
                       for t, hs in ctx.holdings.items()}
            held_mu = {t: getattr(hs, "mu", None)
                       for t, hs in ctx.holdings.items()}
            kept = []
            rejected = 0
            guard_skipped = 0
            for p in pairs:
                c_kt = cand_kt.get(p.buy_ticker)
                h_kt = held_kt.get(p.sell_ticker)
                h_mu = held_mu.get(p.sell_ticker)

                # Fallback: missing Kelly data → keep pair.
                # Audit fix RG-NaN: also fall back on NaN/inf — same
                # pattern as panel + thesis gates. NaN h_kt would slip
                # past `< floor` (NaN<X False) and past mu guard, then
                # land in the comparison where `(c_kt - NaN) >= adv` is
                # False → silent REJECT. Intent is "skip gate when data
                # missing, including non-finite".
                if (c_kt is None or h_kt is None
                        or not math.isfinite(c_kt)
                        or not math.isfinite(h_kt)):
                    kept.append(p)
                    continue
                # Guard: held Kelly too small to drive swap decision.
                if h_kt < kelly_target_floor:
                    kept.append(p)
                    guard_skipped += 1
                    continue
                # Guard: held mu bearish — don't Kelly-block a rational
                # swap based on a stale / noisy Kelly target. RG-NaN:
                # NaN mu also routes to "skip gate" so a corrupted μ
                # doesn't accidentally enforce the gate.
                if h_mu is not None:
                    if not math.isfinite(h_mu):
                        kept.append(p)
                        guard_skipped += 1
                        continue
                    if h_mu <= 0:
                        kept.append(p)
                        guard_skipped += 1
                        continue

                if (c_kt - h_kt) >= kelly_rot_advantage:
                    kept.append(p)
                else:
                    rejected += 1
                    log.info("ROTATION_REJECT  swap=%s→%s  reason=kelly_advantage "
                             "cand_kt=%.3f  held_kt=%.3f  need=%+.3f",
                             p.sell_ticker, p.buy_ticker,
                             c_kt or 0.0, h_kt or 0.0, kelly_rot_advantage)
            if rejected:
                ctx.counters["kelly_rotation_rejects"] = (
                    ctx.counters.get("kelly_rotation_rejects", 0) + rejected
                )
            if guard_skipped:
                ctx.counters["kelly_rotation_guard_skipped"] = (
                    ctx.counters.get("kelly_rotation_guard_skipped", 0) + guard_skipped
                )
            pairs = kept

        ctx.rotations = pairs

        # Decision-tree log: one block per candidate considered.  We replay
        # the comparisons for the top-K candidates so the log is auditable
        # without re-running the whole pipeline.
        chosen_pairs = {p.buy_ticker: p for p in pairs}
        topk = eligible_candidates[: max(5, len(pairs) + 2)]
        for c in topk:
            cand_er = float(getattr(c, "expected_return", 0.0) or 0.0)
            rows: list[dict] = []
            for ht, info in held_diag.items():
                row = dict(info)   # shallow copy
                if row["decision"] is None:
                    raw_adv = cand_er - info["er"]
                    net_adv = raw_adv - info["tax_drag"] - txn_cost
                    row["raw_adv"] = raw_adv
                    row["net_adv"] = net_adv
                    row["decision"] = (
                        "swap" if (chosen_pairs.get(c.ticker)
                                   and chosen_pairs[c.ticker].sell_ticker == ht)
                        else "below_threshold" if net_adv < threshold
                        else "available"
                    )
                rows.append(row)
            chosen = chosen_pairs.get(c.ticker)
            _log_decision_tree(
                cand_ticker = c.ticker,
                cand_er     = cand_er,
                cand_score  = float(c.rank_score),
                held_table  = rows,
                threshold   = threshold,
                txn_cost    = txn_cost,
                horizon     = horizon,
                chosen      = chosen.sell_ticker if chosen else None,
            )

        log.info("BuildPairsTask: %d rotation pair(s) proposed", len(pairs))


class ValidatePairsTask(Task):
    """Drop pairs whose buy ticker fails wash-sale, sector, or corr guards."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.selection import (  # noqa: PLC0415
            is_wash_sale_blocked,
            passes_sector_guard,
            passes_correlation_guard,
        )

        if not ctx.rotations:
            return False

        cfg            = ctx.config
        regime_cfg     = cfg.get("regime", {})
        wash_days      = int(cfg.get("wash_sale_days", 0))
        corr_threshold = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        max_per_sector = int(cfg.get("max_positions_per_sector", 0))
        sector_map     = cfg.get("sector_map", {})
        defensive_set  = set(cfg.get("defensive_tickers", []))

        validated = []
        for pair in ctx.rotations:
            if is_wash_sale_blocked(pair.buy_ticker, ctx.today,
                                    ctx.last_sell_dates or {}, wash_days):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=wash_sale",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            virtual_held = (
                set(ctx.holdings.keys())
                - {p.sell_ticker for p in validated} - {pair.sell_ticker}
                | {p.buy_ticker for p in validated}
            )

            if not passes_sector_guard(
                pair.buy_ticker, list(virtual_held),
                sector_map, max_per_sector, defensive_set,
            ):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=sector_cap",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            if not passes_correlation_guard(
                pair.buy_ticker, list(virtual_held),
                ctx.corr_matrix, corr_threshold,
            ):
                log.info("ROTATION_REJECT  swap=%s→%s  reason=correlation_guard",
                         pair.sell_ticker, pair.buy_ticker)
                continue

            validated.append(pair)

        ctx.rotations = validated
        log.info("ValidatePairsTask: %d pair(s) survived guards", len(validated))


class EmitRotationsTask(Task):
    """Append rotation exits, sized buy orders; prune ranked to avoid double-buy."""

    def run(self, ctx: InferenceContext) -> bool | None:
        from kernel.exits  import ExitSignal                              # noqa: PLC0415
        from kernel.sizing import (  # noqa: PLC0415
            compute_position_size,
            conviction_multiplier,
            sigma_multiplier,
            universe_sigma_median,
        )

        if not ctx.rotations:
            return

        # Hoisted for use inside the per-pair loop (PR1-CASH fix).
        rotation_cfg = ctx.config.get("rotation", {})

        # Audit fix CONF-MULT (2026-04-25): floored confidence multiplier.
        from kernel.regime import confidence_to_size_multiplier  # noqa: PLC0415
        _conf_mult   = confidence_to_size_multiplier(ctx.confidence)
        regime_p     = ctx.config.get("regime_params", {}).get(ctx.regime, {})
        base_max_pct = float(regime_p.get("max_position_pct", 0.15)) * _conf_mult
        # 2026-04-24 sizing parity (#26 #33): apply the same CUSUM
        # wall-time cooldown scaling SizeAndEmitTask uses, so rotation
        # buys aren't oversized while fresh picks are scaled down.
        cooldown_mult = 1.0
        _regime_cfg = ctx.config.get("regime", {})
        if str(_regime_cfg.get("cusum_cooldown_mode", "bar_count")) == "wall_time":
            from kernel.regime import cusum_cooldown_progress  # noqa: PLC0415
            cd_start = (getattr(ctx.regime_state, "cooldown_start", None)
                        if ctx.regime_state is not None else None)
            cd_days  = float(_regime_cfg.get("cusum_cooldown_days", 3.0))
            cooldown_mult = cusum_cooldown_progress(ctx.today, cd_start, cd_days)
        base_max_pct *= cooldown_mult
        reserve_pct  = float(regime_p.get("cash_reserve_pct", 0.0))  * _conf_mult
        sizing_cfg   = (ctx.config.get("ranking", {})
                         .get("panel_scoring", {}).get("sizing", {}))
        sigma_cfg    = (ctx.config.get("ranking", {})
                         .get("panel_scoring", {})
                         .get("sigma_sizing", {}))
        kelly_cfg    = ctx.config.get("ranking", {}).get("kelly_sizing", {})
        kelly_on     = bool(kelly_cfg.get("enabled", False))
        kelly_pure   = bool(kelly_cfg.get("disable_extra_multipliers", False))
        per_session_cap = kelly_cfg.get("per_session_buy_cap")

        sigma_median = universe_sigma_median(
            [getattr(c, "sigma", None) for c in ctx.ranked]
        )

        # Audit fix ROT-BLOCKED-NTFY (Bug L, 2026-04-25): pre-fix, when
        # a rotation pair was found by find_rotation_pairs (counted in
        # ctx.rotations) but later dropped at emit time (Kelly=0,
        # bad price, insufficient cash), the operator never saw it.
        # ntfy showed only successful trades + UNMANAGED; the BLOCKED
        # rotation was log-only. Now: track blocked pairs on ctx so
        # live/runner.py::_notify_decision can surface them in the ntfy
        # body — keeps the operator informed when the system WANTED
        # to swap but a downstream guard vetoed.
        if not hasattr(ctx, "rotations_blocked"):
            ctx.rotations_blocked = []
        rotated_buys: set[str] = set()
        rotated_sells: set[str] = set()

        # Audit fix PR1-CASH (Phase 1 rotation rolling cash, 2026-04-25):
        # pre-fix, every rotation pair was sized against the bar-start
        # `ctx.cash`. With max_rotations_per_bar=2, both pairs each
        # believed they had the full cash balance — and when the actual
        # broker submitted both orders, the second would either over-buy
        # (margin) or fail (cash account). Now: maintain `cash_remaining`
        # that decrements after each accepted rotation buy AND credits
        # the sell-leg's mark-to-market proceeds (RegT same-bar settle).
        cash_remaining = float(ctx.cash)

        for pair in ctx.rotations:
            # 2026-04-24 bug fix: previously the SELL exit was appended
            # FIRST and the BUY constructed second. If the buy failed
            # (no price / shares<1), the position closed without a
            # replacement — the user lost the held but bought nothing.
            # Now we compute the buy fully BEFORE committing the exit.

            # Audit fix ROT-NaN-PRICE (Round 2 deep audit, 2026-04-25):
            # `price <= 0` lets NaN slip through (NaN<=0 is False), then
            # `int(NaN_invest)` later raises and silently aborts the
            # whole rotation pair without the operator seeing why.
            # Same NaN-slip pattern as SE-1 / TR-NaN. Fail-SAFE: skip the
            # pair on non-finite price too, with a clear log.
            price = ctx.prices.get(pair.buy_ticker, 0.0)
            if not math.isfinite(price) or price <= 0:
                log.warning(
                    "EmitRotationsTask: bad price (%s) for %s — skip ENTIRE pair "
                    "(no atomic-rotation orphan exit)",
                    price, pair.buy_ticker,
                )
                ctx.rotations_blocked.append({
                    "sell": pair.sell_ticker, "buy": pair.buy_ticker,
                    "reason": f"bad_price({price})",
                })
                continue

            buy_cand = next((c for c in ctx.ranked if c.ticker == pair.buy_ticker), None)
            if kelly_on and kelly_pure:
                conv, sig_m = 1.0, 1.0
            else:
                conv = conviction_multiplier(
                    getattr(buy_cand, "panel_score", None) if buy_cand else None,
                    sizing_cfg,
                )
                sig_m = sigma_multiplier(
                    getattr(buy_cand, "sigma", None) if buy_cand else None,
                    sigma_median, sigma_cfg,
                )
            # Kelly target if enabled — otherwise legacy regime-cap path.
            if kelly_on and buy_cand is not None and getattr(buy_cand, "kelly_target_pct", None) is not None:
                max_pct = float(buy_cand.kelly_target_pct) * conv * sig_m
                if max_pct <= 0:
                    log.info("EmitRotationsTask: %s Kelly=0 — skip pair", pair.buy_ticker)
                    ctx.rotations_blocked.append({
                        "sell": pair.sell_ticker, "buy": pair.buy_ticker,
                        "reason": "kelly_zero",
                    })
                    continue
            else:
                max_pct = base_max_pct * conv * sig_m

            # Multi-entry cap (matches SizeAndEmitTask).
            if per_session_cap is not None:
                cap = float(per_session_cap)
                if cap > 0 and max_pct > cap:
                    max_pct = cap

            # Audit fix PR1-CASH (2026-04-25): credit sell-leg proceeds
            # to sizing budget (RegT same-bar settlement on rotation
            # pairs). Same fix as Bug M in JointActionTask. Pre-fix, the
            # buy-leg was sized off `ctx.cash` (no held credit, no
            # rolling decrement) → first rotation under-sized when cash
            # was tight, AND second+ rotations over-claimed shared cash.
            # Use defensive getattr — some unit-test mocks pass a
            # SimpleNamespace ctx without `holdings` set up.
            _holdings   = getattr(ctx, "holdings", None) or {}
            held_st     = _holdings.get(pair.sell_ticker) if _holdings else None
            held_shares = int(getattr(held_st, "shares", 0) or 0) if held_st else 0
            held_price  = float(ctx.prices.get(pair.sell_ticker, 0.0) or 0.0)
            sell_proceeds = 0.0
            if held_shares > 0 and math.isfinite(held_price) and held_price > 0:
                # Apply transaction-cost haircut to be honest about
                # realized cash. transaction_cost_pct in rotation_cfg
                # applies to the round-trip; halve for one leg.
                _leg_cost = float(rotation_cfg.get("transaction_cost_pct", 0.0)) / 2.0
                sell_proceeds = held_shares * held_price * (1.0 - _leg_cost)
            cash_for_sizing = cash_remaining + sell_proceeds

            _, shares = compute_position_size(
                ctx.portfolio_value, cash_for_sizing,
                max_pct, reserve_pct, price,
            )
            if shares < 1:
                log.info("EmitRotationsTask: %s insufficient cash — skip ENTIRE pair "
                         "(no atomic-rotation orphan exit)  cash_for_sizing=%.0f",
                         pair.buy_ticker, cash_for_sizing)
                ctx.rotations_blocked.append({
                    "sell": pair.sell_ticker, "buy": pair.buy_ticker,
                    "reason": "insufficient_cash",
                })
                continue

            # Buy confirmed; NOW commit the exit.
            ctx.exits.append((
                pair.sell_ticker,
                ExitSignal(
                    should_exit = True,
                    reason      = (f"rotation→{pair.buy_ticker} "
                                   f"net_adv={pair.net_advantage:+.4f} "
                                   f"horizon={pair.horizon_days}d"),
                    exit_type   = "rotation",
                ),
            ))

            invest     = shares * price
            target_pct = invest / ctx.portfolio_value if ctx.portfolio_value > 0 else 0.0
            ctx.orders.append({
                "ticker":     pair.buy_ticker,
                "shares":     shares,
                "price":      price,
                "invest":     invest,
                "target_pct": target_pct,
                "regime":     ctx.regime,
                "confidence": ctx.confidence,
                "conviction": conv,
                "sigma_mult": sig_m,
                "rank_score": pair.buy_score,
                "rs_score":   0.0,
                "panel_score": getattr(buy_cand, "panel_score", None) if buy_cand else None,
                "mu":         getattr(buy_cand, "mu", None)    if buy_cand else None,
                "sigma":      getattr(buy_cand, "sigma", None) if buy_cand else None,
                "kelly_target_pct": getattr(buy_cand, "kelly_target_pct", None) if buy_cand else None,
                "detail":     (f"rotation←{pair.sell_ticker} "
                               f"net_adv={pair.net_advantage:+.4f} "
                               f"horizon={pair.horizon_days}d"),
                "order_type": "ROTATION",
            })
            rotated_buys.add(pair.buy_ticker)
            # PR1-CASH: roll the cash forward — credit sell, debit buy.
            cash_remaining = cash_remaining + sell_proceeds - invest
            ctx.counters["rotations"] = ctx.counters.get("rotations", 0) + 1
            log.info(
                "ROTATION_EXEC  swap=%s→%s  shares=%d  net_adv=%+.4f  "
                "raw_adv=%+.4f  tax=%.4f  cost=%.4f  threshold=%+.4f  "
                "horizon=%dd  sell_proc=%.0f  cash_after=%.0f",
                pair.sell_ticker, pair.buy_ticker, shares,
                pair.net_advantage, pair.raw_advantage,
                pair.tax_drag, pair.transaction_cost,
                pair.threshold, pair.horizon_days,
                sell_proceeds, cash_remaining,
            )

        if rotated_buys:
            ctx.ranked = [c for c in ctx.ranked if c.ticker not in rotated_buys]
