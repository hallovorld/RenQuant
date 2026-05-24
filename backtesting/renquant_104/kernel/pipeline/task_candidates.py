"""Per-ticker buy candidate scoring tasks."""
from __future__ import annotations

import logging

from .context import TickerInferenceContext
from .pipeline import Task

log = logging.getLogger("kernel.pipeline.candidates")


class EarningsFilterTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_earnings_blocked  # noqa: PLC0415
        earnings_buf = int(tc.config.get("regime", {}).get("earnings_buffer_days", 3))
        if is_earnings_blocked(tc.ticker, tc.today, tc.earnings_calendar or {}, earnings_buf):
            tc.blocked_by = "earnings_blackout"
            log.info("DROP_EarningsFilter [%s]: blocked (within ±%dd of earnings)",
                     tc.ticker, earnings_buf)
            return False


class WashSaleFilterTask(Task):
    """Cost-aware wash-sale gate per IRC §1091.

    The 2026-05-09 economic-cost rewrite: §1091 only disallows the loss
    deduction on a sale that REALIZED A LOSS within ±30d of the buy-back.
    Sales at a GAIN have no §1091 cost. Loss sales have only an NPV
    time-value cost (deferred deduction).

    This task runs at the per-ticker pre-screen stage where μ̂ isn't
    available yet, so:
      - GAIN sales pass (no cost)
      - LOSS sales WITHIN window are still blocked here (conservative;
        the post-NGB economic gate can re-admit them by passing μ̂)
      - sales OUTSIDE window pass (rule doesn't apply)

    Config:
      wash_sale_days       : int — window in days (default 30)
      wash_sale_tax_rate   : float — combined federal+state rate (0.30)
      wash_sale_discount_rate : float — for NPV (0.05)
      wash_sale_hold_years : float — expected hold of replacement (2.0)
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_wash_sale_blocked_with_cost  # noqa: PLC0415
        wash_days = int(tc.config.get("wash_sale_days", 0))
        tax_rate = float(tc.config.get("wash_sale_tax_rate", 0.30))
        disc = float(tc.config.get("wash_sale_discount_rate", 0.05))
        hold_yrs = float(tc.config.get("wash_sale_hold_years", 2.0))
        blocked, reason, cost_npv = is_wash_sale_blocked_with_cost(
            tc.ticker,
            tc.today,
            tc.last_sell_dates or {},
            tc.last_sell_pls or {},
            wash_days,
            tax_rate=tax_rate,
            discount_rate=disc,
            estimated_hold_years=hold_yrs,
            expected_dollar_return=None,   # μ̂ not yet known at this stage
        )
        if blocked:
            tc.blocked_by = f"wash_sale:{reason}"
            log.info("DROP_WashSaleFilter [%s]: %s", tc.ticker, reason)
            return False
        # Not blocked but log the reason so the audit trail shows
        # whether we passed because of "gain sale" / "outside window" /
        # "no recent sale".
        if reason and "no recent sale" not in reason and "disabled" not in reason:
            log.debug("PASS_WashSaleFilter [%s]: %s (cost_npv=$%.2f)",
                      tc.ticker, reason, cost_npv)


class SectorMapGateTask(Task):
    """Require sector metadata before a ticker can enter buy selection.

    Panel-LTR neutralization, relative strength, and QP sector caps all rely
    on ``sector_map``. A missing sector must not silently degrade to
    ``rs_score=0`` and no sector cap, because that creates unmanaged sector
    bets in live trading.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        require = bool(
            tc.config.get("risk", {}).get(
                "require_sector_map_for_buys",
                tc.config.get("ranking", {})
                         .get("panel_scoring", {})
                         .get("enabled", False),
            )
        )
        if not require:
            return None
        benchmark = tc.config.get("benchmark", "SPY")
        if tc.ticker == benchmark:
            return None
        sector_map = tc.config.get("sector_map", {}) or {}
        sector = sector_map.get(tc.ticker)
        if not isinstance(sector, str) or not sector:
            tc.blocked_by = "missing_sector_map"
            log.info(
                "DROP_SectorMapGate [%s]: missing sector_map entry "
                "(required for RS + QP sector caps)",
                tc.ticker,
            )
            return False


class BuildFeaturesTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        # Feature cache optimization (2026-04-24): if SimAdapter pre-built
        # a full-range feature frame for this ticker, slice it up to today
        # instead of rebuilding from OHLCV (10x faster per bar).
        cached = getattr(tc, "feature_cache_frame", None)
        if cached is not None and not cached.empty:
            tc.features = cached.loc[:tc.today]
            if tc.features is None or tc.features.empty:
                log.info("DROP_BuildFeatures [%s]: cached frame slice is empty "
                         "for date %s (cache range: %s → %s)",
                         tc.ticker, tc.today,
                         cached.index.min() if len(cached) else "?",
                         cached.index.max() if len(cached) else "?")
                return False
            return None

        from kernel.indicators import build_feature_frame  # noqa: PLC0415
        stock_df = tc.ohlcv.get(tc.ticker)
        spy_df   = tc.ohlcv.get("SPY")
        if stock_df is None or tc.model is None or spy_df is None:
            missing = []
            if stock_df is None:
                missing.append("stock_ohlcv")
            if tc.model is None:
                missing.append("model")
            if spy_df is None:
                missing.append("spy_ohlcv")
            tc.blocked_by = "missing_input:" + ",".join(missing)
            log.info("DROP_BuildFeatures [%s]: missing input "
                     "(stock_df=%s, model=%s, spy=%s)",
                     tc.ticker, stock_df is not None,
                     tc.model is not None, spy_df is not None)
            return False
        spec    = tc.config.get("indicator_spec", {})
        vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
        tc.features = build_feature_frame(stock_df, spy_df, spec, vol_win)
        if tc.features is None or tc.features.empty:
            tc.blocked_by = "empty_features"
            log.info("DROP_BuildFeatures [%s]: build_feature_frame returned empty",
                     tc.ticker)
            return False


class ScoreBuyTask(Task):
    """Score ticker with per-ticker tournament model.

    Default: drop if `signal != "buy"` — the tournament model acts as a binary
    admission gate. This was the 103 behavior and is why many watchlists sat
    in cash for extended periods when per-ticker models got conservative.

    When `ranking.panel_scoring.bypass_ticker_gate == true`, the tournament's
    signal/threshold is advisory only: we still compute and record raw/rank
    scores for logging, but do NOT filter on them. Panel-LTR (which is a
    cross-sectional ranker) then gets to see every admissible ticker and
    rank them itself. The downstream `min_model_score` tier + panel
    `buy_floor` + selection-loop tiered thresholds still enforce quality.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.models import score_artifact  # noqa: PLC0415
        rotation_horizon = int(tc.config.get("rotation", {}).get("target_horizon_days", 20))
        sr = score_artifact(
            tc.model, tc.features.iloc[-1],
            holdings=0, horizon_days=rotation_horizon,
        )
        tc.model_action = sr.signal
        log.debug("ScoreBuyTask [%s]: action=%s  raw=%.4f  rank=%.4f  er=%.4f",
                  tc.ticker, sr.signal, sr.raw_score, sr.rank_score, sr.expected_return)

        # Always record scores so downstream tasks + logs have them.
        tc._raw_score       = sr.raw_score          # noqa: SLF001
        tc._rank_score      = sr.rank_score         # noqa: SLF001
        tc._expected_return = sr.expected_return    # noqa: SLF001

        bypass = bool(
            tc.config.get("ranking", {})
                      .get("panel_scoring", {})
                      .get("bypass_ticker_gate", False)
        )
        if bypass:
            return
        if sr.signal != "buy":
            tc.blocked_by = f"model_signal:{sr.signal}"
            log.info("DROP_ScoreBuy [%s]: signal=%s (not 'buy')",
                     tc.ticker, sr.signal)
            return False


class ScoreThresholdTask(Task):
    """Reject candidates whose tournament `rank_score` < regime min_model_score.

    Skipped when `ranking.panel_scoring.bypass_ticker_gate == true` — the
    tournament's calibrated rank_score is an unreliable admission signal
    in sparse-buy regimes; Panel-LTR will overwrite rank_score via
    PanelScoringJob and the selection loop then applies its own tiered
    thresholds on the panel-calibrated score.
    """

    def run(self, tc: TickerInferenceContext) -> bool | None:
        bypass = bool(
            tc.config.get("ranking", {})
                      .get("panel_scoring", {})
                      .get("bypass_ticker_gate", False)
        )
        if bypass:
            return
        # Audit fix TC-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN
        # rank passed the `< min_score` gate (NaN < x is False) →
        # candidate proceeded with NaN rank_score. Treat NaN as worst
        # (= rejected).
        import math
        min_score = float(tc.regime_params.get("min_model_score", 0.10))
        rank      = getattr(tc, "_rank_score", 0.0)
        if rank is None or not math.isfinite(rank) or rank < min_score:
            tc.blocked_by = "rank_below_min"
            log.info("DROP_ScoreThreshold [%s]: rank=%s < min=%.4f",
                     tc.ticker, rank, min_score)
            return False


class RelativeStrengthTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import compute_relative_strength  # noqa: PLC0415
        sector_map = tc.config.get("sector_map", {})
        sector_etf = tc.config.get("sector_etf_map", {})
        etf = sector_etf.get(sector_map.get(tc.ticker, "other"))
        if not etf or etf not in tc.ohlcv:
            tc.rs_score = 0.0
            return
        stock_df = tc.ohlcv.get(tc.ticker)
        etf_df   = tc.ohlcv[etf]
        if len(stock_df) >= 21 and len(etf_df) >= 21:
            try:
                stock_r = float(stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1)
                etf_r   = float(etf_df["close"].iloc[-1]   / etf_df["close"].iloc[-21]   - 1)
                tc.rs_score = compute_relative_strength(stock_r, etf_r)
            except Exception:
                tc.rs_score = 0.0
        else:
            tc.rs_score = 0.0
        log.debug("RelativeStrengthTask [%s]: rs=%.4f", tc.ticker, tc.rs_score)


class AssembleCandidateTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import CandidateResult  # noqa: PLC0415
        raw  = getattr(tc, "_raw_score",        0.0)
        rank = getattr(tc, "_rank_score",       0.0)
        er   = getattr(tc, "_expected_return",  0.0)
        tc.candidate = CandidateResult(
            ticker          = tc.ticker,
            raw_score       = raw,
            rank_score      = rank,
            rs_score        = tc.rs_score,
            detail          = (f"raw={raw:.3f} rank={rank:.3f} "
                               f"rs={tc.rs_score:.3f} er={er:+.4f}"),
            expected_return = er,
        )
        log.debug("AssembleCandidateTask [%s]: candidate assembled", tc.ticker)
