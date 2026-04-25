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
            log.debug("EarningsFilterTask [%s]: blocked", tc.ticker)
            return False


class WashSaleFilterTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_wash_sale_blocked  # noqa: PLC0415
        wash_days = int(tc.config.get("wash_sale_days", 0))
        if is_wash_sale_blocked(tc.ticker, tc.today, tc.last_sell_dates or {}, wash_days):
            log.debug("WashSaleFilterTask [%s]: wash-sale blocked", tc.ticker)
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
                return False
            return None

        from kernel.indicators import build_feature_frame  # noqa: PLC0415
        stock_df = tc.ohlcv.get(tc.ticker)
        spy_df   = tc.ohlcv.get("SPY")
        if stock_df is None or tc.model is None or spy_df is None:
            return False
        spec    = tc.config.get("indicator_spec", {})
        vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
        tc.features = build_feature_frame(stock_df, spy_df, spec, vol_win)
        if tc.features is None or tc.features.empty:
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
            log.debug("ScoreThresholdTask [%s]: rank=%s < min=%.4f — rejected",
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
