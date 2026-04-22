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
            tc.candidate_reject_reason = "earnings_blocked"
            log.debug("EarningsFilterTask [%s]: blocked", tc.ticker)
            return False


class WashSaleFilterTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_wash_sale_blocked  # noqa: PLC0415
        wash_days = int(tc.config.get("wash_sale_days", 0))
        if is_wash_sale_blocked(tc.ticker, tc.today, tc.last_sell_dates or {}, wash_days):
            tc.candidate_reject_reason = "wash_sale_blocked"
            log.debug("WashSaleFilterTask [%s]: wash-sale blocked", tc.ticker)
            return False


class BuildFeaturesTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.indicators import build_feature_frame  # noqa: PLC0415
        stock_df = tc.ohlcv.get(tc.ticker)
        spy_df   = tc.ohlcv.get("SPY")
        if stock_df is None or tc.model is None or spy_df is None:
            tc.candidate_reject_reason = "missing_inputs"
            return False
        spec    = tc.config.get("indicator_spec", {})
        vol_win = int(tc.config.get("regime", {}).get("vol_realized_window", 20))
        tc.features = build_feature_frame(stock_df, spy_df, spec, vol_win)
        if tc.features is None or tc.features.empty:
            tc.candidate_reject_reason = "insufficient_feature_history"
            return False


class ScoreBuyTask(Task):
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
        if sr.signal != "buy":
            tc.candidate_reject_reason = "model_not_buy"
            return False
        tc._raw_score       = sr.raw_score          # noqa: SLF001
        tc._rank_score      = sr.rank_score         # noqa: SLF001
        tc._expected_return = sr.expected_return    # noqa: SLF001


class ScoreThresholdTask(Task):
    def run(self, tc: TickerInferenceContext) -> bool | None:
        min_score = float(tc.regime_params.get("min_model_score", 0.10))
        rank      = getattr(tc, "_rank_score", 0.0)
        if rank < min_score:
            tc.candidate_reject_reason = "below_min_model_score"
            log.debug("ScoreThresholdTask [%s]: rank=%.4f < min=%.4f — rejected",
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
        tc.candidate_reject_reason = None
        log.debug("AssembleCandidateTask [%s]: candidate assembled", tc.ticker)
