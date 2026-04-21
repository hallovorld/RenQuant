"""Per-ticker buy candidate scoring tasks.

Each task can return False to discard this ticker from the candidate pool.

Reads:  tc.ticker, tc.ohlcv, tc.model, tc.regime, tc.regime_params,
        tc.config, tc.today, tc.earnings_calendar, tc.last_sell_dates
Writes: tc.features, tc.model_action, tc.rs_score, tc.candidate
"""
from __future__ import annotations

import logging

from ..context import TickerInferenceContext
from ..pipeline import Task

log = logging.getLogger("kernel.pipeline.candidates")


class EarningsFilterTask(Task):
    """Skip this ticker if an earnings event falls within the buffer window."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_earnings_blocked  # noqa: PLC0415

        earnings_buf = int(tc.config.get("regime", {}).get("earnings_buffer_days", 3))
        earnings_cal = tc.earnings_calendar or {}

        if is_earnings_blocked(tc.ticker, tc.today, earnings_cal, earnings_buf):
            log.debug("EarningsFilterTask [%s]: blocked", tc.ticker)
            return False


class WashSaleFilterTask(Task):
    """Skip this ticker if we sold it within wash_sale_days."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import is_wash_sale_blocked  # noqa: PLC0415

        wash_days  = int(tc.config.get("wash_sale_days", 0))
        last_sells = tc.last_sell_dates or {}

        if is_wash_sale_blocked(tc.ticker, tc.today, last_sells, wash_days):
            log.debug("WashSaleFilterTask [%s]: wash-sale blocked", tc.ticker)
            return False


class BuildFeaturesTask(Task):
    """Load stock + SPY data and build the feature frame → tc.features."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
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
    """Score the model; discard ticker if signal is not 'buy' → tc.model_action."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.models import score_artifact  # noqa: PLC0415

        sr = score_artifact(tc.model, tc.features.iloc[-1], holdings=0)
        tc.model_action = sr.signal

        log.debug("ScoreBuyTask [%s]: action=%s  raw=%.4f  rank=%.4f",
                  tc.ticker, sr.signal, sr.raw_score, sr.rank_score)

        if sr.signal != "buy":
            return False

        # Stash scores on context for AssembleCandidateTask
        tc._raw_score  = sr.raw_score   # noqa: SLF001
        tc._rank_score = sr.rank_score  # noqa: SLF001


class ScoreThresholdTask(Task):
    """Discard ticker if calibrated rank_score is below the regime threshold."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        min_score = float(tc.regime_params.get("min_model_score", 0.10))
        rank      = getattr(tc, "_rank_score", 0.0)

        if rank < min_score:
            log.debug("ScoreThresholdTask [%s]: rank=%.4f < min=%.4f — rejected",
                      tc.ticker, rank, min_score)
            return False


class RelativeStrengthTask(Task):
    """Compute 20-day relative strength vs sector ETF → tc.rs_score."""

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
    """Package scored ticker into a CandidateResult → tc.candidate."""

    def run(self, tc: TickerInferenceContext) -> bool | None:
        from kernel.selection import CandidateResult  # noqa: PLC0415

        raw  = getattr(tc, "_raw_score",  0.0)
        rank = getattr(tc, "_rank_score", 0.0)

        tc.candidate = CandidateResult(
            ticker    = tc.ticker,
            raw_score = raw,
            rank_score= rank,
            rs_score  = tc.rs_score,
            detail    = f"raw={raw:.3f} rank={rank:.3f} rs={tc.rs_score:.3f}",
        )
        log.debug("AssembleCandidateTask [%s]: candidate assembled", tc.ticker)
