"""TickerCandidateJob — score one candidate ticker for buy eligibility.

Per-ticker job: reads/writes TickerInferenceContext only.
Run in parallel by InferencePipeline for all non-held tickers.

Reads:  tc.ticker, tc.ohlcv, tc.model, tc.regime, tc.regime_params,
        tc.config, tc.today
Writes: tc.candidate (CandidateResult | None)
"""
from __future__ import annotations

import logging

from ..context import TickerInferenceContext
from ..pipeline import TickerJob

log = logging.getLogger("kernel.pipeline.candidates")


class TickerCandidateJob(TickerJob):
    """Score one ticker and produce a CandidateResult if it qualifies."""

    def run(self, tc: TickerInferenceContext) -> None:
        from kernel.selection  import (                    # noqa: PLC0415
            CandidateResult, compute_relative_strength,
            is_earnings_blocked, is_wash_sale_blocked,
        )
        from kernel.models     import score_artifact       # noqa: PLC0415
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        config       = tc.config
        spec         = config.get("indicator_spec", {})
        vol_win      = int(config.get("regime", {}).get("vol_realized_window", 20))
        earnings_buf = int(config.get("regime", {}).get("earnings_buffer_days", 3))
        wash_days    = int(config.get("wash_sale_days", 0))
        sector_map   = config.get("sector_map", {})
        sector_etf   = config.get("sector_etf_map", {})

        min_score = float(tc.regime_params.get("min_model_score", 0.10))
        # No score floor for defensives in BEAR (bear_only universe, orchestrator decides)

        earnings_cal = tc.earnings_calendar or {}
        last_sells   = tc.last_sell_dates or {}

        # Earnings filter
        if is_earnings_blocked(tc.ticker, tc.today, earnings_cal, earnings_buf):
            log.debug("%s earnings blocked", tc.ticker)
            return

        # Wash-sale pre-filter
        if is_wash_sale_blocked(tc.ticker, tc.today, last_sells, wash_days):
            return

        stock_df = tc.ohlcv.get(tc.ticker)
        spy_df   = tc.ohlcv.get("SPY")
        if stock_df is None or tc.model is None or spy_df is None:
            return

        features = build_feature_frame(stock_df, spy_df, spec, vol_win)
        if features is None or features.empty:
            return

        sr = score_artifact(tc.model, features.iloc[-1], holdings=0)
        log.debug("%s  action=%s  raw=%.4f  rank=%.4f",
                  tc.ticker, sr.signal, sr.raw_score, sr.rank_score)

        if sr.signal != "buy":
            return
        if sr.rank_score < min_score:
            return

        # Relative strength vs sector ETF (20-day)
        rs_score = 0.0
        etf = sector_etf.get(sector_map.get(tc.ticker, "other"))
        if etf and etf in tc.ohlcv:
            etf_df = tc.ohlcv[etf]
            if len(stock_df) >= 21 and len(etf_df) >= 21:
                try:
                    stock_r = float(stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1)
                    etf_r   = float(etf_df["close"].iloc[-1]   / etf_df["close"].iloc[-21] - 1)
                    rs_score = compute_relative_strength(stock_r, etf_r)
                except Exception:
                    pass

        tc.candidate = CandidateResult(
            ticker=tc.ticker,
            raw_score=sr.raw_score,
            rank_score=sr.rank_score,
            rs_score=rs_score,
            detail=f"raw={sr.raw_score:.3f} rank={sr.rank_score:.3f}",
        )
