"""CandidateJob — score all non-held tickers and build candidate list.

Reads:  ctx.buy_blocked, ctx.bear_only, ctx.regime, ctx.models, ctx.holdings,
        ctx.ohlcv, ctx.today, ctx.earnings_calendar, ctx.last_sell_dates,
        ctx.config
Writes: ctx.candidates (list of CandidateResult)
        ctx.counters["earnings_blocks"] incremented
"""
from __future__ import annotations

import logging

from ..context import InferenceContext
from ..pipeline import Job

log = logging.getLogger("kernel.pipeline.candidates")


class CandidateJob(Job):
    """Build the list of buy candidates, applying earnings and wash-sale filters."""

    def should_skip(self, ctx: InferenceContext) -> bool:
        # Skip if buys are fully blocked (but not BEAR — bear_only is still allowed)
        return ctx.buy_blocked and not ctx.bear_only

    def run(self, ctx: InferenceContext) -> None:
        from kernel.selection import (  # noqa: PLC0415
            CandidateResult,
            compute_relative_strength,
            is_earnings_blocked,
            is_wash_sale_blocked,
        )
        from kernel.models import score_artifact          # noqa: PLC0415
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        config        = ctx.config
        spec          = config.get("indicator_spec", {})
        vol_win       = int(config.get("regime", {}).get("vol_realized_window", 20))
        regime_p      = config.get("regime_params", {}).get(ctx.regime, {})
        min_score     = float(regime_p.get("min_model_score", 0.10))
        earnings_buf  = int(config.get("regime", {}).get("earnings_buffer_days", 3))
        wash_days     = int(config.get("wash_sale_days", 0))
        sector_map    = config.get("sector_map", {})
        sector_etf_map = config.get("sector_etf_map", {})
        defensive_set = set(config.get("defensive_tickers", []))

        spy_df  = ctx.ohlcv.get("SPY")
        held    = set(ctx.holdings.keys())
        today   = ctx.today
        earnings_cal = ctx.earnings_calendar or {}
        last_sells   = ctx.last_sell_dates

        # Universe: BEAR → defensives only (with no score floor); else all models
        if ctx.bear_only:
            universe  = [t for t in defensive_set if t in ctx.models]
            min_score = 0.0  # no score floor for defensives in BEAR
        else:
            universe = list(ctx.models.keys())

        candidates = []

        for ticker in universe:
            if ticker in held:
                continue
            if ticker not in ctx.ohlcv:
                continue

            # Earnings filter
            if is_earnings_blocked(ticker, today, earnings_cal, earnings_buf):
                ctx.counters["earnings_blocks"] = ctx.counters.get("earnings_blocks", 0) + 1
                continue

            # Wash-sale pre-filter (second guard is in SelectionJob)
            if is_wash_sale_blocked(ticker, today, last_sells, wash_days):
                continue

            stock_df = ctx.ohlcv[ticker]
            artifact = ctx.models.get(ticker)
            if artifact is None:
                continue

            features = build_feature_frame(stock_df, spy_df, spec, vol_win) if spy_df is not None else None
            if features is None or features.empty:
                continue

            sr = score_artifact(artifact, features.iloc[-1], holdings=0)
            log.debug("%s  action=%s  raw=%.4f  rank=%.4f", ticker, sr.signal, sr.raw_score, sr.rank_score)

            if sr.signal != "buy":
                continue
            if sr.rank_score < min_score:
                continue

            # Relative strength vs sector ETF (20-day)
            rs_score = 0.0
            sector = sector_map.get(ticker, "other")
            etf    = sector_etf_map.get(sector)
            if etf and etf in ctx.ohlcv and len(ctx.ohlcv[etf]) >= 21:
                try:
                    stock_r = float(stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1)
                    etf_r   = float(ctx.ohlcv[etf]["close"].iloc[-1] / ctx.ohlcv[etf]["close"].iloc[-21] - 1)
                    rs_score = compute_relative_strength(stock_r, etf_r)
                except Exception:
                    pass

            candidates.append(CandidateResult(
                ticker=ticker,
                raw_score=sr.raw_score,
                rank_score=sr.rank_score,
                rs_score=rs_score,
                detail=f"raw={sr.raw_score:.3f} rank={sr.rank_score:.3f}",
            ))

        ctx.candidates = candidates
        log.info("CandidateJob: %d candidates (bear_only=%s)", len(candidates), ctx.bear_only)
