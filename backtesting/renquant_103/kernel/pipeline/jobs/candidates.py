"""CandidateJob — scan watchlist, apply filters, compute RS score per candidate."""
from __future__ import annotations

import pandas as pd

from ..base import Job
from ..context import InferenceContext
from ...selection import CandidateResult, compute_relative_strength


class CandidateJob(Job):
    """Scans the full watchlist and builds ctx.candidates.

    Filters (each is a skip condition):
      1. Already held
      2. No price data for today
      3. Wash-sale blocked
      4. Earnings window (±earnings_buffer_days)
      5. Model action != "buy"
      6. Calibrated rank_score < min_model_score (regime-aware)

    Then computes RS score vs sector ETF and appends CandidateResult.

    Reads:  ctx.holdings, ctx.ohlcv, ctx.today, ctx.regime_params, ctx.config,
            ctx.last_sell_dates, ctx.earnings_cal, ctx.action_fn, ctx.score_fn
    Writes: ctx.candidates
    """

    def should_skip(self, ctx: InferenceContext) -> bool:
        return ctx.skip_buys

    def run(self, ctx: InferenceContext) -> None:
        cfg          = ctx.config
        today_ts     = pd.Timestamp(ctx.today)
        rp           = ctx.regime_params
        watchlist    = cfg.get("watchlist", [])
        wash_days    = cfg.get("wash_sale_days", 30)
        earn_buffer  = cfg.get("regime", {}).get("earnings_buffer_days", 3)
        min_score    = float(rp.get("min_model_score", 0.10))
        sector_map   = cfg.get("sector_map", {})
        sector_etf   = cfg.get("sector_etf_map", {})
        exportable   = set(cfg.get("_exportable", []))

        candidates = []

        for t in watchlist:
            if t not in exportable:
                continue
            if t in ctx.holdings:
                continue
            df = ctx.ohlcv.get(t)
            if df is None or today_ts not in df.index:
                continue

            # Wash-sale guard
            last_sell = ctx.last_sell_dates.get(t)
            if last_sell and (ctx.today - last_sell).days < wash_days:
                continue

            # Earnings filter
            if _earnings_blocked(t, ctx.today, ctx.earnings_cal, earn_buffer):
                continue

            # Model buy signal
            if ctx.action_fn(t, today_ts) != "buy":
                continue

            # Min score filter (calibrated rank_score)
            rs = ctx.score_fn(t, today_ts)
            if rs is None or rs < min_score:
                continue

            # Relative strength vs sector ETF
            rs_score = 0.0
            sector = sector_map.get(t, "other")
            etf    = sector_etf.get(sector)
            if etf and etf in ctx.ohlcv and today_ts in ctx.ohlcv[etf].index:
                try:
                    stock_ret = float(df["close"].pct_change(20).loc[today_ts])
                    etf_ret   = float(ctx.ohlcv[etf]["close"].pct_change(20).loc[today_ts])
                    rs_score  = compute_relative_strength(stock_ret, etf_ret)
                except Exception:
                    pass

            candidates.append(CandidateResult(
                ticker=t, raw_score=rs, rank_score=rs, rs_score=rs_score,
            ))

        ctx.candidates = candidates


def _earnings_blocked(
    ticker: str,
    today: object,
    earnings_cal: dict,
    buffer_days: int,
) -> bool:
    for d in earnings_cal.get(ticker, []):
        try:
            diff = abs((pd.Timestamp(d).date() - today).days)
            if diff <= buffer_days:
                return True
        except Exception:
            pass
    return False
