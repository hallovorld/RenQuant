"""InferenceContext — normalized state passed between all inference pipeline jobs.

All three platforms (notebook, LEAN, live runner) provide a thin adapter that
populates this context before calling InferencePipeline().run(ctx).  Jobs only
ever read/write this dataclass — they have no knowledge of the platform.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceContext:
    # ── Inputs: set by adapter before each bar ─────────────────────────────────

    today: datetime.date

    # Normalized OHLCV: {ticker: DataFrame with columns open/high/low/close/volume}
    ohlcv: dict[str, Any]

    # Recent SPY daily log-returns as float array (full history up to today)
    spy_returns: Any   # np.ndarray

    # Yesterday's close per ticker (needed by SellJob single-day loss gate)
    prev_closes: dict[str, float]

    # Portfolio state
    holdings: dict[str, Any]   # {ticker: HoldingState}
    pos_shares: dict[str, float]
    cash: float
    portfolio_value: float

    # Model signal lookups: each is a callable (ticker, today_ts) -> str|float|None
    # Adapters supply these closures so jobs don't import pandas/broker logic.
    action_fn: Any    # (ticker, today_ts) -> "buy"|"hold"|"sell"
    score_fn: Any     # (ticker, today_ts) -> float | None  (calibrated rank_score)

    # Artifacts
    gmm_artifact: dict | None
    corr_dict: dict[str, dict[str, float]]
    earnings_cal: dict[str, list[str]]

    # Config (read-only)
    config: dict

    # ── Persistent state: lives across bars, adapter owns storage ─────────────

    # High-water mark for drawdown breaker
    hwm: float

    # Regime state object (carries CUSUM countdown, last regime, etc.)
    regime_state: Any   # kernel.regime.RegimeState

    # Wash-sale clock: {ticker: date of last sell}
    last_sell_dates: dict[str, datetime.date]

    # ── Outputs: written by jobs, read by downstream jobs ─────────────────────

    # RegimeJob → all downstream
    regime: str = "BULL_CALM"
    regime_confidence: float = 0.5
    in_transition: bool = False
    regime_params: dict = field(default_factory=dict)

    # DrawdownJob → BuyGatesJob, SelectionJob
    skip_buys: bool = False

    # SellJob → ExecutionJob (adapter applies these after pipeline)
    exit_actions: list[dict] = field(default_factory=list)

    # CandidateJob → RankingJob
    candidates: list[Any] = field(default_factory=list)

    # RankingJob → SelectionJob
    ranked: list[Any] = field(default_factory=list)

    # SelectionJob → adapter (adapter applies buy orders)
    orders: list[dict] = field(default_factory=list)

    # DrawdownJob → equity curve (adapter appends this)
    equity_point: dict = field(default_factory=dict)
