"""InferenceContext — shared state passed through the 7-job InferencePipeline.

Self-contained: only stdlib + dataclasses.  No common/ imports.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceContext:
    """All state needed by the 7-job InferencePipeline.

    Callers (LeanAdapter / RunnerAdapter) populate required inputs before
    calling InferencePipeline.run(ctx).  Each job reads upstream fields and
    writes its own output fields.
    """

    # ── Required inputs (set by adapter before pipeline) ─────────────────────
    config: dict
    today: datetime.date

    # Market data — ticker → pd.DataFrame (open/high/low/close/volume)
    ohlcv: dict = field(default_factory=dict)
    # Recent SPY daily returns as plain floats (most recent last)
    spy_returns: list = field(default_factory=list)

    # Artifacts
    models: dict = field(default_factory=dict)         # ticker → artifact dict
    gmm: Any = None                                     # loaded GMM JSON dict or None
    corr_matrix: Any = None                            # dict[ticker][ticker] → float or None
    earnings_calendar: Any = None                      # dict[ticker] → list[str] or None

    # Portfolio state — populated by adapter from LEAN Portfolio / broker
    holdings: dict = field(default_factory=dict)       # ticker → HoldingState
    last_sell_dates: dict = field(default_factory=dict) # ticker → date | None
    portfolio_value: float = 0.0
    cash: float = 0.0
    prices: dict = field(default_factory=dict)         # ticker → float

    # Persisted cross-bar state (owned by adapter, updated by pipeline)
    hwm: float = 0.0
    skip_buys: bool = False
    regime_state: Any = None                           # kernel.regime.RegimeState
    regime_counts: dict = field(default_factory=dict)  # regime → int

    # ── Pipeline outputs (written by jobs) ───────────────────────────────────
    # RegimeJob
    regime: str = "BULL_CALM"
    confidence: float = 0.5

    # DrawdownJob — updates hwm and skip_buys in place (no separate fields)

    # SellJob
    exits: list = field(default_factory=list)           # list of (ticker, ExitSignal)

    # BuyGatesJob
    buy_blocked: bool = False
    bear_only: bool = False

    # CandidateJob
    candidates: list = field(default_factory=list)      # list of CandidateResult

    # RankingJob
    ranked: list = field(default_factory=list)          # list of CandidateResult, sorted

    # RotationJob — list of RotationPair (held → candidate swaps)
    rotations: list = field(default_factory=list)

    # SelectionJob
    orders: list = field(default_factory=list)          # list of order dicts

    # Telemetry counters — incremented by jobs
    counters: dict = field(default_factory=dict)

    # MonitorIdleStreakTask state — populated by adapter from persisted
    # state file. The Task reads the prior streak counters, updates them,
    # and writes back. Adapter persists across bar boundaries.
    monitor_state: dict = field(default_factory=dict)

    # Feature cache (performance optimization, 2026-04-24): SimAdapter
    # pre-computes per-ticker full-range feature frames ONCE at init.
    # Per-bar tasks (BuildFeaturesTask, ScoreModelTask) slice up to
    # today instead of rebuilding from OHLCV. Live runner leaves this
    # None (fresh data each bar makes cache stale). Key: ticker; Value:
    # full feature DataFrame indexed by bar date.
    feature_cache: dict = field(default_factory=dict)


@dataclass
class TickerInferenceContext:
    """Per-ticker context for parallel sell/candidate jobs.

    Created by the pipeline orchestrator from InferenceContext fields.
    Jobs write only to output fields; they never touch InferenceContext directly.
    """
    # Inputs (read-only)
    ticker: str
    ohlcv: dict                  # shared reference to InferenceContext.ohlcv
    model: Any                   # model artifact dict
    config: dict
    today: datetime.date
    regime: str
    regime_params: dict
    exit_params: dict            # pre-built from regime_params + config

    # Sell-job inputs (None for candidate jobs)
    holding: Any = None          # HoldingState | None
    price: float = 0.0

    # Candidate-job inputs (None for sell jobs)
    earnings_calendar: Any = None  # dict[ticker → list[str]] | None
    last_sell_dates: Any = None    # dict[ticker → date | None] | None

    # Intermediate task outputs — written by one task, read by the next
    features: Any = None         # built feature DataFrame (shared by sell + candidate tasks)
    model_action: str = "hold"   # scored model signal
    rs_score: float = 0.0        # relative-strength score vs sector ETF

    # Optional pre-built feature cache (performance optimization, 2026-04-24).
    # SimAdapter pre-computes full-range feature frames ONCE at init and
    # passes them here. BuildFeaturesTask then slices `[:today]` instead
    # of rebuilding from OHLCV each bar. Live runner doesn't use this —
    # each bar has "new" OHLCV so cache would be stale. Cache should be
    # the FULL feature frame indexed by bar date.
    feature_cache_frame: Any = None

    # Final outputs (written by TickerSellJob or TickerCandidateJob)
    exit_signal: Any = None      # ExitSignal | None
    candidate: Any = None        # CandidateResult | None
