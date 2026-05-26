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
    # Wall-clock timestamp for live/session-aware checks. Sim/LEAN may leave
    # this None to preserve bar-date-only historical semantics.
    run_timestamp: datetime.datetime | None = None

    # Broker-isolation tag (set by RunnerAdapter from broker.broker_name).
    # None for sim/lean paths. When None the legacy live_state.json is read.
    # See kernel/state_paths.py for the path convention.
    broker_name: str | None = None

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
    # 2026-05-09 cost-aware wash-sale: realized $ P/L of the most recent
    # full liquidation per ticker (FIFO). None = unknown (treated as binary
    # block by is_wash_sale_blocked_with_cost). Negative = LOSS → §1091
    # applies → NPV deferred-tax cost computed.
    last_sell_pls: dict = field(default_factory=dict)   # ticker → float | None
    # 2026-05-04 G8 (post-stop re-entry blackout, refactor doc):
    # ticker → date when a path-rule exit (trailing_stop / stop_loss /
    # single_day_loss) last fired. Used by PostStopCooldownFilterTask
    # to block re-entry within a configurable window. Distinct from
    # last_sell_dates (which tracks ANY sell for wash-sale / 30d window
    # on losses only). post-stop blackout fires regardless of P&L sign.
    last_stop_exit_dates: dict = field(default_factory=dict)  # ticker → date
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
    # Rotation V1 persistence gate (2026-04-24): prior bars' proposed
    # (sell, buy) pair sets. Most recent bar last. Adapter seeds from
    # persisted state; task_rotation pushes this bar's proposals after
    # finalizing. Empty when persistence gate disabled.
    prior_rotation_proposals: list = field(default_factory=list)

    # SelectionJob
    orders: list = field(default_factory=list)          # list of order dicts

    # Telemetry counters — incremented by jobs
    counters: dict = field(default_factory=dict)

    # MonitorIdleStreakTask state — populated by adapter from persisted
    # state file. The Task reads the prior streak counters, updates them,
    # and writes back. Adapter persists across bar boundaries.
    monitor_state: dict = field(default_factory=dict)

    # Feature cache (performance optimization, 2026-04-24; live parity
    # 2026-05-25): adapters build run-local per-ticker feature frames from
    # that run's already-fresh OHLCV. Per-bar tasks (BuildFeaturesTask,
    # ScoreModelTask) slice up to today instead of rebuilding from OHLCV.
    # Key: ticker; Value: full feature DataFrame indexed by bar date.
    feature_cache: dict = field(default_factory=dict)

    # ── ExecutionPipeline plumbing (slice 2 of P0 consolidation) ─────────────
    # Adapter sets execution_backend before pp_execution.ExecutionPipeline.run.
    # Each ExecutionTask reads it via ctx.execution_backend.place_market_order.
    # fills accumulates confirmed Fill records produced THIS BAR; adapters
    # drain it in their post-pipeline hooks (trade-log write, equity curve).
    # Both fields are transient: cleared at the start of every ExecutionPipeline
    # run so a stale value from the previous bar can't poison this bar.
    execution_backend: Any = None        # kernel.execution.ExecutionBackend | None

    # ── Meta-label snapshot logger (P4.1, 2026-05-11) ────────────────────────
    # Adapter attaches a kernel.meta_label.SnapshotLogger when
    # `meta_label_training.enabled` is true in config. The
    # MetaLabelLoggingJob (last in InferencePipeline) calls
    # ``snapshot_logger.record(row)`` for every held ticker per bar so the
    # post-sim training pipeline (P4.2+P4.3) has per-day per-position
    # features to fit a triple-barrier exit classifier on. None in prod /
    # untrained sims — Job's should_skip handles that case.
    snapshot_logger: Any = None          # kernel.meta_label.SnapshotLogger | None
    fills: list = field(default_factory=list)  # list[kernel.execution.Fill]


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

    # earnings_calendar: shared by candidate AND sell tctx so the
    # buy-side EarningsFilterTask and the sell-side EarningsBlackoutSellTask
    # both see the same calendar. dict[ticker → list[ISO date strings]] | None.
    earnings_calendar: Any = None
    # last_sell_dates: candidate-job input (None for sell jobs)
    last_sell_dates: Any = None    # dict[ticker → date | None] | None
    # 2026-05-09 cost-aware wash-sale (mirror of InferenceContext field —
    # populated by adapter from broker FIFO-matched fills or sim trade tape)
    last_sell_pls: Any = None      # dict[ticker → float | None] | None

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
    blocked_by: str | None = None  # candidate-gate reason when chain stops before assembly
