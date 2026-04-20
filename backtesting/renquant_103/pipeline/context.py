"""PipelineContext — shared state passed between pipeline jobs."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineContext:
    """All state needed by DataJob → SignalJob → ExecutionJob.

    Callers set the required inputs before calling Pipeline.run(ctx).
    Jobs read their upstream fields and write their own output fields.
    """

    # ── Required inputs (set by caller) ───────────────────────────────────────
    config: dict                   # strategy_config.json as a dict
    strategy_dir: Path             # backtesting/renquant_103/
    sell_only: bool                # True = skip buy phase
    broker: Any                    # BaseBroker (live.broker)
    models: dict[str, Any]        # {symbol: kernel artifact dict or common model}

    # ── Populated by DataJob ──────────────────────────────────────────────────
    ohlcv: dict[str, Any] = field(default_factory=dict)      # {symbol: pd.DataFrame}
    df_spy: Any = None                                          # pd.DataFrame | None
    gmm_artifact: dict | None = None
    corr_matrix: dict = field(default_factory=dict)
    earnings_cal: dict = field(default_factory=dict)

    # ── Populated by SignalJob ────────────────────────────────────────────────
    # Regime
    regime: str = "BULL_CALM"
    confidence: float = 0.5
    in_transition: bool = False
    transition_countdown: int = 0
    # Resolved per-regime params
    regime_params: dict = field(default_factory=dict)   # e.g. stop_loss_pct, max_hold_days
    # Market gate results
    spy_price: float = 0.0
    spy_above_ema50: bool = True
    spy_vel_ok: bool = True
    # Ranked buy candidates from scanner
    candidates: list = field(default_factory=list)   # list[CandidateResult]

    # ── Populated by SignalJob (from broker + live_state.json) ────────────────
    account_value: float = 0.0
    cash_avail: float = 0.0
    positions_cache: dict = field(default_factory=dict)   # {symbol: broker pos dict}
    pending_orders: set = field(default_factory=set)       # symbols with open orders
    held: list = field(default_factory=list)               # symbols currently held
    circuit_open: bool = False                             # drawdown breaker active

    # Live state (loaded from live_state.json, persisted back at end)
    state: dict = field(default_factory=dict)
    entry_dates: dict = field(default_factory=dict)        # {symbol: "YYYY-MM-DD"}
    sell_streaks: dict = field(default_factory=dict)       # {symbol: int}
    last_sell_dates: dict = field(default_factory=dict)   # {symbol: "YYYY-MM-DD"}
    position_hwm: dict = field(default_factory=dict)      # {symbol: float}

    # ── Convenience ──────────────────────────────────────────────────────────
    today: datetime.date = field(default_factory=datetime.date.today)
    today_str: str = field(default_factory=lambda: datetime.date.today().isoformat())
