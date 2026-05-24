"""Execution-model primitives — single source of truth for fees,
slippage, and T+N cash settlement.

Per CLAUDE.md §5.13.5, every business decision flowing through sim and
LEAN routes through exactly one function here. SimAdapter and LeanAdapter
both consume these modules; no parallel implementations.

Modules:
- fees:        SEC + TAF + custom-bps commission schedule (Alpaca-style)
- slippage:    half-spread + linear-impact (Almgren-Chriss simplified)
- t2_settlement: NYSE-aware T+N cash queue for sell proceeds

Defaults match Alpaca's commission schedule (Q4 2025) and a conservative
2 bps half-spread (≈ 4 bps round-trip on liquid S&P names).
"""

from .backend import ExecutionBackend, FakeBackend
from .fees import FeeConfig, compute_buy_fees, compute_sell_fees
from .slippage import SlippageConfig, slip_fill_price
from .t2_settlement import PendingCashEntry, T2CashQueue
from .types import Fill, OrderIntent, OrderSide

__all__ = [
    # Backend ABC + reference impl (slice 1 of ExecutionPipeline refactor)
    "ExecutionBackend",
    "FakeBackend",
    "Fill",
    "OrderIntent",
    "OrderSide",
    # Fees / slippage / settlement primitives
    "FeeConfig",
    "compute_buy_fees",
    "compute_sell_fees",
    "SlippageConfig",
    "slip_fill_price",
    "PendingCashEntry",
    "T2CashQueue",
]
