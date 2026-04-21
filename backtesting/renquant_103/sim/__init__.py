"""Backtest simulation driver for renquant_103.

Glue layer between the kernel and the notebook: walks the OOS date range,
calls kernel primitives in the same order as LEAN/live, and records a
portfolio simulation matching the live pipeline's behavior.
"""
from .runner import SimResult, run_backtest

__all__ = ["SimResult", "run_backtest"]
