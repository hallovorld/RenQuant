"""Top-level `kernel` namespace package.

This file exists ONLY to satisfy pytest test collection (which previously
errored on `kernel/walk_forward_splits.py` import when the directory had
no __init__.py). Using `pkgutil.extend_path` keeps `kernel` a namespace
package so the strategy-local `backtesting/renquant_104/kernel/` continues
to resolve when sys.path includes the strategy dir.

Without this extender, a plain empty __init__.py would shadow the strategy
kernel (no `kernel.pipeline`, no `kernel.execution`, etc.) and break
sim/runner imports under pytest. (2026-05-20: P0-7 fix shipped an empty
__init__.py which silently broke test_no_trade_monitor + sim adapter
imports — caught when fixing the per-trading-day streak counter bug.)
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
