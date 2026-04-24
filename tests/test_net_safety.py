"""Regression tests for kernel.net_safety — the hard-timeout + fetch-budget
primitives used by every external network call after the 2026-04-23 yfinance
hang incident (PanelDataJob stuck 10+ min with 19 CLOSE_WAIT sockets).

Pins:
  1. `call_with_timeout` returns None on timeout, no exception raised.
  2. Timeout actually bounds wall time (< 2× the configured limit).
  3. Successful calls return the result and log only on slow.
  4. `FetchBudget` exhaustion short-circuits subsequent calls.
  5. Exceptions inside the wrapped callable return None, not raise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.net_safety import FetchBudget, call_with_timeout  # noqa: E402


class TestCallWithTimeout:
    def test_success_returns_value(self):
        assert call_with_timeout(lambda: 42, timeout_sec=5.0) == 42

    def test_timeout_returns_none(self):
        def _sleeper():
            time.sleep(5.0)
            return "never"
        t0 = time.monotonic()
        result = call_with_timeout(_sleeper, timeout_sec=0.5, label="test")
        elapsed = time.monotonic() - t0
        assert result is None
        assert elapsed < 2.0, f"timeout leaked: {elapsed}s"

    def test_exception_returns_none(self):
        def _raiser():
            raise RuntimeError("boom")
        assert call_with_timeout(_raiser, timeout_sec=5.0) is None

    def test_label_used_in_logs(self, caplog):
        """Label appears in log output on timeout for diagnosability."""
        import logging
        caplog.set_level(logging.WARNING, logger="kernel.net_safety")
        call_with_timeout(lambda: time.sleep(2.0),
                          timeout_sec=0.3, label="yf.earnings_dates(AAPL)")
        assert any("yf.earnings_dates(AAPL)" in rec.message for rec in caplog.records)

    def test_forwards_args_and_kwargs(self):
        def _add(a, b, *, offset=0):
            return a + b + offset
        assert call_with_timeout(_add, 2, 3, offset=10, timeout_sec=5.0) == 15


class TestFetchBudget:
    def test_starts_not_exhausted(self):
        b = FetchBudget(total_sec=10.0)
        assert b.exhausted() is False
        assert b.remaining() == pytest.approx(10.0)

    def test_charge_consumes(self):
        b = FetchBudget(total_sec=10.0)
        b.charge(3.5)
        assert b.consumed == pytest.approx(3.5)
        assert b.remaining() == pytest.approx(6.5)

    def test_exhausted_when_over_budget(self):
        b = FetchBudget(total_sec=5.0)
        b.charge(5.0)
        assert b.exhausted() is True

    def test_call_with_timeout_short_circuits_when_budget_exhausted(self):
        """Once budget is spent, further call_with_timeout calls skip
        the network entirely."""
        b = FetchBudget(total_sec=1.0)
        b.charge(2.0)
        assert b.exhausted() is True
        # This call should NOT run the lambda — would sleep 10s otherwise
        t0 = time.monotonic()
        result = call_with_timeout(
            lambda: time.sleep(10), timeout_sec=5.0, budget=b,
        )
        assert result is None
        assert time.monotonic() - t0 < 0.1, "short-circuit failed"

    def test_budget_caps_per_call_timeout(self):
        """Remaining budget caps the per-call timeout. If budget has 2 s
        left but you asked for 10 s, effective timeout is 2 s."""
        b = FetchBudget(total_sec=10.0)
        b.charge(8.0)   # 2 s remaining
        t0 = time.monotonic()
        result = call_with_timeout(
            lambda: time.sleep(5), timeout_sec=10.0, budget=b,
        )
        elapsed = time.monotonic() - t0
        assert result is None
        assert elapsed < 3.5, f"budget-cap leaked: {elapsed}s"


class TestYFinanceCallSitesWrapped:
    """Source-level check: every `yf.Ticker(sym).*` call outside of cache
    or test code should go through `call_with_timeout`."""

    def test_earnings_surprise_uses_wrapper(self):
        src = (_STRATEGY_DIR / "kernel" / "earnings_surprise.py").read_text()
        assert "call_with_timeout" in src, (
            "earnings_surprise.py must wrap yfinance calls in call_with_timeout"
        )

    def test_fundamentals_uses_wrapper(self):
        src = (_STRATEGY_DIR / "kernel" / "fundamentals.py").read_text()
        assert "call_with_timeout" in src

    def test_watchlist_loops_use_budget(self):
        for fn_name in ("fetch_fundamentals_watchlist",
                        "fetch_earnings_surprise_watchlist",
                        "fetch_insider_trades_watchlist"):
            for filename in ("kernel/fundamentals.py",
                             "kernel/earnings_surprise.py",
                             "kernel/insider_trades.py"):
                src = (_STRATEGY_DIR / filename).read_text()
                if f"def {fn_name}(" in src:
                    assert "FetchBudget" in src, (
                        f"{fn_name} in {filename} must use FetchBudget to "
                        "cap total wall time"
                    )
                    break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
