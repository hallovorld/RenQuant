"""Tests for scripts/preopen_cancel_gate.py

Pins:
  * Below-threshold severity → no cancel attempt
  * Above-threshold severity → cancels ALL pending market orders
  * dry-run skips actual cancel API calls
  * Only market orders considered (not limit / stop)
  * Cancellation failure on one order doesn't abort the batch
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _mock_order(symbol="X", order_type="OrderType.MARKET",
                  side="buy", qty="1", id="o-1"):
    o = MagicMock()
    o.symbol = symbol
    o.order_type = order_type
    o.side = side
    o.qty = qty
    o.id = id
    o.position_intent = "buy_to_open"
    return o


class TestBelowThresholdPasses:

    def test_severity_below_threshold_no_cancel(self):
        from scripts import preopen_cancel_gate as gate
        metrics = {
            "source": "ES=F", "prior_close": 5000.0, "latest": 5005.0,
            "current_pct": 0.001, "sigma_60d": 0.005,
            "severity": 0.2, "n_obs": 100,
        }
        with patch.object(gate, "compute_overnight_severity",
                          return_value=metrics):
            # TradingClient should NEVER be instantiated when severity passes
            with patch("alpaca.trading.client.TradingClient") as MockClient:
                result = gate.cancel_stale_market_orders(
                    threshold_sigma=2.0, dry_run=False,
                )
                assert result["action"] == "pass"
                assert result["cancelled"] == []
                MockClient.assert_not_called()


class TestAboveThresholdCancels:

    def test_severity_above_threshold_cancels_market_orders(self):
        from scripts import preopen_cancel_gate as gate
        metrics = {
            "source": "ES=F", "prior_close": 5000.0, "latest": 4700.0,
            "current_pct": -0.06, "sigma_60d": 0.005,
            "severity": -12.0, "n_obs": 100,
        }
        # 3 pending orders: 2 MARKET, 1 LIMIT (should NOT be cancelled by gate)
        m1 = _mock_order(symbol="META", id="o-1")
        m2 = _mock_order(symbol="TXN",  id="o-2")
        lim = _mock_order(symbol="AAPL", order_type="OrderType.LIMIT", id="o-3")

        with patch.object(gate, "compute_overnight_severity",
                          return_value=metrics), \
             patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            with patch("alpaca.trading.client.TradingClient") as MockClient:
                client_inst = MockClient.return_value
                client_inst.get_orders.return_value = [m1, m2, lim]
                result = gate.cancel_stale_market_orders(
                    threshold_sigma=2.0, dry_run=False,
                )
                # Both market orders cancelled; limit untouched
                assert sorted(result["cancelled"]) == ["META", "TXN"]
                # Cancel was called for each market order
                cancel_calls = client_inst.cancel_order_by_id.call_args_list
                cancelled_ids = sorted(c.args[0] for c in cancel_calls)
                assert cancelled_ids == ["o-1", "o-2"]

    def test_dry_run_does_not_actually_cancel(self):
        from scripts import preopen_cancel_gate as gate
        metrics = {
            "source": "ES=F", "prior_close": 5000.0, "latest": 4700.0,
            "current_pct": -0.06, "sigma_60d": 0.005,
            "severity": -12.0, "n_obs": 100,
        }
        m1 = _mock_order(symbol="META", id="o-1")
        with patch.object(gate, "compute_overnight_severity",
                          return_value=metrics), \
             patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            with patch("alpaca.trading.client.TradingClient") as MockClient:
                client_inst = MockClient.return_value
                client_inst.get_orders.return_value = [m1]
                result = gate.cancel_stale_market_orders(
                    threshold_sigma=2.0, dry_run=True,
                )
                assert result["action"] == "dry-run"
                assert result["cancelled"] == []  # nothing actually cancelled
                assert result["considered"] == 1
                client_inst.cancel_order_by_id.assert_not_called()

    def test_cancel_failure_does_not_abort_batch(self):
        from scripts import preopen_cancel_gate as gate
        metrics = {
            "source": "ES=F", "prior_close": 5000.0, "latest": 4700.0,
            "current_pct": -0.06, "sigma_60d": 0.005,
            "severity": -12.0, "n_obs": 100,
        }
        m1 = _mock_order(symbol="META", id="o-1")
        m2 = _mock_order(symbol="TXN",  id="o-2")

        with patch.object(gate, "compute_overnight_severity",
                          return_value=metrics), \
             patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            with patch("alpaca.trading.client.TradingClient") as MockClient:
                client_inst = MockClient.return_value
                client_inst.get_orders.return_value = [m1, m2]
                # First cancel fails; second succeeds. Batch should continue.
                client_inst.cancel_order_by_id.side_effect = [
                    Exception("alpaca timeout"),
                    None,
                ]
                result = gate.cancel_stale_market_orders(
                    threshold_sigma=2.0, dry_run=False,
                )
                # TXN succeeded; META failure logged but not raised
                assert result["cancelled"] == ["TXN"]


class TestSeverityCompute:

    def test_severity_from_synthetic_history(self):
        """Validates the math of compute_overnight_severity with mocked
        yfinance output."""
        import pandas as pd
        from scripts import preopen_cancel_gate as gate

        # Build 100 days of synthetic OHLC with known overnight σ
        idx = pd.date_range("2026-01-01", periods=100, freq="D")
        opens = pd.Series([100.0] * 100, index=idx)
        closes = pd.Series([99.5] * 100, index=idx)  # overnight returns = (100-99.5)/99.5 ≈ +0.5%
        df = pd.DataFrame({"Open": opens, "Close": closes})

        with patch("yfinance.download", return_value=df):
            m = gate.compute_overnight_severity(symbol="ES=F",
                                                  lookback_days=120,
                                                  sigma_window=60)
            # All overnight returns identical → sigma ≈ 0 (within float-eps)
            assert abs(m["sigma_60d"]) < 1e-12
            assert abs(m["severity"])  < 1e-9
            assert m["source"] == "ES=F"
            assert m["n_obs"] == 99  # 100 bars - 1 shift
