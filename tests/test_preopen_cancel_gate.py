"""Tests for scripts/preopen_cancel_gate.py

Pins:
  * Below-threshold severity → no cancel attempt
  * Above-threshold severity → cancels ALL pending market orders
  * dry-run skips actual cancel API calls
  * Only market orders considered (not limit / stop)
  * Cancellation failure on one order doesn't abort the batch
  * Missing/stale price data fails open: no broker cancel, no ntfy
  * Current move uses ES 5m price vs prior NYSE cash close, not daily Open
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

    def test_data_unavailable_no_cancel(self):
        from scripts import preopen_cancel_gate as gate

        with patch.object(gate, "compute_overnight_severity",
                          side_effect=ValueError("stale ES=F data")):
            with patch("alpaca.trading.client.TradingClient") as MockClient:
                result = gate.cancel_stale_market_orders(
                    threshold_sigma=2.0, dry_run=False,
                )
                assert result["action"] == "data-unavailable"
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

    def test_severity_uses_intraday_current_price_not_daily_open(self):
        """Daily ES opens can describe the futures session, not the current
        pre-open print. The gate must use 5m ES current-vs-cash-close move."""
        import pandas as pd
        from scripts import preopen_cancel_gate as gate

        now = pd.Timestamp("2026-05-22 13:15:00Z")  # Fri 09:15 ET
        intraday_idx = pd.DatetimeIndex([
            pd.Timestamp("2026-05-21 20:00:00Z"),  # prior NYSE close
            pd.Timestamp("2026-05-22 13:10:00Z"),
            now,
        ])
        intraday = pd.DataFrame(
            {"Close": [5000.0, 4955.0, 4950.0]},
            index=intraday_idx,
        )

        daily_idx = pd.date_range("2026-01-01", periods=100, freq="B")
        closes = pd.Series([100.0] * 100, index=daily_idx)
        rets = pd.Series(
            [0.01 if i % 2 == 0 else -0.01 for i in range(100)],
            index=daily_idx,
        )
        opens = closes.shift(1).fillna(100.0) * (1.0 + rets)
        # Would have been a false +50% trigger in the old daily-Open code.
        opens.iloc[-1] = 150.0
        daily = pd.DataFrame({"Open": opens, "Close": closes}, index=daily_idx)
        expected_sigma = float(
            ((daily["Open"] - daily["Close"].shift(1)) / daily["Close"].shift(1))
            .dropna()
            .tail(60)
            .std()
        )

        def fake_download(_sym, *args, **kwargs):
            return intraday if kwargs.get("interval") == "5m" else daily

        with patch("yfinance.download", side_effect=fake_download):
            m = gate.compute_overnight_severity(
                symbol="ES=F",
                lookback_days=120,
                sigma_window=60,
                now=now,
            )
            expected_move = (4950.0 - 5000.0) / 5000.0
            assert m["source"] == "ES=F"
            assert m["sigma_source"] == "SPY"
            assert abs(m["current_pct"] - expected_move) < 1e-12
            assert abs(m["sigma_60d"] - expected_sigma) < 1e-12
            assert abs(m["severity"] - expected_move / expected_sigma) < 1e-12
            assert m["n_obs"] == 99  # 100 bars - 1 shift

    def test_main_skips_closed_market_day_before_fetching_or_cancelling(self):
        from scripts import preopen_cancel_gate as gate

        with patch.object(gate, "_is_nyse_session_date", return_value=False), \
             patch.object(gate, "cancel_stale_market_orders") as cancel, \
             patch.object(sys, "argv", ["preopen_cancel_gate.py"]):
            gate.main()
            cancel.assert_not_called()
