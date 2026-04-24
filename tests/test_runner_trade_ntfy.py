"""live/runner.py MUST send ntfy on any trade.

User rule (2026-04-23): "任何脚本，只要发生交易，一定要 ntfy"

Previously ntfy was only fired by shell wrappers (daily_104.sh,
live_only_104.sh). Direct `python -m live.runner --once` invocations
could place real orders silently. This suite pins that `live/runner.py`
itself — at the point of commit — publishes ntfy whenever ctx.orders
or ctx.exits is non-empty.

These are source-level + monkey-patched-network tests so no real ntfy
POST happens during CI.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestSourceLevel:
    """Contract: _notify_trades exists and is called after commit."""

    def test_helper_defined(self):
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        assert "def _notify_trades(" in src, (
            "live/runner.py must define _notify_trades helper"
        )

    def test_called_after_commit(self):
        """The helper must be invoked right after `adapter.commit(ctx)` so
        every order-placing path surfaces to ntfy."""
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        # Both lines in the same cycle function
        idx_commit = src.find("adapter.commit(ctx)")
        idx_notify = src.find("_notify_trades(", idx_commit)
        assert idx_commit > 0
        assert idx_notify > idx_commit
        # No more than ~500 chars of comment between them
        assert idx_notify - idx_commit < 1200


class TestNotifyTradesBehaviour:
    """Behavioural tests — call the helper with stub contexts."""

    def _import(self):
        from live.runner import _notify_trades
        return _notify_trades

    def test_silent_when_no_orders_and_no_exits(self):
        notify = self._import()
        ctx = SimpleNamespace(orders=[], exits=[])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_not_called(), (
            "Silent run must not emit ntfy — avoid notification spam"
        )

    def test_fires_on_buy_order(self):
        notify = self._import()
        ctx = SimpleNamespace(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66, "invest": 2295.96}],
            exits=[],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()
        # Inspect the Request constructed
        req = m.call_args[0][0]
        assert "BUY TSM x6" in req.data.decode()
        assert req.headers.get("Title") == "RENQUANT-104 [full] TRADE"

    def test_fires_on_exit(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="XLU", exit_type="trailing_stop")
        ctx = SimpleNamespace(orders=[], exits=[exit_sig])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only", ctx)
        m.assert_called_once()
        req = m.call_args[0][0]
        assert "EXIT XLU (trailing_stop)" in req.data.decode()

    def test_combines_buys_and_exits(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="XLU", exit_type="rotation")
        ctx = SimpleNamespace(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            exits=[exit_sig],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()
        body = m.call_args[0][0].data.decode()
        assert "BUY TSM" in body
        assert "EXIT XLU" in body

    def test_network_failure_does_not_raise(self, caplog):
        """If ntfy.sh is unreachable, the caller must NOT raise — trade
        already executed on Alpaca, local failure can't roll back."""
        notify = self._import()
        ctx = SimpleNamespace(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            exits=[],
        )
        import logging
        caplog.set_level(logging.WARNING)
        with patch("urllib.request.urlopen",
                   side_effect=ConnectionError("no network")):
            notify("RENQUANT-104", "full", ctx)   # must not raise
        assert any("ntfy publish FAILED" in rec.message
                   for rec in caplog.records)

    def test_respects_topic_env_var(self):
        notify = self._import()
        ctx = SimpleNamespace(
            orders=[{"ticker": "AAPL", "shares": 1, "price": 100.0}],
            exits=[],
        )
        with patch("urllib.request.urlopen") as m:
            with patch.dict("os.environ", {"RENQUANT_NTFY_TOPIC": "alt-topic"}):
                notify("RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        assert req.full_url == "https://ntfy.sh/alt-topic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
