"""Tests for Z9 runner integration — broker-side stop placement in commit().

Z9 (2026-04-28 NVTS post-mortem): polled stop_loss is gated by 30-min
cron cadence; broker-side GTC stops trigger in ms regardless of poll.

Layered tests:
  1. Source-level wiring (string contracts on the runner)
  2. _z9_enabled / _z9_stop_pct semantics
  3. _z9_place_or_replace_stop never-loosen invariant
  4. _z9_cancel_stop idempotence
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

RUNNER_PATH = REPO_ROOT / "backtesting/renquant_104/adapters/runner.py"
RUNNER_SRC = RUNNER_PATH.read_text()


# ── Source-level wiring contracts ──────────────────────────────────────────

class TestZ9WiringPresent:
    def test_z9_audit_tag(self):
        assert "Z9 (2026-04-28)" in RUNNER_SRC

    def test_state_field_loaded(self):
        # state.get("stop_orders", {}) read at make_context start
        assert 'state.get("stop_orders",     {})' in RUNNER_SRC
        assert "self._stop_orders    = stop_orders" in RUNNER_SRC

    def test_state_field_persisted(self):
        # written back to live_state.json on commit
        assert '"stop_orders":       self._stop_orders' in RUNNER_SRC

    def test_helpers_defined(self):
        assert "def _z9_enabled" in RUNNER_SRC
        assert "def _z9_stop_pct" in RUNNER_SRC
        assert "def _z9_place_or_replace_stop" in RUNNER_SRC
        assert "def _z9_cancel_stop" in RUNNER_SRC

    def test_buy_path_wired(self):
        # After successful BUY, place stop with current total qty
        assert "self._z9_place_or_replace_stop" in RUNNER_SRC

    def test_full_sell_cancels_stop(self):
        # On full liquidation (not partial), cancel the stop
        assert 'self._z9_cancel_stop(ticker, reason="full liquidation")' in RUNNER_SRC

    def test_trim_replaces_stop(self):
        # Partial sell (TRIM) reduces stop qty without loosening price
        assert 'reason="trim → flat"' in RUNNER_SRC

    def test_external_sell_cancels_stop(self):
        # Z2 STATE-EXT-SELL detection also cancels orphan stops
        assert 'self._z9_cancel_stop(t, reason="external disposition")' in RUNNER_SRC

    def test_gc_drops_orphan_stops(self):
        assert 'STATE-GC: dropped %d stale stop_orders' in RUNNER_SRC

    def test_default_disabled(self):
        # Config flag must default to False
        assert 'cfg.get("enabled", False)' in RUNNER_SRC

    def test_capability_check(self):
        # supports_broker_side_stops gate prevents calling unsupported brokers
        assert 'supports_broker_side_stops' in RUNNER_SRC


# ── Helper-method semantics (mocked broker) ────────────────────────────────

@pytest.fixture
def adapter():
    """Build a minimal RunnerAdapter without invoking __init__ (which
    pulls in conda env, parquet, db, etc.). We just want the helper
    methods bound to a stub instance."""
    from adapters.runner import RunnerAdapter  # noqa: PLC0415
    a = RunnerAdapter.__new__(RunnerAdapter)
    a._stop_orders = {}
    a._last_ctx_stop_pct = 0.06
    a._broker = MagicMock()
    a._broker.supports_broker_side_stops = MagicMock(return_value=True)
    a._broker.place_stop_order = MagicMock(
        return_value={"order_id": "stop-1", "status": "accepted"},
    )
    a._broker.cancel_order = MagicMock(return_value=True)
    return a


class _FakeCtx:
    def __init__(self, regime="BULL_CALM", enabled=True, sdl_pct=0.06):
        self.regime = regime
        self.config = {
            "live": {"broker_side_stops": {"enabled": enabled}},
            "regime_params": {regime: {"max_single_day_loss_pct": sdl_pct}},
        }


class TestZ9Enabled:
    def test_enabled_when_config_true_and_broker_supports(self, adapter):
        ctx = _FakeCtx(enabled=True)
        assert adapter._z9_enabled(ctx) is True

    def test_disabled_when_config_false(self, adapter):
        ctx = _FakeCtx(enabled=False)
        assert adapter._z9_enabled(ctx) is False

    def test_disabled_when_broker_unsupported(self, adapter):
        adapter._broker.supports_broker_side_stops = MagicMock(return_value=False)
        ctx = _FakeCtx(enabled=True)
        assert adapter._z9_enabled(ctx) is False


class TestZ9StopPct:
    def test_reads_from_regime_params(self, adapter):
        ctx = _FakeCtx(regime="BULL_CALM", sdl_pct=0.06)
        assert adapter._z9_stop_pct(ctx) == 0.06

    def test_default_when_missing(self, adapter):
        ctx = _FakeCtx()
        ctx.config = {"regime_params": {}}
        ctx.regime = "BULL_CALM"
        assert adapter._z9_stop_pct(ctx) == 0.06


class TestZ9PlaceOrReplaceStop:
    def test_places_at_correct_target(self, adapter):
        adapter._z9_place_or_replace_stop("AAPL", 100, 200.0, "2026-04-28")
        # 200 × (1 - 0.06) = 188.0
        adapter._broker.place_stop_order.assert_called_once_with("AAPL", 100, 188.0)
        assert "AAPL" in adapter._stop_orders
        assert adapter._stop_orders["AAPL"]["stop_price"] == 188.0

    def test_never_loosens_on_topup(self, adapter):
        # Initial: BUY 100 @ $200 → stop @ $188
        adapter._z9_place_or_replace_stop("AAPL", 100, 200.0, "2026-04-28")
        # TOPUP: add 50 @ $220. Naive: stop = 220 × 0.94 = $206.8 (LOOSER)
        # Invariant: must keep stop at $188 (the tighter level).
        adapter._z9_place_or_replace_stop("AAPL", 150, 220.0, "2026-04-29")
        last_call = adapter._broker.place_stop_order.call_args
        symbol, qty, stop_price = last_call.args
        assert symbol == "AAPL"
        assert qty == 150
        assert stop_price == 188.0, (
            f"TOPUP must not loosen stop — got ${stop_price:.2f}, "
            "expected ≤ $188 (the existing tighter level)"
        )

    def test_tightens_on_average_down(self, adapter):
        # Initial: BUY 100 @ $200 → stop @ $188
        adapter._z9_place_or_replace_stop("AAPL", 100, 200.0, "2026-04-28")
        # Average down: BUY 100 more @ $180. New target = 180 × 0.94 = $169.2
        # That's TIGHTER than $188 → use the new value.
        adapter._z9_place_or_replace_stop("AAPL", 200, 180.0, "2026-04-29")
        last_call = adapter._broker.place_stop_order.call_args
        symbol, qty, stop_price = last_call.args
        assert stop_price < 188.0
        assert abs(stop_price - 169.2) < 1e-6

    def test_cancels_existing_before_placing_new(self, adapter):
        adapter._z9_place_or_replace_stop("AAPL", 100, 200.0, "2026-04-28")
        adapter._broker.cancel_order.assert_not_called()
        adapter._z9_place_or_replace_stop("AAPL", 150, 220.0, "2026-04-29")
        adapter._broker.cancel_order.assert_called_once()

    def test_zero_qty_no_op(self, adapter):
        adapter._z9_place_or_replace_stop("AAPL", 0, 200.0, "2026-04-28")
        adapter._broker.place_stop_order.assert_not_called()

    def test_zero_price_no_op(self, adapter):
        adapter._z9_place_or_replace_stop("AAPL", 100, 0.0, "2026-04-28")
        adapter._broker.place_stop_order.assert_not_called()

    def test_broker_failure_does_not_corrupt_state(self, adapter):
        adapter._broker.place_stop_order = MagicMock(
            side_effect=RuntimeError("API down"),
        )
        adapter._z9_place_or_replace_stop("AAPL", 100, 200.0, "2026-04-28")
        # On failure, no entry recorded
        assert "AAPL" not in adapter._stop_orders


class TestZ9CancelStop:
    def test_cancels_existing(self, adapter):
        adapter._stop_orders["AAPL"] = {
            "order_id": "stop-1", "stop_price": 188.0, "qty": 100, "stamped_at": "x",
        }
        adapter._z9_cancel_stop("AAPL", reason="test")
        adapter._broker.cancel_order.assert_called_once_with("stop-1")
        assert "AAPL" not in adapter._stop_orders

    def test_no_op_when_no_existing(self, adapter):
        adapter._z9_cancel_stop("AAPL", reason="test")
        adapter._broker.cancel_order.assert_not_called()

    def test_state_cleared_even_if_broker_fails(self, adapter):
        adapter._stop_orders["AAPL"] = {
            "order_id": "stop-1", "stop_price": 188.0, "qty": 100, "stamped_at": "x",
        }
        adapter._broker.cancel_order = MagicMock(side_effect=RuntimeError("down"))
        adapter._z9_cancel_stop("AAPL", reason="test")
        # Broker call failed but local state is cleared (so we don't keep
        # trying to cancel a phantom stop)
        assert "AAPL" not in adapter._stop_orders
