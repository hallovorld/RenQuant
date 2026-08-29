"""RenQuant#618 class B — entry date = CURRENT trip start; entry state
cleared on the realized broker qty; `min_reentry_days` on every BUY path.

Incident (2026-08-24..28, live): `first_fill_map` was built from the OLDEST
BUY fill ever per symbol, so a re-entered name inherited its previous
trip's date (`ENTRY-DATE-SEED NVDA ← 2026-04-17` on 2026-08-25 → hold=130d
one session after the buy) and `min_hold_days=5` never protected it; the
`is_topup` stamp keyed on the same stale map; and `min_reentry_days=5` was
enforced only by the QP wash mask, so the non-QP SELECT path re-bought VLO
7h after its exit filled for a gain.

Three layers pinned here:
  1. the pure trip-lifecycle replay (adapters/runner_trip_lifecycle)
     — single trip, exit + re-entry, partial sell, multiple round trips,
     price-less SELLs NOT dropped, order_id de-dup, broker-anchored
     correction when the history is inconsistent (class C);
  2. the entry-date decision table (resolve_entry_date): seed / sentinel /
     keep / backfill (inside the trip) / RESEED (previous trip);
  3. the REAL RunnerAdapter.commit path: a partial-intent sell that leaves
     the broker flat clears entry state + stamps the wash-sale clock; a
     fresh BUY inside the cooldown is skipped (days 0-4) and allowed at
     day 5+; a held name (top-up) is never blocked; a stale entry for a
     flat name does not turn a fresh BUY into a TOPUP.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.runner_trip_lifecycle import (  # noqa: E402
    TripState,
    days_since_last_exit,
    fill_trade_date,
    last_exit_map,
    normalize_fills,
    reentry_blocked,
    replay_trip_lifecycle,
    resolve_entry_date,
    trip_start_map,
)

D = datetime.date


def _fill(sym, side, qty, day, *, price=100.0, order_id=None, hour=13):
    """Alpaca-shaped fill (umbrella live/alpaca_broker.get_filled_orders)."""
    return {
        "order_id": order_id or f"{sym}-{side}-{day}-{qty}",
        "symbol": sym,
        "action": side,
        "qty": float(qty),
        "filled_at": f"{day}T{hour:02d}:30:01+00:00",
        "avg_price": price,
        "partial": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. Trip-lifecycle replay (pure)
# ═════════════════════════════════════════════════════════════════════════════

class TestReplayTripLifecycle:
    def test_single_trip_start_is_first_buy(self):
        fills = [_fill("VLO", "BUY", 2, "2026-08-05"),
                 _fill("VLO", "BUY", 3, "2026-08-12")]   # top-up keeps the trip
        states, dropped = replay_trip_lifecycle(fills)
        assert dropped == 0
        assert states["VLO"].trip_start == D(2026, 8, 5)
        assert states["VLO"].last_exit is None
        assert states["VLO"].replay_qty == 5.0
        assert trip_start_map(states) == {"VLO": D(2026, 8, 5)}

    def test_exit_and_reentry_gives_the_reentry_date(self):
        """The incident shape: NVDA bought 04-17, fully sold, re-bought
        08-25 → the trip start is 08-25, NOT 04-17."""
        fills = [_fill("NVDA", "BUY", 7, "2026-04-17"),
                 _fill("NVDA", "SELL", 7, "2026-05-02"),
                 _fill("NVDA", "BUY", 7, "2026-08-25")]
        states, _ = replay_trip_lifecycle(fills)
        st = states["NVDA"]
        assert st.trip_start == D(2026, 8, 25)
        assert st.last_exit == D(2026, 5, 2)
        assert st.replay_qty == 7.0

    def test_partial_sell_keeps_the_trip(self):
        fills = [_fill("APH", "BUY", 8, "2026-07-01"),
                 _fill("APH", "SELL", 3, "2026-07-20"),   # trim
                 _fill("APH", "BUY", 2, "2026-08-01")]    # top-up
        states, _ = replay_trip_lifecycle(fills)
        st = states["APH"]
        assert st.trip_start == D(2026, 7, 1)
        assert st.last_exit is None
        assert st.replay_qty == 7.0

    def test_multiple_round_trips(self):
        """VLO/NVDA ping-pong: four full round trips; the open trip is the
        last BUY, last_exit is the last flattening SELL."""
        fills = [_fill("VLO", "BUY", 2, "2026-08-05"),
                 _fill("VLO", "SELL", 2, "2026-08-25"),
                 _fill("VLO", "BUY", 5, "2026-08-26"),
                 _fill("VLO", "SELL", 5, "2026-08-27"),
                 _fill("VLO", "BUY", 5, "2026-08-28")]
        states, _ = replay_trip_lifecycle(fills)
        st = states["VLO"]
        assert st.trip_start == D(2026, 8, 28)
        assert st.last_exit == D(2026, 8, 27)
        assert st.replay_qty == 5.0
        # Flat after the final SELL → no open trip, last_exit = that SELL.
        states2, _ = replay_trip_lifecycle(fills[:-1])
        assert states2["VLO"].trip_start is None
        assert states2["VLO"].last_exit == D(2026, 8, 27)
        assert trip_start_map(states2) == {}
        assert last_exit_map(states2) == {"VLO": D(2026, 8, 27)}

    def test_price_less_sell_is_not_dropped(self):
        """runner_tax_lots drops price<=0 rows (it needs a basis); the
        lifecycle replay is qty-only and MUST still close the trip."""
        fills = [_fill("VLO", "BUY", 2, "2026-08-05"),
                 _fill("VLO", "SELL", 2, "2026-08-25", price=0.0),
                 _fill("VLO", "BUY", 5, "2026-08-26")]
        fills[1]["avg_price"] = 0.0
        states, dropped = replay_trip_lifecycle(fills)
        assert dropped == 0
        assert states["VLO"].trip_start == D(2026, 8, 26)
        assert states["VLO"].last_exit == D(2026, 8, 25)

    def test_sell_before_any_buy_closes_and_counts_as_exit(self):
        """History window starts mid-trip: the SELL is the exit event, the
        next BUY opens a fresh trip."""
        fills = [_fill("X", "SELL", 4, "2026-06-01"),
                 _fill("X", "BUY", 4, "2026-06-10")]
        states, _ = replay_trip_lifecycle(fills)
        assert states["X"].last_exit == D(2026, 6, 1)
        assert states["X"].trip_start == D(2026, 6, 10)

    def test_duplicate_order_id_is_deduped(self):
        """Class C: the page walk re-fetches the boundary order. Without
        de-dup the duplicated BUY keeps the running qty above zero and the
        trip never closes → the stale date comes back."""
        first = _fill("NVDA", "BUY", 7, "2026-04-17", order_id="o-1")
        fills = [first, dict(first),
                 _fill("NVDA", "SELL", 7, "2026-05-02"),
                 _fill("NVDA", "BUY", 7, "2026-08-25")]
        states, dropped = replay_trip_lifecycle(fills)
        assert dropped == 1
        assert states["NVDA"].trip_start == D(2026, 8, 25)

    def test_unusable_rows_are_counted_never_silently_dropped(self):
        fills = [
            {"symbol": "A", "action": "BUY", "qty": 1},                     # no date
            {"symbol": "A", "qty": 1, "filled_at": "2026-01-01T13:00:00Z"},  # no side
            {"symbol": "A", "action": "BUY", "qty": 0, "filled_at": "2026-01-01T13:00:00Z"},
            {"action": "BUY", "qty": 1, "filled_at": "2026-01-01T13:00:00Z"},  # no symbol
            "not-a-dict",
            _fill("A", "BUY", 1, "2026-01-02"),
        ]
        by_symbol, dropped = normalize_fills(fills)
        assert dropped == 5
        assert [f.date for f in by_symbol["A"]] == [D(2026, 1, 2)]

    def test_anchored_correction_when_duplicate_buy_inflates_history(self):
        """Forward replay lands on 14 while the broker holds 7 (the exact
        `LIVE-TAX-LOTS: NVDA reconstructed lot qty 14 != broker 7` line):
        walk back from the broker qty → the latest BUY opened the trip."""
        fills = [_fill("NVDA", "BUY", 7, "2026-04-17", order_id="a"),
                 _fill("NVDA", "BUY", 7, "2026-04-17", order_id="b"),  # bogus dup, new id
                 _fill("NVDA", "SELL", 7, "2026-05-02"),
                 _fill("NVDA", "BUY", 7, "2026-08-25")]
        states, _ = replay_trip_lifecycle(fills, current_qty={"NVDA": 7.0})
        st = states["NVDA"]
        assert st.consistent is False and st.anchored is True
        assert st.replay_qty == 14.0 and st.broker_qty == 7.0
        assert st.trip_start == D(2026, 8, 25)

    def test_anchored_unknown_when_history_does_not_reach_flat(self):
        """A top-up is visible but the trip's first BUY is outside the
        window: never guess a LATER date — unknown keeps the state."""
        fills = [_fill("PANW", "BUY", 2, "2026-08-10")]
        states, _ = replay_trip_lifecycle(fills, current_qty={"PANW": 3.0})
        assert states["PANW"].consistent is False
        assert states["PANW"].trip_start is None

    def test_consistent_history_keeps_forward_result(self):
        fills = [_fill("VLO", "BUY", 5, "2026-08-26")]
        states, _ = replay_trip_lifecycle(fills, current_qty={"VLO": 5.0})
        assert states["VLO"].consistent is True and states["VLO"].anchored is False
        assert states["VLO"].trip_start == D(2026, 8, 26)

    def test_flat_at_broker_with_inconsistent_history_uses_latest_sell(self):
        fills = [_fill("VLO", "BUY", 2, "2026-08-05", order_id="a"),
                 _fill("VLO", "BUY", 2, "2026-08-05", order_id="b"),
                 _fill("VLO", "SELL", 2, "2026-08-25")]
        states, _ = replay_trip_lifecycle(fills, current_qty={"VLO": 0.0})
        st = states["VLO"]
        assert st.trip_start is None
        assert st.last_exit == D(2026, 8, 25)

    def test_symbols_absent_from_current_qty_keep_forward(self):
        fills = [_fill("Z", "BUY", 1, "2026-08-01")]
        states, _ = replay_trip_lifecycle(fills, current_qty={"OTHER": 1.0})
        assert states["Z"].broker_qty is None and states["Z"].consistent is True

    def test_execution_subrepo_schema_is_accepted(self):
        """side/filled_qty/id keys (renquant-execution broker) replay too."""
        fills = [{"id": "e1", "symbol": "GE", "side": "buy", "filled_qty": 3,
                  "filled_at": "2026-05-14T14:00:00Z"},
                 {"id": "e2", "symbol": "GE", "side": "sell", "filled_qty": 3,
                  "filled_at": "2026-06-01T14:00:00Z"}]
        states, dropped = replay_trip_lifecycle(fills)
        assert dropped == 0
        assert states["GE"].trip_start is None
        assert states["GE"].last_exit == D(2026, 6, 1)


class TestFillTradeDate:
    def test_aware_utc_maps_to_new_york_trade_date(self):
        # 00:30Z on the 26th is 20:30 ET on the 25th.
        assert fill_trade_date("2026-08-26T00:30:00+00:00") == D(2026, 8, 25)
        assert fill_trade_date("2026-08-26T00:30:00Z") == D(2026, 8, 25)
        assert fill_trade_date("2026-08-25T13:30:01Z") == D(2026, 8, 25)

    def test_naive_and_date_only_fall_back_to_the_calendar_date(self):
        assert fill_trade_date("2026-08-25T13:30:01") == D(2026, 8, 25)
        assert fill_trade_date("2026-08-25") == D(2026, 8, 25)

    def test_garbage_is_none(self):
        assert fill_trade_date("") is None
        assert fill_trade_date(None) is None
        assert fill_trade_date("not a date") is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Entry-date decision table
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveEntryDate:
    TODAY = D(2026, 8, 25)

    def test_seed_from_trip_start(self):
        assert resolve_entry_date(None, D(2026, 8, 25), self.TODAY) == ("2026-08-25", "seed")

    def test_sentinel_when_no_history(self):
        val, action = resolve_entry_date(None, None, self.TODAY, sentinel_days=31)
        assert (val, action) == ("2026-07-25", "sentinel")

    def test_keep_when_equal_or_trip_unknown(self):
        assert resolve_entry_date("2026-08-25", D(2026, 8, 25), self.TODAY) == ("2026-08-25", "keep")
        assert resolve_entry_date("2026-08-05", None, self.TODAY) == ("2026-08-05", "keep")

    def test_backfill_inside_the_trip(self):
        """State stamped LATER than the trip's first fill → move back to it
        (the pre-existing ENTRY-DATE-BACKFILL, bounded by the trip)."""
        assert resolve_entry_date("2026-08-20", D(2026, 8, 12), self.TODAY) == ("2026-08-12", "backfill")

    def test_reseed_when_state_belongs_to_a_previous_trip(self):
        """The incident: entry_dates.VLO = 2026-08-05 (previous trip) while
        the current trip started 2026-08-26 → RESEED, not preserve."""
        assert resolve_entry_date("2026-08-05", D(2026, 8, 26), D(2026, 8, 28)) == ("2026-08-26", "reseed")
        assert resolve_entry_date("2026-04-17", D(2026, 8, 25), self.TODAY) == ("2026-08-25", "reseed")

    def test_unparseable_state_is_treated_as_today(self):
        # today (08-25) > trip start (08-12) → backfill
        assert resolve_entry_date("garbage", D(2026, 8, 12), self.TODAY) == ("2026-08-12", "backfill")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Re-entry cooldown rule (pure)
# ═════════════════════════════════════════════════════════════════════════════

class TestReentryBlocked:
    TODAY = D(2026, 8, 25)

    @pytest.mark.parametrize("days_ago,expected", [(0, True), (1, True), (4, True),
                                                   (5, False), (6, False), (30, False)])
    def test_five_day_cooldown(self, days_ago, expected):
        last = (self.TODAY - datetime.timedelta(days=days_ago)).isoformat()
        blocked, days, when = reentry_blocked(
            "VLO", self.TODAY, min_reentry_days=5,
            state_last_sell=last, replay_last_exit=None,
        )
        assert blocked is expected and days == days_ago
        assert when == self.TODAY - datetime.timedelta(days=days_ago)

    def test_later_ledger_wins(self):
        blocked, days, _ = reentry_blocked(
            "VLO", self.TODAY, min_reentry_days=5,
            state_last_sell="2026-07-01",                    # stale state
            replay_last_exit=D(2026, 8, 25),                 # today's fill
        )
        assert blocked is True and days == 0

    def test_replay_only_ledger(self):
        blocked, days, _ = reentry_blocked(
            "VLO", self.TODAY, min_reentry_days=5,
            state_last_sell=None, replay_last_exit=D(2026, 8, 22),
        )
        assert blocked is True and days == 3

    def test_no_exit_or_disabled_never_blocks(self):
        assert reentry_blocked("VLO", self.TODAY, min_reentry_days=5,
                               state_last_sell=None, replay_last_exit=None) == (False, None, None)
        assert reentry_blocked("VLO", self.TODAY, min_reentry_days=0,
                               state_last_sell="2026-08-25", replay_last_exit=None) == (False, None, None)
        assert days_since_last_exit("VLO", self.TODAY, state_last_sell="bad",
                                    replay_last_exit=None) == (None, None)


# ═════════════════════════════════════════════════════════════════════════════
# 4. The REAL RunnerAdapter.commit path
# ═════════════════════════════════════════════════════════════════════════════

TODAY = D(2026, 8, 25)


class FakeBroker:
    """Minimal qty-aware fake driving RunnerAdapter.commit end-to-end
    (same shape as tests/test_s_frac_stage0_commit_contract.FakeBroker)."""

    broker_name = "paper"

    def __init__(self, fills=None, positions=None):
        self.fills = dict(fills or {})          # ticker -> broker order result
        self.positions = dict(positions or {})  # ticker -> held qty (broker truth)
        self.place_order_calls: list[tuple] = []
        self.cancel_calls: list[str] = []

    def get_open_orders(self):
        return set()

    def get_position(self, ticker):
        return float(self.positions.get(ticker, 0.0))

    def place_order(self, ticker, side, qty):
        self.place_order_calls.append((ticker, side, qty))
        result = dict(self.fills[ticker])
        filled = float(result.get("filled_qty") or 0.0)
        if filled > 0:
            if side == "BUY":
                self.positions[ticker] = self.positions.get(ticker, 0.0) + filled
            else:
                self.positions[ticker] = self.positions.get(ticker, 0.0) - filled
                if abs(self.positions[ticker]) <= 1e-12:
                    self.positions.pop(ticker, None)
        return result

    def supports_broker_side_stops(self, symbol=None, qty=None):
        return False

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)


def _config(*, min_reentry_days=5):
    return {
        "model_name": "renquant_104",
        "watchlist": ["VLO", "NVDA", "HPE"],
        "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.06}},
        "live": {"broker_side_stops": {"enabled": False}},
        "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                "long_term_threshold_days": 365},
        "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
        "persistence": {"enabled": False},
        "min_reentry_days": min_reentry_days,
        "wash_sale_days": 30,
    }


def _make_adapter(tmp_path, *, config, broker, positions=None, entry_dates=None,
                  entry_signals=None, position_hwm=None, last_sell_dates=None,
                  trip_states=None):
    """RunnerAdapter shell with exactly the state commit() touches (bypasses
    __init__; drives the REAL commit() body)."""
    from adapters.runner import RunnerAdapter  # noqa: PLC0415

    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    ra = RunnerAdapter.__new__(RunnerAdapter)
    ra._config = config
    ra._models = {}
    ra._broker = broker
    ra._strategy_dir = strategy_dir
    ra._sell_only = False
    ra._broker_name = "paper"
    ra._db = None
    ra._universe_rejections = {}
    ra._software_stops = None
    ra._positions_cache = dict(positions or {})
    ra._entry_dates = dict(entry_dates or {})
    ra._entry_signals = dict(entry_signals or {})
    ra._sell_streaks = {}
    ra._protection_breaches = {}
    ra._position_hwm = dict(position_hwm or {})
    ra._last_sell_dates_str = dict(last_sell_dates or {})
    ra._last_stop_exit_dates_str = {}
    ra._stop_orders = {}
    ra._recent_sell_orders = {}
    ra._state = {}
    ra._last_ctx_stop_pct = 0.06
    if trip_states is not None:
        ra._trip_states = dict(trip_states)
    return ra


def _make_ctx(config, *, today=TODAY, orders=(), exits=(), holdings=None,
              prices=None, cash=10_000.0):
    return SimpleNamespace(
        today=today, config=config, orders=list(orders), exits=list(exits),
        holdings=dict(holdings or {}), prices=dict(prices or {}), cash=cash,
        regime="BULL_CALM", confidence=0.9, hwm=100_000.0, skip_buys=False,
        monitor_state={}, regime_state=None, counters={}, candidates=[],
        buy_blocked=False, bear_only=False, pending_broker_tickers=set(),
        rotations=[],
    )


def _saved_state(tmp_path):
    f = tmp_path / "backtesting" / "renquant_104" / "live_state.paper.json"
    assert f.exists(), "commit() must persist live_state"
    return json.loads(f.read_text())


class TestCommitClearsEntryStateOnRealizedQtyZero:
    def _sell(self, tmp_path, *, broker_qty_before_fill, filled_qty, intent_qty):
        from kernel.exits import ExitSignal, HoldingState  # noqa: PLC0415
        config = _config()
        broker = FakeBroker(
            fills={"VLO": {"status": "filled", "order_id": "sell-1",
                           "filled_qty": filled_qty, "filled_avg_price": 349.0}},
            positions={"VLO": broker_qty_before_fill},
        )
        ra = _make_adapter(
            tmp_path, config=config, broker=broker,
            positions={"VLO": {"qty": 10.0, "qty_available": 10.0,
                               "avg_entry_price": 340.0}},
            entry_dates={"VLO": "2026-08-05"},
            entry_signals={"VLO": {"rank_score": 0.5}},
            position_hwm={"VLO": 350.0},
        )
        hs = HoldingState(entry_price=340.0, entry_date=D(2026, 8, 5),
                          high_watermark=350.0, shares=10.0)
        sig = ExitSignal(should_exit=True, reason="kelly delta",
                         exit_type="kelly_trim", quantity=float(intent_qty))
        ctx = _make_ctx(config, exits=[("VLO", sig)], holdings={"VLO": hs},
                        prices={"VLO": 349.0})
        return ra, ctx, broker

    def test_partial_intent_but_broker_flat_after_fill_is_a_full_exit(self, tmp_path, caplog):
        """Intent says TRIM 6 of 10; the broker actually held 6 (a pending
        sell filled after the snapshot) and is FLAT after the fill →
        entry_dates / entry_signals / position_hwm cleared, wash-sale clock
        stamped, exit logged as SELL not TRIM."""
        ra, ctx, broker = self._sell(tmp_path, broker_qty_before_fill=6.0,
                                     filled_qty=6.0, intent_qty=6)
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.place_order_calls == [("VLO", "SELL", 6.0)]
        assert broker.get_position("VLO") == 0.0
        assert any("ENTRY-DATE-CLEAR VLO: broker qty 0 after fill" in r.message
                   for r in caplog.records)
        assert any(r.message.startswith("SELL  VLO") for r in caplog.records)
        assert not any(r.message.startswith("TRIM  VLO") for r in caplog.records)
        assert "VLO" not in ra._entry_dates
        assert "VLO" not in ra._entry_signals
        assert "VLO" not in ra._position_hwm
        assert ra._last_sell_dates_str["VLO"] == TODAY.isoformat()
        state = _saved_state(tmp_path)
        assert "VLO" not in state["entry_dates"]
        assert state["last_sell_dates"]["VLO"] == TODAY.isoformat()

    def test_partial_with_shares_still_held_keeps_entry_state(self, tmp_path, caplog):
        ra, ctx, broker = self._sell(tmp_path, broker_qty_before_fill=10.0,
                                     filled_qty=6.0, intent_qty=6)
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.get_position("VLO") == 4.0
        assert not any("ENTRY-DATE-CLEAR" in r.message for r in caplog.records)
        assert any(r.message.startswith("TRIM  VLO") for r in caplog.records)
        assert ra._entry_dates["VLO"] == "2026-08-05"
        assert "VLO" not in ra._last_sell_dates_str


class TestCommitReentryCooldownOnNonQPPath:
    """Every BUY intent — SELECT / rotation buy leg / QP — passes through
    the runner's BUY loop; the cooldown is applied there from the persisted
    ``last_sell_dates`` ledger and the fill replay, so the non-QP SELECT
    path (`SELECT [slot N]` in daily_104 logs) is now covered."""

    def _buy(self, tmp_path, *, last_sell_dates=None, trip_states=None,
             positions=None, entry_dates=None, broker_positions=None,
             min_reentry_days=5, order_type="NEW_BUY"):
        config = _config(min_reentry_days=min_reentry_days)
        broker = FakeBroker(
            fills={"VLO": {"status": "filled", "order_id": "buy-1",
                           "filled_qty": 5, "filled_avg_price": 340.53}},
            positions=broker_positions,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           positions=positions, entry_dates=entry_dates,
                           last_sell_dates=last_sell_dates,
                           trip_states=trip_states)
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "VLO", "shares": 5, "price": 340.53,
                     "order_type": order_type}],
            prices={"VLO": 340.53}, cash=5_000.0,
        )
        return ra, ctx, broker

    @pytest.mark.parametrize("days_ago", [0, 1, 2, 3, 4])
    def test_blocked_inside_the_cooldown(self, tmp_path, caplog, days_ago):
        last = (TODAY - datetime.timedelta(days=days_ago)).isoformat()
        ra, ctx, broker = self._buy(tmp_path, last_sell_dates={"VLO": last})
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.place_order_calls == []
        assert ctx.orders_placed == []
        assert [o["skip_reason"] for o in ctx.orders_skipped] == ["min_reentry_days"]
        msgs = [r.message for r in caplog.records if r.message.startswith("ANTI-CHURN VLO")]
        assert len(msgs) == 1
        assert f"last full exit {last} is {days_ago}d ago < min_reentry_days=5" in msgs[0]
        assert "VLO" not in ra._entry_dates

    @pytest.mark.parametrize("days_ago", [5, 6, 12])
    def test_allowed_at_or_after_the_cooldown(self, tmp_path, caplog, days_ago):
        last = (TODAY - datetime.timedelta(days=days_ago)).isoformat()
        ra, ctx, broker = self._buy(tmp_path, last_sell_dates={"VLO": last})
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.place_order_calls == [("VLO", "BUY", 5)]
        assert [o["ticker"] for o in ctx.orders_placed] == ["VLO"]
        assert ctx.orders_skipped == []
        assert not any("ANTI-CHURN" in r.message for r in caplog.records)
        # Fresh entry: stamped today, wash-sale entry popped.
        assert ra._entry_dates["VLO"] == TODAY.isoformat()
        assert "VLO" not in ra._last_sell_dates_str

    def test_blocked_from_the_fill_replay_ledger_alone(self, tmp_path):
        """No `last_sell_dates` entry (e.g. state GC'd or the sell was
        never reconciled) — the replay's last flat point still blocks."""
        ra, ctx, broker = self._buy(
            tmp_path,
            trip_states={"VLO": TripState(trip_start=None,
                                          last_exit=TODAY - datetime.timedelta(days=2))},
        )
        ra.commit(ctx)
        assert broker.place_order_calls == []
        assert [o["skip_reason"] for o in ctx.orders_skipped] == ["min_reentry_days"]

    def test_topup_of_a_held_name_is_not_a_reentry(self, tmp_path, caplog):
        """A name held at bar start is topped up, never cooldown-blocked
        (a stale `last_sell_dates` entry must not freeze additions)."""
        ra, ctx, broker = self._buy(
            tmp_path,
            last_sell_dates={"VLO": TODAY.isoformat()},
            positions={"VLO": {"qty": 5.0, "qty_available": 5.0,
                               "avg_entry_price": 330.0}},
            entry_dates={"VLO": "2026-08-20"},
            broker_positions={"VLO": 5.0},
            order_type="TOP_UP",
        )
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.place_order_calls == [("VLO", "BUY", 5)]
        assert any(r.message.startswith("TOPUP  VLO") for r in caplog.records)
        assert ra._entry_dates["VLO"] == "2026-08-20"   # trip start preserved

    def test_disabled_cooldown_never_blocks(self, tmp_path):
        ra, ctx, broker = self._buy(tmp_path, min_reentry_days=0,
                                    last_sell_dates={"VLO": TODAY.isoformat()})
        ra.commit(ctx)
        assert broker.place_order_calls == [("VLO", "BUY", 5)]

    def test_stale_entry_for_a_flat_name_is_a_fresh_entry_not_a_topup(self, tmp_path, caplog):
        """entry_dates still carries the previous trip (the SELL filled
        after the last GC) but the broker is flat at bar start → the BUY
        is a fresh entry stamped today; the stale date must not survive
        as a TOPUP."""
        ra, ctx, broker = self._buy(
            tmp_path,
            entry_dates={"VLO": "2026-08-05"},
            last_sell_dates={"VLO": "2026-08-10"},   # outside the cooldown
        )
        with caplog.at_level("INFO", logger="adapters.runner"):
            ra.commit(ctx)
        assert broker.place_order_calls == [("VLO", "BUY", 5)]
        assert any("ENTRY-DATE-CLEAR VLO: stale entry state 2026-08-05" in r.message
                   for r in caplog.records)
        assert any(r.message.startswith("BUY  VLO") for r in caplog.records)
        assert ra._entry_dates["VLO"] == TODAY.isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Source-level pins on the runner wiring (refactor tripwires)
# ═════════════════════════════════════════════════════════════════════════════

RUNNER_SOURCE = (_STRATEGY / "adapters" / "runner.py").read_text()


class TestRunnerWiring:
    def test_seed_map_is_the_trip_start_not_the_oldest_buy(self):
        assert "first_fill_map = trip_start_map(trip_states)" in RUNNER_SOURCE
        assert "replay_trip_lifecycle(" in RUNNER_SOURCE
        assert 'f.get("action") != "BUY"' not in RUNNER_SOURCE
        assert "take the OLDEST" not in RUNNER_SOURCE

    def test_reseed_and_clear_tags_present(self):
        assert "ENTRY-DATE-RESEED %s %s → %s (trip start)" in RUNNER_SOURCE
        assert "ENTRY-DATE-CLEAR" in RUNNER_SOURCE
        assert "ANTI-CHURN %s: BUY skipped" in RUNNER_SOURCE

    def test_topup_keyed_on_held_at_bar_start(self):
        assert "is_topup = ticker in self._entry_dates and _held_at_start" in RUNNER_SOURCE
        assert "ticker in (getattr(ctx, \"holdings\", None) or {})" in RUNNER_SOURCE

    def test_reentry_ledger_keys_exist_in_live_state_schema(self):
        """The cooldown reads `last_sell_dates` — a key the pipeline's
        live_state_v2 schema defines (never an invented key)."""
        assert 'state_last_sell=self._last_sell_dates_str.get(ticker)' in RUNNER_SOURCE
        assert '"last_sell_dates":   self._last_sell_dates_str' in RUNNER_SOURCE
