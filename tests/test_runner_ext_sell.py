"""runner.py decomposition slice 8 — runner_ext_sell pure/parameterized tests."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_ext_sell import (  # noqa: E402
    EXT_SELL_LOOKBACK_DAYS,
    attribute_ext_sell,
    bar_date,
    ext_sell_fill_date,
    ext_sell_stamp_decision,
    lookup_ext_sell_fills,
    normalize_fill_record,
)


class TestNormalizeFillRecord:
    def test_umbrella_schema(self):
        n = normalize_fill_record({"action": "SELL", "avg_price": 10.0,
                                   "qty": 5, "order_id": "o1", "filled_at": "t"})
        assert n == {"order_id": "o1", "side": "sell", "price": 10.0,
                     "qty": 5.0, "filled_at": "t"}

    def test_execution_schema_no_side(self):
        # execution subrepo: no side field, filled_avg_price/filled_qty/id
        n = normalize_fill_record({"filled_avg_price": 7.0, "filled_qty": 3,
                                   "id": "x", "filled_at": "t"})
        assert n["side"] == "" and n["price"] == 7.0 and n["qty"] == 3.0

    def test_zero_price_skipped(self):
        assert normalize_fill_record({"avg_price": 0, "action": "buy"})["price"] is None


class TestBarDate:
    def test_date_passthrough(self):
        assert bar_date(SimpleNamespace(today=datetime.date(2026, 6, 12))) == \
            datetime.date(2026, 6, 12)

    def test_datetime_normalized_to_date(self):
        dt = datetime.datetime(2026, 6, 12, 15, 30)
        assert bar_date(SimpleNamespace(today=dt)) == datetime.date(2026, 6, 12)


class _AfterAwareBroker:
    """Simulates a REAL broker: ``get_filled_orders(after=...)`` only
    returns fills at/after the requested date, exactly like Alpaca's
    server-side ``after=`` filter. The other mocks in this file (and the
    pre-review version of the D1/D2 regression test below) ignore
    ``after`` entirely — which is exactly why codex #428's review found
    the prior regression test 'passes trivially': it never actually
    exercised the lookback boundary, regardless of how short the window
    was. This mock makes the boundary load-bearing."""

    def __init__(self, fills):
        self._fills = fills

    def get_filled_orders(self, after):
        return [f for f in self._fills if str(f["filled_at"])[:10] >= after]


class TestLookupExtSellFills:
    def test_no_broker_api(self):
        assert lookup_ext_sell_fills(object(), SimpleNamespace(today=datetime.date(2026, 6, 12)), ["MU"]) == {}

    def test_empty_disappeared(self):
        assert lookup_ext_sell_fills(object(), SimpleNamespace(today=datetime.date(2026, 6, 12)), []) == {}

    def test_picks_latest_sell_for_ticker(self):
        class _B:
            def get_filled_orders(self, after):
                return [
                    {"symbol": "MU", "action": "SELL", "avg_price": 10, "qty": 2,
                     "order_id": "o1", "filled_at": "2026-06-10T15:00:00"},
                    {"symbol": "MU", "action": "SELL", "avg_price": 11, "qty": 2,
                     "order_id": "o2", "filled_at": "2026-06-11T15:00:00"},
                ]
        out = lookup_ext_sell_fills(_B(), SimpleNamespace(today=datetime.date(2026, 6, 12)), ["MU"])
        assert out["MU"]["order_id"] == "o2"  # most recent

    def test_lookback_query_boundary_covers_wash_sale_window_plus_buffer(self):
        """Codex #428 review, finding 1: assert the ACTUAL ``after=`` value
        passed to the broker covers at least the 30-day wash-sale window
        plus an operational buffer — not the old fixed 5-day window."""
        captured = {}

        class _Spy:
            def get_filled_orders(self, after):
                captured["after"] = after
                return []

        ctx = SimpleNamespace(today=datetime.date(2026, 6, 26))
        lookup_ext_sell_fills(_Spy(), ctx, ["META"])

        after_date = datetime.date.fromisoformat(captured["after"])
        gap_days = (ctx.today - after_date).days
        assert gap_days >= 35, (
            f"lookback only covers {gap_days}d before reconciliation; must "
            f"cover at least the 30d wash-sale window plus an operational "
            f"buffer (got after={captured['after']!r})"
        )
        assert EXT_SELL_LOOKBACK_DAYS >= 35

    def test_real_meta_incident_24_day_gap_through_real_lookback_boundary(self):
        """REQUIRED regression (codex #428 review, finding 1): the EXACT
        production incident this fix cites — broker SELL fill on
        2026-06-02, discovered by reconciliation on 2026-06-26 — a 24-day
        gap. Uses ``_AfterAwareBroker``, which actually RESPECTS ``after=``
        like a real broker, so this only passes if the lookback window
        genuinely covers the gap through the REAL ``lookup_ext_sell_fills``
        boundary — not because the mock ignores it."""
        fills = [{
            "symbol": "META", "action": "SELL", "avg_price": 590.0, "qty": 10,
            "order_id": "real-fill-1", "filled_at": "2026-06-02T14:31:00-04:00",
        }]
        broker = _AfterAwareBroker(fills)
        ctx = SimpleNamespace(today=datetime.date(2026, 6, 26))

        out = lookup_ext_sell_fills(broker, ctx, ["META"])

        assert "META" in out, (
            "the 24-day META incident gap (fill 2026-06-02, reconciliation "
            "2026-06-26) was not found — lookback window is still too short"
        )
        assert ext_sell_fill_date(out["META"]) == datetime.date(2026, 6, 2)

    def test_old_five_day_window_would_have_missed_the_real_incident(self):
        """Fixture sanity check: confirms ``_AfterAwareBroker`` genuinely
        enforces the ``after`` boundary (so the test above is not
        vacuously true) by showing the OLD 5-day window would have
        excluded the same fill."""
        fills = [{
            "symbol": "META", "action": "SELL", "avg_price": 590.0, "qty": 10,
            "order_id": "real-fill-1", "filled_at": "2026-06-02T14:31:00-04:00",
        }]
        broker = _AfterAwareBroker(fills)
        old_after = (datetime.date(2026, 6, 26) - datetime.timedelta(days=5)).isoformat()
        assert broker.get_filled_orders(after=old_after) == [], (
            "fixture bug: the OLD 5-day window should not have reached "
            "the 2026-06-02 fill from a 2026-06-26 reconciliation date"
        )


class TestAttributeExtSell:
    def test_z9_stop(self):
        s = attribute_ext_sell({"MU": {"order_id": "z9"}}, {},
                               "MU", {"MU": {"order_id": "z9"}})
        assert "source=z9_stop" in s

    def test_runner_exit(self):
        s = attribute_ext_sell({}, {"o1": {"exit_type": "stop_loss"}},
                               "MU", {"MU": {"order_id": "o1"}})
        assert "source=runner_stop_loss" in s

    def test_external_or_manual(self):
        s = attribute_ext_sell({}, {}, "MU", {"MU": {"order_id": "unknown"}})
        assert "source=external_or_manual" in s

    def test_no_fill_record(self):
        assert attribute_ext_sell({}, {}, "MU", {}) == "no_broker_fill_record"


class TestExtSellFillDate:
    """2026-07-01 fix: STATE-EXT-SELL must stamp the wash-sale clock from
    the ACTUAL broker SELL fill date it already looked up via
    ``lookup_ext_sell_fills``, not from ``today_str`` (the date the
    reconciliation code happens to run).

    Confirmed live: META's ``last_sell_dates`` was wrongly stamped with
    the reconciliation RUN date (2026-06-26) instead of the real broker
    SELL fill date (2026-06-02) — a 24-day wash-sale over-extension.

    Codex #428 review additionally requires: (2) a CONFIRMED SELL side
    before the date is authoritative, and (3) proper timezone-aware
    parsing (America/New_York trade date) instead of first-10-chars
    string slicing. Both covered below.
    """

    def test_extracts_date_from_normalized_fill(self):
        fill = {"order_id": "o1", "side": "sell", "price": 590.0,
                "qty": 5.0, "filled_at": "2026-06-02T14:31:00-04:00"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 2)

    def test_none_fill_returns_none(self):
        assert ext_sell_fill_date(None) is None

    def test_empty_dict_returns_none(self):
        assert ext_sell_fill_date({}) is None

    def test_missing_filled_at_returns_none(self):
        assert ext_sell_fill_date({"order_id": "o1", "side": "sell"}) is None

    def test_unparseable_filled_at_returns_none(self):
        # side="sell" so this isolates the unparseable-timestamp path
        # from the (separately tested) confirmed-side requirement.
        assert ext_sell_fill_date({"side": "sell", "filled_at": "not-a-date"}) is None

    def test_reconciliation_delay_scenario_uses_real_fill_date_not_today(self):
        """REQUIRED regression (codex #428 review, finding 1): the EXACT
        production incident, reproduced at the function-composition level
        ``adapters/runner.py``'s STATE-EXT-SELL block uses
        (``lookup_ext_sell_fills`` then ``ext_sell_fill_date`` on its
        result) — through the REAL ``lookup_ext_sell_fills`` lookback
        boundary, using a broker mock that actually RESPECTS ``after=``
        (``_AfterAwareBroker``, not one that ignores it and 'passes
        trivially' regardless of window size).

        D1 = 2026-06-02 (real broker SELL fill date). D2 = 2026-06-26
        (date reconciliation actually runs) — a 24-day gap, matching the
        live META incident exactly.

        The stamped wash-sale date must be D1 (the real fill), never D2
        (today, when reconciliation happened to catch it)."""
        d1 = datetime.date(2026, 6, 2)    # real broker SELL fill date
        d2 = datetime.date(2026, 6, 26)   # date reconciliation actually runs

        broker = _AfterAwareBroker([{
            "symbol": "META", "action": "SELL", "avg_price": 590.0,
            "qty": 10, "order_id": "real-fill-1",
            "filled_at": d1.isoformat() + "T14:31:00-04:00",
        }])

        ctx = SimpleNamespace(today=d2)
        ext_sell_fills = lookup_ext_sell_fills(broker, ctx, ["META"])

        stamped = ext_sell_fill_date(ext_sell_fills.get("META"))

        assert stamped == d1, (
            f"stamped={stamped}, expected the REAL fill date {d1}, not the "
            f"reconciliation-run date {d2} (META incident regression, "
            f"24-day gap)"
        )
        assert stamped != d2

    def test_no_broker_fill_found_yields_none_not_a_fabricated_date(self):
        """Genuine unknown-cause disappearance: the broker has no matching
        SELL fill for the ticker at all (corporate action / account
        transfer / a disposition the broker API can't attribute to a dated
        fill). The composed lookup must yield ``None`` so the caller in
        ``runner.py`` takes the explicit fallback path instead of
        inventing a non-today date."""
        class _Broker:
            def get_filled_orders(self, after):
                return []   # no fills at all

        ctx = SimpleNamespace(today=datetime.date(2026, 6, 5))
        ext_sell_fills = lookup_ext_sell_fills(_Broker(), ctx, ["ZZZZ"])

        assert ext_sell_fill_date(ext_sell_fills.get("ZZZZ")) is None


class TestExtSellFillDateConfirmedSideRequired:
    """Codex #428 review, finding 2: the lookup accepts fills with no
    confirmed ``side``/``action`` (tolerable for the log-only attribution
    string) — but ``ext_sell_fill_date`` must refuse to use such a fill,
    or an actual BUY fill, to stamp the wash-sale clock. Only a CONFIRMED
    ``side == "sell"`` is authoritative."""

    def test_ambiguous_no_side_fill_never_stamps(self):
        # execution-subrepo schema: no side/action field at all.
        fill = normalize_fill_record({
            "order_id": "x", "filled_avg_price": 7.0, "filled_qty": 3,
            "filled_at": "2026-06-02T14:31:00-04:00",
        })
        assert fill["side"] == ""
        assert ext_sell_fill_date(fill) is None

    def test_confirmed_buy_fill_never_stamps(self):
        fill = normalize_fill_record({
            "order_id": "b1", "action": "BUY", "avg_price": 590.0,
            "qty": 5, "filled_at": "2026-06-02T14:31:00-04:00",
        })
        assert fill["side"] == "buy"
        assert ext_sell_fill_date(fill) is None

    def test_confirmed_sell_fill_stamps(self):
        fill = normalize_fill_record({
            "order_id": "s1", "action": "SELL", "avg_price": 590.0,
            "qty": 5, "filled_at": "2026-06-02T14:31:00-04:00",
        })
        assert fill["side"] == "sell"
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 2)

    def test_ambiguous_fill_still_returned_by_lookup_for_attribution(self):
        """The ambiguous fill must still come back from
        ``lookup_ext_sell_fills`` (log-attribution needs it) — the
        rejection only happens in ``ext_sell_fill_date`` (the
        wash-sale-authoritative path)."""
        broker = _AfterAwareBroker([{
            "order_id": "x", "symbol": "DUK", "filled_qty": 20,
            "filled_avg_price": 75.0, "filled_at": "2026-06-02T18:00:00-04:00",
            "status": "filled",
        }])
        ctx = SimpleNamespace(today=datetime.date(2026, 6, 5))
        out = lookup_ext_sell_fills(broker, ctx, ["DUK"])
        assert "DUK" in out and out["DUK"]["side"] == ""
        assert ext_sell_fill_date(out["DUK"]) is None


class TestExtSellFillDateTimezoneAware:
    """REQUIRED tests (codex #428 review, finding 3): UTC/Z-suffix,
    explicit-offset, DST-boundary, and partial-fill timestamps, all
    asserting the correct America/New_York TRADE date — not the naive
    first-10-characters slice of the raw string, which can be off by one
    calendar day near the UTC midnight boundary."""

    def test_utc_z_suffix_near_midnight_shifts_to_prior_ny_date(self):
        """The exact example from the review: a fill at 00:30 UTC belongs
        to the PRIOR America/New_York trading date. The naive
        first-10-chars slice of '2026-06-02T00:30:00Z' would read
        '2026-06-02' — WRONG. The correct NY trade date is 2026-06-01."""
        fill = {"side": "sell", "filled_at": "2026-06-02T00:30:00Z"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 1)

    def test_explicit_offset_timestamp(self):
        fill = {"side": "sell", "filled_at": "2026-06-02T14:31:00-04:00"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 2)

    def test_explicit_offset_timestamp_lowercase_z(self):
        fill = {"side": "sell", "filled_at": "2026-06-02T00:30:00z"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 6, 1)

    def test_dst_spring_forward_boundary(self):
        """2026 US DST spring-forward is 2026-03-08 02:00 America/New_York
        (07:00 UTC). A fill at 04:30 UTC — before the transition, while NY
        is still EST (UTC-5) — is 2026-03-07 23:30 EST: the PRIOR calendar
        date. A naive slice of '2026-03-08T04:30:00Z' would read
        '2026-03-08' — WRONG by one day. Correctness here requires a real
        DST-aware timezone database (zoneinfo), not a fixed UTC offset."""
        fill = {"side": "sell", "filled_at": "2026-03-08T04:30:00Z"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 3, 7)

    def test_dst_spring_forward_just_after_transition(self):
        fill = {"side": "sell", "filled_at": "2026-03-08T07:01:00Z"}
        assert ext_sell_fill_date(fill) == datetime.date(2026, 3, 8)

    def test_partial_fill_still_extracts_correct_date(self):
        """A partial fill (broker's ``partial: True`` marker) must extract
        the same correct date as a full fill — the fill-completeness flag
        is orthogonal to date parsing."""
        raw = {"symbol": "META", "action": "SELL", "avg_price": 590.0,
               "qty": 3, "order_id": "partial-1", "partial": True,
               "filled_at": "2026-06-02T14:31:00-04:00"}
        normalized = normalize_fill_record(raw)
        assert ext_sell_fill_date(normalized) == datetime.date(2026, 6, 2)

    def test_naive_timestamp_rejected_fail_closed(self):
        """No timezone/offset at all — must fail closed (None), never
        guess which timezone the broker meant."""
        fill = {"side": "sell", "filled_at": "2026-06-02T14:31:00"}
        assert ext_sell_fill_date(fill) is None

    def test_garbage_timestamp_with_confirmed_side_rejected(self):
        fill = {"side": "sell", "filled_at": "definitely-not-a-timestamp"}
        assert ext_sell_fill_date(fill) is None


class TestExtSellStampDecision:
    """Codex #428 review ("ALSO reconsider"): the no-fill fallback must
    not blindly overwrite an existing OLDER ``last_sell_dates`` value with
    today's reconciliation date — that destroys known evidence and
    recreates the over-extension bug in a different form. Preserve the
    existing value and flag it as UNRESOLVED instead; only fall back to
    ``today_str`` when there is truly no prior information at all."""

    def test_actual_fill_date_wins_over_everything(self):
        stamp, path = ext_sell_stamp_decision(
            datetime.date(2026, 6, 2), "2026-05-01", "2026-06-26",
        )
        assert (stamp, path) == ("2026-06-02", "actual_fill")

    def test_no_fill_but_prior_stamp_exists_preserves_prior(self):
        stamp, path = ext_sell_stamp_decision(None, "2026-05-01", "2026-06-26")
        assert (stamp, path) == ("2026-05-01", "unresolved_preserve")
        # Must NOT be today's reconciliation date.
        assert stamp != "2026-06-26"

    def test_no_fill_and_no_prior_stamp_falls_back_to_today(self):
        stamp, path = ext_sell_stamp_decision(None, None, "2026-06-26")
        assert (stamp, path) == ("2026-06-26", "no_fill_fallback")

    def test_no_fill_and_empty_prior_stamp_falls_back_to_today(self):
        # An empty string prior_stamp (falsy) is treated as "no prior
        # value", not a real date to preserve.
        stamp, path = ext_sell_stamp_decision(None, "", "2026-06-26")
        assert (stamp, path) == ("2026-06-26", "no_fill_fallback")
