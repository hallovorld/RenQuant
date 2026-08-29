"""RQ#618 class C — the live fill replay never drops a price-less fill, and
the broker's order pagination never duplicates the boundary order.

Defects pinned (forensics in hallovorld/RenQuant#618):

(a) ``adapters/runner_tax_lots.py`` ``reconstruct_live_tax_lots_from_fills``
    used to ``continue`` past any fill whose ``filled_avg_price`` was
    missing/0. A price-less SELL therefore never decremented lots, so the
    reconstructed lot qty exceeded the broker qty on every run (VLO 7 vs 5,
    PANW 6 vs 3, APH 14 vs 8) and the lots fell back to the broker average.
    Now a price-less SELL reduces lots at the disposed lots' cost basis
    (tagged ``price_missing=True``), a price-less BUY is appended at qty with
    a NaN/None-safe stand-in basis and a flagged lot, each is warned ONCE and
    counted in ``.stats``; ``adopt_live_tax_lots`` back-fills the flagged
    basis from the broker average and logs a diagnosable invariant on a
    mismatch (signed delta + degraded counts).

(b) ``live/alpaca_broker.py`` ``get_filled_orders`` paginated with
    ``until_cursor = oldest`` while its comment promised "minus 1µs", so the
    boundary order of every full page came back on the next page — a
    duplicated BUY doubles a lot. Now the cursor is oldest − 1µs AND orders
    are deduplicated by ``id`` across pages.

Lean-CI friendly: ``kernel.exits`` is self-contained and the alpaca SDK
modules the broker imports lazily are stubbed via ``sys.modules``.
"""
from __future__ import annotations

import datetime
import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "backtesting" / "renquant_104"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adapters.runner_tax_lots import (  # noqa: E402
    LiveTaxLotReconstruction,
    adopt_live_tax_lots,
    degraded_counts_for,
    reconstruct_live_tax_lots_from_fills,
)
from kernel.exits import HoldingState, TaxLot  # noqa: E402
from live.alpaca_broker import AlpacaBroker  # noqa: E402

D0 = datetime.date(2026, 8, 5)


def _fill(symbol, action, qty, price, day, order_id=None):
    """A broker fill dict in the shape ``get_filled_orders`` returns."""
    return {
        "order_id": order_id or f"{symbol}-{action}-{day}",
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "filled_at": f"2026-08-{day:02d}T13:30:01+00:00",
        "avg_price": price,
    }


def _lot_qty(result, ticker):
    return sum(L.shares for L in result.get(ticker, []))


def _lines(caplog, needle):
    return [r for r in caplog.records if needle in r.getMessage()]


# ── (a) the replay ────────────────────────────────────────────────────────────

class TestPriceLessSellReducesLots:

    def test_price_less_sell_reduces_qty(self):
        fills = [
            _fill("VLO", "BUY", 5, 340.53, 26),
            _fill("VLO", "SELL", 2, None, 27),  # filled_avg_price missing
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "VLO") == 3
        assert result.stats["price_missing_sell"] == 1
        rec = result.stats["degraded_fills"][0]
        assert rec["symbol"] == "VLO" and rec["action"] == "SELL"
        assert rec["price_missing"] is True and rec["applied"] is True
        assert rec["kind"] == "price_missing_sell"
        # stand-in price for the realized-P&L record = disposed cost basis
        assert rec["stand_in_price"] == pytest.approx(340.53)
        assert rec["order_id"] == "VLO-SELL-27"

    @pytest.mark.parametrize("missing", [None, 0, 0.0, "", "n/a", float("nan"), -1])
    def test_every_missing_price_shape_is_applied(self, missing):
        fills = [
            _fill("PANW", "BUY", 6, 200.0, 20),
            _fill("PANW", "SELL", 3, missing, 21),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "PANW") == 3
        assert result.stats["price_missing_sell"] == 1

    def test_filled_avg_price_key_is_honoured_before_degrading(self):
        fills = [
            _fill("APH", "BUY", 8, 100.0, 20),
            {"symbol": "APH", "action": "SELL", "qty": 3,
             "filled_at": "2026-08-21T13:30:01+00:00",
             "avg_price": None, "filled_avg_price": 101.0},
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "APH") == 5
        assert result.stats["price_missing_sell"] == 0
        assert result.stats["fills_applied"] == 2

    def test_full_price_less_exit_flattens_the_ticker(self):
        fills = [
            _fill("NVDA", "BUY", 7, 210.0, 25),
            _fill("NVDA", "SELL", 7, None, 26),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert "NVDA" not in result
        assert result.stats["price_missing_sell"] == 1

    def test_issue_618_vlo_ledger_reconciles_to_broker_qty(self):
        """The VLO ping-pong from the issue with every SELL price-less: the
        old replay left 2+5+5 = 12 shares; the broker holds 5."""
        fills = [
            _fill("VLO", "BUY", 2, 308.56, 5),
            _fill("VLO", "SELL", 2, None, 25),
            _fill("VLO", "BUY", 5, 340.53, 26),
            _fill("VLO", "SELL", 5, None, 27),
            _fill("VLO", "BUY", 5, 346.50, 28),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "VLO") == 5
        assert [L.price for L in result["VLO"]] == [346.50]
        assert result.stats["price_missing_sell"] == 2
        assert result.stats["fills_applied"] == 3

    def test_hifo_and_avg_methods_also_reduce(self):
        fills = [
            _fill("MU", "BUY", 4, 100.0, 1),
            _fill("MU", "BUY", 4, 120.0, 2),
            _fill("MU", "SELL", 4, None, 3),
        ]
        for method in ("hifo", "avg", "fifo"):
            cfg = {"tax": {"lot_method": method}}
            result = reconstruct_live_tax_lots_from_fills(fills, config=cfg)
            assert _lot_qty(result, "MU") == pytest.approx(4), method
        hifo = reconstruct_live_tax_lots_from_fills(fills, config={"tax": {"lot_method": "hifo"}})
        assert [L.price for L in hifo["MU"]] == [100.0]  # the 120 lot was disposed


class TestFullExitReentryDoesNotResurrectTheSoldLot:
    """The arithmetic behind the observed mismatches, with PRICED fills:
    ``HoldingState.total_shares()`` falls back to the legacy ``shares`` field
    once the lot list is empty, so a full sell never popped the ticker and
    the next BUY's ``ensure_lots`` re-synthesised the sold lot. Old replay:
    VLO 2+5=7 vs broker 5, NVDA 7+7=14 vs 7, PANW 3+3=6 vs 3, APH 6+8=14 vs 8
    (logs/daily_104/2026-08-28.log:392-394)."""

    @pytest.mark.parametrize("ticker,legs,broker_qty", [
        ("VLO", [("BUY", 2, 308.56), ("SELL", 2, 341.69), ("BUY", 5, 340.53),
                 ("SELL", 5, 349.00), ("BUY", 5, 346.50)], 5),
        ("NVDA", [("BUY", 7, 210.0), ("SELL", 7, 212.598), ("BUY", 7, 222.9)], 7),
        ("PANW", [("BUY", 3, 180.0), ("SELL", 3, 190.0), ("BUY", 3, 185.0)], 3),
        ("APH", [("BUY", 6, 90.0), ("SELL", 6, 95.0), ("BUY", 8, 96.0)], 8),
    ])
    def test_reconstructed_qty_equals_broker_qty(self, ticker, legs, broker_qty):
        fills = [_fill(ticker, a, q, px, day + 1) for day, (a, q, px) in enumerate(legs)]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, ticker) == broker_qty
        assert result.stats["degraded_fills"] == []  # nothing degraded: priced fills
        # the surviving lots are the re-entry lots only, never the sold one
        last_buy_price = [px for a, _, px in legs if a == "BUY"][-1]
        assert {L.price for L in result[ticker]} <= {px for a, _, px in legs if a == "BUY"}
        assert result[ticker][-1].price == last_buy_price
        assert legs[0][2] not in {L.price for L in result[ticker]}

    def test_full_exit_flattens_and_reentry_starts_clean(self):
        fills = [
            _fill("NVDA", "BUY", 7, 210.0, 25),
            _fill("NVDA", "SELL", 7, 212.598, 26),
        ]
        assert "NVDA" not in reconstruct_live_tax_lots_from_fills(fills)
        fills.append(_fill("NVDA", "BUY", 7, 222.9, 27))
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert [(L.shares, L.price) for L in result["NVDA"]] == [(7.0, 222.9)]
        assert result["NVDA"][0].date == datetime.date(2026, 8, 27)


class TestPriceLessBuyIsAppliedNotDropped:

    def test_price_less_buy_with_prior_lots_uses_running_basis(self):
        fills = [
            _fill("HPE", "BUY", 10, 50.0, 20),
            _fill("HPE", "BUY", 19, None, 28),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "HPE") == 29
        flagged = [L for L in result["HPE"] if getattr(L, "price_missing", False)]
        assert len(flagged) == 1
        assert flagged[0].shares == 19
        assert flagged[0].price == 50.0  # stand-in = running weighted basis
        assert math.isfinite(flagged[0].price)
        assert result.stats["price_missing_buy"] == 1
        rec = result.stats["degraded_fills"][0]
        assert rec["kind"] == "price_missing_buy" and rec["price_missing"] is True

    def test_price_less_first_buy_has_a_zero_not_nan_basis(self):
        fills = [_fill("NET", "BUY", 3, None, 28)]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "NET") == 3
        lot = result["NET"][0]
        assert lot.price == 0.0 and getattr(lot, "price_missing", False) is True
        assert not math.isnan(lot.price)

    def test_price_less_buy_then_sell_still_reduces(self):
        fills = [
            _fill("NET", "BUY", 3, None, 28),
            _fill("NET", "SELL", 1, 60.0, 29),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert _lot_qty(result, "NET") == 2

    def test_flag_survives_the_return_copy(self):
        """The function returns TaxLot copies; the flag must ride along or
        the hydration site cannot back-fill the basis."""
        result = reconstruct_live_tax_lots_from_fills([_fill("NET", "BUY", 3, None, 28)])
        lot = result["NET"][0]
        assert isinstance(lot, TaxLot)
        assert lot.price_missing is True


class TestWarningsAndCounts:

    def test_one_warning_per_price_less_fill(self, caplog):
        fills = [
            _fill("VLO", "BUY", 2, 308.56, 5),
            _fill("VLO", "SELL", 2, None, 25),
            _fill("VLO", "BUY", 5, None, 26),
            _fill("VLO", "SELL", 5, None, 27),
        ]
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            reconstruct_live_tax_lots_from_fills(fills)
        warned = [r for r in _lines(caplog, "has no fill price") if r.levelno == logging.WARNING]
        assert len(warned) == 3
        assert sum("SELL qty=2.0000" in r.getMessage() for r in warned) == 1
        assert sum("SELL qty=5.0000" in r.getMessage() for r in warned) == 1
        assert sum("BUY qty=5.0000" in r.getMessage() for r in warned) == 1
        for r in warned:
            assert "price_missing=True" in r.getMessage()
            assert "order_id=VLO-" in r.getMessage()

    def test_summary_line_carries_every_count(self, caplog):
        fills = [
            _fill("VLO", "BUY", 2, 308.56, 5),
            _fill("VLO", "SELL", 2, None, 25),          # price_missing_sell
            _fill("VLO", "BUY", 5, None, 26),           # price_missing_buy
            _fill("XYZ", "SELL", 1, 10.0, 3),           # sell before any buy
            _fill("VLO", "SELL", 9, 300.0, 27),         # oversell 9 > 5 held
            {"symbol": "ABC", "action": "BUY", "qty": "many", "avg_price": 1,
             "filled_at": "2026-08-01T13:30:01+00:00"},  # unparseable qty
            _fill("ABC", "DIVIDEND", 1, 1.0, 2),         # unknown action
            "not-a-dict",
        ]
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            result = reconstruct_live_tax_lots_from_fills(fills)
        st = result.stats
        assert st["fills_total"] == 7
        assert st["fills_applied"] == 2          # BUY 2 @308.56, SELL 9 @300
        assert st["price_missing_sell"] == 1
        assert st["price_missing_buy"] == 1
        assert st["dropped_sell_without_lots"] == 1
        assert st["oversell_clamped"] == 1
        assert st["dropped_unparseable"] == 1
        assert st["dropped_unknown_action"] == 1
        assert st["fills_by_ticker"] == {"VLO": 4, "XYZ": 1}
        assert degraded_counts_for(st, "VLO") == {
            "price_missing_sell": 1, "price_missing_buy": 1,
            "dropped_sell_without_lots": 0, "oversell_clamped": 1,
        }
        assert degraded_counts_for(st, "XYZ")["dropped_sell_without_lots"] == 1
        assert degraded_counts_for(st, "NOPE") == {k: 0 for k in degraded_counts_for(st, "VLO")}
        summary = _lines(caplog, "LIVE-TAX-LOTS replay summary")
        assert len(summary) == 1
        msg = summary[0].getMessage()
        for frag in ("fills=7", "applied=2", "price_missing_sell=1",
                     "price_missing_buy=1", "dropped_unparseable=1",
                     "dropped_sell_without_lots=1", "dropped_unknown_action=1",
                     "oversell_clamped=1", "tickers_with_lots=0"):
            assert frag in msg, frag
        assert summary[0].levelno == logging.WARNING  # degraded > 0

    def test_clean_history_summary_is_info_and_nothing_degraded(self, caplog):
        fills = [_fill("MU", "BUY", 10, 100.0, 1), _fill("MU", "SELL", 4, 110.0, 2)]
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            result = reconstruct_live_tax_lots_from_fills(fills)
        assert result.stats["fills_applied"] == 2
        assert result.stats["degraded_fills"] == []
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        summary = _lines(caplog, "LIVE-TAX-LOTS replay summary")
        assert len(summary) == 1 and summary[0].levelno == logging.INFO

    def test_empty_input_is_a_plain_empty_dict_with_stats(self, caplog):
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            result = reconstruct_live_tax_lots_from_fills(None)
        assert result == {}
        assert isinstance(result, dict) and isinstance(result, LiveTaxLotReconstruction)
        assert result.stats["fills_total"] == 0
        assert not _lines(caplog, "replay summary")  # nothing to summarise

    def test_unparseable_fills_are_still_dropped_not_applied(self):
        fills = [
            _fill("MU", "BUY", 0, 100.0, 1),
            _fill("MU", "BUY", -3, 100.0, 1),
            {"symbol": "MU", "action": "BUY", "qty": 5, "avg_price": 100.0, "filled_at": None},
            {"symbol": "", "action": "BUY", "qty": 5, "avg_price": 100.0,
             "filled_at": "2026-08-01T13:30:01+00:00"},
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert result == {}
        assert result.stats["dropped_unparseable"] == 4
        assert result.stats["degraded_fills"] == []


# ── (a) the hydration site ────────────────────────────────────────────────────

def _holding(avg=100.0, qty=5.0):
    return HoldingState(entry_price=avg, entry_date=D0, high_watermark=avg, shares=qty)


class TestAdoptLiveTaxLots:

    def test_match_attaches_lots_and_weighted_entry(self):
        h = _holding(avg=999.0, qty=10.0)
        lots = [TaxLot(4, 100.0, D0), TaxLot(6, 200.0, D0)]
        assert adopt_live_tax_lots(h, "MU", lots, 10.0, 999.0) is True
        assert h.lots is lots
        assert h.entry_price == pytest.approx(160.0)

    def test_mismatch_logs_delta_and_degraded_counts(self, caplog):
        """The hydration invariant: reconstructed != broker → the warning
        names the signed delta, the fills seen, and the degraded counts."""
        h = _holding(avg=340.0, qty=5.0)
        lots = [TaxLot(2, 308.56, D0), TaxLot(5, 340.53, D0)]  # 7 vs 5
        stats = reconstruct_live_tax_lots_from_fills([
            _fill("VLO", "BUY", 2, 308.56, 5),
            _fill("VLO", "SELL", 2, None, 25),
            _fill("VLO", "BUY", 5, 340.53, 26),
        ]).stats
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            ok = adopt_live_tax_lots(h, "VLO", lots, 5.0, 340.0, stats=stats)
        assert ok is False
        assert h.lots == [] and h.entry_price == 340.0  # broker fallback kept
        rec = _lines(caplog, "LIVE-TAX-LOTS: VLO reconstructed lot qty 7.0000 != broker qty 5.0000")
        assert len(rec) == 1 and rec[0].levelno == logging.WARNING
        msg = rec[0].getMessage()
        assert "using broker avg_entry_price fallback" in msg  # grep-compat with #618
        assert "delta=+2.0000" in msg
        assert "replay saw 3 fill(s)" in msg
        assert "price_missing_sell=1" in msg
        assert "price_missing_buy=0" in msg
        assert "sell_without_lots=0" in msg
        assert "oversell_clamped=0" in msg

    def test_price_missing_lot_is_backfilled_from_broker_residual(self, caplog):
        h = _holding(avg=110.0, qty=10.0)
        known = TaxLot(5, 100.0, D0)
        missing = TaxLot(5, 0.0, D0)
        missing.price_missing = True
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            ok = adopt_live_tax_lots(h, "MU", [known, missing], 10.0, 110.0)
        assert ok is True
        # (110 × 10 − 5 × 100) / 5 = 120
        assert missing.price == pytest.approx(120.0)
        assert missing.price_missing is True  # flag kept for tax reporting
        assert known.price == 100.0
        assert h.entry_price == pytest.approx(110.0)
        rec = _lines(caplog, "basis back-filled at 120.0000 from broker_avg_residual")
        assert len(rec) == 1 and rec[0].levelno == logging.WARNING

    def test_non_positive_residual_falls_back_to_broker_avg(self):
        h = _holding(avg=90.0, qty=10.0)
        known = TaxLot(5, 200.0, D0)  # known cost 1000 > 90 × 10 = 900
        missing = TaxLot(5, 0.0, D0)
        missing.price_missing = True
        assert adopt_live_tax_lots(h, "MU", [known, missing], 10.0, 90.0) is True
        assert missing.price == pytest.approx(90.0)

    @pytest.mark.parametrize("bad_avg", [0.0, -1.0, float("nan"), None, "x"])
    def test_unknown_basis_with_no_broker_avg_is_not_attached(self, bad_avg, caplog):
        h = _holding(avg=0.0, qty=3.0)
        h.entry_price = 0.0
        missing = TaxLot(3, 0.0, D0)
        missing.price_missing = True
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            ok = adopt_live_tax_lots(h, "NET", [missing], 3.0, bad_avg)
        assert ok is False
        assert h.lots == []
        assert _lines(caplog, "lots NOT attached")

    def test_no_lots_with_degraded_fills_warns_no_lots_clean_is_silent(self, caplog):
        h = _holding(avg=50.0, qty=8.0)
        stats = reconstruct_live_tax_lots_from_fills([
            _fill("APH", "SELL", 3, None, 21),  # sell before any buy in window
        ]).stats
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            assert adopt_live_tax_lots(h, "APH", None, 8.0, 50.0, stats=stats) is False
        rec = _lines(caplog, "LIVE-TAX-LOTS: APH no reconstructed lots")
        assert len(rec) == 1 and "sell_without_lots=1" in rec[0].getMessage()
        assert "delta=-8.0000" in rec[0].getMessage()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            assert adopt_live_tax_lots(h, "APH", [], 8.0, 50.0, stats=None) is False
        assert not caplog.records  # pre-change behaviour: silent when nothing degraded

    def test_end_to_end_issue_shape_now_reconciles_and_adopts(self):
        fills = [
            _fill("VLO", "BUY", 2, 308.56, 5),
            _fill("VLO", "SELL", 2, None, 25),
            _fill("VLO", "BUY", 5, 340.53, 26),
            _fill("VLO", "SELL", 5, None, 27),
            _fill("VLO", "BUY", 5, 346.50, 28),
        ]
        result = reconstruct_live_tax_lots_from_fills(fills)
        h = _holding(avg=346.5, qty=5.0)
        assert adopt_live_tax_lots(h, "VLO", result.get("VLO"), 5.0, 346.5, stats=result.stats)
        assert sum(L.shares for L in h.lots) == 5
        assert h.entry_price == pytest.approx(346.5)


class TestRunnerHandlingSiteIsWired:
    """The runner's hydration site must call the helper (the old inline
    warning block is gone) — a static pin, since a RunnerAdapter end-to-end
    needs the strategy deps that the lean CI lacks."""

    def test_runner_calls_adopt_live_tax_lots(self):
        src = (REPO / "backtesting" / "renquant_104" / "adapters" / "runner.py").read_text()
        assert "adopt_live_tax_lots(" in src
        assert 'stats=getattr(live_tax_lots, "stats", None)' in src
        assert '"LIVE-TAX-LOTS: %s reconstructed lot qty %.4f != broker "' not in src


# ── (b) pagination ────────────────────────────────────────────────────────────

class _GetOrdersRequest:
    def __init__(self, **kw):
        self.status = kw.get("status")
        self.limit = kw.get("limit")
        self.direction = kw.get("direction")
        self.after = None
        self.until = None


@pytest.fixture
def sdk(monkeypatch):
    """Stub ``alpaca.trading.requests`` / ``.enums`` (imported lazily inside
    ``get_filled_orders``) so the walk runs without the SDK installed."""
    requests = types.ModuleType("alpaca.trading.requests")
    requests.GetOrdersRequest = _GetOrdersRequest
    enums = types.ModuleType("alpaca.trading.enums")
    enums.QueryOrderStatus = SimpleNamespace(CLOSED="closed", OPEN="open")
    alpaca = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    alpaca.trading = trading
    trading.requests = requests
    trading.enums = enums
    for name, mod in (
        ("alpaca", alpaca), ("alpaca.trading", trading),
        ("alpaca.trading.requests", requests), ("alpaca.trading.enums", enums),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return SimpleNamespace(requests=requests, enums=enums)


def _orders(n, start=datetime.datetime(2026, 8, 28, 20, 55, tzinfo=datetime.timezone.utc)):
    """n filled BUY orders with strictly decreasing submitted_at (1s apart)."""
    out = []
    for i in range(n):
        ts = start - datetime.timedelta(seconds=i)
        out.append(SimpleNamespace(
            id=f"ord-{i:05d}", symbol="NVDA", side="buy", status="filled",
            qty=7, filled_qty=7, filled_avg_price=210.0,
            submitted_at=ts, filled_at=ts,
        ))
    return out


class _PagingClient:
    """A closed-orders endpoint. ``inclusive_until`` mirrors a server whose
    ``until`` filter is ``submitted_at <= until`` (the boundary order comes
    back on the next page); the exclusive variant uses ``<``."""

    def __init__(self, orders, *, inclusive_until):
        self.orders = sorted(orders, key=lambda o: o.submitted_at, reverse=True)
        self.inclusive_until = inclusive_until
        self.requests: list[_GetOrdersRequest] = []

    def get_orders(self, filter):
        self.requests.append(filter)
        rows = self.orders
        if filter.until is not None:
            if self.inclusive_until:
                rows = [o for o in rows if o.submitted_at <= filter.until]
            else:
                rows = [o for o in rows if o.submitted_at < filter.until]
        if filter.after is not None:
            rows = [o for o in rows if o.submitted_at > filter.after]
        return rows[: filter.limit]


def _broker(client) -> AlpacaBroker:
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._trading_client = client
    return b


class TestPaginationBoundary:

    @pytest.mark.parametrize("inclusive", [True, False])
    def test_overlapping_boundary_yields_no_duplicates(self, sdk, inclusive):
        client = _PagingClient(_orders(1250), inclusive_until=inclusive)
        fills = _broker(client).get_filled_orders()
        ids = [f["order_id"] for f in fills]
        assert len(ids) == 1250
        assert len(set(ids)) == 1250, "boundary order duplicated across pages"
        assert len(client.requests) == 3  # 500 + 500 + 250

    def test_cursor_is_oldest_minus_one_microsecond(self, sdk):
        client = _PagingClient(_orders(1000), inclusive_until=True)
        _broker(client).get_filled_orders()
        assert len(client.requests) == 3  # 500 + 500 + empty page → stop
        first, second, third = client.requests
        assert first.until is None
        oldest_on_page_1 = min(o.submitted_at for o in client.orders[:500])
        assert second.until == oldest_on_page_1 - datetime.timedelta(microseconds=1)
        oldest_on_page_2 = min(o.submitted_at for o in client.orders[500:1000])
        assert third.until == oldest_on_page_2 - datetime.timedelta(microseconds=1)

    def test_after_bound_is_forwarded_and_dedupe_keeps_first_seen(self, sdk):
        client = _PagingClient(_orders(700), inclusive_until=True)
        fills = _broker(client).get_filled_orders(after="2026-08-28")
        assert client.requests[0].after == datetime.datetime(2026, 8, 28, tzinfo=datetime.timezone.utc)
        assert len(fills) == len({f["order_id"] for f in fills})

    def test_id_less_orders_are_kept_not_deduped(self, sdk):
        """An order without an id cannot be deduplicated; it must not be
        dropped either (older SDK shapes)."""
        rows = _orders(3)
        for o in rows:
            o.id = None
        client = _PagingClient(rows, inclusive_until=True)
        fills = _broker(client).get_filled_orders()
        assert len(fills) == 3
        assert all(f["order_id"] == "" for f in fills)

    def test_non_datetime_cursor_does_not_crash_and_still_dedupes(self, sdk):
        """SDK drift: a string ``submitted_at`` cannot take the 1µs step; the
        walk keeps the inclusive cursor and the id dedupe still holds."""
        rows = _orders(600)
        for o in rows:
            o.submitted_at = o.submitted_at.isoformat()
        client = _PagingClient(rows, inclusive_until=True)
        client.orders = sorted(rows, key=lambda o: o.submitted_at, reverse=True)
        fills = _broker(client).get_filled_orders()
        ids = [f["order_id"] for f in fills]
        assert len(ids) == 600 and len(set(ids)) == 600

    def test_source_no_longer_assigns_the_bare_cursor(self):
        src = (REPO / "live" / "alpaca_broker.py").read_text()
        assert "until_cursor = oldest\n" not in src
        assert "oldest - timedelta(microseconds=1)" in src
        assert "seen_ids" in src
