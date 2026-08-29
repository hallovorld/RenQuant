"""Property/invariant tests for runner live tax-lot reconstruction.

Eng plan S2 item 6 (test-ladder rebalance). The example-based suites
(test_wash_sale_economic, test_runner_sell_attribution, test_partial_sell)
pin specific tax-lot scenarios; this module pins the ACCOUNTING INVARIANTS
that must hold over arbitrary broker fill histories. The subjects are the
pure functions extracted to `adapters/runner_tax_lots.py` (decomposition
slice 6) — they reconstruct FIFO cost basis from Alpaca fills (which expose
only average entry), so a violated invariant here is a mis-stated realized
basis: wrong P/L, wrong wash-sale, wrong tax.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): valid
and adversarial fill histories are generated over a deterministic seeded
grid; failures print the generating sequence for replay.

Invariants pinned:
- conservation: for a valid history (no oversell), Σ surviving lot shares ==
  Σ buys − Σ sells per ticker; a fully-sold ticker disappears.
- no zero/negative lots ever survive.
- FIFO ordering: surviving lots are in non-decreasing acquisition date.
- every surviving lot price equals an actual BUY fill price (lots are never
  blended — only the legacy weighted-avg entry is).
- determinism: the result is invariant under input fill ORDER (the function
  re-sorts by filled_at); only chronology matters.
- robustness: malformed fills (qty<=0, missing/!ISO date, unknown action,
  missing symbol, non-dict) are ignored — same result as without.
- price-less fills (RQ#618 class C, 2026-08-29): a fill with qty>0 and NO
  fill price is APPLIED, not ignored — a price-less SELL reduces lots and a
  price-less BUY appends a flagged lot — so conservation holds over the
  full history including them (the old `continue` left every price-less
  SELL un-applied → reconstructed qty > broker qty every run).
- sell_event_price: prefers the broker fill price, falls back, never returns
  a non-finite or non-positive number.
"""
from __future__ import annotations

import datetime
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner_tax_lots import (  # noqa: E402
    reconstruct_live_tax_lots_from_fills,
    sell_event_price,
)

SEED = 0x7A21
N = 600
TICKERS = ["AAPL", "MSFT", "NVDA"]
_BASE = datetime.datetime(2026, 1, 1, 9, 30)


def _ts(i: int) -> str:
    # unique, chronologically-monotone ISO stamp (hours apart so date varies
    # and the string sort matches chronological order).
    return (_BASE + datetime.timedelta(hours=i)).isoformat() + "Z"


def _gen_valid(rng):
    """A realizable fill history: a SELL never exceeds shares currently held,
    so no clamping happens and conservation is exact. Returns (fills, net)."""
    n = rng.randint(0, 40)
    held = {t: 0.0 for t in TICKERS}
    buy_prices = {t: set() for t in TICKERS}
    fills = []
    for i in range(n):
        tkr = rng.choice(TICKERS)
        price = float(rng.randint(1, 5000))  # integral prices → exact compares
        if held[tkr] > 1.0 and rng.random() < 0.4:
            qty = held[tkr] * rng.uniform(0.05, 0.95)  # strictly < held
            held[tkr] -= qty
            action = "SELL"
        else:
            qty = rng.uniform(1.0, 200.0)
            held[tkr] += qty
            buy_prices[tkr].add(price)
            action = "BUY"
        fills.append({"symbol": tkr, "action": action, "qty": qty,
                      "avg_price": price, "filled_at": _ts(i)})
    return fills, held, buy_prices


def _norm(result):
    """Comparable, order-independent view of the reconstructed lots."""
    return {t: [(round(L.shares, 6), L.price, L.date) for L in lots]
            for t, lots in result.items()}


class TestReconstructInvariants:

    def test_conservation_and_dropout(self):
        rng = random.Random(SEED)
        for _ in range(N):
            fills, held, _ = _gen_valid(rng)
            result = reconstruct_live_tax_lots_from_fills(fills)
            for tkr in TICKERS:
                net = held[tkr]
                if net > 1e-6:
                    assert tkr in result, (tkr, net, fills)
                    got = sum(L.shares for L in result[tkr])
                    assert math.isclose(got, net, rel_tol=1e-6, abs_tol=1e-6), (
                        f"{tkr}: reconstructed {got} != net {net}")
                else:
                    assert tkr not in result, (tkr, net)

    def test_no_zero_or_negative_lots(self):
        rng = random.Random(SEED + 1)
        for _ in range(N):
            fills, _, _ = _gen_valid(rng)
            result = reconstruct_live_tax_lots_from_fills(fills)
            for tkr, lots in result.items():
                assert lots, (tkr, "empty lot list must be dropped, not kept")
                for L in lots:
                    assert L.shares > 1e-9, (tkr, L.shares)

    def test_fifo_date_order(self):
        rng = random.Random(SEED + 2)
        for _ in range(N):
            fills, _, _ = _gen_valid(rng)
            result = reconstruct_live_tax_lots_from_fills(fills)
            for tkr, lots in result.items():
                dates = [L.date for L in lots]
                assert dates == sorted(dates), (tkr, dates)

    def test_lot_prices_are_actual_buy_prices(self):
        rng = random.Random(SEED + 3)
        for _ in range(N):
            fills, _, buy_prices = _gen_valid(rng)
            result = reconstruct_live_tax_lots_from_fills(fills)
            for tkr, lots in result.items():
                for L in lots:
                    assert L.price in buy_prices[tkr], (tkr, L.price, buy_prices[tkr])

    def test_order_independent(self):
        """The function re-sorts by filled_at, so a shuffled input fill list
        reconstructs identically — only chronology is load-bearing."""
        rng = random.Random(SEED + 4)
        for _ in range(N):
            fills, _, _ = _gen_valid(rng)
            ref = _norm(reconstruct_live_tax_lots_from_fills(fills))
            shuffled = fills[:]
            rng.shuffle(shuffled)
            got = _norm(reconstruct_live_tax_lots_from_fills(shuffled))
            assert got == ref, "result depends on input order, not just time"

    def test_malformed_fills_ignored(self):
        rng = random.Random(SEED + 5)
        junk = [
            {"symbol": "AAPL", "action": "BUY", "qty": 0, "avg_price": 10,
             "filled_at": _ts(10_000)},                      # qty <= 0
            {"symbol": "AAPL", "action": "BUY", "qty": 5, "avg_price": 10,
             "filled_at": None},                             # no date
            {"symbol": "AAPL", "action": "BUY", "qty": 5, "avg_price": 10,
             "filled_at": "not-a-date"},                     # unparseable date
            {"symbol": "", "action": "BUY", "qty": 5, "avg_price": 10,
             "filled_at": _ts(10_002)},                      # no symbol
            {"symbol": "AAPL", "action": "DIVIDEND", "qty": 5, "avg_price": 10,
             "filled_at": _ts(10_003)},                      # unknown action
            "not-a-dict", None, 42,                           # non-dict entries
        ]
        for _ in range(N // 4):
            fills, _, _ = _gen_valid(rng)
            ref = _norm(reconstruct_live_tax_lots_from_fills(fills))
            polluted = fills[:]
            for j in junk:
                polluted.insert(rng.randint(0, len(polluted)), j)
            got = _norm(reconstruct_live_tax_lots_from_fills(polluted))
            assert got == ref, "malformed fills changed the reconstruction"

    def test_price_missing_fills_are_applied_not_ignored(self):
        """RQ#618 class C: strip the price from a random subset of fills of a
        valid history — conservation must STILL hold (a price-less SELL
        reduces lots; a price-less BUY appends a flagged lot), and the
        replay counts exactly the fills it degraded."""
        rng = random.Random(SEED + 6)
        for _ in range(N // 4):
            fills, net, _ = _gen_valid(rng)
            stripped = []
            n_sell = n_buy = 0
            for f in fills:
                f = dict(f)
                if rng.random() < 0.3:
                    f["avg_price"] = rng.choice([None, 0, "", float("nan")])
                    if f["action"] == "SELL":
                        n_sell += 1
                    else:
                        n_buy += 1
                stripped.append(f)
            result = reconstruct_live_tax_lots_from_fills(stripped)
            for tkr in TICKERS:
                got = sum(L.shares for L in result.get(tkr, []))
                assert math.isclose(got, net[tkr], abs_tol=1e-6), (tkr, got, net[tkr])
            assert result.stats["price_missing_sell"] == n_sell
            assert result.stats["price_missing_buy"] == n_buy
            assert result.stats["dropped_unparseable"] == 0
            flagged = [L for lots in result.values() for L in lots
                       if getattr(L, "price_missing", False)]
            # every surviving flagged lot came from a price-less BUY
            assert len(flagged) <= n_buy

    def test_full_exit_then_reentry_conserves(self):
        """RQ#618: `_gen_valid` never sells the FULL position, which is
        exactly the history that used to resurrect the sold lot (the legacy
        `total_shares()` fallback). Pin conservation over full round trips."""
        rng = random.Random(SEED + 7)
        for _ in range(N // 4):
            fills, held = [], 0.0
            i = 0
            for _trip in range(rng.randint(1, 5)):
                q = float(rng.randint(1, 50))
                fills.append({"symbol": "AAPL", "action": "BUY", "qty": q,
                              "avg_price": float(rng.randint(1, 500)),
                              "filled_at": _ts(i)}); i += 1
                held += q
                if rng.random() < 0.7:  # full exit
                    fills.append({"symbol": "AAPL", "action": "SELL", "qty": held,
                                  "avg_price": float(rng.randint(1, 500)),
                                  "filled_at": _ts(i)}); i += 1
                    held = 0.0
            got = sum(L.shares for L in reconstruct_live_tax_lots_from_fills(fills).get("AAPL", []))
            assert math.isclose(got, held, abs_tol=1e-6), (fills, got, held)

    def test_empty_and_none(self):
        assert reconstruct_live_tax_lots_from_fills(None) == {}
        assert reconstruct_live_tax_lots_from_fills([]) == {}

    def test_sell_without_prior_buy_is_noop(self):
        fills = [{"symbol": "AAPL", "action": "SELL", "qty": 5, "avg_price": 10,
                  "filled_at": _ts(0)}]
        result = reconstruct_live_tax_lots_from_fills(fills)
        assert result == {}
        # RQ#618 class C: the drop is counted, never silent.
        assert result.stats["dropped_sell_without_lots"] == 1

    def test_oversell_drops_ticker_no_negative(self):
        # Adversarial: sell more than held mid-history. The excess is clamped
        # (never a negative lot) and the position is dropped to flat.
        fills = [
            {"symbol": "AAPL", "action": "BUY", "qty": 10, "avg_price": 100,
             "filled_at": _ts(0)},
            {"symbol": "AAPL", "action": "SELL", "qty": 25, "avg_price": 110,
             "filled_at": _ts(1)},
        ]
        assert reconstruct_live_tax_lots_from_fills(fills) == {}


class TestSellEventPrice:

    def test_prefers_broker_then_fallback_then_zero(self):
        class _Sig:
            def __init__(self, p):
                self.sell_price = p

        assert sell_event_price(_Sig(42.0), 99.0) == 42.0     # broker wins
        assert sell_event_price(_Sig(None), 99.0) == 99.0     # fall back
        assert sell_event_price(_Sig(0.0), 99.0) == 99.0      # 0 not valid
        assert sell_event_price(_Sig(-5.0), 99.0) == 99.0     # negative skip
        assert sell_event_price(_Sig(float("nan")), 99.0) == 99.0
        assert sell_event_price(_Sig(None), None) == 0.0      # nothing valid
        assert sell_event_price(object(), -1.0) == 0.0        # no attr + bad fb

    def test_never_nonfinite_or_negative(self):
        rng = random.Random(SEED + 6)
        choices = [None, 0.0, -1.0, float("nan"), float("inf"),
                   float("-inf"), "x", 12.5, 1e7]

        class _Sig:
            pass

        for _ in range(2000):
            sig = _Sig()
            sig.sell_price = rng.choice(choices)
            fb = rng.choice(choices)
            out = sell_event_price(sig, fb)
            assert math.isfinite(out) and out >= 0.0, (sig.sell_price, fb, out)
