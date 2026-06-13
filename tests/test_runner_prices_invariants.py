"""Property/invariant tests for compute_broker_mark_prices (RU-PRICE-1).

Eng plan S2 item 6 (test-ladder rebalance). The RU-PRICE-1 dust guard is
currently pinned by a STRING-SCAN test
(test_broker_nan_guards.py::test_runner_micro_qty_dust_does_not_inflate_price
asserts the literal ``qty >= 0.5`` appears in the adapter source). This module
establishes the BEHAVIORAL contract that string-scan stands in for: it calls
the extracted pure function (runner.py make_context decomposition slice) and
proves the guard holds across the whole input space — which is strictly
stronger than checking that a particular line of source text exists, and lets
the source-scan retire in a follow-up.

No `hypothesis` dependency (the project pins a hermetic requirements.lock.txt
without it): each property is swept over a deterministic seeded grid and
prints offending inputs on failure for replay.

Invariants pinned:
- RU-PRICE-1 dust guard: a position with qty < 0.5 NEVER yields a price (the
  micro-qty inflation bug of 2026-05-09 cannot recur).
- every emitted price is finite and in the open interval (0, 1e6).
- an emitted price equals market_value / qty exactly.
- prices ⊆ broker_mark_prices (same value) — prices only ever holds marks the
  fallback dict also holds.
- mode gating: a full daily run (not sell_only, not intraday) emits an EMPTY
  prices dict (real-time marks never mix with daily closes); sell-only OR
  intraday makes prices == broker_mark_prices.
- broker_mark_prices is mode-INDEPENDENT: identical for the same positions
  regardless of the two flags.
- non-finite / non-positive / dust / over-cap inputs are dropped from BOTH
  dicts.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner_prices import compute_broker_mark_prices  # noqa: E402

SEED = 0x9A1C
N = 3000
MODES = [(False, False), (True, False), (False, True), (True, True)]


def _positions(rng):
    """A randomized positions_cache spanning admit / dust / non-finite /
    over-cap / zero-mkt regimes."""
    out = {}
    for i in range(rng.randint(0, 6)):
        qty = rng.choice([
            rng.uniform(0.0, 0.5),          # dust (below floor)
            rng.uniform(0.5, 5000),         # admissible
            0.0, -rng.uniform(0, 10),       # zero / negative
            float("nan"), float("inf"),
        ])
        mkt = rng.choice([
            rng.uniform(0.0, 5_000_000),
            0.0, -rng.uniform(0, 1000),
            float("nan"), float("inf"),
            rng.uniform(1e9, 1e12),         # huge → px may exceed cap
        ])
        out[f"T{i}"] = {"qty": qty, "market_value": mkt}
    return out


def _sweep():
    rng = random.Random(SEED)
    for _ in range(N):
        pos = _positions(rng)
        sell_only, intraday = rng.choice(MODES)
        yield pos, sell_only, intraday


class TestPriceGuards:

    def test_emitted_prices_are_finite_and_bounded(self):
        """Every value in EITHER returned dict is finite and in (0, 1e6) — the
        sanity cap that rejects the inflated mkt/qty dust price."""
        for pos, so, intr in _sweep():
            prices, marks = compute_broker_mark_prices(
                pos, sell_only=so, use_intraday_prices=intr)
            for d in (prices, marks):
                for tkr, px in d.items():
                    assert math.isfinite(px) and 0 < px < 1e6, (tkr, px, pos)

    def test_dust_qty_never_priced(self):
        """RU-PRICE-1: a position whose qty is below the 0.5-share floor must
        not appear in either dict, no matter how large its market_value."""
        for pos, so, intr in _sweep():
            prices, marks = compute_broker_mark_prices(
                pos, sell_only=so, use_intraday_prices=intr)
            for tkr, p in pos.items():
                qty = float(p["qty"])
                if math.isfinite(qty) and qty < 0.5:
                    assert tkr not in marks, (tkr, p)
                    assert tkr not in prices, (tkr, p)

    def test_price_equals_mkt_over_qty(self):
        """When admitted, the mark is exactly market_value / qty."""
        for pos, so, intr in _sweep():
            _, marks = compute_broker_mark_prices(
                pos, sell_only=so, use_intraday_prices=intr)
            for tkr, px in marks.items():
                qty = float(pos[tkr]["qty"])
                mkt = float(pos[tkr]["market_value"])
                assert px == mkt / qty, (tkr, px, mkt, qty)

    def test_prices_subset_of_marks(self):
        """prices never carries a ticker (or value) absent from
        broker_mark_prices — the fallback dict is the superset."""
        for pos, so, intr in _sweep():
            prices, marks = compute_broker_mark_prices(
                pos, sell_only=so, use_intraday_prices=intr)
            for tkr, px in prices.items():
                assert tkr in marks and marks[tkr] == px, (tkr, px, marks)


class TestModeGating:

    def test_full_daily_emits_no_prices(self):
        """A full daily run (neither sell-only nor intraday) must NOT seed
        prices from broker marks — daily closes own that dict, mixing in
        real-time marks is the bug this gate prevents."""
        rng = random.Random(SEED + 1)
        for _ in range(N):
            pos = _positions(rng)
            prices, marks = compute_broker_mark_prices(
                pos, sell_only=False, use_intraday_prices=False)
            assert prices == {}, (prices, pos)

    def test_sell_only_or_intraday_fills_prices_from_marks(self):
        """In sell-only OR intraday mode, every trustworthy mark also lands in
        prices — i.e. prices == broker_mark_prices exactly."""
        rng = random.Random(SEED + 2)
        for _ in range(N):
            pos = _positions(rng)
            for so, intr in [(True, False), (False, True), (True, True)]:
                prices, marks = compute_broker_mark_prices(
                    pos, sell_only=so, use_intraday_prices=intr)
                assert prices == marks, (so, intr, prices, marks)

    def test_marks_are_mode_independent(self):
        """broker_mark_prices depends only on the positions, never on the
        flags — it is the OHLCV-missing fallback in every mode."""
        rng = random.Random(SEED + 3)
        for _ in range(N):
            pos = _positions(rng)
            ref = compute_broker_mark_prices(
                pos, sell_only=False, use_intraday_prices=False)[1]
            for so, intr in MODES:
                marks = compute_broker_mark_prices(
                    pos, sell_only=so, use_intraday_prices=intr)[1]
                assert marks == ref, (so, intr, marks, ref)


class TestRejectionBoundaries:

    def test_explicit_boundary_table(self):
        # (qty, mkt) -> admitted?  exercises each guard arm at the boundary.
        cases = [
            ((0.5, 100.0), True),      # exactly the floor, admitted
            ((0.4999, 100.0), False),  # just under the floor → dust
            ((1.0, 0.0), False),       # mkt must be > 0
            ((1.0, -5.0), False),      # negative mkt rejected
            ((float("nan"), 100.0), False),
            ((float("inf"), 100.0), False),
            ((1.0, float("nan")), False),
            ((1.0, float("inf")), False),
            ((1e-6, 1e6), False),      # the original dust-inflation shape
            ((1.0, 2e6), False),       # px == 2e6 ≥ cap → rejected
            ((2.0, 1e6 - 1), True),    # px just under cap → admitted
        ]
        for (qty, mkt), admit in cases:
            _, marks = compute_broker_mark_prices(
                {"X": {"qty": qty, "market_value": mkt}},
                sell_only=True, use_intraday_prices=False)
            assert ("X" in marks) == admit, (qty, mkt, admit, marks)

    def test_missing_fields_default_to_zero_and_drop(self):
        # absent qty/market_value default to 0 → rejected, no KeyError.
        _, marks = compute_broker_mark_prices(
            {"A": {}, "B": {"qty": 3.0}, "C": {"market_value": 300.0}},
            sell_only=True, use_intraday_prices=False)
        assert marks == {}, marks

    def test_empty_positions(self):
        prices, marks = compute_broker_mark_prices(
            {}, sell_only=True, use_intraday_prices=True)
        assert prices == {} and marks == {}
