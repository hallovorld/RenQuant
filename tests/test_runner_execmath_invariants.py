"""Property/invariant tests for the runner execution-math primitives.

Eng plan S2 item 6 (test-ladder rebalance): the existing
`test_runner_state_fixes.py` pins these functions with worked EXAMPLES; this
module pins the SAFETY INVARIANTS that must hold across the whole input
space — the properties a single example can't guarantee. The subjects are the
pure functions extracted to `adapters/runner_execmath.py` (decomposition
slice 5): they sit on the live-cash and broker-response edges, so a violated
invariant here is real money mis-sized or rejected fills mutating state.

No `hypothesis` dependency: the project pins a hermetic `requirements.lock.txt`
and hypothesis is not in it, so adding it would un-hermetic CI. Instead each
property is swept over a deterministic, seeded grid of thousands of cases
(`random.Random(SEED)`), which is reproducible and CI-safe. If a case fails,
the assertion message prints the exact inputs so the failure is replayable.

Invariants:
- cap_buy_order_to_cash NEVER admits spend > remaining cash (the core money
  safety property), resizes only DOWNWARD to a positive integer, is monotone
  non-decreasing in cash, and is a no-op when the order already fits.
- broker_order_execution: filled/pending/rejected are mutually exclusive and
  exhaustive; a non-filled classification zeroes filled_qty (rejected/pending
  fills must never mutate live state); filled_avg_price is always positive.
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

from adapters import runner as _runner  # noqa: E402
from adapters.runner_execmath import (  # noqa: E402
    broker_order_execution,
    cap_buy_order_to_cash,
    normalize_order_status,
)

SEED = 0x5EED
N = 4000  # cases per property — large enough to exercise the grid corners


def _cases(seed_offset: int, n: int = N):
    """Deterministic stream of (cash, shares, price) triples spanning the
    interesting regime boundaries (afford-all / resize / reject / degenerate)."""
    rng = random.Random(SEED + seed_offset)
    for _ in range(n):
        # mix scales so we hit "fits", "must resize", and "can't afford 1"
        cash = rng.choice([
            0.0, rng.uniform(0, 5), rng.uniform(5, 1_000),
            rng.uniform(1_000, 250_000),
        ])
        price = rng.choice([
            rng.uniform(0.01, 5), rng.uniform(5, 500), rng.uniform(500, 4000),
        ])
        shares = rng.choice([
            float(rng.randint(1, 3)), float(rng.randint(1, 2000)),
            rng.uniform(0.0, 5.0),
        ])
        yield cash, shares, price


class TestCapBuyOrderToCash:

    def test_never_overspends(self):
        """THE money-safety invariant: whatever comes back, its realized spend
        (shares × price) is ≤ remaining cash (within the function's 1e-6
        slack). A breach is a live overdraft."""
        for cash, shares, price in _cases(1):
            order = {"shares": shares, "price": price, "ticker": "T"}
            capped, reason = cap_buy_order_to_cash(order, cash)
            if capped is None:
                # rejection never spends — nothing to check beyond the reason
                assert reason in ("cash_budget_exhausted", "bad_order"), (
                    cash, shares, price, reason)
                continue
            spend = float(capped["shares"]) * price
            assert spend <= cash + 1e-6, (
                f"OVERSPEND cash={cash} shares={capped['shares']} "
                f"price={price} spend={spend} reason={reason}")

    def test_resizes_only_downward_to_positive_int(self):
        """A budget resize must reduce share count, land on a positive whole
        number of shares, and never invent shares the caller didn't ask for."""
        for cash, shares, price in _cases(2):
            order = {"shares": shares, "price": price, "ticker": "T"}
            capped, reason = cap_buy_order_to_cash(order, cash)
            if reason != "cash_budget_resized":
                continue
            new = capped["shares"]
            assert new == int(new) and new >= 1, (cash, shares, price, new)
            assert new < shares, (
                f"resize did not shrink: {shares} -> {new}")
            assert capped["original_shares"] == shares

    def test_noop_when_affordable(self):
        """If the order already fits the budget, shares are untouched and no
        adjustment reason is emitted (idempotent admit)."""
        for cash, shares, price in _cases(3):
            if not (math.isfinite(cash) and shares > 0 and price > 0):
                continue
            if shares * price > cash + 1e-6:
                continue  # this case is a resize/reject, not a no-op
            order = {"shares": shares, "price": price, "ticker": "T"}
            capped, reason = cap_buy_order_to_cash(order, cash)
            assert reason is None, (cash, shares, price, reason)
            assert capped is not None and capped["shares"] == shares
            assert "budget_adjustment" not in capped

    def test_monotone_nondecreasing_in_cash(self):
        """More budget can only ever buy at least as many shares — sizing is
        monotone in cash. Guards against a resize branch that non-monotonically
        rounds (e.g. an off-by-one in floor division)."""
        rng = random.Random(SEED + 4)
        for _ in range(N):
            price = rng.uniform(0.5, 800)
            shares = float(rng.randint(1, 1500))
            c1 = rng.uniform(0, 200_000)
            c2 = c1 + rng.uniform(0, 50_000)  # c2 >= c1
            order = {"shares": shares, "price": price, "ticker": "T"}

            def _got(cash):
                capped, _ = cap_buy_order_to_cash(dict(order), cash)
                return float(capped["shares"]) if capped else 0.0

            s1, s2 = _got(c1), _got(c2)
            assert s2 >= s1, (
                f"non-monotone: cash {c1}->{c2} gave shares {s1}->{s2} "
                f"(price={price}, want={shares})")

    def test_reject_implies_unaffordable_or_bad(self):
        """A None return is justified: 'cash_budget_exhausted' only when even
        one share is unaffordable; 'bad_order' only on non-finite/non-positive
        inputs."""
        for cash, shares, price in _cases(5):
            order = {"shares": shares, "price": price, "ticker": "T"}
            capped, reason = cap_buy_order_to_cash(order, cash)
            if capped is not None:
                continue
            inputs_ok = (math.isfinite(cash) and math.isfinite(shares)
                         and math.isfinite(price) and price > 0 and shares > 0)
            if reason == "cash_budget_exhausted":
                assert inputs_ok and cash < price, (cash, shares, price)
            else:
                assert reason == "bad_order"

    def test_non_finite_inputs_are_bad_order(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            for order in ({"shares": bad, "price": 10.0},
                          {"shares": 5.0, "price": bad},
                          {"shares": 5.0, "price": -1.0},
                          {"shares": 0.0, "price": 10.0}):
                capped, reason = cap_buy_order_to_cash(order, 1000.0)
                assert capped is None and reason == "bad_order", (order, reason)
            capped, reason = cap_buy_order_to_cash({"shares": 5.0, "price": 10.0}, bad)
            assert capped is None and reason == "bad_order", (bad, reason)


class TestBrokerOrderExecution:

    _STATUSES = [
        "filled", "partially_filled", "new", "accepted", "pending_new",
        "rejected", "canceled", "cancelled", "expired", "stopped",
        "suspended", "done_for_day", "OrderStatus.FILLED", "", None,
    ]

    def _cases(self):
        rng = random.Random(SEED + 6)
        for _ in range(N):
            status = rng.choice(self._STATUSES)
            requested = rng.choice([0.0, float(rng.randint(1, 500))])
            filled = rng.choice([0.0, None, float(rng.randint(0, 500))])
            avg = rng.choice([0.0, None, rng.uniform(1, 4000)])
            fallback = rng.uniform(1, 4000)
            result = {"status": status, "filled_qty": filled,
                      "filled_avg_price": avg, "quantity": requested}
            yield result, requested, fallback

    def test_pending_is_exactly_the_unresolved_complement(self):
        """`pending` partitions the order space: an order is pending IFF it is
        neither filled nor rejected. This is the property the runner relies on
        to decide "wait for next session vs act now" — there is no fourth
        state and no gap."""
        for result, requested, fallback in self._cases():
            out = broker_order_execution(result, requested, fallback)
            assert out["pending"] == (not out["filled"] and not out["rejected"]), out

    def test_partial_fill_then_terminal_status_honors_the_fill(self):
        """A partial fill that is then terminally canceled is BOTH filled (the
        already-executed shares are real and must mutate state) AND rejected
        (no further fills will arrive). filled and rejected legitimately
        co-occur here — the safety rule is that the executed quantity is never
        discarded just because the order later went terminal."""
        out = broker_order_execution(
            {"status": "canceled", "filled_qty": 50, "filled_avg_price": 20.0},
            requested_qty=100, fallback_price=10.0)
        assert out["filled"] and out["rejected"]
        assert out["partial"]
        assert out["filled_qty"] == 50.0
        assert out["filled_avg_price"] == 20.0

    def test_non_filled_zeroes_quantity(self):
        """The state-mutation guard: only filled quantity may move live state,
        so a non-filled classification MUST report filled_qty == 0.0."""
        for result, requested, fallback in self._cases():
            out = broker_order_execution(result, requested, fallback)
            if not out["filled"]:
                assert out["filled_qty"] == 0.0, out

    def test_avg_price_always_positive(self):
        """Downstream P/L and cash math divide/multiply by avg price; it must
        never be 0 or negative — it falls back to the bar price."""
        for result, requested, fallback in self._cases():
            out = broker_order_execution(result, requested, fallback)
            assert out["filled_avg_price"] > 0, out

    def test_clean_rejection_with_no_fill_never_mutates(self):
        """The realistic terminal-reject shape (broker reports zero fill): the
        order is rejected, NOT filled, and filled_qty is zeroed so it can never
        move live state, cash, or P/L."""
        for status in ("rejected", "canceled", "expired", "done_for_day"):
            out = broker_order_execution(
                {"status": status, "filled_qty": 0, "filled_avg_price": 0},
                requested_qty=99, fallback_price=10.0)
            assert out["rejected"] and not out["filled"]
            assert out["filled_qty"] == 0.0
            assert out["filled_avg_price"] == 10.0  # falls back to bar price


class TestNormalizeOrderStatus:

    def test_strips_enum_prefix_and_lowercases(self):
        rng = random.Random(SEED + 7)
        for raw, want in [
            ("OrderStatus.FILLED", "filled"),
            ("  Accepted  ", "accepted"),
            ("PENDING_NEW", "pending_new"),
            (None, ""),
            ("", ""),
        ]:
            assert normalize_order_status(raw) == want
        # idempotent: normalizing a normalized token is a fixpoint
        for _ in range(200):
            tok = rng.choice(["filled", "rejected", "new", "partially_filled"])
            assert normalize_order_status(tok) == tok


class TestReexportContract:
    """The decomposition must stay transparent: runner re-exports the same
    objects (S2.5 contract that S2.6 tests lean on)."""

    def test_runner_reexports_same_objects(self):
        assert _runner.cap_buy_order_to_cash is cap_buy_order_to_cash
        assert _runner.broker_order_execution is broker_order_execution
        assert _runner.normalize_order_status is normalize_order_status
