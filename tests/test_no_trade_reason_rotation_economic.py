"""The 2026-08-20 mislabel: a vol gate blamed for an economic no-trade.

WHAT HAPPENED. The live message read

    no trade (risk_gate_vol_dropped(30))

while the same run's funnel-integrity line read

    verdict=ECONOMIC_NO_TRADE fired=0 structural=False candidates_final=84 buys=0

and 61 rotations were blocked `nonpositive_expected_return_no_long`. So 84
candidates survived every gate, WERE scored, and the model declined all of
them on economics. The vol gate was not the binding constraint — it was the
last entry in the priority list, reached by fall-through.

`_no_trade_reason`'s own docstring says the 2026-06-01 rewrite reordered the
list for exactly this failure ("Old ordering put risk_gate_vol_dropped ahead of
admission/QP and surfaced 'no trade (vol_dropped(10))' even when 72 of 82
candidates survived"). It fixed the ordering; it never gave the rotation-side
economic block a counter, so the same wrong answer returned through the gap.

WHY IT MATTERED. The operator read that message as evidence that the vol cap
was starving the book and moved to loosen a live risk limit. A message naming
the wrong cause does not merely confuse — it steers capital decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live.runner import _no_trade_reason  # noqa: E402


def _ctx(**kw):
    base = dict(bear_only=False, regime_state=SimpleNamespace(in_transition=False),
                counters={}, ranked=[1, 2, 3], skip_buys=False, buy_blocked=False,
                rotations_blocked=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_2026_08_20_session_does_not_blame_the_vol_gate():
    """The exact shape of that run: vol dropped 30, and 61 rotations were
    declined on economics after scoring."""
    ctx = _ctx(
        counters={"risk_gate_vol_dropped": 30},
        rotations_blocked=[{"sell": None, "buy": f"T{i}",
                            "reason": "nonpositive_expected_return_no_long"}
                           for i in range(61)],
    )
    reason = _no_trade_reason(ctx)
    assert reason == "rotation_nonpositive_expected_return_no_long(61)", reason
    assert "risk_gate_vol_dropped" not in reason


def test_negative_raw_signal_counts_too():
    """The sibling rotation reason from the same live message."""
    ctx = _ctx(counters={"risk_gate_vol_dropped": 9},
               rotations_blocked=[{"buy": "X", "reason": "negative_raw_signal_no_long"}])
    assert _no_trade_reason(ctx) == "rotation_negative_raw_signal_no_long(1)"


def test_the_dominant_reason_is_named_never_pooled():
    """The real 2026-08-20 payload was MIXED: 13 nonpositive-expected-return
    and 47 negative-raw-signal. A pooled total labelled with one of the two
    names would repeat, one layer finer, the very defect this file exists for.

    It matters materially: of those 47, twenty-five had a POSITIVE expected
    return (AFRM +34%, META +19%, SOFI +9%) and were declined on panel score
    alone. Calling that 'nonpositive expected return' would be false.
    """
    ctx = _ctx(
        counters={"risk_gate_vol_dropped": 30},
        rotations_blocked=(
            [{"buy": f"E{i}", "reason": "nonpositive_expected_return_no_long"}
             for i in range(13)]
            + [{"buy": f"S{i}", "reason": "negative_raw_signal_no_long"}
               for i in range(47)]
        ),
    )
    reason = _no_trade_reason(ctx)
    assert reason == "rotation_negative_raw_signal_no_long(47)", reason
    assert "(60)" not in reason, "the two gates must not be pooled into one total"


def test_the_vol_gate_is_still_named_when_it_IS_the_cause():
    """The fall-through must survive: with nothing downstream, a pre-scoring
    vol drop is the honest answer and must still be reported."""
    ctx = _ctx(counters={"risk_gate_vol_dropped": 30})
    assert _no_trade_reason(ctx) == "risk_gate_vol_dropped(30)"


def test_a_non_economic_rotation_block_does_not_hijack_the_reason():
    """Rotations blocked for OTHER causes (e.g. tradeability) are not an
    economic decline and must not mask a genuine vol-gate no-trade."""
    ctx = _ctx(counters={"risk_gate_vol_dropped": 30},
               rotations_blocked=[{"buy": "X", "reason": "already_has_exit"}])
    assert _no_trade_reason(ctx) == "risk_gate_vol_dropped(30)"


def test_earlier_binding_blocks_still_outrank_the_rotation_reason():
    """Ordering contract: an admission/QP block is still more binding."""
    ctx = _ctx(counters={"regime_admission_blocked": 4, "risk_gate_vol_dropped": 30},
               rotations_blocked=[{"buy": "X",
                                   "reason": "nonpositive_expected_return_no_long"}])
    assert _no_trade_reason(ctx) == "regime_admission_blocked(4)"


class TestTheOutputCannotDependOnPayloadORDER:
    """[codex on RenQuant#599] The first version tie-broke with
    `-ord(kv[0][0])`, which compares ONE character. Both reasons start with
    "n", so equal counts fell back to dict insertion order — i.e. to the order
    `rotations_blocked` happened to arrive in. The PR text claimed a
    deterministic alphabetical tie-break that the code did not implement.
    """

    @staticmethod
    def _mixed(order):
        er = [{"buy": f"E{i}", "reason": "nonpositive_expected_return_no_long"}
              for i in range(7)]
        rs = [{"buy": f"S{i}", "reason": "negative_raw_signal_no_long"}
              for i in range(7)]
        return _ctx(counters={"risk_gate_vol_dropped": 30},
                    rotations_blocked=(er + rs) if order == "er_first" else (rs + er))

    def test_equal_counts_give_the_same_answer_in_both_orders(self):
        a = _no_trade_reason(self._mixed("er_first"))
        b = _no_trade_reason(self._mixed("rs_first"))
        assert a == b, f"payload order changed the notification: {a!r} vs {b!r}"

    def test_the_tie_goes_to_the_alphabetically_first_FULL_string(self):
        """Not the first character — the whole string, which is what makes it
        stable as reason names are added."""
        assert _no_trade_reason(self._mixed("er_first")) == \
            "rotation_negative_raw_signal_no_long(7)"


class TestOnlyKnownSignalReasonsAreClassified:
    """[same review] Substring matching on `expected_return` would classify a
    future `missing_expected_return` — a plumbing fault, not an economic
    decline — as the model declining on economics.

    The allowlist is enumerated, and enumerated allow-lists go stale. The
    polarity is what makes it safe HERE: an unlisted reason merely fails to be
    ELEVATED above the vol-gate fall-through, so the default is "do not claim
    this is the cause". Contrast orch#1013, where an unlisted order type was
    silently DROPPED from a collection and the default had to be "include".
    """

    def test_a_missing_value_fault_is_not_an_economic_decline(self):
        ctx = _ctx(counters={"risk_gate_vol_dropped": 30},
                   rotations_blocked=[{"buy": f"X{i}",
                                       "reason": "missing_expected_return"}
                                      for i in range(40)])
        reason = _no_trade_reason(ctx)
        assert reason == "risk_gate_vol_dropped(30)", reason
        assert "expected_return" not in reason

    def test_a_prefixed_lookalike_does_not_match_either(self):
        ctx = _ctx(counters={"risk_gate_vol_dropped": 30},
                   rotations_blocked=[{"buy": "X",
                                       "reason": "stale_negative_raw_signal_no_long"}])
        assert _no_trade_reason(ctx) == "risk_gate_vol_dropped(30)"
