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
    assert reason == "rotation_nonpositive_expected_return(61)", reason
    assert "risk_gate_vol_dropped" not in reason


def test_negative_raw_signal_counts_too():
    """The sibling rotation reason from the same live message."""
    ctx = _ctx(counters={"risk_gate_vol_dropped": 9},
               rotations_blocked=[{"buy": "X", "reason": "negative_raw_signal_no_long"}])
    assert _no_trade_reason(ctx) == "rotation_nonpositive_expected_return(1)"


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
               rotations_blocked=[{"buy": "X", "reason": "nonpositive_expected_return"}])
    assert _no_trade_reason(ctx) == "regime_admission_blocked(4)"
