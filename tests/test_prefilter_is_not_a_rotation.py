"""A prefiltered buy candidate is not a blocked rotation (2026-08-20).

WHAT THE OPERATOR SAW. The live 2026-08-20 DECISION message opened with three
of these, then said "+58 more (61 total)":

    BLOCKED-ROTATION None→APH (nonpositive_expected_return_no_long)

and the operator asked the two questions the message provoked: why is every
rotation failing, and why is the sell leg NULL?

NEITHER PREMISE WAS TRUE. No rotation failed, because no rotation was ever
attempted for those names. `BuildPairsTask` declines a buy CANDIDATE before any
sell leg is chosen, and the producer writes `sell=None` together with
`stage="prefilter"` on purpose — its own comment reads "no pair exists yet ...
so monitors can tell the stages apart"
[renquant-pipeline kernel/pipeline/task_rotation.py, VERIFIED in the running
tree at RenQuant/.subrepo_runtime/repos/].

AND THE INVERSE, WHICH IS WORSE. Of the 61 entries that day, 60 were prefilter
and exactly ONE was a genuine blocked rotation: `SPG→CRWD reason=correlation_
guard`, recorded by `ValidatePairsTask` AFTER all 60 prefilter appends
[VERIFIED — log line 550 at 13:57:10,492 vs the prefilter block at ,491]. So it
sat at position 61, inside the "+58 more". The renderer showed the operator
three rotations that never existed and hid the only one that did.

So the producer was correct and self-describing, and the DEFECT WAS PURELY IN
THIS RENDERER: it ignored `stage`, and `rb.get("sell", "?")` could never fire
its own default because the key is present with a null value. The message
invented a rotation that never existed and then reported it as broken.

That is the same class as the two mislabels before it (#598, #599): a
notification naming a cause that is not the cause. It is worth pinning hard,
because this one produced a direct operator question about a subsystem that was
working fine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ctx(**kw) -> SimpleNamespace:
    base: dict = dict(
        orders=[], orders_placed=[], orders_skipped=[], exits=[],
        regime="BULL_CALM", confidence=0.63, portfolio_value=10768.0,
        holdings={f"H{i}": None for i in range(6)},
        bear_only=False, regime_state=SimpleNamespace(in_transition=False),
        skip_buys=False, buy_blocked=False, counters={}, ranked=[],
        rotations_blocked=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _body(ctx) -> str:
    from live.runner import _notify_decision
    with patch("urllib.request.urlopen") as m:
        _notify_decision("RENQUANT-104", "full", ctx)
    return m.call_args[0][0].data.decode()


def _prefilter(ticker: str, reason: str = "nonpositive_expected_return_no_long") -> dict:
    """The exact shape the running pipeline appends."""
    return {"sell": None, "buy": ticker, "reason": reason, "stage": "prefilter"}


class TestTheOperatorsTwoQuestions:
    """Both premises of "why did rotation fail, and why NULL" must disappear."""

    def _real_payload(self):
        """2026-08-20 as it actually was: 61 prefilter entries, no pairs."""
        names = ["APH", "WELL", "CVS", "ROST"] + [f"T{i}" for i in range(57)]
        return _ctx(counters={"risk_gate_vol_dropped": 30},
                    rotations_blocked=[_prefilter(t) for t in names])

    def test_the_word_None_never_reaches_the_operator(self):
        body = _body(self._real_payload())
        assert "None→" not in body, body
        assert "None" not in body, "a null internal value must never be rendered"

    def test_a_prefilter_decline_is_not_called_a_rotation(self):
        body = _body(self._real_payload())
        assert "BLOCKED-ROTATION" not in body, (
            "no rotation was attempted for any of these names — calling it one "
            "is what made the operator ask why rotation was broken"
        )

    def test_the_count_and_the_reason_still_reach_the_operator(self):
        """Renaming must not cost information. The 61 and the reason are the
        diagnostic; only the false framing goes away."""
        body = _body(self._real_payload())
        assert "DECLINED-BUY x61" in body, body
        assert "nonpositive_expected_return_no_long 61" in body, body
        assert "APH" in body


class TestARealRotationBlockIsStillReportedAsOne:
    """The fix must not erase the case the segment was built for."""

    def test_a_paired_block_still_renders_with_both_legs(self):
        body = _body(_ctx(rotations_blocked=[
            {"sell": "GE", "buy": "AMZN", "reason": "preexisting_exit"}]))
        assert "BLOCKED-ROTATION GE→AMZN (preexisting_exit)" in body, body
        assert "DECLINED-BUY" not in body

    def test_the_two_kinds_are_reported_separately_never_pooled(self):
        ctx = _ctx(rotations_blocked=[
            {"sell": "GE", "buy": "AMZN", "reason": "insufficient_cash"},
            _prefilter("APH"), _prefilter("WELL"),
        ])
        body = _body(ctx)
        assert "BLOCKED-ROTATION GE→AMZN (insufficient_cash)" in body, body
        assert "DECLINED-BUY x2" in body, body
        assert "x3" not in body, "one real rotation + two prefilters is not three of either"


class TestTheStageFieldIsNotTheOnlyDefence:
    """`stage` is what the current producer writes, but a null sell leg is
    itself proof that no pair exists — an older or divergent producer that
    omits `stage` must not fall back to printing `None→`."""

    def test_a_null_sell_without_a_stage_field_is_still_a_prefilter(self):
        body = _body(_ctx(rotations_blocked=[
            {"sell": None, "buy": "APH", "reason": "negative_raw_signal_no_long"}]))
        assert "DECLINED-BUY x1" in body, body
        assert "None" not in body

    def test_a_missing_sell_key_is_treated_the_same(self):
        body = _body(_ctx(rotations_blocked=[
            {"buy": "APH", "reason": "negative_raw_signal_no_long"}]))
        assert "DECLINED-BUY x1" in body, body
        assert "?→" not in body


class TestMixedReasonsAreSplitAndOrderIndependent:
    """Same lesson as #599: what the operator sees must not depend on the
    order the payload happened to arrive in."""

    @staticmethod
    def _mixed(order: str):
        er = [_prefilter(f"E{i}", "nonpositive_expected_return_no_long") for i in range(13)]
        rs = [_prefilter(f"S{i}", "negative_raw_signal_no_long") for i in range(47)]
        return _ctx(rotations_blocked=(er + rs) if order == "er_first" else (rs + er))

    def test_each_reason_keeps_its_own_count(self):
        body = _body(self._mixed("er_first"))
        assert "negative_raw_signal_no_long 47" in body, body
        assert "nonpositive_expected_return_no_long 13" in body, body
        assert "x60" in body, "the TOTAL is still 60 — the split is within it"

    def test_reversing_the_payload_does_not_change_the_REASON_SPLIT(self):
        """Only the split is asserted, deliberately — see the next test for
        why the sample tickers must NOT be order-independent."""
        import re
        a, b = _body(self._mixed("er_first")), _body(self._mixed("rs_first"))
        why = lambda s: re.search(r"DECLINED-BUY x\d+ \(([^)]*)\)", s).group(1)
        assert why(a) == why(b), f"{why(a)} vs {why(b)}"

    def test_equal_counts_still_order_by_the_full_reason_string(self):
        ctx = _ctx(rotations_blocked=(
            [_prefilter(f"S{i}", "negative_raw_signal_no_long") for i in range(4)]
            + [_prefilter(f"E{i}", "nonpositive_expected_return_no_long") for i in range(4)]))
        body = _body(ctx)
        i_neg = body.index("negative_raw_signal_no_long 4")
        i_non = body.index("nonpositive_expected_return_no_long 4")
        assert i_neg < i_non, "ties resolve alphabetically on the full string"


class TestTheBodyBudgetIsProtected:
    """The original 2026-08-19 cap existed because 60 segments (~3.3 KB) pushed
    the body past _NTFY_BODY_MAX_BYTES and the truncation fell INSIDE the
    blocked list, evicting the regime/equity tail. Collapsing to one segment
    must not reintroduce that."""

    def test_sixty_one_prefilters_produce_exactly_one_segment(self):
        body = _body(_ctx(rotations_blocked=[_prefilter(f"T{i}") for i in range(61)]))
        assert sum(1 for p in body.split(" | ") if p.startswith("DECLINED-BUY")) == 1, body

    def test_the_context_the_operator_reads_survives(self):
        body = _body(_ctx(rotations_blocked=[_prefilter(f"T{i}") for i in range(61)]))
        assert "regime=BULL_CALM" in body, "the tail must not be truncated away"
        assert "eq=$10,768" in body
        assert "[truncated]" not in body


class TestTheSampleTickersFollowCANDIDATERANK:
    """The tickers shown are payload-order, and that is CORRECT, not a leak.

    I first wrote an order-independence test over the whole segment and it
    failed here. The code was right and the test was wrong: the producer builds
    its list as `[c for c in ctx.ranked if c.ticker not in held_set]`
    [VERIFIED — task_rotation.py:250 in the running tree], so `rotations_blocked`
    arrives in descending candidate rank. The 2026-08-20 log confirms it to the
    decimal — APH 2.434, WELL 2.210, CVS 2.209, ROST 1.934, strictly
    descending.

    So the first three names are "the highest-ranked candidates the model
    declined", which is the single most useful sample to show. Sorting them for
    determinism would destroy real information in exchange for a property
    nobody needs. The reason SPLIT must be order-independent; the sample must
    not be.
    """

    def test_the_top_ranked_declines_are_the_ones_shown(self):
        ranked_order = ["APH", "WELL", "CVS", "ROST"] + [f"T{i}" for i in range(20)]
        body = _body(_ctx(rotations_blocked=[_prefilter(t) for t in ranked_order]))
        assert "APH, WELL, CVS +21 more" in body, body
        assert "ROST" not in body, "the 4th-ranked name is inside the +N, not shown"


class TestTheONERealRotationThatDayIsNoLongerHidden:
    """The inverse of the headline bug, and the more damaging half.

    2026-08-20 had 61 entries: 60 prefilter declines, then ONE genuine blocked
    rotation — `SPG→CRWD reason=correlation_guard` from `ValidatePairsTask`,
    appended last [VERIFIED — logs/daily_104/2026-08-20.log:550]. Under the old
    single flat list the visible slots went to the first three, all prefilter,
    and the real rotation block sat at position 61 inside "+58 more".

    So the message did not merely mislabel: it spent the operator's three
    visible slots on rotations that never existed while concealing the one that
    actually happened. Splitting the kinds fixes both directions at once.
    """

    def _the_real_payload(self):
        names = ["APH", "WELL", "CVS", "ROST"] + [f"T{i}" for i in range(56)]
        return _ctx(counters={"risk_gate_vol_dropped": 30}, rotations_blocked=(
            [_prefilter(t) for t in names]
            + [{"sell": "SPG", "buy": "CRWD", "reason": "correlation_guard"}]))

    def test_the_real_block_surfaces_as_its_own_segment(self):
        body = _body(self._the_real_payload())
        assert "BLOCKED-ROTATION SPG→CRWD (correlation_guard)" in body, body

    def test_it_is_not_buried_behind_the_sixty(self):
        """It must not be summarised away: one paired block is under the cap,
        so there is no `+N more` for the paired list at all."""
        body = _body(self._the_real_payload())
        assert "BLOCKED-ROTATION +" not in body, body

    def test_the_sixty_are_still_counted_separately(self):
        body = _body(self._the_real_payload())
        assert "DECLINED-BUY x60" in body, body
        assert "x61" not in body, "60 declines + 1 rotation is not 61 of either"
