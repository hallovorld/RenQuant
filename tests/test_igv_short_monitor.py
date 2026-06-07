"""IGV short-plan state machine + options-executor safety tests (no live API)."""
import pytest

from live.igv_short_state import (
    Bar, Market, PlanConfig, PlanState, step,
    ENTER, CLOSE_HALF, CLOSE_MOST, CLOSE_ALL, VOID,
)
from live import options_executor as ox

CFG = PlanConfig()


def _bars(*closes_highs_lows):
    return [Bar(high=h, low=lo, close=c) for (c, h, lo) in closes_highs_lows]


# ── entry path A: reject at 97.5–99 ─────────────────────────────────────────
def test_path_a_entry_on_rejection():
    # touched 97.5–99 (bar high 98.4), latest hourly closes back below 97.5
    bars = _bars((98.2, 98.4, 97.6), (97.1, 97.9, 96.9))
    st, actions = step(PlanState(), Market(price=97.0, hourly_bars=bars), CFG)
    assert st.state == "IN_POSITION" and st.entry_path == "A"
    assert [a.kind for a in actions] == [ENTER]


def test_no_entry_without_rejection_close():
    # touched the zone but latest bar still closes inside it -> no entry
    bars = _bars((98.0, 98.5, 97.6), (98.1, 98.6, 97.8))
    st, actions = step(PlanState(), Market(price=98.1, hourly_bars=bars), CFG)
    assert st.state == "WATCH" and actions == []


def test_standby_blocks_entry_when_reclaimed_100():
    bars = _bars((98.2, 98.4, 97.6), (97.1, 97.9, 96.9))  # would be a path-A reject
    st, actions = step(PlanState(), Market(price=100.3, hourly_bars=bars), CFG)
    assert st.state == "WATCH" and actions == []  # price reclaimed 100 -> stand down


# ── void ────────────────────────────────────────────────────────────────────
def test_void_on_recovery():
    st, actions = step(PlanState(), Market(price=101.6, hourly_bars=[]), CFG)
    assert st.state == "VOIDED" and [a.kind for a in actions] == [VOID]


# ── entry path B: needs breakdown first ─────────────────────────────────────
def test_path_b_requires_breakdown():
    # bounce 95–96 then close <95, but lows never dipped below 94.8 -> no breakdown
    bars = _bars((95.6, 95.9, 95.3), (94.95, 95.4, 94.92))
    st, actions = step(PlanState(), Market(price=94.95, hourly_bars=bars), CFG)
    assert st.broke_below_breakdown is False  # never broke 94.8
    assert st.state == "WATCH" and actions == []  # reject seen, but path B not armed


def test_path_b_entry_after_breakdown():
    bars = _bars((95.5, 95.9, 95.1), (94.6, 95.4, 94.4))
    seeded = PlanState(broke_below_breakdown=True)
    st, actions = step(seeded, Market(price=94.6, hourly_bars=bars), CFG)
    assert st.state == "IN_POSITION" and st.entry_path == "B"
    assert [a.kind for a in actions] == [ENTER]


def test_breakdown_flag_sets_on_sub_948():
    st, _ = step(PlanState(), Market(price=94.5, hourly_bars=[]), CFG)
    assert st.broke_below_breakdown is True


# ── management: take profit / stops ─────────────────────────────────────────
def _in_pos():
    return PlanState(state="IN_POSITION", entry_path="A", contracts=4)


def test_tp_half_once():
    st, actions = step(_in_pos(), Market(price=92.5, hourly_bars=[]), CFG)
    assert st.tp_half_done and [a.kind for a in actions] == [CLOSE_HALF]
    # second eval at same level: no repeat
    st2, actions2 = step(st, Market(price=92.5, hourly_bars=[]), CFG)
    assert actions2 == []


def test_tp_most_closes():
    st, actions = step(_in_pos(), Market(price=89.0, hourly_bars=[]), CFG)
    assert st.state == "CLOSED" and [a.kind for a in actions] == [CLOSE_MOST]


def test_sl_half_once():
    st, actions = step(_in_pos(), Market(price=100.6, hourly_bars=[]), CFG)
    assert st.sl_half_done and [a.kind for a in actions] == [CLOSE_HALF]


def test_sl_exit_on_daily_close():
    st, actions = step(_in_pos(), Market(price=101.0, hourly_bars=[], daily_close=101.7), CFG)
    assert st.state == "CLOSED" and [a.kind for a in actions] == [CLOSE_ALL]


def test_terminal_states_noop():
    for term in ("VOIDED", "CLOSED"):
        st, actions = step(PlanState(state=term), Market(price=95.0, hourly_bars=[]), CFG)
        assert st.state == term and actions == []


# ── executor safety caps (no network) ───────────────────────────────────────
class _FakeClient:
    def __init__(self): self.submitted = []
    def submit_order(self, order): self.submitted.append(order); return {"id": "x", "coid": order.client_order_id}


def _legs():
    import datetime
    return ox.SpreadLegs(long_put_occ="IGV260612P00098000", short_put_occ="IGV260612P00090000",
                         expiry=datetime.date(2026, 6, 12), long_strike=98.0, short_strike=90.0)


def test_cap_rejects_over_max_contracts():
    with pytest.raises(ValueError, match="exceeds hard cap"):
        ox.open_put_spread(_legs(), ox.MAX_CONTRACTS + 1, 3.0, plan_id="t", paper=True, client=_FakeClient())


def test_cap_rejects_insane_debit():
    with pytest.raises(ValueError, match="outside sane bound"):
        ox.open_put_spread(_legs(), 1, 9.0, plan_id="t", paper=True, client=_FakeClient())  # debit > width(8)


def test_open_is_idempotent_client_order_id():
    c = _FakeClient()
    ox.open_put_spread(_legs(), 1, 3.0, plan_id="t", paper=True, client=c)
    ox.open_put_spread(_legs(), 1, 3.0, plan_id="t", paper=True, client=c)
    coids = {o.client_order_id for o in c.submitted}
    assert len(coids) == 1  # same deterministic id -> broker dedupes the resubmit
