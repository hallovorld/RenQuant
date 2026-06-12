"""P0.5 dead-man switch tests (intraday roadmap §4; G2 TRADING_OFF reuse)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from deadman_check import decide, heartbeat_age, in_rth  # noqa: E402

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


def _ny(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=NY).astimezone(UTC)


class TestDecide:

    def test_stale_in_rth_trips(self):
        assert decide(age=300, rth=True, flag_exists=False) == "trip"

    def test_missing_heartbeat_counts_as_stale(self):
        assert decide(age=None, rth=True, flag_exists=False) == "trip"

    def test_fresh_in_rth_ok(self):
        assert decide(age=30, rth=True, flag_exists=False) == "ok"

    def test_outside_rth_never_trips(self):
        assert decide(age=None, rth=False, flag_exists=False) == "skip"
        assert decide(age=9999, rth=False, flag_exists=True) == "skip"

    def test_recovered_with_flag_reminds_never_deletes(self):
        # Re-enable is the operator's act (G2 contract): healthy heartbeat
        # + existing flag = remind, not remove.
        assert decide(age=30, rth=True, flag_exists=True) == "remind"

    def test_stale_with_flag_already_present_no_double_trip(self):
        assert decide(age=999, rth=True, flag_exists=True) == "ok_flag_present"


class TestRth:

    def test_midday_trading_day(self):
        assert in_rth(_ny(2026, 6, 12, 12, 0)) is True  # Friday

    def test_pre_open_and_post_close(self):
        assert in_rth(_ny(2026, 6, 12, 9, 15)) is False
        assert in_rth(_ny(2026, 6, 12, 16, 1)) is False

    def test_weekend(self):
        assert in_rth(_ny(2026, 6, 13, 12, 0)) is False  # Saturday

    def test_dst_boundary_is_wall_clock(self):
        # 2026-11-02 (Monday after the 2026-11-01 DST fall-back): 12:00
        # New York is in RTH regardless of the UTC offset change — the
        # check uses NY wall clock, the 217-naive-sources class of bug
        # cannot occur here.
        assert in_rth(_ny(2026, 11, 2, 12, 0)) is True
        assert in_rth(_ny(2026, 11, 2, 8, 0)) is False


class TestHeartbeatAge:

    def test_age_computed(self, tmp_path):
        hb = tmp_path / "heartbeat"
        hb.write_text("1000.0")
        assert heartbeat_age(1042.0, hb) == 42.0

    def test_missing_file_is_none(self, tmp_path):
        assert heartbeat_age(1000.0, tmp_path / "absent") is None

    def test_garbled_file_is_none(self, tmp_path):
        hb = tmp_path / "heartbeat"
        hb.write_text("not-a-number")
        assert heartbeat_age(1000.0, hb) is None
