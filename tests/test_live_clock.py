"""P0.3 session-clock authority tests (intraday roadmap §4)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.clock import NY, trading_date  # noqa: E402

UTC = dt.timezone.utc


class TestTradingDate:

    def test_late_pacific_evening_is_next_ny_date(self):
        # 22:00 PT on 2026-06-12 = 01:00 NY on 06-13: the machine's local
        # date (PT) and the exchange date DISAGREE — the bug class P0.3
        # kills. trading_date must say 06-13.
        t = dt.datetime(2026, 6, 13, 1, 0, tzinfo=NY)
        assert trading_date(t) == dt.date(2026, 6, 13)

    def test_dst_fallback_boundary(self):
        # 2026-11-01 01:30 EDT vs EST ambiguity: aware-arithmetic only;
        # both folds map to NY date 11-01.
        t = dt.datetime(2026, 11, 1, 6, 30, tzinfo=UTC)  # 01:30/02:30 NY
        assert trading_date(t) == dt.date(2026, 11, 1)

    def test_utc_input_converted(self):
        t = dt.datetime(2026, 6, 13, 3, 0, tzinfo=UTC)  # 23:00 NY 06-12
        assert trading_date(t) == dt.date(2026, 6, 12)


class TestBreakerUsesTradingDate:

    def test_roll_uses_exchange_date(self, monkeypatch, tmp_path):
        import live.clock as clock
        from live.agent_breaker import AgentBreaker

        monkeypatch.setattr(clock, "ny_now",
                            lambda: dt.datetime(2026, 6, 13, 1, 0, tzinfo=NY))
        b = AgentBreaker(off_flag=tmp_path / "OFF")
        b.admit(symbol="MU", notional=10.0)
        assert b._day == dt.date(2026, 6, 13), \
            "G2 cap day must roll with the exchange, not midnight PT"
