"""Earnings-calendar staleness rail + incremental-selection tests.

2026-08-30 data-audit fixes under test:
  * assess_earnings_calendar_freshness — the rail that makes a stale
    calendar LOUD (the Apr-24-frozen prod artifact silently disabled
    the pre/post-earnings buffer for every Aug/Sep 2026 print);
  * select_recent_prints — the incremental selection for the daily
    earnings-surprise refresh (only tickers with a print in the last
    N days need a PEAD/SUE refetch);
  * fetch_earnings_calendar helpers — output path now targets the
    CONSUMED artifacts/prod/ location (the pre-fix script wrote to
    artifacts/, one level off since the 2026-05-10 sim/prod refactor),
    and merge keeps known dates through transient vendor failures;
  * the earnings_calendar_rail.py CLI exit-code contract the shell
    wrappers (daily_104.sh Step 0c, refresh_earnings_calendar.sh,
    daily_earnings_surprise_refresh.sh) depend on.

Stdlib + pytest only — this file is run by
.github/workflows/earnings-freshness-contract.yml.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.earnings_freshness import (  # noqa: E402
    assess_earnings_calendar_freshness,
    earnings_calendar_horizon,
    select_recent_prints,
)


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TODAY = datetime.date(2026, 8, 30)


# ── Freshness rail ───────────────────────────────────────────────────────────

class TestFreshnessRail:
    def test_fresh_calendar_is_ok(self):
        cal = {"NVDA": ["2026-08-26", "2026-11-25"]}
        v = assess_earnings_calendar_freshness(cal, today=TODAY,
                                               min_horizon_days=5)
        assert v["status"] == "ok"
        assert v["last_date"] == datetime.date(2026, 11, 25)

    def test_boundary_exactly_today_plus_horizon_is_ok(self):
        cal = {"HPE": [(TODAY + datetime.timedelta(days=5)).isoformat()]}
        v = assess_earnings_calendar_freshness(cal, today=TODAY,
                                               min_horizon_days=5)
        assert v["status"] == "ok"

    def test_below_horizon_is_stale(self):
        # The actual defect shape: last date 2026-07-24 vs today 2026-08-30.
        cal = {"AAPL": ["2026-04-30"], "PANW": ["2026-07-24"]}
        v = assess_earnings_calendar_freshness(cal, today=TODAY,
                                               min_horizon_days=5)
        assert v["status"] == "stale"
        assert v["last_date"] == datetime.date(2026, 7, 24)
        assert "buffer" in v["message"]

    def test_one_day_short_is_stale(self):
        cal = {"HPE": [(TODAY + datetime.timedelta(days=4)).isoformat()]}
        v = assess_earnings_calendar_freshness(cal, today=TODAY,
                                               min_horizon_days=5)
        assert v["status"] == "stale"

    @pytest.mark.parametrize("cal", [None, {}, [], "nope",
                                     {"AAPL": ["garbage", 42]}])
    def test_missing_or_unparseable_is_missing(self, cal):
        v = assess_earnings_calendar_freshness(cal, today=TODAY,
                                               min_horizon_days=5)
        assert v["status"] == "missing"
        assert v["last_date"] is None

    def test_horizon_ignores_garbage_dates(self):
        cal = {"A": ["not-a-date", "2026-09-04"], "B": [None]}
        assert earnings_calendar_horizon(cal) == datetime.date(2026, 9, 4)


# ── Incremental selection ────────────────────────────────────────────────────

class TestSelectRecentPrints:
    def test_selects_prints_within_lookback_only(self):
        cal = {
            "NVDA": ["2026-08-26"],            # 4d ago — selected
            "CRWD": ["2026-08-26"],            # selected
            "PANW": ["2026-08-20"],            # 10d ago — NOT selected
            "HPE":  ["2026-09-03"],            # future — NOT selected
            "AAPL": ["2026-04-30"],            # ancient — NOT selected
        }
        assert select_recent_prints(cal, today=TODAY, lookback_days=7) == \
            ["CRWD", "NVDA"]

    def test_window_is_inclusive_both_ends(self):
        cal = {
            "EDGE0": [TODAY.isoformat()],
            "EDGE7": [(TODAY - datetime.timedelta(days=7)).isoformat()],
            "EDGE8": [(TODAY - datetime.timedelta(days=8)).isoformat()],
        }
        assert select_recent_prints(cal, today=TODAY, lookback_days=7) == \
            ["EDGE0", "EDGE7"]

    def test_dedupes_and_uppercases(self):
        cal = {"nvda": ["2026-08-26", "2026-08-27"]}
        assert select_recent_prints(cal, today=TODAY, lookback_days=7) == ["NVDA"]

    def test_non_dict_returns_empty(self):
        assert select_recent_prints(None, today=TODAY) == []
        assert select_recent_prints(["x"], today=TODAY) == []


# ── fetch_earnings_calendar helpers ─────────────────────────────────────────

class TestFetchCalendarHelpers:
    @pytest.fixture(scope="class")
    def fetch_mod(self):
        return _load_script("fetch_earnings_calendar")

    def test_output_path_prefers_consumed_prod_dir(self, fetch_mod, tmp_path):
        (tmp_path / "artifacts" / "prod").mkdir(parents=True)
        out = fetch_mod.resolve_output_path(tmp_path)
        # The consumers (main.py, adapters/runner_artifacts.py) read
        # prod/earnings-calendar.json — the pre-fix script wrote one
        # level up and never refreshed the consumed artifact.
        assert out == tmp_path / "artifacts" / "prod" / "earnings-calendar.json"

    def test_output_path_falls_back_to_artifacts_then_root(self, fetch_mod, tmp_path):
        (tmp_path / "artifacts").mkdir()
        assert fetch_mod.resolve_output_path(tmp_path) == \
            tmp_path / "artifacts" / "earnings-calendar.json"
        bare = tmp_path / "bare"
        bare.mkdir()
        assert fetch_mod.resolve_output_path(bare) == \
            bare / "earnings-calendar.json"

    def test_merge_keeps_previous_dates_on_empty_fetch(self, fetch_mod):
        prev = {"NVDA": ["2026-08-26", "2026-11-25"]}
        fetched = {"NVDA": []}     # transient vendor failure
        merged = fetch_mod.merge_calendars(prev, fetched, ["NVDA"], TODAY)
        assert merged["NVDA"] == ["2026-08-26", "2026-11-25"]

    def test_merge_unions_and_drops_old_dates(self, fetch_mod):
        prev = {"HPE": ["2026-04-24", "2026-09-03"]}
        fetched = {"HPE": ["2026-09-03", "2026-12-02"]}
        merged = fetch_mod.merge_calendars(prev, fetched, ["HPE"], TODAY,
                                           keep_past_days=45)
        assert merged["HPE"] == ["2026-09-03", "2026-12-02"]  # 04-24 dropped

    def test_merge_scopes_to_watchlist_and_tolerates_garbage(self, fetch_mod):
        prev = {"GONE": ["2026-09-10"], "NVDA": ["junk"]}
        fetched = {"NVDA": ["2026-09-10"], "CRWD": ["2026-09-11"]}
        merged = fetch_mod.merge_calendars(prev, fetched, ["NVDA", "CRWD"], TODAY)
        assert set(merged) == {"NVDA", "CRWD"}
        assert merged["NVDA"] == ["2026-09-10"]

    def test_calendar_last_date(self, fetch_mod):
        assert fetch_mod.calendar_last_date(
            {"A": ["2026-09-04"], "B": ["2026-12-02", "bad"]}) == "2026-12-02"
        assert fetch_mod.calendar_last_date({"A": ["bad"]}) is None


# ── CLI exit-code contract (what the shell wrappers script against) ─────────

class TestRailCli:
    @pytest.fixture(scope="class")
    def rail(self):
        return _load_script("earnings_calendar_rail")

    def _write(self, tmp_path, payload) -> str:
        p = tmp_path / "earnings-calendar.json"
        p.write_text(json.dumps(payload))
        return str(p)

    def test_check_fresh_exits_0(self, rail, tmp_path):
        cal = self._write(tmp_path, {"NVDA": ["2026-11-25"]})
        assert rail.main(["check", "--calendar", cal,
                          "--min-horizon-days", "5",
                          "--today", "2026-08-30"]) == 0

    def test_check_stale_exits_3(self, rail, tmp_path):
        cal = self._write(tmp_path, {"PANW": ["2026-07-24"]})
        assert rail.main(["check", "--calendar", cal,
                          "--min-horizon-days", "5",
                          "--today", "2026-08-30"]) == 3

    def test_check_missing_file_exits_4(self, rail, tmp_path):
        assert rail.main(["check", "--calendar",
                          str(tmp_path / "nope.json"),
                          "--today", "2026-08-30"]) == 4

    def test_check_unreadable_json_exits_4(self, rail, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert rail.main(["check", "--calendar", str(p),
                          "--today", "2026-08-30"]) == 4

    def test_select_recent_prints_tickers(self, rail, tmp_path, capsys):
        cal = self._write(tmp_path, {
            "NVDA": ["2026-08-26"], "PANW": ["2026-08-20"],
            "HPE": ["2026-09-03"],
        })
        rc = rail.main(["select-recent", "--calendar", cal,
                        "--lookback-days", "7", "--today", "2026-08-30"])
        assert rc == 0
        assert capsys.readouterr().out.split() == ["NVDA"]

    def test_select_recent_missing_calendar_exits_4(self, rail, tmp_path):
        assert rail.main(["select-recent", "--calendar",
                          str(tmp_path / "nope.json"),
                          "--today", "2026-08-30"]) == 4
