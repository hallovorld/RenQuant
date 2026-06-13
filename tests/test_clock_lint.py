"""Clock/tz lint ratchet tests (eng plan §III.4 / P0.3)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from clock_lint import RATCHET_FILE, scan, scan_file, _is_naive_call  # noqa: E402


class TestDetection:
    def _labels(self, src, tmp_path):
        p = tmp_path / "m.py"
        p.write_text(src)
        return [k for _, k in scan_file(p)]

    def test_datetime_now_flagged(self, tmp_path):
        assert self._labels("import datetime\nx = datetime.datetime.now()\n",
                            tmp_path) == ["datetime.now()"]

    def test_date_today_flagged(self, tmp_path):
        assert self._labels("import datetime\nx = datetime.date.today()\n",
                            tmp_path) == ["date.today()"]

    def test_utcnow_flagged(self, tmp_path):
        assert self._labels("import datetime\nx = datetime.datetime.utcnow()\n",
                            tmp_path) == ["datetime.utcnow()"]

    def test_aware_now_not_flagged(self, tmp_path):
        # now(tz=...) and now(NY) are timezone-aware → clean
        src = ("import datetime\nfrom zoneinfo import ZoneInfo\n"
               "NY = ZoneInfo('America/New_York')\n"
               "a = datetime.datetime.now(tz=NY)\n"
               "b = datetime.datetime.now(NY)\n")
        assert self._labels(src, tmp_path) == []

    def test_time_time_not_flagged(self, tmp_path):
        # epoch time is tz-agnostic → not a hazard
        assert self._labels("import time\nx = time.time()\n", tmp_path) == []

    def test_comment_string_not_flagged(self, tmp_path):
        # AST-based: 'datetime.now()' in a comment/string must not count
        src = "# datetime.now() in a comment\ns = 'date.today()'\n"
        assert self._labels(src, tmp_path) == []


class TestRatchet:
    def test_session_paths_within_ratchet(self):
        hits = scan()
        cap = json.loads(RATCHET_FILE.read_text())["max_naive_time_sources"]
        assert len(hits) <= cap, (
            f"naive time sources grew to {len(hits)} > ratchet {cap}: "
            f"{[(h['file'], h['line'], h['kind']) for h in hits]} — use "
            f"live.clock for trading-day semantics")

    def test_ratchet_well_formed(self):
        r = json.loads(RATCHET_FILE.read_text())
        assert isinstance(r["max_naive_time_sources"], int)
        assert r["floor"] == 0

    def test_clock_module_exempt(self):
        # live/clock.py is the authority; its aware ny_now() must not be
        # counted (and it has no naive sources anyway).
        files = {h["file"] for h in scan()}
        assert "live/clock.py" not in files
