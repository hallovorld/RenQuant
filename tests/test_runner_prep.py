"""runner.py decomposition slice 7 — runner_prep pure-helper tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.runner_prep import (  # noqa: E402
    _parse_iso_dt,
    persisted_skip_buys,
    resolve_hwm,
)


class TestResolveHwm:
    def test_stale_hwm_snaps_to_equity(self):
        # hwm=$100k seed, equity=$10k → ratio 10× > 1.5 → snap down
        hwm, snapped = resolve_hwm(100_000.0, 10_000.0)
        assert hwm == 10_000.0 and snapped is True

    def test_real_drawdown_preserved(self):
        # a real 33% drawdown (ratio 1.49 < 1.5) is NOT snapped
        hwm, snapped = resolve_hwm(14_900.0, 10_000.0)
        assert hwm == 14_900.0 and snapped is False

    def test_ratchets_up_to_equity(self):
        hwm, snapped = resolve_hwm(9_000.0, 10_000.0)
        assert hwm == 10_000.0 and snapped is False


class TestPersistedSkipBuys:
    def test_true(self):
        assert persisted_skip_buys({"skip_buys": True}) is True

    def test_false_default(self):
        assert persisted_skip_buys({}) is False
        assert persisted_skip_buys(None) is False


class TestParseIsoDt:
    def test_valid(self):
        assert _parse_iso_dt("2026-06-12T15:00:00") is not None

    def test_invalid_returns_none(self):
        assert _parse_iso_dt("garbage") is None
        assert _parse_iso_dt(None) is None
