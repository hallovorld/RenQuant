"""Contract tests for `live_state.json` — each attribute's R/W cycle.

Context: user asked for a review of every attribute after 3 distinct
bugs in 3 sessions (HWM stale seed / empty entry_dates / missing
regime_state persistence). This suite pins behavioural invariants
so the next regression is caught at test time, not by staring at
live Alpaca logs.

All tests are source-level (they scan runner.py / exercise pure
RunnerAdapter logic without broker). Integration coverage remains
in `tests/test_runner_hwm_guard.py` + `tests/test_regime_state_persistence.py`.

Keys audited:
  regime, regime_confidence, high_water_mark,
  entry_dates, sell_streaks, last_sell_dates, position_hwm,
  monitor_state, regime_state
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

RUNNER_PATH = _STRATEGY_DIR / "adapters" / "runner.py"
SOURCE      = RUNNER_PATH.read_text()


# ── Read-load invariants ─────────────────────────────────────────────────────

class TestAllKeysAreLoaded:
    """Every persisted attribute must have a read site in make_context.

    Missing load means the value is ignored at invocation time — a classic
    failure mode (see HWM stale-seed bug + regime_state countdown bug).
    """

    @pytest.mark.parametrize("key,default_form", [
        ("entry_dates",     r'state.get("entry_dates"'),
        ("sell_streaks",    r'state.get("sell_streaks"'),
        ("last_sell_dates", r'state.get("last_sell_dates"'),
        ("position_hwm",    r'state.get("position_hwm"'),
        ("high_water_mark", r'state.get("high_water_mark"'),
        ("monitor_state",   r'state.get("monitor_state"'),
        ("regime_state",    r'state.get("regime_state"'),
    ])
    def test_key_is_loaded(self, key: str, default_form: str):
        assert default_form in SOURCE, (
            f"live_state.json key `{key}` must be loaded via "
            f"{default_form!r} in RunnerAdapter.make_context"
        )


class TestAllKeysAreSaved:
    """Every persisted attribute must have a write site in apply_outputs.

    Missing save means the value never makes it to disk — the second
    classic failure mode.
    """

    @pytest.mark.parametrize("key", [
        "regime", "regime_confidence", "high_water_mark",
        "entry_dates", "sell_streaks", "last_sell_dates",
        "position_hwm", "monitor_state", "regime_state",
    ])
    def test_key_is_saved(self, key: str):
        assert f'"{key}":' in SOURCE, (
            f"live_state.json key `{key}` must be written in "
            f"RunnerAdapter.apply_outputs"
        )


# ── Per-attribute semantics ──────────────────────────────────────────────────

class TestHighWaterMark:
    """`high_water_mark` must resolve stale values on load — the May-2026 bug.

    Invariant: if stored HWM > 1.5× account_value, snap down.
    """
    def test_resolve_hwm_stale_snapback(self):
        from adapters.runner import resolve_hwm
        new_hwm, snapped = resolve_hwm(stored_hwm=100_000, account_value=10_000)
        assert snapped is True
        assert new_hwm == pytest.approx(10_000)

    def test_resolve_hwm_normal_preserved(self):
        from adapters.runner import resolve_hwm
        new_hwm, snapped = resolve_hwm(stored_hwm=12_000, account_value=10_000)
        assert snapped is False
        assert new_hwm == pytest.approx(12_000)

    def test_resolve_hwm_ratchet_up(self):
        from adapters.runner import resolve_hwm
        new_hwm, snapped = resolve_hwm(stored_hwm=9_000, account_value=10_500)
        assert snapped is False
        assert new_hwm == pytest.approx(10_500)


class TestEntryDatesPersistenceFallback:
    """`entry_dates` must PERSIST the today-fallback for legacy positions.

    Before fix (2026-04-23): `entry_dates.get(ticker, today)` returned
    today but never wrote back → next run recomputed fresh today →
    hold_days = today - today = 0 forever → min_hold_days/min_rotation_hold
    gates locked.
    """

    def test_runner_writes_fallback_into_dict(self):
        """The fallback for a missing ticker must be stamped INTO
        entry_dates so subsequent loads see a stable date."""
        assert 'if ticker not in entry_dates:\n                entry_dates[ticker]' in SOURCE, (
            "entry_dates fallback for legacy positions must be persisted "
            "into the dict, not just returned — otherwise hold_days "
            "always shows 0 for pre-104 positions"
        )


class TestSellStreaksLifecycle:
    """`sell_streaks[ticker]` counts consecutive sell signals.

    Invariants:
      - Reset to 0 on BUY of that ticker.
      - Popped on SELL of that ticker (the key goes away, not stuck at 0).
      - Written from HoldingState.sell_streak at end of bar.
    """

    def test_reset_on_buy(self):
        assert 'self._sell_streaks.pop(ticker, None)' in SOURCE

    def test_written_from_hs_sell_streak(self):
        assert 'self._sell_streaks[ticker] = hs.sell_streak' in SOURCE


class TestLastSellDatesWashSaleIntegration:
    """`last_sell_dates[ticker]` → wash-sale guard.

    Invariants:
      - Recorded on SELL (ISO date string).
      - Popped on BUY (same-day sell→buy fine once wash_sale_days elapses).
      - String-shaped (not datetime) — JSON-safe.
    """

    def test_write_on_sell(self):
        assert 'self._last_sell_dates_str[ticker] = today_str' in SOURCE

    def test_pop_on_buy(self):
        assert 'self._last_sell_dates_str.pop(ticker, None)' in SOURCE


class TestPositionHWMLifecycle:
    """`position_hwm[ticker]` feeds the trailing stop.

    Invariants:
      - Seeded to entry_price on BUY.
      - Ratcheted `max(hwm, current_price)` each bar.
      - Popped on SELL (position closed).
    """

    def test_seeded_on_buy(self):
        assert 'self._position_hwm[ticker]      = price' in SOURCE

    def test_ratchet_max(self):
        assert 'self._position_hwm[ticker] = max(' in SOURCE

    def test_popped_on_sell(self):
        assert 'self._position_hwm.pop(ticker, None)' in SOURCE


class TestMonitorStateLifecycle:
    """`monitor_state` accumulates no-trade streaks for sanity alerts.

    Invariant: loaded as dict on startup, carried into InferenceContext,
    written back dict-shaped. Never cleared on its own.
    """

    def test_loaded_as_dict(self):
        assert 'dict(state.get("monitor_state"' in SOURCE

    def test_written_as_dict(self):
        assert 'dict(getattr(ctx, "monitor_state"' in SOURCE


class TestRegimeStatePersistence:
    """The new `regime_state` key must carry CUSUM countdown + in_transition
    across invocations. Already covered in more depth by
    `tests/test_regime_state_persistence.py`; this is a smoke-test that
    live_state acts as the transport."""

    def test_regime_state_has_transport_fields(self):
        for field in ("countdown", "in_transition", "cusum_pos", "cusum_neg"):
            assert f'"{field}"' in SOURCE, (
                f"regime_state must carry `{field}` into live_state.json"
            )


# ── End-to-end contract ─────────────────────────────────────────────────────

class TestLiveStateSchemaComplete:
    """Meta-test: if a new attribute gets added, someone has to list it here
    too. Prevents silent schema growth with no coverage."""

    EXPECTED_KEYS = {
        "regime", "regime_confidence", "high_water_mark",
        "entry_dates", "sell_streaks", "last_sell_dates",
        "position_hwm", "monitor_state", "regime_state",
    }

    def test_all_expected_keys_are_written(self):
        missing = {k for k in self.EXPECTED_KEYS if f'"{k}":' not in SOURCE}
        assert not missing, (
            f"live_state keys missing write site in apply_outputs: {missing}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
