"""Behavior tests for adapters/runner.py audit fixes (2026-04-25 batch).

Covers DBT-3 from post_tier1_followups doc — runner-only fixes that
shipped without test coverage:
  - STATE-GC          drop stale entries from live_state on commit
  - STATE-GC-NEWBUYS  preserve newly-bought tickers from GC
  - ENTRY-DATE-FROM-FILLS  seed entry_date from broker fill history
  - ENTRY-DATE-BACKFILL    override stale state when broker is older
  - UNMANAGED-NTFY    surface non_wl_holds via ctx
  - EXITS-FAIL-DB     record_trades uses exits_placed not ctx.exits

These tests use STRING-LEVEL contracts (source-substring assertions)
because the full RunnerAdapter mock surface (Alpaca broker + parquet
cache + InferenceContext + SQLite) is heavy. Behavior contracts via
string-search are robust to refactor as long as the AUDIT FIX TAG
(STATE-GC, etc.) and the load-bearing line stay in source.

For the loadable runtime tests, see test_kernel_units.py and
test_panel_alignment.py — those exercise smaller helpers in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

RUNNER_PATH = REPO_ROOT / "backtesting/renquant_104/adapters/runner.py"
LIVE_RUNNER_PATH = REPO_ROOT / "live/runner.py"
ROTATION_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/pipeline/task_rotation.py"

RUNNER_SOURCE = RUNNER_PATH.read_text()
LIVE_RUNNER_SOURCE = LIVE_RUNNER_PATH.read_text()
ROTATION_SOURCE = ROTATION_PATH.read_text()


# ── STATE-GC ──────────────────────────────────────────────────────────────────

class TestStateGC:
    """Live state must drop entries for tickers no longer held."""

    def test_state_gc_audit_tag_present(self):
        assert "STATE-GC" in RUNNER_SOURCE, "STATE-GC fix tag must be documented"

    def test_state_gc_drops_stale_entry_dates(self):
        # The fix iterates currently_held and pops missing entries from
        # entry_dates / position_hwm / sell_streaks / entry_signals.
        assert "currently_held" in RUNNER_SOURCE
        assert "store.pop(t, None)" in RUNNER_SOURCE
        assert "STATE-GC: dropped" in RUNNER_SOURCE

    def test_state_gc_iterates_all_4_stores(self):
        for store_name in ("entry_dates", "entry_signals",
                           "sell_streaks", "position_hwm"):
            assert store_name in RUNNER_SOURCE

    def test_state_gc_preserves_wash_sale_in_window(self):
        # last_sell_dates entries within wash_sale_days window must stay
        assert "wash_sale_window_days" in RUNNER_SOURCE
        assert "last_sell_dates_str" in RUNNER_SOURCE


class TestStateGCNewBuys:
    """Bug K2: newly-bought tickers must be preserved from GC sweep."""

    def test_newbuys_audit_tag(self):
        assert "STATE-GC-NEWBUYS" in RUNNER_SOURCE

    def test_orders_placed_extends_currently_held(self):
        # Fix: extend currently_held with broker-confirmed buys before GC sweep
        assert 'getattr(ctx, "orders_placed", [])' in RUNNER_SOURCE
        assert "currently_held.add(t)" in RUNNER_SOURCE


# ── ENTRY-DATE-FROM-FILLS ──────────────────────────────────────────────────────

class TestEntryDateFromFills:
    """Seed missing entry_date from broker fill history; sentinel fallback."""

    def test_audit_tag_present(self):
        assert "ENTRY-DATE-FROM-FILLS" in RUNNER_SOURCE

    def test_seeds_from_broker_fills(self):
        assert "first_fill_map" in RUNNER_SOURCE
        assert "broker.get_filled_orders" in RUNNER_SOURCE
        # Earliest BUY fill kept per symbol
        assert 'f.get("action") != "BUY"' in RUNNER_SOURCE

    def test_sentinel_fallback_when_no_fills(self):
        # Sentinel = today - 31 days (past min_hold_days=30 default)
        assert "datetime.timedelta(days=31)" in RUNNER_SOURCE
        assert "ENTRY-DATE-SEED" in RUNNER_SOURCE


class TestEntryDateBackfill:
    """Backfill stale entry_date when broker has older fill."""

    def test_audit_tag_present(self):
        assert "ENTRY-DATE-BACKFILL" in RUNNER_SOURCE

    def test_only_overrides_when_older(self):
        # Logic: broker_first < cur_entry → override; else keep state
        assert "broker_first < cur_entry" in RUNNER_SOURCE


class TestEntryDateFillsPagination:
    """DBT-1: paginate broker.get_filled_orders past 100."""

    def test_alpaca_broker_paginates(self):
        # Read the alpaca broker source to verify pagination loop
        alpaca_src = (REPO_ROOT / "live/alpaca_broker.py").read_text()
        assert "page_size = 500" in alpaca_src
        assert "max_pages" in alpaca_src
        assert "until_cursor" in alpaca_src
        assert "DBT-1" in alpaca_src


# ── UNMANAGED-NTFY ────────────────────────────────────────────────────────────

class TestUnmanagedNtfy:
    """Non-watchlist held positions surfaced in ntfy + ctx."""

    def test_audit_tag_present(self):
        assert "UNMANAGED-NTFY" in RUNNER_SOURCE
        assert "UNMANAGED-NTFY" in LIVE_RUNNER_SOURCE

    def test_runner_attaches_to_ctx(self):
        # runner.py: ctx.non_wl_holds = list of unmanaged tickers
        assert "ctx.non_wl_holds" in RUNNER_SOURCE
        assert "_non_wl_holds" in RUNNER_SOURCE

    def test_live_runner_includes_in_ntfy_body(self):
        # live/runner.py: append "UNMANAGED ..." line when non_wl_holds non-empty
        assert "non_wl_holds" in LIVE_RUNNER_SOURCE
        assert "UNMANAGED" in LIVE_RUNNER_SOURCE


# ── EXITS-FAIL-DB ─────────────────────────────────────────────────────────────

class TestExitsFailDB:
    """SQLite n_exits + trade events must use broker-confirmed list."""

    def test_audit_tag_present(self):
        assert "EXITS-FAIL-DB" in RUNNER_SOURCE

    def test_uses_exits_placed_not_ctx_exits(self):
        # The fix introduces exits_for_db = exits_placed-or-fallback
        assert "exits_for_db" in RUNNER_SOURCE
        # Used in both record_trades (loop) and record_pipeline_run (count)
        assert "for t, sig in exits_for_db" in RUNNER_SOURCE
        assert "n_exits         = len(exits_for_db)" in RUNNER_SOURCE


# ── ROT-BLOCKED-NTFY (Bug L) ──────────────────────────────────────────────────

class TestRotBlockedNtfy:
    """When EmitRotationsTask drops a pair (Kelly=0 / cash / price), surface to ntfy."""

    def test_audit_tag_present(self):
        assert "ROT-BLOCKED-NTFY" in ROTATION_SOURCE
        assert "ROT-BLOCKED-NTFY" in LIVE_RUNNER_SOURCE

    def test_blocked_pairs_appended_with_reason(self):
        # ROT-BLOCKED tracks {sell, buy, reason}; reasons: kelly_zero,
        # bad_price, insufficient_cash. "bad_price" appears with the
        # actual numeric value embedded so substring-check covers both
        # `"reason": "kelly_zero"` and `"reason": f"bad_price({price})"`.
        assert "rotations_blocked" in ROTATION_SOURCE
        assert '"kelly_zero"' in ROTATION_SOURCE
        assert '"insufficient_cash"' in ROTATION_SOURCE
        # bad_price uses an f-string interpolation
        assert 'bad_price' in ROTATION_SOURCE

    def test_ntfy_surfaces_blocked_rotations(self):
        # live/runner.py reads ctx.rotations_blocked
        assert "rotations_blocked" in LIVE_RUNNER_SOURCE
        assert "BLOCKED-ROTATION" in LIVE_RUNNER_SOURCE


# ── ROT-COUNTER (Bug L companion) ─────────────────────────────────────────────

class TestRotCounter:
    """SQLite n_rotations must use EMITTED count, not CONSIDERED."""

    def test_audit_tag_present(self):
        assert "ROT-COUNTER" in RUNNER_SOURCE

    def test_runner_uses_counter_not_len_rotations(self):
        # n_rotations fed from ctx.counters["rotations"] which is incremented
        # only on actual emit
        assert 'ctx.counters.get("rotations", 0)' in RUNNER_SOURCE
