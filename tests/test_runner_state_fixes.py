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
SIM_ADAPTER_PATH = REPO_ROOT / "backtesting/renquant_104/adapters/sim.py"
LIVE_RUNNER_PATH = REPO_ROOT / "live/runner.py"
ROTATION_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/pipeline/task_rotation.py"
DECISION_TRACE_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/decision_trace.py"

RUNNER_SOURCE = RUNNER_PATH.read_text()
SIM_ADAPTER_SOURCE = SIM_ADAPTER_PATH.read_text()
LIVE_RUNNER_SOURCE = LIVE_RUNNER_PATH.read_text()
ROTATION_SOURCE = ROTATION_PATH.read_text()
DECISION_TRACE_SOURCE = DECISION_TRACE_PATH.read_text()

from adapters.runner import (  # noqa: E402
    cap_buy_order_to_cash,
    effective_live_holdings_after_orders,
    same_bar_sell_credit,
)
from adapters.sim import _model_type_from_artifact as sim_model_type_from_artifact  # noqa: E402


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


class TestSimModelTypeTrace:
    def test_sim_model_type_helper_reads_dict_metadata(self):
        artifact = {"_metadata": {"model_type": "xgb"}}

        assert sim_model_type_from_artifact(artifact) == "xgb"


class TestStateGCNewBuys:
    """Bug K2: newly-bought tickers must be preserved from GC sweep."""

    def test_newbuys_audit_tag(self):
        assert "STATE-GC-NEWBUYS" in RUNNER_SOURCE

    def test_orders_placed_extends_currently_held(self):
        # Fix: extend currently_held with broker-confirmed buys before GC sweep
        assert 'getattr(ctx, "orders_placed", [])' in RUNNER_SOURCE
        assert "effective_live_holdings_after_orders" in RUNNER_SOURCE

    def test_full_exit_is_removed_before_state_gc(self):
        current = effective_live_holdings_after_orders(
            ["AAPL", "MSFT"],
            {"AAPL"},
            [{"ticker": "NVDA"}],
        )

        assert current == {"MSFT", "NVDA"}

    def test_runner_skips_holding_state_persistence_for_full_exits(self):
        assert "full_exit_tickers: set[str] = set()" in RUNNER_SOURCE
        assert "full_exit_tickers.add(ticker)" in RUNNER_SOURCE
        assert "if ticker in full_exit_tickers:" in RUNNER_SOURCE


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
        # The fix introduces explicit placed-vs-intent selection. An empty
        # broker-confirmed list is meaningful and must not fall back to intent.
        assert "exits_for_db" in RUNNER_SOURCE
        # Used in both record_trades (loop) and record_pipeline_run (count)
        assert "for t, sig in exits_for_db" in RUNNER_SOURCE
        assert "n_exits         = len(exits_for_db)" in RUNNER_SOURCE
        assert "orders_for_db" in RUNNER_SOURCE
        assert "for o in orders_for_db" in RUNNER_SOURCE
        assert "n_buys          = len(orders_for_db)" in RUNNER_SOURCE


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


class TestRunnerCashBudgetGuard:
    """Live runner must not rely on broker rejects for over-budget buy baskets."""

    def test_cash_guard_allows_fully_funded_order(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "AAPL", "shares": 3, "price": 100.0},
            500.0,
        )
        assert reason is None
        assert order["shares"] == 3
        assert order["invest"] == 300.0

    def test_cash_guard_resizes_to_remaining_cash(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "AAPL", "shares": 10, "price": 100.0},
            350.0,
        )
        assert reason == "cash_budget_resized"
        assert order["shares"] == 3
        assert order["invest"] == 300.0
        assert order["budget_adjustment"] == "cash_budget_resized"
        assert order["original_shares"] == 10

    def test_cash_guard_rejects_when_no_share_affordable(self):
        order, reason = cap_buy_order_to_cash(
            {"ticker": "AAPL", "shares": 10, "price": 100.0},
            99.0,
        )
        assert order is None
        assert reason == "cash_budget_exhausted"

    def test_runner_commit_contains_live_cash_ledger(self):
        assert "buy_cash_remaining" in RUNNER_SOURCE
        assert "cash_budget_exhausted" in RUNNER_SOURCE
        assert "cash_budget_resized" in RUNNER_SOURCE

    def test_same_bar_sell_credit_sums_confirmed_exit_proceeds(self):
        class Sig:
            shares_sold = 3
            sell_price = 101.25

        class Ctx:
            exits_placed = [("SPY", Sig())]

        assert same_bar_sell_credit(Ctx()) == 303.75

    def test_runner_buy_budget_credits_confirmed_sells(self):
        assert "LIVE-SAME-BAR-SELL-CREDIT" in RUNNER_SOURCE
        assert "buy_cash_remaining += sell_credit" in RUNNER_SOURCE
        assert "same_bar_sell_credit(ctx)" in RUNNER_SOURCE


# ── ticker_daily_state writer (round-5) ───────────────────────────────────────

class TestTickerDailyStateWiring:
    """Per user spec round-5 (2026-04-26): every watchlist ticker must
    get a ticker_daily_state row per bar — including those filtered at
    universe / broker / no-model gates. Validates wiring is in place at
    source level (the writer's own behavior is covered by
    test_ticker_daily_state.py)."""

    def test_writer_imported(self):
        assert "record_ticker_daily_state" in RUNNER_SOURCE

    def test_iterates_full_watchlist_not_just_cands(self):
        # The wiring loop must iterate the full decision-trace universe,
        # not just ctx.candidates — that is the entire point of round-5.
        # decision_trace_tickers(config) includes the watchlist and any
        # benchmark sleeve ticker that is part of the decision surface.
        assert "build_ticker_daily_state_rows" in RUNNER_SOURCE
        assert "build_ticker_daily_state_rows" in SIM_ADAPTER_SOURCE
        assert "decision_trace_tickers" in DECISION_TRACE_SOURCE
        assert "for tk in decision_trace_tickers(config):" in DECISION_TRACE_SOURCE

    def test_blocked_by_preserves_exact_universe_rejection_reason(self):
        # When ticker has no model loaded, keep LoadUniverseJob's exact
        # rejection reason instead of collapsing every failure to
        # "universe_floor".
        assert "_universe_rejections" in LIVE_RUNNER_SOURCE
        for source in (RUNNER_SOURCE, SIM_ADAPTER_SOURCE):
            assert "_universe_rejections" in source
        assert "universe:{reason}" in DECISION_TRACE_SOURCE

    def test_pending_at_broker_is_recorded(self):
        # broker_pending must surface as both pending_at_broker=1 AND
        # blocked_by="broker_pending" when nothing else has blocked.
        assert "pending_broker_tickers" in RUNNER_SOURCE
        assert "pending_at_broker" in DECISION_TRACE_SOURCE
        assert '"broker_pending"' in DECISION_TRACE_SOURCE

    def test_non_selected_tickers_have_explicit_reason(self):
        # Null blocked_by on non-selected rows makes the daily decision tree
        # ambiguous. Live and sim adapters must stamp a terminal reason for
        # no-position/no-candidate and held/no-new-buy cases.
        for source in (RUNNER_SOURCE, SIM_ADAPTER_SOURCE):
            assert "build_ticker_daily_state_rows" in source
        assert '"held_no_new_buy"' in DECISION_TRACE_SOURCE
        assert '"no_model_signal"' in DECISION_TRACE_SOURCE
        assert '"not_selected"' in DECISION_TRACE_SOURCE

    def test_in_universe_uses_models_keys(self):
        # in_universe = 1 iff ticker passed universe floor (i.e. has a
        # loaded per-ticker model). Source must reference self._models.
        assert "model_keys=set(self._models or {})" in RUNNER_SOURCE
        assert "model_keys=set(self._models or {})" in SIM_ADAPTER_SOURCE
        assert "in_universe" in DECISION_TRACE_SOURCE
        assert "tk in model_keys" in DECISION_TRACE_SOURCE


# ── STATE-EXT-SELL: external/manual disposition stamps wash-sale clock ────

class TestExternalSellWashSaleClock:
    """Z2 (2026-04-28 NVTS post-mortem): a position that disappears from
    the broker between bars must stamp last_sell_dates regardless of who
    sold it. Pre-fix the runner only stamped its own SELLs; manual sells
    via Alpaca app or broker-side liquidations let the bot re-buy the
    same ticker the next bar within the 30-day wash-sale window.
    """

    def test_audit_tag_present(self):
        assert "STATE-EXT-SELL" in RUNNER_SOURCE

    def test_disappeared_iterates_entry_dates(self):
        # The fix scans entry_dates for tickers no longer in currently_held
        assert "disappeared = [t for t in self._entry_dates" in RUNNER_SOURCE
        assert "if t not in currently_held" in RUNNER_SOURCE

    def test_excludes_runner_initiated_sells_today(self):
        # Tickers we already stamped today (runner-side full sells) must
        # not be double-stamped — they got their date in the SELL loop.
        assert (
            "self._last_sell_dates_str.get(t) != today_str"
            in RUNNER_SOURCE
        )

    def test_stamps_today_str(self):
        # External-sell ticker gets `last_sell_dates_str[t] = today_str`
        assert "self._last_sell_dates_str[t] = today_str" in RUNNER_SOURCE

    def test_warns_loudly_so_operator_sees_it(self):
        # Use log.warning, not log.info — manual sells should be
        # surfaced clearly so the operator knows the clock started.
        assert (
            "STATE-EXT-SELL" in RUNNER_SOURCE
            and "log.warning" in RUNNER_SOURCE
        )

    def test_ordering_external_sell_runs_before_gc_pop(self):
        # External-sell stamping must happen BEFORE the GC pops the
        # entry from entry_dates — otherwise we lose the "ticker was
        # held last bar" signal.
        ext_idx = RUNNER_SOURCE.index("STATE-EXT-SELL")
        gc_pop_idx = RUNNER_SOURCE.index("STATE-GC: dropped")
        assert ext_idx < gc_pop_idx, (
            "External-sell detection must run before STATE-GC pops "
            "entry_dates, or the disappeared list will already be empty."
        )


# ── 2026-05-17 STATE-EXT-SELL pending-order false-positive fix ───────────────

class TestStateExtSellPendingOrderFix:
    """Pre-fix: a Sunday/after-hours BUY queued at Alpaca (status=accepted,
    filled_qty=0) was misclassified as STATE-EXT-SELL on the next runner
    invocation because broker positions don't yet show it. 2026-05-17 HON
    (and 5/15 META) both got 30-day wash-sale blocks for this reason.

    Fix: exclude tickers with pending broker orders from the `disappeared`
    list AND extend GC's `currently_held` with pending_broker_tickers to
    preserve in-flight buy state (entry_date / entry_signal / position_hwm)
    until the order eventually fills.

    Invariant: pending-at-broker ≠ externally-sold. The STATE-EXT-SELL
    stamp is reserved for "ticker was a real position, then disappeared".
    """

    def test_excludes_pending_from_disappeared(self):
        assert "pending_broker_tickers" in RUNNER_SOURCE
        assert "t not in pending_broker" in RUNNER_SOURCE, \
            "disappeared filter must exclude pending broker orders"

    def test_skipped_pending_logged(self):
        assert "skipping wash-sale stamp" in RUNNER_SOURCE, \
            "Skipped tickers should be logged for operator visibility"

    def test_gc_preserves_pending_state(self):
        assert "held_or_pending = currently_held | pending_broker" in RUNNER_SOURCE, \
            "GC sweep must also consider pending broker orders"

    def test_entry_signals_persist_entry_regime(self):
        assert '"regime":           order.get("regime")' in RUNNER_SOURCE
        assert "entry_regime           = es.get(\"regime\")" in RUNNER_SOURCE

    def test_fix_tag_present(self):
        assert "2026-05-17 Bug fix" in RUNNER_SOURCE
        assert "pending-at-broker ≠ externally-sold" in RUNNER_SOURCE


# ── 2026-05-23 preopen-cancel stale state fix ───────────────────────────────

class TestPreopenCancelDoesNotStampWashSale:
    """A queued buy cancelled before open never became a position.

    If broker no longer reports the order as pending, the runner must clear
    local optimistic entry state without converting the vanished order into an
    external sell / wash-sale block.
    """

    def test_reads_preopen_cancel_ledger(self):
        assert "_preopen_cancel_symbols" in RUNNER_SOURCE
        assert "preopen_cancel_ledger.jsonl" in RUNNER_SOURCE

    def test_canceled_state_excluded_from_external_sell(self):
        assert "t not in preopen_canceled" in RUNNER_SOURCE
        assert "STALE_STATE" in RUNNER_SOURCE
        assert "clearing local entry state without" in RUNNER_SOURCE
        assert "wash-sale stamp" in RUNNER_SOURCE


class TestSelectedMeansActuallyPlaced:
    """Decision telemetry selected=1 means accepted/applied order, not intent."""

    def test_live_selected_uses_broker_confirmed_orders(self):
        assert "selected_tickers = selected_buy_tickers(orders_for_db)" in RUNNER_SOURCE
        assert "def selected_buy_tickers" in DECISION_TRACE_SOURCE
        assert "broker_skip:" in RUNNER_SOURCE
