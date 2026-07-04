"""S-FRAC stage 0 — active-path audit tests for the fractional-capable
RunnerAdapter.commit contract.

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
§2.2 (stage-0 deliverable) + §2.3 (the enumerated audit tests). The v1
fatal lesson: fractional capability was built and proven on a NON-ACTIVE
path (ExecutionPipeline / FakeBackend) while the ACTIVE live path — the
umbrella ``RunnerAdapter.commit`` — still int-truncated fractional fills
(``shares = int(execution["filled_qty"] or shares)``, the exact line Codex
cited to block renquant-pipeline#153). These tests carry the burden on the
REAL commit path.

The §2.3 enumeration covered here:
  1. E2E through the real commit path (fractional BUY round-trip + the
     reverse fractional SELL with zero residual dust)      → TestE2ECommitPath
  2. Truncation audit (static, AST-based)                  → TestTruncationAudit
  3. Active-path liveness proof (commit_path_fingerprint
     stamped by commit + recorded in the run bundle +
     the live entry-point chain walk)                      → TestActivePathLiveness
  4. Flag-off regression (whole-share byte-identical)      → TestFlagOffWholeShareRegression
  5. Partial-fill and cancel-replace state coverage        → TestPartialFillAndCancelReplace
  6. Stop-reconciliation-on-restart                        → TestStopReconciliationOnRestart
  7. Fail-closed entry ⇒ stage-0 outage-window loss
     budget $0 by construction                             → TestFailClosedEntry

Plus unit pins for the contract module itself (normalize_fill_qty /
fmt_qty / routing / capability gate) and the float-fill round-trip through
``broker_order_execution`` on a fake execution result.

Stage 0 is default-inert: no config anywhere enables
``execution.fractional_shares``; whole-share behavior is byte-identical
(test 4 pins it against the killed legacy ``int()`` semantics).
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TODAY = datetime.date(2026, 7, 3)

# The broker fill quantity the design names for the E2E audit.
FRACTIONAL_QTY = 0.435578


# ── Fakes ───────────────────────────────────────────────────────────────────

class FakeBroker:
    """Qty-aware fake broker driving RunnerAdapter.commit end-to-end.

    Mirrors the live AlpacaBroker stop-capability semantics: broker-side
    (GTC) stops cover WHOLE-SHARE quantities only — the qty-aware
    ``supports_broker_side_stops(symbol, qty)`` probe answers False for a
    fractional qty (Alpaca fractional orders are TIF=DAY only, design §4).
    Every qty-aware probe is recorded so tests can assert the capability
    is re-evaluated against the CURRENT quantity (never cached).
    """

    broker_name = "paper"

    def __init__(self, fills=None, positions=None, fractional_contract=False):
        self.fills = dict(fills or {})          # ticker -> broker order result
        self.positions = dict(positions or {})  # ticker -> held qty (broker truth)
        self.place_order_calls: list[tuple] = []
        self.stop_probe_calls: list[tuple] = []  # (symbol, qty) qty-aware probes
        self.place_stop_calls: list[tuple] = []
        self.cancel_calls: list[str] = []
        if fractional_contract:
            # The renquant-execution#19 broker contract surface the
            # capability gate (§2.2.3) probes for: fractionable lookup +
            # no-submit classification. Stage 1 lands the real impl.
            self.is_fractionable = lambda symbol: True
            self.classify_broker_result = lambda result: "submitted"

    def get_open_orders(self):
        return set()

    def get_position(self, ticker):
        return float(self.positions.get(ticker, 0.0))

    def place_order(self, ticker, side, qty):
        self.place_order_calls.append((ticker, side, qty))
        result = dict(self.fills[ticker])
        filled = float(result.get("filled_qty") or 0.0)
        if filled > 0:
            if side == "BUY":
                self.positions[ticker] = self.positions.get(ticker, 0.0) + filled
            else:
                self.positions[ticker] = self.positions.get(ticker, 0.0) - filled
                if abs(self.positions[ticker]) <= 1e-12:
                    self.positions.pop(ticker, None)
        return result

    def supports_broker_side_stops(self, symbol=None, qty=None):
        if qty is None:
            return True  # legacy broker-level Z9 enable check
        self.stop_probe_calls.append((symbol, float(qty)))
        q = float(qty)
        if not math.isfinite(q) or q <= 0:
            return False
        return abs(q - round(q)) <= 1e-9

    def place_stop_order(self, symbol, quantity, stop_price):
        self.place_stop_calls.append((symbol, quantity, stop_price))
        return {"order_id": f"stop-{symbol}-{len(self.place_stop_calls)}"}

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)


class ArmedSoftwareStops:
    """Minimal stage-3 registry satisfying commit_contract.software_stops_armed."""

    def is_armed(self):
        return True


class UnarmedSoftwareStops:
    def is_armed(self):
        return False


# ── Harness ─────────────────────────────────────────────────────────────────

def _config(*, fractional=False, z9=True):
    cfg = {
        "model_name": "renquant_104",
        "watchlist": ["BLK", "OXY"],
        "regime_params": {"BULL_CALM": {"max_single_day_loss_pct": 0.06}},
        "live": {"broker_side_stops": {"enabled": z9, "pct": 0.2}},
        "tax": {"short_term_rate": 0.37, "long_term_rate": 0.20,
                "long_term_threshold_days": 365},
        "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
        "persistence": {"enabled": False},
    }
    if fractional:
        cfg["execution"] = {"fractional_shares": {"enabled": True}}
    return cfg


def _make_adapter(tmp_path, *, config, broker, software_stops=None,
                  positions=None, entry_dates=None, stop_orders=None,
                  position_hwm=None):
    """RunnerAdapter shell with exactly the state commit() touches.

    Bypasses __init__ (broker connection, DB, artifact loads) — the same
    pattern as tests/test_runner_z9_integration.py — but drives the REAL
    commit() body end-to-end.
    """
    from adapters.runner import RunnerAdapter  # noqa: PLC0415

    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    ra = RunnerAdapter.__new__(RunnerAdapter)
    ra._config = config
    ra._models = {}
    ra._broker = broker
    ra._strategy_dir = strategy_dir
    ra._sell_only = False
    ra._broker_name = "paper"
    ra._db = None
    ra._universe_rejections = {}
    ra._software_stops = software_stops
    ra._positions_cache = dict(positions or {})
    ra._entry_dates = dict(entry_dates or {})
    ra._entry_signals = {}
    ra._sell_streaks = {}
    ra._protection_breaches = {}
    ra._position_hwm = dict(position_hwm or {})
    ra._last_sell_dates_str = {}
    ra._last_stop_exit_dates_str = {}
    ra._stop_orders = dict(stop_orders or {})
    ra._recent_sell_orders = {}
    ra._state = {}
    ra._last_ctx_stop_pct = 0.06
    return ra


def _make_ctx(config, *, today=TODAY, orders=(), exits=(), holdings=None,
              prices=None, cash=10_000.0):
    return SimpleNamespace(
        today=today,
        config=config,
        orders=list(orders),
        exits=list(exits),
        holdings=dict(holdings or {}),
        prices=dict(prices or {}),
        cash=cash,
        regime="BULL_CALM",
        confidence=0.9,
        hwm=100_000.0,
        skip_buys=False,
        monitor_state={},
        regime_state=None,
        counters={},
        candidates=[],
        buy_blocked=False,
        bear_only=False,
        pending_broker_tickers=set(),
        rotations=[],
    )


def _journal_records(tmp_path, action=None):
    """Read the trade journal commit() writes via _log_trade."""
    log_dir = tmp_path / "live" / "logs" / "renquant_104"
    records = []
    for f in sorted(log_dir.glob("*.json")) if log_dir.exists() else []:
        records.extend(json.loads(f.read_text()))
    if action is not None:
        records = [r for r in records if r.get("action") == action]
    return records


def _saved_state(tmp_path):
    state_file = (tmp_path / "backtesting" / "renquant_104"
                  / "live_state.paper.json")
    assert state_file.exists(), "commit() must persist live_state"
    return json.loads(state_file.read_text())


def _load_truncation_audit():
    spec = importlib.util.spec_from_file_location(
        "check_commit_path_no_int_truncation",
        REPO_ROOT / "scripts" / "check_commit_path_no_int_truncation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 1 — E2E through the real commit path
# ═════════════════════════════════════════════════════════════════════════════

class TestE2ECommitPath:
    """Fake broker returns filled_qty=0.435578 → the float survives
    end-to-end: orders_placed, live_state, journal, cash accounting, and
    stop routing selects software (stage-3 armed) — never a truncated
    broker stop. Then the reverse: fractional SELL → position removed →
    zero residual dust."""

    def _buy_setup(self, tmp_path, software_stops=ArmedSoftwareStops()):
        config = _config(fractional=True)
        broker = FakeBroker(
            fills={
                "BLK": {"status": "filled", "order_id": "ord-BLK",
                        "filled_qty": FRACTIONAL_QTY,
                        "filled_avg_price": 100.0},
            },
            fractional_contract=True,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=software_stops)
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": FRACTIONAL_QTY, "price": 100.0}],
            prices={"BLK": 100.0},
            cash=1_000.0,
        )
        return ra, ctx, broker

    def test_fractional_buy_round_trip(self, tmp_path, caplog):
        ra, ctx, broker = self._buy_setup(tmp_path)
        with caplog.at_level("INFO", logger="live.runner"):
            ra.commit(ctx)

        # orders_placed carries the broker float VERBATIM.
        assert len(ctx.orders_placed) == 1
        placed = ctx.orders_placed[0]
        assert placed["shares"] == FRACTIONAL_QTY
        assert placed["filled_qty"] == FRACTIONAL_QTY
        assert not ctx.orders_skipped

        # live_state position state is fractional-entry aware.
        state = _saved_state(tmp_path)
        assert state["entry_dates"]["BLK"] == TODAY.isoformat()
        assert state["position_hwm"]["BLK"] == 100.0

        # journal filled qty fractional + cash decremented by the EXACT
        # fractional notional.
        buys = _journal_records(tmp_path, action="BUY")
        assert len(buys) == 1
        assert buys[0]["shares"] == FRACTIONAL_QTY
        assert buys[0]["invest"] == FRACTIONAL_QTY * 100.0

        # Stop routing: broker-side stop CANNOT cover 0.435578 (probe
        # answered with the actual qty) → routed to the armed software
        # layer; no truncated broker stop is ever placed.
        assert broker.place_stop_calls == []
        assert ("BLK", FRACTIONAL_QTY) in broker.stop_probe_calls
        assert "routed to the software-stop layer" in caplog.text
        assert state["stop_orders"] == {}

    def test_cash_accounting_is_float_exact(self, tmp_path):
        """Discriminating budget: cash = exact fractional notional + $49.
        The follow-on 1-share @ $50 buy must be rejected for cash — under
        the killed int() truncation the fractional invest would have been
        $0 and the second buy would (wrongly) have proceeded."""
        config = _config(fractional=True, z9=False)
        broker = FakeBroker(
            fills={
                "BLK": {"status": "filled", "order_id": "o1",
                        "filled_qty": FRACTIONAL_QTY,
                        "filled_avg_price": 100.0},
                "OXY": {"status": "filled", "order_id": "o2",
                        "filled_qty": 1.0, "filled_avg_price": 50.0},
            },
            fractional_contract=True,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=ArmedSoftwareStops())
        ctx = _make_ctx(
            config,
            orders=[
                {"ticker": "BLK", "shares": FRACTIONAL_QTY, "price": 100.0},
                {"ticker": "OXY", "shares": 1, "price": 50.0},
            ],
            prices={"BLK": 100.0, "OXY": 50.0},
            cash=FRACTIONAL_QTY * 100.0 + 49.0,
        )
        ra.commit(ctx)
        assert [o["ticker"] for o in ctx.orders_placed] == ["BLK"]
        assert [(o["ticker"], o["skip_reason"]) for o in ctx.orders_skipped] == [
            ("OXY", "cash_budget_exhausted"),
        ]

    def test_fractional_sell_zero_residual_dust(self, tmp_path):
        """Reverse leg (the #19 round-2 lifecycle demand, now on the real
        path): fractional full SELL → position removed → zero residual
        dust in live_state; wash-sale stamped; exit qty float-exact."""
        from kernel.exits import ExitSignal, HoldingState  # noqa: PLC0415

        config = _config(fractional=False)  # exits are never gated
        broker = FakeBroker(
            fills={
                "BLK": {"status": "filled", "order_id": "sell-1",
                        "filled_qty": FRACTIONAL_QTY,
                        "filled_avg_price": 101.0},
            },
            positions={"BLK": FRACTIONAL_QTY},
        )
        ra = _make_adapter(
            tmp_path, config=config, broker=broker,
            positions={"BLK": {"qty": FRACTIONAL_QTY,
                               "qty_available": FRACTIONAL_QTY,
                               "avg_entry_price": 100.0}},
            entry_dates={"BLK": "2026-06-20"},
            position_hwm={"BLK": 105.0},
            stop_orders={"BLK": {"order_id": "stop-old", "stop_price": 80.0,
                                 "qty": FRACTIONAL_QTY,
                                 "stamped_at": "2026-06-20"}},
        )
        hs = HoldingState(entry_price=100.0,
                          entry_date=datetime.date(2026, 6, 20),
                          high_watermark=105.0)
        sig = ExitSignal(should_exit=True, reason="model sell",
                         exit_type="model_sell")
        ctx = _make_ctx(config, exits=[("BLK", sig)], holdings={"BLK": hs},
                        prices={"BLK": 101.0})
        ra.commit(ctx)

        # Broker-confirmed exit with the float verbatim.
        assert [t for t, _ in ctx.exits_placed] == ["BLK"]
        assert sig.shares_sold == FRACTIONAL_QTY
        assert sig.sell_price == 101.0

        # Zero residual dust: every per-position store is fully reaped.
        state = _saved_state(tmp_path)
        assert "BLK" not in state["entry_dates"]
        assert "BLK" not in state["position_hwm"]
        assert "BLK" not in state["sell_streaks"]
        assert state["stop_orders"] == {}          # Z9 stop cancelled
        assert broker.cancel_calls == ["stop-old"]
        # Wash-sale clock stamped on the full fractional liquidation.
        assert state["last_sell_dates"]["BLK"] == TODAY.isoformat()
        # Journal SELL row carries the float.
        sells = _journal_records(tmp_path, action="SELL")
        assert len(sells) == 1
        assert sells[0]["qty"] == FRACTIONAL_QTY
        assert sells[0]["partial"] is False


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 2 — static truncation audit
# ═════════════════════════════════════════════════════════════════════════════

class TestTruncationAudit:
    def test_commit_path_modules_are_clean(self):
        """No int() cast on fill quantities anywhere on the commit path
        (runner / runner_execmath / runner_ext_sell / broker_sync /
        z9_stops / commit_contract) outside the sanctioned whole-share
        branch."""
        mod = _load_truncation_audit()
        assert mod.run_audit() == []

    def test_auditor_catches_the_killed_legacy_cast(self, tmp_path):
        """The exact v1 blocker expression must fail the audit if it is
        ever reintroduced (under this or any spelling that names a fill
        quantity)."""
        mod = _load_truncation_audit()
        bad = tmp_path / "reintroduced.py"
        bad.write_text(
            "def f(execution, shares):\n"
            '    return int(execution["filled_qty"] or shares)\n'
        )
        violations = mod.run_audit((bad,))
        assert len(violations) == 1
        assert "filled_qty" in violations[0]

    def test_auditor_catches_alternate_spellings(self, tmp_path):
        mod = _load_truncation_audit()
        bad = tmp_path / "alt.py"
        bad.write_text(
            "def g(fill, sell_qty):\n"
            '    a = int(fill["qty"])\n'
            "    b = int(sell_qty)\n"
            "    return a + b\n"
        )
        assert len(mod.run_audit((bad,))) == 2

    def test_auditor_flags_missing_commit_path_module(self, tmp_path):
        """A renamed/deleted commit-path module must fail the audit, not
        silently shrink its coverage."""
        mod = _load_truncation_audit()
        missing = tmp_path / "runner_gone.py"
        violations = mod.run_audit((missing,))
        assert len(violations) == 1
        assert "MISSING" in violations[0]

    def test_sanctioned_whole_share_branch_is_allowed(self, tmp_path):
        """normalize_fill_qty's eps-guarded int() snap is the ONE allowed
        whole-share branch."""
        mod = _load_truncation_audit()
        ok = tmp_path / "sanctioned.py"
        ok.write_text(
            "def normalize_fill_qty(filled_qty, fallback):\n"
            "    q = float(filled_qty or fallback)\n"
            "    return int(round(q)) if abs(q - round(q)) <= 1e-9 else q\n"
        )
        assert mod.run_audit((ok,)) == []


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 3 — active-path liveness proof
# ═════════════════════════════════════════════════════════════════════════════

class TestActivePathLiveness:
    """"The live runner exercises the contract-carrying commit path" must
    be a recorded fact per run, not an assumption — the direct
    anti-regression for merged-is-not-deployed / deployed-but-dark."""

    def test_live_entry_point_chain(self):
        """Walk the live entry points the design names:
        daily_104.sh / intraday_sell_104.sh → live runner →
        RunnerAdapter.commit (the implementation carrying the contract)."""
        daily = (REPO_ROOT / "scripts" / "daily_104.sh").read_text()
        intraday = (REPO_ROOT / "scripts" / "intraday_sell_104.sh").read_text()
        # Both entry scripts invoke the live runner (umbrella baseline or
        # the orchestrator bridge — both execute live/runner.py's cycle).
        assert "live.runner" in daily and "daily-bridge" in daily
        assert "live.runner" in intraday and "live-bridge" in intraday

        live_runner = (REPO_ROOT / "live" / "runner.py").read_text()
        assert "from adapters.runner import RunnerAdapter" in live_runner
        assert "adapter.commit(ctx)" in live_runner

        runner_src = (_STRATEGY / "adapters" / "runner.py").read_text()
        assert ("ctx.commit_path_fingerprint = commit_path_fingerprint()"
                in runner_src), (
            "RunnerAdapter.commit must stamp the commit-path fingerprint "
            "on every commit (S-FRAC stage 0 liveness proof)"
        )
        bundle_src = (_STRATEGY / "kernel" / "artifact_contract.py").read_text()
        assert 'bundle["commit_path_fingerprint"]' in bundle_src, (
            "build_run_bundle must record ctx.commit_path_fingerprint in "
            "the persisted run bundle"
        )

    def test_commit_stamps_fingerprint_with_executed_source_sha(self, tmp_path):
        config = _config()
        broker = FakeBroker(fills={
            "BLK": {"status": "filled", "order_id": "o", "filled_qty": 5.0,
                    "filled_avg_price": 100.0},
        })
        ra = _make_adapter(tmp_path, config=config, broker=broker)
        ctx = _make_ctx(config,
                        orders=[{"ticker": "BLK", "shares": 5, "price": 100.0}],
                        prices={"BLK": 100.0})
        ra.commit(ctx)

        fp = ctx.commit_path_fingerprint
        assert fp["contract"] == "fractional-v2-stage0"
        assert fp["commit_impl"] == "adapters.runner.RunnerAdapter.commit"
        runner_file = _STRATEGY / "adapters" / "runner.py"
        assert fp["runner_sha256"] == hashlib.sha256(
            runner_file.read_bytes()).hexdigest(), (
            "fingerprint must hash the EXECUTED runner source"
        )

    def test_run_bundle_records_the_fingerprint(self, tmp_path):
        """The persisted run bundle (what record_pipeline_run stores per
        run) carries the fingerprint — a daily-full bundle showing
        contract == fractional-v2-stage0 is the per-run liveness fact."""
        from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415

        config = _config()
        broker = FakeBroker(fills={
            "BLK": {"status": "filled", "order_id": "o", "filled_qty": 5.0,
                    "filled_avg_price": 100.0},
        })
        ra = _make_adapter(tmp_path, config=config, broker=broker)
        ctx = _make_ctx(config,
                        orders=[{"ticker": "BLK", "shares": 5, "price": 100.0}],
                        prices={"BLK": 100.0})
        ra.commit(ctx)

        bundle = build_run_bundle(
            config, ra._strategy_dir, run_id="stage0-test", run_type="live",
            ctx=ctx, broker_mode="paper",
        )
        assert bundle["commit_path_fingerprint"]["contract"] == (
            "fractional-v2-stage0"
        )
        assert bundle["commit_path_fingerprint"]["runner_sha256"]

    def test_bundle_omits_fingerprint_for_non_commit_ctx(self):
        """Sim/train ctxs that never passed through the live commit path
        must NOT carry a liveness claim."""
        from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415

        ctx = _make_ctx(_config())
        bundle = build_run_bundle(
            _config(), _STRATEGY, run_id="x", run_type="sim", ctx=ctx,
        )
        assert "commit_path_fingerprint" not in bundle


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 4 — flag-off whole-share regression (byte-identical)
# ═════════════════════════════════════════════════════════════════════════════

class TestFlagOffWholeShareRegression:
    """With execution.fractional_shares absent (all of production today),
    a whole-share fill must produce EXACTLY what the killed legacy
    ``int(execution["filled_qty"] or shares)`` produced — same values,
    same int types, same JSON bytes, no new order-dict fields."""

    def _run(self, tmp_path, caplog=None):
        config = _config()  # no execution.fractional_shares key at all
        broker = FakeBroker(fills={
            "BLK": {"status": "filled", "order_id": "ord-BLK",
                    "filled_qty": 5.0, "filled_avg_price": 100.0},
        })
        ra = _make_adapter(tmp_path, config=config, broker=broker)
        ctx = _make_ctx(config,
                        orders=[{"ticker": "BLK", "shares": 5, "price": 100.0}],
                        prices={"BLK": 100.0}, cash=1_000.0)
        ra.commit(ctx)
        return ra, ctx, broker

    def test_order_dict_bytes_identical_to_legacy_semantics(self, tmp_path):
        _, ctx, _ = self._run(tmp_path)
        placed = ctx.orders_placed[0]

        # Exactly what the legacy line produced for this fill:
        legacy_shares = int(5.0 or 5)  # the killed expression, whole-share
        expected = {
            "ticker": "BLK",
            "shares": legacy_shares,
            "price": 100.0,
            "invest": 500.0,                 # stamped by cap_buy_order_to_cash
            "order_id": "ord-BLK",
            "status": "filled",
            "filled_qty": legacy_shares,
            "filled_avg_price": 100.0,
        }
        assert placed == expected
        # No new fields leak into flag-off order dicts.
        assert set(placed) == set(expected)
        # Byte-identical serialization ("shares": 5, never 5.0).
        assert (json.dumps(placed, sort_keys=True)
                == json.dumps(expected, sort_keys=True))
        assert type(placed["shares"]) is int
        assert type(placed["filled_qty"]) is int

    def test_journal_and_state_carry_legacy_int(self, tmp_path):
        _, _, _ = self._run(tmp_path)
        buys = _journal_records(tmp_path, action="BUY")
        assert buys[0]["shares"] == 5
        assert json.dumps(buys[0]["shares"]) == "5"
        state = _saved_state(tmp_path)
        assert state["entry_dates"]["BLK"] == TODAY.isoformat()

    def test_whole_share_stop_placement_unchanged(self, tmp_path, caplog):
        """Z9 still places the broker-side GTC stop for whole shares —
        routing only diverges for fractional quantities — and the stop
        log renders '× 5 shares' byte-identical to the legacy int cast."""
        with caplog.at_level("INFO", logger="live.runner"):
            ra, _, broker = self._run(tmp_path)
        assert len(broker.place_stop_calls) == 1
        symbol, qty, stop_price = broker.place_stop_calls[0]
        assert (symbol, qty) == ("BLK", 5.0)
        assert stop_price == pytest.approx(80.0)  # 100 × (1 − 0.2)
        assert ra._stop_orders["BLK"]["qty"] == 5.0
        assert "× 5 shares" in caplog.text

    def test_no_skip_and_no_gate_noise_when_flag_absent(self, tmp_path):
        _, ctx, _ = self._run(tmp_path)
        assert ctx.orders_skipped == []


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 5 — partial-fill and cancel-replace state coverage
# ═════════════════════════════════════════════════════════════════════════════

class TestPartialFillAndCancelReplace:
    def test_partial_fractional_fill_held_at_exact_float(self, tmp_path):
        """Broker reports filled_qty=0.20000 against submitted 0.341052,
        order remains partially_filled: the commit path holds the position
        at the EXACT partial float, never rounds/truncates/zeroes it, and
        does not mark the order terminal."""
        config = _config(fractional=True, z9=False)
        broker = FakeBroker(
            fills={"BLK": {"status": "partially_filled", "order_id": "p1",
                           "filled_qty": 0.20000,
                           "filled_avg_price": 100.0}},
            fractional_contract=True,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=ArmedSoftwareStops())
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": 0.341052, "price": 100.0}],
            prices={"BLK": 100.0},
        )
        ra.commit(ctx)

        placed = ctx.orders_placed[0]
        assert placed["shares"] == 0.2
        assert placed["filled_qty"] == 0.2
        assert placed["status"] == "partially_filled"   # not terminal
        buys = _journal_records(tmp_path, action="BUY")
        assert buys[0]["shares"] == 0.2
        assert buys[0]["invest"] == 0.2 * 100.0

    def test_cancel_replace_fills_accumulate_never_overwrite(self, tmp_path):
        """Cancel-replace modeled as its live order-of-events: leg 1
        partially fills 0.15 and is observed; the original is canceled and
        a replacement submitted for the residual 0.191052, which fills on
        the next commit. The accumulated quantity float-sums across both
        legs (union of fills) — entry state is never overwritten by leg 2.

        NOTE (stage-1 boundary): a terminal-canceled response that is the
        FIRST sighting of a partial fill is classified rejected by
        broker_order_execution today; terminal-state handling (DAY-expiry,
        cancel-with-fill) is explicitly stage-1 scope (design §6 stage 1).
        Stage 0 pins the accumulate-not-overwrite contract."""
        config = _config(fractional=True, z9=False)
        broker = FakeBroker(
            fills={"BLK": {"status": "partially_filled", "order_id": "leg1",
                           "filled_qty": 0.15, "filled_avg_price": 100.0}},
            fractional_contract=True,
        )
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=ArmedSoftwareStops())
        ctx1 = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": 0.341052, "price": 100.0}],
            prices={"BLK": 100.0},
        )
        ra.commit(ctx1)
        assert ctx1.orders_placed[0]["shares"] == 0.15
        assert ra._entry_dates["BLK"] == TODAY.isoformat()

        # Cancel-replace: replacement for the residual fills next commit.
        from kernel.exits import HoldingState  # noqa: PLC0415
        d2 = TODAY + datetime.timedelta(days=1)
        broker.fills["BLK"] = {"status": "filled", "order_id": "leg2",
                               "filled_qty": 0.191052,
                               "filled_avg_price": 100.0}
        hs = HoldingState(entry_price=100.0, entry_date=TODAY,
                          high_watermark=100.0)
        ctx2 = _make_ctx(
            config, today=d2,
            orders=[{"ticker": "BLK", "shares": 0.191052, "price": 100.0}],
            holdings={"BLK": hs}, prices={"BLK": 100.0},
        )
        ra.commit(ctx2)

        # Leg 2 is a TOP-UP: entry_date from leg 1 is preserved, never
        # overwritten (the design's never-overwrites demand).
        assert ctx2.orders_placed[0]["shares"] == 0.191052
        assert ra._entry_dates["BLK"] == TODAY.isoformat()

        # Union of both fills: journal rows float-sum to the submitted
        # total; broker position is the float accumulation.
        buys = _journal_records(tmp_path, action="BUY")
        assert [b["shares"] for b in buys] == [0.15, 0.191052]
        assert sum(b["shares"] for b in buys) == pytest.approx(
            0.341052, abs=1e-12)
        assert broker.get_position("BLK") == pytest.approx(
            0.341052, abs=1e-12)
        # Cash reflects both legs, not either alone.
        assert [b["invest"] for b in buys] == [0.15 * 100.0, 0.191052 * 100.0]


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 6 — stop-reconciliation on restart
# ═════════════════════════════════════════════════════════════════════════════

class TestStopReconciliationOnRestart:
    def test_capability_reevaluated_against_current_qty_not_cached(
            self, tmp_path, caplog):
        """Restart mid-session with a position that is NOW fractional
        while the pre-restart stop bookkeeping says whole-share: the
        qty-aware probe must run against the CURRENT held quantity (from
        the broker/live-state reconciliation), never the cached value —
        and the unprotectable fractional qty must not get a truncated
        broker stop."""
        config = _config()  # flag off; z9 on
        current_qty = 5.435578  # externally acquired fraction post-restart
        broker = FakeBroker(
            fills={"BLK": {"status": "filled", "order_id": "topup",
                           "filled_qty": 1.0, "filled_avg_price": 100.0}},
            positions={"BLK": current_qty - 1.0},
        )
        # Simulated restart: a FRESH adapter loads pre-restart state — the
        # stop_orders entry still claims qty=5.0 from the prior session.
        stale_stop = {"order_id": "stop-pre-restart", "stop_price": 80.0,
                      "qty": 5.0, "stamped_at": "2026-07-01"}
        ra = _make_adapter(
            tmp_path, config=config, broker=broker,
            positions={"BLK": {"qty": current_qty - 1.0,
                               "qty_available": current_qty - 1.0,
                               "avg_entry_price": 90.0}},
            entry_dates={"BLK": "2026-07-01"},
            position_hwm={"BLK": 100.0},
            stop_orders={"BLK": dict(stale_stop)},
        )
        from kernel.exits import HoldingState  # noqa: PLC0415
        hs = HoldingState(entry_price=90.0,
                          entry_date=datetime.date(2026, 7, 1),
                          high_watermark=100.0)
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": 1, "price": 100.0}],
            holdings={"BLK": hs}, prices={"BLK": 100.0},
        )
        with caplog.at_level("INFO", logger="live.runner"):
            ra.commit(ctx)

        # The probe ran against the CURRENT post-trade quantity …
        assert broker.stop_probe_calls[-1] == ("BLK", current_qty)
        # … not against the pre-restart cached whole-share qty.
        assert ("BLK", 5.0) not in broker.stop_probe_calls
        # Fractional + no software layer ⇒ loudly unprotectable, and NO
        # truncated broker stop is placed.
        assert broker.place_stop_calls == []
        assert "broker-side stop UNAVAILABLE" in caplog.text

    def test_fail_closed_entry_rederived_after_restart(self, tmp_path):
        """The fail-closed-entry invariant is re-derived from reconciled
        state, not assumed from a prior session: a fractional BUY intent
        on the restarted adapter still never reaches the broker."""
        config = _config()
        broker = FakeBroker()
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           entry_dates={"BLK": "2026-07-01"},
                           positions={"BLK": {"qty": 0.4,
                                              "qty_available": 0.4,
                                              "avg_entry_price": 90.0}})
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "BLK", "shares": 0.6, "price": 100.0}],
            prices={"BLK": 100.0},
        )
        ra.commit(ctx)
        assert broker.place_order_calls == []
        assert ctx.orders_skipped[0]["skip_reason"] == (
            "fractional_intent_flag_off")


# ═════════════════════════════════════════════════════════════════════════════
# §2.3 audit 7 — fail-closed entry ⇒ stage-0 outage budget $0 by construction
# ═════════════════════════════════════════════════════════════════════════════

class TestFailClosedEntry:
    """No fractional BUY ever reaches the broker while the software-stop
    layer is absent — so no fractional position can exist at stage 0, and
    the outage-window loss budget is $0 by construction (§2.3)."""

    def test_flag_on_without_stop_layer_blocks_all_buys(self, tmp_path):
        config = _config(fractional=True)
        broker = FakeBroker(fractional_contract=True)  # stage-1 contract OK
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=None)  # stage 3 absent
        ctx = _make_ctx(
            config,
            orders=[
                {"ticker": "BLK", "shares": 0.435578, "price": 100.0},
                {"ticker": "OXY", "shares": 2, "price": 50.0},
            ],
            prices={"BLK": 100.0, "OXY": 50.0},
        )
        ra.commit(ctx)
        # The gate fail-closes EVERY buy (the flag landed ahead of its
        # dependencies — the strategy#36 failure mode) with a dedicated
        # audit reason; nothing reaches the broker.
        assert broker.place_order_calls == []
        assert ctx.orders_placed == []
        assert sorted(o["skip_reason"] for o in ctx.orders_skipped) == [
            "fractional_capability_gate_failed:software_stop_layer",
            "fractional_capability_gate_failed:software_stop_layer",
        ]

    def test_flag_on_without_broker_contract_blocks_all_buys(self, tmp_path):
        config = _config(fractional=True)
        broker = FakeBroker(fractional_contract=False)
        ra = _make_adapter(tmp_path, config=config, broker=broker,
                           software_stops=ArmedSoftwareStops())
        ctx = _make_ctx(
            config,
            orders=[{"ticker": "OXY", "shares": 2, "price": 50.0}],
            prices={"OXY": 50.0},
        )
        ra.commit(ctx)
        assert broker.place_order_calls == []
        assert ctx.orders_skipped[0]["skip_reason"] == (
            "fractional_capability_gate_failed:broker_fractional_contract")

    def test_flag_off_fractional_intent_never_submits(self, tmp_path):
        """A fractional intent with the flag off is a contract violation
        upstream (stage-2 sizing leaked past its flag) — fail closed for
        that order; whole-share orders in the same batch are unaffected."""
        config = _config(z9=False)
        broker = FakeBroker(fills={
            "OXY": {"status": "filled", "order_id": "o", "filled_qty": 2.0,
                    "filled_avg_price": 50.0},
        })
        ra = _make_adapter(tmp_path, config=config, broker=broker)
        ctx = _make_ctx(
            config,
            orders=[
                {"ticker": "BLK", "shares": 0.435578, "price": 100.0},
                {"ticker": "OXY", "shares": 2, "price": 50.0},
            ],
            prices={"BLK": 100.0, "OXY": 50.0},
        )
        ra.commit(ctx)
        assert [c[0] for c in broker.place_order_calls] == ["OXY"]
        assert [(o["ticker"], o["skip_reason"]) for o in ctx.orders_skipped] == [
            ("BLK", "fractional_intent_flag_off"),
        ]
        assert [o["ticker"] for o in ctx.orders_placed] == ["OXY"]

    def test_exits_are_never_blocked_by_the_gate(self, tmp_path):
        """Fail-close applies to BUY emission only — a fractional position
        (however acquired) must always remain exitable."""
        from kernel.exits import ExitSignal, HoldingState  # noqa: PLC0415

        config = _config(fractional=True, z9=False)  # gate FAILS (no stops)
        broker = FakeBroker(
            fills={"BLK": {"status": "filled", "order_id": "s",
                           "filled_qty": 0.4, "filled_avg_price": 100.0}},
            positions={"BLK": 0.4},
            fractional_contract=True,
        )
        ra = _make_adapter(
            tmp_path, config=config, broker=broker,
            positions={"BLK": {"qty": 0.4, "qty_available": 0.4,
                               "avg_entry_price": 90.0}},
            entry_dates={"BLK": "2026-06-20"},
        )
        hs = HoldingState(entry_price=90.0,
                          entry_date=datetime.date(2026, 6, 20),
                          high_watermark=100.0)
        sig = ExitSignal(should_exit=True, reason="stop", exit_type="stop_loss")
        ctx = _make_ctx(config, exits=[("BLK", sig)], holdings={"BLK": hs},
                        prices={"BLK": 100.0})
        ra.commit(ctx)
        assert [t for t, _ in ctx.exits_placed] == ["BLK"]
        assert sig.shares_sold == 0.4


# ═════════════════════════════════════════════════════════════════════════════
# Contract-module unit pins + float-fill round-trip on a fake execution result
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeFillQty:
    def test_fractional_preserved_verbatim(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        assert normalize_fill_qty(0.435578, 5) == 0.435578

    def test_whole_share_snaps_to_int_like_legacy(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        out = normalize_fill_qty(5.0, 3)
        assert out == 5 and type(out) is int

    def test_broker_float_noise_snaps(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        out = normalize_fill_qty(5.0 + 5e-10, 3)
        assert out == 5 and type(out) is int

    def test_falsy_fill_falls_back_like_legacy(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        assert normalize_fill_qty(0, 7) == 7
        assert normalize_fill_qty(None, 3) == 3
        assert normalize_fill_qty("", 2) == 2

    def test_string_float_from_broker_json(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        assert normalize_fill_qty("0.435578", 1) == 0.435578

    def test_non_finite_never_propagates(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        assert normalize_fill_qty(float("nan"), 3) == 0
        assert normalize_fill_qty(float("inf"), 3) == 0


class TestFloatFillRoundTrip:
    """§2.3 float-fill round-trip on a fake broker execution result:
    broker_order_execution → normalize_fill_qty preserves the broker
    float verbatim (never re-derived, never truncated)."""

    def test_round_trip_preserves_broker_float(self):
        from adapters.commit_contract import normalize_fill_qty  # noqa: PLC0415
        from adapters.runner_execmath import broker_order_execution  # noqa: PLC0415

        execution = broker_order_execution(
            {"status": "filled", "order_id": "x",
             "filled_qty": "0.435578", "filled_avg_price": "101.25"},
            requested_qty=0.435578, fallback_price=100.0,
        )
        assert execution["filled"] is True
        assert execution["filled_qty"] == 0.435578
        assert execution["filled_avg_price"] == 101.25
        assert normalize_fill_qty(execution["filled_qty"], 1) == 0.435578

    def test_fractional_full_fill_is_not_partial(self):
        from adapters.runner_execmath import broker_order_execution  # noqa: PLC0415
        execution = broker_order_execution(
            {"status": "filled", "filled_qty": 0.341052,
             "filled_avg_price": 100.0},
            requested_qty=0.341052, fallback_price=100.0,
        )
        assert execution["partial"] is False

    def test_fractional_partial_fill_classified(self):
        from adapters.runner_execmath import broker_order_execution  # noqa: PLC0415
        execution = broker_order_execution(
            {"status": "partially_filled", "filled_qty": 0.2,
             "filled_avg_price": 100.0},
            requested_qty=0.341052, fallback_price=100.0,
        )
        assert execution["partial"] is True
        assert execution["pending"] is False
        assert execution["filled_qty"] == 0.2


class TestFmtQty:
    def test_whole_share_byte_identical_to_legacy_int_format(self):
        from adapters.commit_contract import fmt_qty  # noqa: PLC0415
        assert fmt_qty(5) == "5"
        assert fmt_qty(5.0) == "5"          # old "%d" % int(5.0)
        assert fmt_qty(10.0) == "10"        # old "%.0f" % 10.0

    def test_fractional_rendered_verbatim(self):
        from adapters.commit_contract import fmt_qty  # noqa: PLC0415
        assert fmt_qty(0.435578) == "0.435578"


class TestStopRouting:
    def test_whole_share_routes_broker(self):
        from adapters.commit_contract import route_stop_protection  # noqa: PLC0415
        assert route_stop_protection(FakeBroker(), "BLK", 5.0) == "broker"

    def test_fractional_routes_software_when_armed(self):
        from adapters.commit_contract import route_stop_protection  # noqa: PLC0415
        assert route_stop_protection(
            FakeBroker(), "BLK", 0.4, ArmedSoftwareStops()) == "software"

    def test_fractional_unprotectable_without_layer(self):
        from adapters.commit_contract import route_stop_protection  # noqa: PLC0415
        assert route_stop_protection(FakeBroker(), "BLK", 0.4) == "unprotectable"
        assert route_stop_protection(
            FakeBroker(), "BLK", 0.4, UnarmedSoftwareStops()) == "unprotectable"

    def test_legacy_no_arg_broker_fails_closed_for_fractional(self):
        """A broker predating the qty-aware signature submits whole-share
        GTC stops — a fractional qty is NOT protectable there."""
        from adapters.commit_contract import (  # noqa: PLC0415
            supports_broker_side_stops_for,
        )

        class LegacyBroker:
            def supports_broker_side_stops(self):
                return True

        assert supports_broker_side_stops_for(LegacyBroker(), "BLK", 5.0) is True
        assert supports_broker_side_stops_for(LegacyBroker(), "BLK", 0.4) is False

    def test_armed_probe_is_fail_closed(self):
        from adapters.commit_contract import software_stops_armed  # noqa: PLC0415

        class Raising:
            def is_armed(self):
                raise RuntimeError("registry corrupt")

        class Truthy:
            def is_armed(self):
                return "yes"  # truthy but not True

        assert software_stops_armed(None) is False
        assert software_stops_armed(object()) is False
        assert software_stops_armed(Raising()) is False
        assert software_stops_armed(Truthy()) is False
        assert software_stops_armed(ArmedSoftwareStops()) is True


class TestCapabilityGate:
    def test_flag_off_is_trivially_ok_and_inert(self):
        from adapters.commit_contract import fractional_capability_gate  # noqa: PLC0415
        for cfg in (None, {}, {"execution": {}},
                    {"execution": {"fractional_shares": {"enabled": False}}}):
            gate = fractional_capability_gate(cfg, FakeBroker())
            assert gate["ok"] is True
            assert gate["enabled"] is False
            assert gate["missing"] == []

    def test_flag_on_requires_both_capabilities(self):
        from adapters.commit_contract import fractional_capability_gate  # noqa: PLC0415
        cfg = {"execution": {"fractional_shares": {"enabled": True}}}
        gate = fractional_capability_gate(cfg, FakeBroker())
        assert gate["ok"] is False
        assert gate["missing"] == [
            "broker_fractional_contract", "software_stop_layer",
        ]
        gate = fractional_capability_gate(
            cfg, FakeBroker(fractional_contract=True), ArmedSoftwareStops())
        assert gate["ok"] is True
        assert gate["missing"] == []

    def test_gate_carries_the_contract_tag(self):
        from adapters.commit_contract import (  # noqa: PLC0415
            COMMIT_QTY_CONTRACT,
            fractional_capability_gate,
        )
        gate = fractional_capability_gate(None, FakeBroker())
        assert gate["contract"] == COMMIT_QTY_CONTRACT == "fractional-v2-stage0"

    def test_defense_in_depth_unprotectable_entry_reason(self):
        """Even with a (forged) ok gate, a fractional entry whose qty no
        layer can protect fails closed — belt and braces under §2.2.2."""
        from adapters.commit_contract import (  # noqa: PLC0415
            fractional_entry_fail_closed_reason,
        )
        reason = fractional_entry_fail_closed_reason(
            0.4, {"enabled": True, "ok": True}, broker=FakeBroker(),
            symbol="BLK", software_stops=None,
        )
        assert reason == "fractional_entry_unprotectable_no_stop_layer"
