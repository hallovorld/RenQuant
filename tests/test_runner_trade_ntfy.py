"""live/runner.py MUST send ntfy on EVERY decision cycle.

User rules (2026-04-23):
  1. "任何脚本，只要发生交易，一定要 ntfy"  — cover trades
  2. "每一次计算尝试下单的决策（不管下没下）都要发"  — cover no-trades too

Design: `_notify_decision(label, run_mode, ctx)` fires after every
pipeline commit. Trades → high-priority TRADE ntfy with order detail.
Zero-trades → default-priority DECISION ntfy with a rollup of why
(regime, transition state, no_candidates, sector_full, …).

All tests use monkey-patched urllib so no real POST happens.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: This file IS the operator-notification contract: it calls the composition
#: helpers and asserts on what the operator would read. The marker — not a
#: substring scan over the source — is what
#: `TestTheCONTRACTWorkflowActuallyRunsTheseTests` and the
#: operator-notification-contract workflow agree on
#: [codex on RenQuant#601: "a substring scan cannot encode that semantic
#: boundary"]. Files that merely mention or monkeypatch a helper away must NOT
#: carry it.
pytestmark = pytest.mark.notification_contract

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _allow_mocked_ntfy(monkeypatch):
    """pytest.ini suppresses real notifications; these tests mock urlopen."""
    monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)


class TestSourceLevel:
    """Contract: _notify_decision exists and is called after commit."""

    def test_helper_defined(self):
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        assert "def _notify_decision(" in src, (
            "live/runner.py must define _notify_decision helper"
        )

    def test_called_after_commit(self):
        """The helper must be invoked right after `adapter.commit(ctx)` so
        every cycle surfaces to ntfy."""
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        idx_commit = src.find("adapter.commit(ctx)")
        idx_notify = src.find("_notify_decision(", idx_commit)
        assert idx_commit > 0
        assert idx_notify > idx_commit
        assert idx_notify - idx_commit < 1200

    def test_daily_wrapper_can_suppress_inner_preflight_ntfy(self):
        """AUDIT REGRESSION GUARD: daily_104.sh owns fallback alerts.

        If live.runner sends its own urgent preflight ntfy before the daily
        wrapper can fall back to sell-only, the operator gets a false ERROR
        even though risk exits complete successfully.
        """
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        assert "RENQUANT_SUPPRESS_PREFLIGHT_NTFY" in src
        assert "preflight ntfy suppressed" in src
        assert "log_fn = log.warning if suppress_preflight_ntfy else log.error" in src

    def test_buy_side_preflight_alert_is_not_urgent_error(self):
        """Model-contract blocks should not page like broker outages."""
        from live.runner import _preflight_alert_payload

        msg = (
            "2 hard pre-flight check(s) failed:\n"
            "  ✗ P-WF-GATE: active panel artifact failed WF\n"
            "  ✗ P-RUN-ID: run_id not stamped\n"
        )
        alert = _preflight_alert_payload("RENQUANT-104", "full", msg)

        assert alert["title"] == "RENQUANT-104 [full] BUY-BLOCKED"
        assert alert["priority"] == "default"
        assert alert["taxonomy"] == "DECISION"
        assert "No orders placed" in alert["body"]

    def test_non_buy_side_preflight_alert_stays_urgent(self):
        """Broker/account failures remain action-required."""
        from live.runner import _preflight_alert_payload

        msg = (
            "1 hard pre-flight check(s) failed:\n"
            "  ✗ P-BROKER-CONNECT: broker disconnected\n"
        )
        alert = _preflight_alert_payload("RENQUANT-104", "full", msg)

        assert alert["title"] == "RENQUANT-104 [full] PREFLIGHT-FAIL"
        assert alert["priority"] == "urgent"
        assert alert["taxonomy"] == "ACTION_REQUIRED"

    def test_daily_shadow_wrappers_suppress_inner_preflight_ntfy(self):
        """Shadow failures are non-fatal; each shadow leg suppresses the
        runner's inner preflight ntfy (the daily wrapper sends one alert).

        2026-08-04 re-anchor: the original anchor "Step 4: Shadow e2e"
        rotted when Step 4 was RETIRED (2026-08-03) — this test failed for
        the wrong reason from that day on. The maintained pattern is the
        Step-5 family: EVERY shadow-blend leg must set
        RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 immediately before its heredoc."""
        src = (REPO_ROOT / "scripts" / "daily_104.sh").read_text()
        assert "Step 4: RETIRED" in src  # the retirement is the anchor now
        for heading in ("--- Step 5:", "--- Step 5b:", "--- Step 5c:",
                        "--- Step 5d:", "--- Step 5e:"):
            idx = src.find(heading)
            assert idx > 0, heading
            idx_suppress = src.find("RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1", idx)
            idx_python = src.find('"$PYTHON" - <<PY', idx)
            assert idx_suppress > idx, heading
            assert idx_python > idx_suppress, heading
            assert idx_python - idx_suppress < 120, heading

    def test_live_only_wrapper_is_a_thin_shim_with_no_own_ntfy(self):
        """2026-08-04 re-anchor: live_only_104.sh became a compatibility shim
        that execs intraday_sell_104.sh — the original assertion text moved
        out with the old body. The property that must HOLD: the shim itself
        composes no success ntfy of its own (live.runner posts the decision),
        and it delegates via exec (no second wrapper layer that could
        duplicate alerts)."""
        src = (REPO_ROOT / "scripts" / "live_only_104.sh").read_text()
        assert 'exec bash "$REPO_DIR/scripts/intraday_sell_104.sh"' in src
        assert 'notify "RenQuant 104 [$TAG]" "$FULL_MSG"' not in src
        assert "t.get('signal'" not in src


def _stub_ctx(**kwargs) -> SimpleNamespace:
    """Baseline ctx with sensible defaults; override fields via kwargs."""
    defaults: dict = dict(
        orders           = [],
        orders_placed    = [],   # broker-confirmed (post-guard)
        orders_skipped   = [],   # pipeline-intent but guard-blocked
        exits            = [],
        regime           = "BULL_CALM",
        confidence       = 0.50,
        portfolio_value  = 10071.0,
        holdings         = {"AAA": None, "BBB": None},
        bear_only        = False,
        regime_state     = SimpleNamespace(in_transition=False),
        skip_buys        = False,
        buy_blocked      = False,
        counters         = {},
        ranked           = [],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestAlwaysFiresOnCycle:
    """Both trade and no-trade cycles must post to ntfy."""

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_fires_on_zero_trade_cycle(self):
        """Every cycle notifies — even the quiet ones (safety: prove alive)."""
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()
        req = m.call_args[0][0]
        assert req.headers.get("Title") == "RENQUANT-104 [full] DECISION"
        assert req.headers.get("Priority") == "default"

    def test_fires_on_buy_order_high_priority(self):
        """Orders that actually reached the broker — orders_placed."""
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            orders_placed=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()
        req = m.call_args[0][0]
        assert "BUY TSM x6" in req.data.decode()
        assert req.headers.get("Title").startswith("RENQUANT-104 [full] TRADE"), (
            "the tag contract is the title PREFIX; 2026-08-19 appends the action")
        assert "BUY TSM x6" in req.headers.get("Title"), (
            "the action must reach the TITLE — the body is collapsed on a phone")
        assert req.headers.get("Priority") == "high"

    def test_skipped_order_reports_skip_not_buy(self):
        """2026-04-23 incident: pipeline wanted BUY TSM, but
        duplicate-order guard skipped (pending order exists from
        earlier bar). ntfy must NOT say "BUY TSM" — that misleads
        the user into thinking a 2nd fill happened. Must say SKIPPED
        with reason."""
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            orders_placed=[],
            orders_skipped=[{
                "ticker": "TSM", "shares": 6, "price": 382.66,
                "skip_reason": "pending_order_exists",
            }],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        title = m.call_args[0][0].headers.get("Title")
        assert "SKIPPED" in body
        assert "TSM" in body
        assert "pending_order_exists" in body
        assert "BUY TSM x6" not in body, (
            "Must not report a phantom buy when broker skipped the order"
        )
        assert title == "RENQUANT-104 [full] DECISION", (
            "Title is DECISION (not TRADE) when nothing actually filled"
        )

    def test_fires_on_exit_high_priority(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="XLU", exit_type="trailing_stop")
        ctx = _stub_ctx(exits=[exit_sig])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only", ctx)
        m.assert_called_once()
        req = m.call_args[0][0]
        assert "EXIT XLU (trailing_stop)" in req.data.decode()
        assert req.headers.get("Title").startswith("RENQUANT-104 [sell-only] TRADE")

    def test_shadow_exit_is_marked_hypothetical_not_live_trade(self):
        """2026-07-01: title prefix renamed [SHADOW] -> [READONLY] so the
        broker-mode title token no longer collides with the per-model
        SHADOW[name]/SHADOW-PICKS[name] body segments (see
        TestShadowRecommendations below and the title-prefix disambiguation
        tests)."""
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="FTNT", exit_type="qp_sell")
        ctx = _stub_ctx(exits=[exit_sig])
        with patch.dict("os.environ", {"RENQUANT_SHADOW_NTFY": "1"}), \
                patch("urllib.request.urlopen") as m:
            notify("[READONLY]RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        body = req.data.decode()
        assert req.headers.get("Title").startswith(
            "[READONLY]RENQUANT-104 [full] SHADOW-ACTION"
        )
        assert req.headers.get("Priority") == "default"
        # 2026-08-04 asserted `"SHADOW/HYPOTHETICAL" not in body`, on the
        # reasoning that "the shadow identity lives in the title". INVERTED
        # 2026-08-19: that assertion is what let the marker's removal stand
        # for 15 days (2026-08-05..2026-08-19, zero disclaimers across all six
        # shadow lanes) until the operator received a shadow body and asked
        # whether it was real money.
        #
        # The title is not what the reader always gets: a phone collapses the
        # body, and a body copied out of the app carries NO title. So the body
        # must be self-identifying, and it must be so at parts[0] where
        # truncation cannot reach it. Terse (the 2026-08-04 直 directive
        # stands) but present.
        assert body.startswith("SHADOW — not real"), (
            "a shadow body must say so as its FIRST token — the title is not "
            "guaranteed to reach the reader", body[:120],
        )
        assert "EXIT FTNT (qp_sell)" in body

    def test_combines_buys_and_exits(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="XLU", exit_type="rotation")
        ctx = _stub_ctx(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            orders_placed=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            exits=[exit_sig],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "BUY TSM" in body and "EXIT XLU" in body


class TestWhyNoTradeRollup:
    """No-trade cycles must SURFACE the reason in the body."""

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_transition_window(self):
        notify = self._import()
        ctx = _stub_ctx(regime_state=SimpleNamespace(in_transition=True))
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert "transition_window" in m.call_args[0][0].data.decode()

    def test_drawdown_halt(self):
        notify = self._import()
        ctx = _stub_ctx(skip_buys=True)
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert "drawdown_halt" in m.call_args[0][0].data.decode()

    def test_panel_contract_block_beats_drawdown_rollup(self):
        notify = self._import()
        ctx = _stub_ctx(
            skip_buys=True,
            buy_blocked=True,
            counters={"panel_scoring_fail_closed": 91},
            ranked=[SimpleNamespace(ticker="X")],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "panel_scoring_fail_closed(91)" in body
        assert "drawdown_halt" not in body

    def test_qp_mu_contract_block_beats_drawdown_rollup(self):
        notify = self._import()
        ctx = _stub_ctx(
            skip_buys=True,
            counters={"qp_mu_contract_block": 1},
            ranked=[SimpleNamespace(ticker="X")],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "qp_mu_contract_block(1)" in body
        assert "drawdown_halt" not in body

    def test_bear_only(self):
        notify = self._import()
        ctx = _stub_ctx(bear_only=True)
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert "bear_only" in m.call_args[0][0].data.decode()

    def test_no_candidates(self):
        notify = self._import()
        ctx = _stub_ctx()   # ranked=[] default
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert "no_candidates" in m.call_args[0][0].data.decode()

    def test_sector_full_shows_count(self):
        notify = self._import()
        ctx = _stub_ctx(
            counters={"sector_blocks": 3},
            ranked=[SimpleNamespace(ticker="X")],   # have candidates
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "sector_full(3)" in body

    def test_qp_reason_beats_unsuppressing_buy_gate(self):
        """A buy gate is context, not cause, when QP produced no buy-sized delta."""
        notify = self._import()
        ctx = _stub_ctx(
            buy_blocked=True,
            counters={"qp_delta_below_min_dw": 70, "qp_skipped_band": 6},
            ranked=[SimpleNamespace(ticker="X")],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "qp_delta_below_min_dw(70)" in body

    def test_buy_gate_surfaces_when_it_suppressed_qp_buys(self):
        notify = self._import()
        ctx = _stub_ctx(
            buy_blocked=True,
            counters={"qp_blocked_buys": 2, "qp_delta_below_min_dw": 70},
            ranked=[SimpleNamespace(ticker="X")],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "buy_blocked" in body


class TestBodyIncludesContextSnapshot:
    """Every notification must include regime / confidence / holdings /
    equity so audit state is captured even on quiet cycles."""

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_snapshot_fields_present(self):
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "regime=BULL_CALM" in body
        assert "conf=0.50" in body
        assert "held=2" in body
        assert "eq=$10,071" in body

    def test_transition_flag_surfaced(self):
        """When regime just changed, in_transition=True must show in ntfy
        so operator can distinguish a fresh flip from a stable regime."""
        notify = self._import()
        ctx = _stub_ctx(regime_state=SimpleNamespace(
            in_transition=True, hard_bear=False, hurst=0.55, hurst_regime="AMBIGUOUS"))
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "transition=T" in body

    def test_hard_bear_flag_surfaced(self):
        """hard_bear=True (extreme vol/return forced BEAR) is the most
        important regime diagnostic — must always surface."""
        notify = self._import()
        ctx = _stub_ctx(regime="BEAR", confidence=1.0,
                        regime_state=SimpleNamespace(
                            in_transition=False, hard_bear=True,
                            hurst=0.42, hurst_regime="REVERSION"))
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "regime=BEAR" in body
        assert "hard_bear=T" in body

    def test_hurst_value_surfaced(self):
        """Hurst exponent tells operator how trending the market is."""
        notify = self._import()
        ctx = _stub_ctx(regime_state=SimpleNamespace(
            in_transition=False, hard_bear=False, hurst=0.72,
            hurst_regime="MOMENTUM"))
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "hurst=0.72" in body
        # MOMENTUM truncated to first 3 chars to keep body terse
        assert "hurst_reg=MOM" in body

    def test_ambiguous_hurst_not_surfaced(self):
        """AMBIGUOUS Hurst regime is the default — don't clutter ntfy."""
        notify = self._import()
        ctx = _stub_ctx(regime_state=SimpleNamespace(
            in_transition=False, hard_bear=False, hurst=0.55,
            hurst_regime="AMBIGUOUS"))
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "hurst=0.55" in body
        assert "hurst_reg=" not in body

    def test_stable_regime_no_transition_marker(self):
        """When in_transition=False and hard_bear=False, those flags
        should NOT appear (clean body when conditions are normal)."""
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "transition=" not in body
        assert "hard_bear=" not in body


class TestSilentIntradayNoOp:
    """User rule (2026-04-27): the every-30-min intraday sell-only cycle
    must NOT ntfy on no-op cycles (would push 12× per day with nothing
    actionable). Trades + failed exits + unmanaged + rotation-blocks +
    skipped intents still notify so anything the operator should see
    still reaches their phone."""

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_silent_when_quiet_intraday(self):
        notify = self._import()
        ctx = _stub_ctx()  # no orders, no exits, no failures
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_not_called()

    def test_loud_when_silent_flag_off_default(self):
        """Backward-compat: callers that don't pass silent_if_quiet still
        get a ntfy on every cycle (full / open / preclose / EOD)."""
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()

    def test_loud_on_trade_even_when_silent_flag(self):
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "AAPL", "shares": 5, "price": 200.0}],
            orders_placed=[{"ticker": "AAPL", "shares": 5, "price": 200.0}],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_called_once()
        assert "] TRADE" in m.call_args[0][0].headers.get("Title")

    def test_loud_on_exit_even_when_silent_flag(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="XLU", exit_type="trailing_stop")
        ctx = _stub_ctx(exits=[exit_sig])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_called_once()
        body = m.call_args[0][0].data.decode()
        assert "EXIT XLU" in body

    def test_loud_on_failed_exit_even_when_silent_flag(self):
        """Broker rejected a sell — operator MUST see it."""
        notify = self._import()
        ctx = _stub_ctx(
            exits_failed=[{"ticker": "AAPL", "exit_type": "stop_loss",
                            "qty": 5, "error": "insufficient_qty"}],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_called_once()
        body = m.call_args[0][0].data.decode()
        assert "FAILED-EXIT AAPL" in body
        assert "no trade" not in body
        assert "] FAILED-EXIT" in m.call_args[0][0].headers.get("Title")
        assert m.call_args[0][0].headers.get("Priority") == "urgent"

    def test_loud_on_pending_order_even_when_silent_flag(self):
        notify = self._import()
        ctx = _stub_ctx(
            orders_pending=[{
                "ticker": "AAPL", "shares": 5,
                "status": "accepted", "order_id": "abc",
            }],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_called_once()
        req = m.call_args[0][0]
        body = req.data.decode()
        assert "PENDING-BUY AAPL" in body
        assert "no trade" not in body
        assert "] PENDING" in req.headers.get("Title")
        assert req.headers.get("Priority") == "high"

    def test_loud_on_unmanaged_even_when_silent_flag(self):
        notify = self._import()
        ctx = _stub_ctx(non_wl_holds=["BA"])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "sell-only (intraday)", ctx, silent_if_quiet=True)
        m.assert_called_once()
        assert "UNMANAGED BA" in m.call_args[0][0].data.decode()

    def test_pipeline_passes_flag_only_for_intraday_sell_only(self):
        """The runner wiring must only set silent_if_quiet when BOTH
        sell_only=True AND use_intraday_prices=True."""
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        assert "silent_if_quiet = bool(sell_only and use_intraday_prices)" in src, (
            "_run_once_multi_pipeline must scope the silent flag exactly to "
            "the every-30-min intraday sell-only cycle"
        )


class TestFailSafe:
    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_network_failure_does_not_raise(self, caplog):
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
            orders_placed=[{"ticker": "TSM", "shares": 6, "price": 382.66}],
        )
        import logging
        caplog.set_level(logging.WARNING)
        with patch.dict("os.environ", {
            "RENQUANT_NTFY_RETRIES": "2",
            "RENQUANT_NTFY_BACKOFF_SECONDS": "0",
            "RENQUANT_NTFY_DISABLE_CURL_FALLBACK": "1",
        }):
            with patch("urllib.request.urlopen",
                       side_effect=ConnectionError("no network")):
                notify("RENQUANT-104", "full", ctx)   # must not raise
        assert any("ntfy publish FAILED" in rec.message
                   for rec in caplog.records)

    def test_retries_transient_urlopen_failure(self):
        """Trade ntfy must not be single-shot: transient SSL timeouts happen."""
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "BAC", "shares": 13, "price": 51.80}],
            orders_placed=[{"ticker": "BAC", "shares": 13, "price": 51.80}],
        )
        ok_response = SimpleNamespace(read=lambda: b"ok")
        with patch.dict("os.environ", {
            "RENQUANT_NTFY_RETRIES": "3",
            "RENQUANT_NTFY_BACKOFF_SECONDS": "0",
        }):
            with patch("urllib.request.urlopen",
                       side_effect=[TimeoutError("ssl handshake"), ok_response]) as m:
                notify("RENQUANT-104", "full", ctx)
        assert m.call_count == 2

    def test_uses_curl_fallback_after_urllib_retries(self):
        """If Python urllib keeps failing, fall back to curl before giving up."""
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "WFC", "shares": 7, "price": 76.40}],
            orders_placed=[{"ticker": "WFC", "shares": 7, "price": 76.40}],
        )
        with patch.dict("os.environ", {
            "RENQUANT_NTFY_RETRIES": "2",
            "RENQUANT_NTFY_BACKOFF_SECONDS": "0",
        }):
            with patch("urllib.request.urlopen",
                       side_effect=TimeoutError("ssl handshake")) as urlopen:
                with patch("subprocess.run") as curl:
                    notify("RENQUANT-104", "full", ctx)
        assert urlopen.call_count == 2
        curl.assert_called_once()
        assert b"BUY WFC x7" in curl.call_args.kwargs["input"]

    def test_respects_topic_env_var(self):
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "AAPL", "shares": 1, "price": 100.0}],
            orders_placed=[{"ticker": "AAPL", "shares": 1, "price": 100.0}],
        )
        with patch("urllib.request.urlopen") as m:
            with patch.dict("os.environ", {"RENQUANT_NTFY_TOPIC": "alt-topic"}):
                notify("RENQUANT-104", "full", ctx)
        assert m.call_args[0][0].full_url == "https://ntfy.sh/alt-topic"

    def test_repeated_no_trade_decision_suppressed_within_cooldown(self, tmp_path):
        notify = self._import()
        ctx = _stub_ctx()
        with patch.dict("os.environ", {
            "RENQUANT_ALERT_STATE_PATH": str(tmp_path / "alert_state.json"),
            "RENQUANT_NTFY_BACKOFF_SECONDS": "0",
        }):
            with patch("urllib.request.urlopen") as m:
                notify("RENQUANT-104", "full", ctx)
                notify("RENQUANT-104", "full", ctx)
        assert m.call_count == 1

    def test_failed_exit_still_sends_every_time(self, tmp_path):
        notify = self._import()
        ctx = _stub_ctx(
            exits_failed=[{"ticker": "AAPL", "exit_type": "stop_loss",
                            "qty": 5, "error": "insufficient_qty"}],
        )
        with patch.dict("os.environ", {
            "RENQUANT_ALERT_STATE_PATH": str(tmp_path / "alert_state.json"),
            "RENQUANT_NTFY_BACKOFF_SECONDS": "0",
        }):
            with patch("urllib.request.urlopen") as m:
                notify("RENQUANT-104", "sell-only", ctx)
                notify("RENQUANT-104", "sell-only", ctx)
        assert m.call_count == 2


def _shadow_summary_entry(name="patchtst_v1", picks=None, admission=None, **overrides):
    """Build a ctx._shadow_summary entry matching the shape produced by
    kernel.panel_pipeline.shadow_scoring._compute_shadow_summary.

    2026-07-01 ROUND 2 (Codex CHANGES_REQUESTED on umbrella PR #426):
    defaults to an ACTIONABLE admission verdict (fresh artifact, full
    coverage) so the pre-existing rendering tests below keep exercising the
    actionable path unchanged. TestShadowPicksAdmissionGate overrides
    `admission` directly to cover the NOT-ACTIONABLE gate.
    """
    default_admission = {
        "verdict": "healthy",
        "actionable": True,
        "trained_date": "2026-06-30",
        "age_days": 1.0,
        "artifact_fingerprint": "sha256:testfp1234567890",
        "n_scored": 83,
        "n_expected": 83,
        "coverage": 1.0,
        "min_coverage": 0.80,
        "reasons": [],
        "run_id": f"2026-07-01:{name}:sha256:testfp1",
    }
    admission = admission if admission is not None else default_admission
    entry = dict(
        name=name, kind="patchtst",
        top3=["OXY", "OKE", "CVX"],
        top10_overlap=4,
        n_candidates=83,
        spearman_vs_primary=0.42,
        top_picks=picks if picks is not None else [
            # shadow_percentile: 100.0 = best (rank 1) — FIXED direction
            # 2026-07-01 round 2 (was rank/n*100, best name near 1st
            # percentile). Not itself rendered in the ntfy body (only
            # rank + z-score are), kept here for fixture realism.
            {"ticker": "NVDA", "shadow_score": 0.05, "shadow_rank": 1,
             "shadow_percentile": 100.0, "shadow_zscore": 2.10,
             "in_primary_admitted": None, "in_primary_topN": True},
            {"ticker": "OXY", "shadow_score": -0.15, "shadow_rank": 15,
             "shadow_percentile": 83.1, "shadow_zscore": 0.88,
             "in_primary_admitted": None, "in_primary_topN": False},
        ],
        top_picks_n=5,
        admission=admission,
        actionable=admission.get("actionable"),
        run_id=admission.get("run_id"),
    )
    entry.update(overrides)
    return entry


class TestShadowTopPicksNtfy:
    """2026-07-01: shadow-model top-N RAW RANK diagnostic (operator
    incident: a "[SHADOW]...BUY OXY" ntfy was misread as "the shadow
    PatchTST model recommends OXY" — see
    doc/progress/2026-07-01-shadow-ntfy-top-picks.md).

    ROUND 3 (Codex #426 review point 3): deliberately never called a
    "recommendation" or "confidence" score in the rendered ntfy text — see
    TestShadowPicksAdmissionGate for the freshness/coverage admission gate
    that suppresses picks entirely when they would not be actionable.
    """

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_shadow_picks_segment_rendered_with_rank_and_zscore(self):
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW-PICKS[patchtst_v1]:" in body
        assert "OXY(rank 15/83, z=+0.88)" in body
        assert "NVDA(rank 1/83, z=+2.10" in body

    def test_shadow_picks_segment_is_labeled_relative_not_confidence(self):
        """2026-07-01 ROUND 3 (Codex #426 review point 3, "stop calling the
        line a recommendation or confidence"): the message TEXT itself (not
        just a code comment) must say this is a raw, unvalidated rank —
        without using the word "confidence" at all in the actionable-path
        tag."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "[raw rank (unvalidated, see freshness verdict)]" in body
        assert "recommend" not in body.lower()

    def test_no_fabricated_probability_confidence_wording(self):
        """Must never render a bare '%' confidence claim (e.g. '73%
        confidence') — only rank/percentile/z-score, always honestly
        labeled. shadow_percentile itself is NOT rendered as a standalone
        '%' claim in the compact ntfy line (only rank + z-score are)."""
        import re
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert re.search(r"\d+(\.\d+)?%\s*confidence", body) is None
        assert "confidence=" not in body

    def test_legacy_shadow_line_kept_byte_for_byte_backward_compat(self):
        """The pre-existing SHADOW[name] top3=.../top10.../ρ=... line must
        be unchanged — some downstream tooling may parse this exact
        format."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW[patchtst_v1] top3=OXY/OKE/CVX top10∩prim=4/10 ρ=+0.42 n=83" in body

    def test_also_bought_tag_when_primary_actually_bought_the_pick(self):
        """in_primary_admitted is None at shadow_scoring.py build time (not
        determinable — SelectionJob hasn't run yet). live/runner.py must
        overlay the REAL value from ctx.orders_placed at ntfy-render time,
        since the full pipeline has run by _notify_decision."""
        notify = self._import()
        ctx = _stub_ctx(
            orders=[{"ticker": "NVDA", "shares": 2, "price": 900.0}],
            orders_placed=[{"ticker": "NVDA", "shares": 2, "price": 900.0}],
            _shadow_summary=[_shadow_summary_entry()],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "NVDA(rank 1/83, z=+2.10, ALSO-BOUGHT)" in body
        # OXY was NOT bought today — must not carry the ALSO-BOUGHT tag.
        assert "OXY(rank 15/83, z=+0.88, ALSO-BOUGHT)" not in body
        assert "OXY(rank 15/83, z=+0.88)" in body

    def test_no_also_bought_tag_when_primary_bought_nothing(self):
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "ALSO-BOUGHT" not in body

    def test_multiple_shadow_models_each_get_own_picks_segment(self):
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(name="patchtst_v1"),
            _shadow_summary_entry(name="ngboost_v2", top3=["MU", "AMD", "TSM"]),
        ])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW-PICKS[patchtst_v1]:" in body
        assert "SHADOW-PICKS[ngboost_v2]:" in body

    def test_missing_top_picks_does_not_break_legacy_line(self):
        """Older/degenerate summary dicts without top_picks (e.g. from a
        stale cached run) must not raise — legacy SHADOW[...] line still
        renders, SHADOW-PICKS[...] segment simply omitted."""
        notify = self._import()
        legacy_entry = _shadow_summary_entry()
        del legacy_entry["top_picks"]
        ctx = _stub_ctx(_shadow_summary=[legacy_entry])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW[patchtst_v1]" in body
        assert "SHADOW-PICKS[patchtst_v1]" not in body


class TestShadowPicksAdmissionGate:
    """2026-07-01 ROUND 2 (Codex CHANGES_REQUESTED on umbrella PR #426): a
    raw shadow rank must never be presented as an ACTIONABLE pick when the
    artifact is stale or the scored universe is a censored subset. Covers
    the two REAL known examples cited in the review: PatchTST confirmed
    ~140 days stale, and an 83/292 (~28%) censored universe.

    2026-07-01 ROUND 4 (Codex CHANGES_REQUESTED — scope narrowing, see
    doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum #4): the
    diagnostic rank/z-score list is now ALWAYS rendered, even in the NOT
    ACTIONABLE branch (previously fully suppressed). shadow_scoring.py's
    ``_compute_admission`` now defaults ``actionable`` to False regardless
    of the computed verdict (see tests/test_shadow_scoring.py
    ``TestComputeAdmission``/``TestComputeShadowSummaryAdmissionIntegration``
    for that gate itself) — since every cycle now renders NOT ACTIONABLE by
    default, fully suppressing the list here would make the entire feature
    permanently dark, defeating the PR's original observability-only
    intent. The gate is
    STILL enforced: the ranks are labeled diagnostic-only, never presented
    as an actionable pick.
    """

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def _breach_admission(self, **overrides):
        admission = dict(
            verdict="breach", actionable=False, gates_passed=False,
            trained_date="2026-02-11", age_days=140.0,
            artifact_fingerprint="sha256:staleabc1234",
            n_scored=83, n_expected=83, coverage=1.0, min_coverage=0.80,
            reasons=["artifact 140d stale (breach>=35d)"],
            run_id="2026-07-01:patchtst_v1:sha256:staleab",
        )
        admission.update(overrides)
        return admission

    def test_stale_artifact_picks_are_not_actionable_but_ranks_still_shown(self):
        """Real known example: PatchTST confirmed ~140 days stale — the
        line must be clearly labeled NOT ACTIONABLE, but (round 4) the raw
        diagnostic rank/z-score breakdown is still shown for observability,
        never presented as a pick."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(admission=self._breach_admission())
        ])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW-PICKS[patchtst_v1]: NOT ACTIONABLE" in body
        assert "140d stale" in body
        assert "verdict=breach" in body
        assert "NVDA(rank 1/83" in body
        assert "OXY(rank 15/83" in body
        assert "[diagnostic rank only, not actionable]" in body

    def test_incomplete_coverage_picks_are_not_actionable_but_ranks_still_shown(self):
        """Real known example: rank 1 of an 83-name censored subset is not
        comparable to rank 1 of the intended ~292-name watchlist — still
        NOT ACTIONABLE, but (round 4) diagnostics remain visible."""
        notify = self._import()
        admission = self._breach_admission(
            verdict="healthy", actionable=False, gates_passed=False,
            n_scored=83, n_expected=292, coverage=83 / 292,
            reasons=["coverage 83/292 (28%) < 80%"],
        )
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(admission=admission)
        ])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW-PICKS[patchtst_v1]: NOT ACTIONABLE" in body
        assert "83/292" in body
        assert "NVDA(rank 1/83" in body

    def test_missing_admission_field_defaults_to_not_actionable_but_ranks_still_shown(self):
        """Fail-closed: a summary with no `admission` key at all (e.g. a
        cached ctx from before this fix landed) must not be silently
        treated as actionable — but (round 4) the raw top_picks data that
        IS present still renders as diagnostic-only."""
        notify = self._import()
        entry = _shadow_summary_entry()
        del entry["admission"]
        del entry["actionable"]
        del entry["run_id"]
        ctx = _stub_ctx(_shadow_summary=[entry])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW-PICKS[patchtst_v1]: NOT ACTIONABLE" in body
        assert "no admission verdict computed" in body
        assert "NVDA(rank 1/83" in body

    def test_actionable_case_surfaces_verdict_coverage_and_run_id(self):
        """When actionable, the picks line binds the ranks to provenance —
        verdict, scored-vs-expected coverage, and a run id — not just a
        bare ranked list."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "[healthy cov=83/83 run=" in body

    def test_not_actionable_body_has_no_confidence_or_recommendation_wording(self):
        import re
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(admission=self._breach_admission())
        ])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert re.search(r"\d+(\.\d+)?%\s*confidence", body) is None
        assert "confidence=" not in body
        assert "recommend" not in body.lower()

    def test_legacy_shadow_line_still_renders_when_picks_not_actionable(self):
        """The pre-existing SHADOW[name] top3/overlap/spearman diagnostic
        line is unaffected by the admission gate — it never claimed
        actionability in the first place, so it is not gated."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(admission=self._breach_admission())
        ])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "SHADOW[patchtst_v1] top3=OXY/OKE/CVX top10∩prim=4/10 ρ=+0.42 n=83" in body


class TestNtfyBodyLengthBudget:
    """ntfy's practical body limit is ~4096 bytes. No truncation guard
    existed before 2026-07-01; the new SHADOW-PICKS segments (one per
    configured shadow model) made an unbounded body more likely, so an
    explicit, honest truncation was added rather than letting the
    transport silently cut the message."""

    def test_truncate_helper_noop_under_budget(self):
        from live.runner import _truncate_ntfy_body
        assert _truncate_ntfy_body("short body") == "short body"

    def test_truncate_helper_caps_and_marks_long_body(self):
        from live.runner import _truncate_ntfy_body, _NTFY_BODY_MAX_BYTES
        body = "x" * (_NTFY_BODY_MAX_BYTES * 2)
        out = _truncate_ntfy_body(body)
        assert len(out.encode("utf-8")) <= _NTFY_BODY_MAX_BYTES
        assert out.endswith("…[truncated]")

    def test_truncate_helper_is_utf8_safe(self):
        """Must not cut in the middle of a multi-byte character (ρ is used
        throughout the shadow segments)."""
        from live.runner import _truncate_ntfy_body
        body = "ρ" * 5000
        out = _truncate_ntfy_body(body, max_bytes=101)
        # Must decode cleanly (no UnicodeDecodeError) and respect budget.
        assert len(out.encode("utf-8")) <= 101
        out.encode("utf-8").decode("utf-8")  # raises if a char was split

    def test_many_shadow_models_still_yields_bounded_body(self):
        """End-to-end: enough shadow models/picks to blow past budget if
        unbounded must still respect the byte cap."""
        from live.runner import _NTFY_BODY_MAX_BYTES
        from live.runner import _notify_decision
        picks = [
            {"ticker": f"TCK{i:03d}", "shadow_score": 0.01 * i, "shadow_rank": i + 1,
             "shadow_percentile": 1.0, "shadow_zscore": 0.12,
             "in_primary_admitted": None, "in_primary_topN": False}
            for i in range(20)
        ]
        ctx = _stub_ctx(_shadow_summary=[
            _shadow_summary_entry(name=f"shadow_model_{i}", picks=picks)
            for i in range(10)
        ])
        with patch("urllib.request.urlopen") as m:
            _notify_decision("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert len(body.encode("utf-8")) <= _NTFY_BODY_MAX_BYTES


class TestTitlePrefixDisambiguation:
    """2026-07-01 fix: [SHADOW] title prefix ("this ran via the readonly
    broker") was colliding with the unrelated body segments SHADOW[name] /
    SHADOW-PICKS[name] ("this alternate MODEL's own view"), which is what
    caused the operator to misread a "[SHADOW]...BUY OXY" ntfy as a
    PatchTST recommendation. Renamed the broker-mode title token to
    [READONLY] (Option A — repo-wide grep found no external consumer
    pattern-matching the literal "[SHADOW]" title substring)."""

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def test_readonly_prefix_triggers_shadow_broker_behavior(self):
        notify = self._import()
        ctx = _stub_ctx()
        with patch.dict("os.environ", {"RENQUANT_SHADOW_NTFY": "1"}), \
                patch("urllib.request.urlopen") as m:
            notify("[READONLY]RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        # is_shadow=True (readonly broker) still tags SHADOW-DECISION /
        # SHADOW-ACTION — that classification is unaffected by the rename,
        # only the title token changed from [SHADOW] to [READONLY].
        assert req.headers.get("Title") == "[READONLY]RENQUANT-104 [full] SHADOW-DECISION"
        body = req.data.decode()
        assert "SHADOW/HYPOTHETICAL" not in body  # 2026-08-04 terse-body change

    def test_legacy_shadow_prefix_no_longer_treated_as_readonly_broker(self):
        """Documents the intentional behavior change: an old caller passing
        the stale "[SHADOW]" prefix is NOT treated as a readonly-broker run
        any more — is_shadow now keys off "[READONLY]" only."""
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("[SHADOW]RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        body = req.data.decode()
        assert req.headers.get("Title") == "[SHADOW]RENQUANT-104 [full] DECISION"
        assert "SHADOW/HYPOTHETICAL (no live orders)" not in body

    def test_title_prefix_token_distinct_from_body_shadow_model_segments(self):
        """The title prefix (broker mode) and the body's per-model labels
        (alternate MODEL's own view) must not share the same ambiguous
        "[SHADOW]" token any more."""
        notify = self._import()
        ctx = _stub_ctx(_shadow_summary=[_shadow_summary_entry()])
        with patch.dict("os.environ", {"RENQUANT_SHADOW_NTFY": "1"}), \
                patch("urllib.request.urlopen") as m:
            notify("[READONLY]RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        title = req.headers.get("Title")
        body = req.data.decode()
        assert title.startswith("[READONLY]")
        assert "[SHADOW]" not in title
        # Body segments use the OTHER concept (alternate model's own view) —
        # still literally "SHADOW[...]" / "SHADOW-PICKS[...]", but no longer
        # colliding with the title's broker-mode meaning since the title no
        # longer uses that token at all.
        assert "SHADOW[patchtst_v1]" in body
        assert "SHADOW-PICKS[patchtst_v1]" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOperatorCanActuallyReadIt:
    """2026-08-19 regression guards for the readability contract.

    All three defects fixed that day were the same failure — the message did
    not carry the fact the reader needs at the place the reader looks — and
    NONE of them was caught by a test. The shadow marker's removal was in fact
    LOCKED IN by an assertion that it must be absent
    (`test_shadow_exit_is_marked_hypothetical_not_live_trade`, inverted above).
    These tests exist so the next edit to this surface has to argue with CI.
    """

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    # --- title carries the action -------------------------------------
    def test_failed_exits_sort_first_in_the_title(self):
        """A rejected sell may need a human at the broker, so it must never be
        the token that `+N` summarises away."""
        from live.runner import _action_headline
        head = _action_headline(
            [{"ticker": "AAA", "shares": 1}], [], [], [],
            [{"ticker": "ZZZ", "exit_type": "sell"}],
        )
        assert head.startswith("FAILED-EXIT ZZZ"), head

    def test_title_summarises_beyond_the_cap(self):
        from live.runner import _action_headline, _TITLE_ACTION_MAX
        orders = [{"ticker": f"T{i}", "shares": i} for i in range(_TITLE_ACTION_MAX + 3)]
        head = _action_headline(orders, [], [], [], [])
        assert head.count(",") == _TITLE_ACTION_MAX - 1, head
        assert head.endswith("+3"), head

    def test_no_action_leaves_the_title_byte_identical(self):
        """Quiet cycles must not grow a trailing separator — existing
        title-matching consumers depend on the exact no-action form."""
        from live.runner import _action_headline
        assert _action_headline([], [], [], [], []) == ""
        notify = self._import()
        ctx = _stub_ctx()
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert m.call_args[0][0].headers.get("Title") == "RENQUANT-104 [full] DECISION"

    def test_tag_is_not_repeated_when_the_headline_leads_with_it(self):
        notify = self._import()
        ctx = _stub_ctx(exits_failed=[{"ticker": "AAPL", "exit_type": "sell",
                                       "error": "rejected"}])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        title = m.call_args[0][0].headers.get("Title")
        assert title.count("FAILED-EXIT") == 1, title
        assert "AAPL" in title, title

    # --- the body keeps room for the decision --------------------------
    def test_blocked_rotations_are_capped_with_an_honest_remainder(self):
        """2026-08-19: 59 BLOCKED-ROTATION segments pushed the body to 3830 B
        against a 3800 B cap, so the truncation fell INSIDE the diagnostic and
        the trailing regime/equity context never arrived. The count carries the
        signal; the full list stays in the run log.

        FIXTURE CORRECTED 2026-08-20. I wrote this test with `"sell": None`
        while asserting the paired `S→B` rendering — a payload shape that never
        produces a paired rotation at all, since a null sell leg IS the
        pre-pairing prefilter marker. The assertion happened to pass only
        because the renderer was printing `None→B` for it, i.e. the test was
        pinned to the very defect fixed in the same change. Real sell legs
        now."""
        from live.runner import _ROT_BLOCKED_NTFY_MAX
        notify = self._import()
        blocked = [{"sell": f"H{i}", "buy": f"B{i}", "reason": "insufficient_cash"}
                   for i in range(59)]
        ctx = _stub_ctx(rotations_blocked=blocked,
                        orders_placed=[{"ticker": "PANW", "shares": 3, "price": 1.0}])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert body.count("→B") == _ROT_BLOCKED_NTFY_MAX, body[:200]
        assert f"+{59 - _ROT_BLOCKED_NTFY_MAX} more (59 total" in body, body[:400]
        assert "BUY PANW x3" in body, "the decision must survive its own diagnostic"

    def test_the_decision_survives_a_flood_of_blocked_rotations(self):
        """The property that actually failed in production: with the cap, a
        pathological blocked list can no longer evict the context that follows
        it."""
        notify = self._import()
        ctx = _stub_ctx(
            rotations_blocked=[{"sell": f"HELD{i:04d}", "buy": f"TICKER{i:04d}",
                                "reason": "insufficient_cash"}
                               for i in range(400)],
            orders_placed=[{"ticker": "PANW", "shares": 3, "price": 1.0}],
        )
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert "…[truncated]" not in body, "the cap should make truncation unnecessary"
        assert "regime=" in body, "trailing context must still arrive"

    # --- shadow bodies are self-identifying ----------------------------
    def test_shadow_marker_is_the_first_token_of_the_body(self):
        notify = self._import()
        ctx = _stub_ctx(orders_placed=[{"ticker": "NVDA", "shares": 5, "price": 217.56}])
        with patch.dict("os.environ", {"RENQUANT_SHADOW_NTFY": "1"}), \
                patch("urllib.request.urlopen") as m:
            notify("[READONLY][ALPACA_SHADOW_VOL_WINDOW]RENQUANT-104", "full", ctx)
        body = m.call_args[0][0].data.decode()
        assert body.startswith("SHADOW — not real"), body[:120]

    def test_live_bodies_carry_no_shadow_marker(self):
        notify = self._import()
        ctx = _stub_ctx(orders_placed=[{"ticker": "NVDA", "shares": 5, "price": 217.56}])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        assert "SHADOW" not in m.call_args[0][0].data.decode()


class TestShadowLanesAreLogOnly:
    """Readonly/shadow lanes compose the alert but do NOT push it (2026-08-30).

    Inventory 08-23..08-30 [VERIFIED read-only pass over logs/alerts]: 64 ntfy
    messages, 22 of them "[READONLY][<lane>]RENQUANT-104 [full] SHADOW-ACTION:
    FAILED-EXIT VLO …" — six shadow lanes each reporting that the LIVE run's
    pending SELL made THEIR hypothetical exit fail. Pure noise, and `force=True`
    on the actionable path meant dedupe never applied. The fix lives in the
    single send site of `_notify_decision`; the composition and the live path
    are untouched, which the tests below pin from both sides.
    """

    def _import(self):
        from live.runner import _notify_decision
        return _notify_decision

    def _noise_ctx(self):
        # The shape of the 22 messages ("SHADOW-ACTION: FAILED-EXIT VLO …"):
        # a shadow lane that placed its own hypothetical BUY and whose sell of
        # VLO "failed" because the LIVE book already has a pending SELL VLO.
        # SHADOW-ACTION needs a trade/pending; the failed exit alone would tag
        # SHADOW-DECISION — both are gated identically below.
        return _stub_ctx(
            orders_placed=[{"ticker": "NVDA", "shares": 5, "price": 217.56}],
            exits_placed=[], exits_pending=[],
            exits_failed=[{"ticker": "VLO", "exit_type": "qp_sell",
                           "error": "pending SELL VLO already open (live book)"}],
        )

    def test_readonly_variant_makes_no_http_call_by_default(self, monkeypatch, caplog):
        monkeypatch.delenv("RENQUANT_SHADOW_NTFY", raising=False)
        notify = self._import()
        with patch("urllib.request.urlopen") as m, \
                patch("live.alerts.subprocess.run") as curl, \
                caplog.at_level("INFO", logger="live.runner"):
            notify("[READONLY][V]RENQUANT-104", "full", self._noise_ctx())
        m.assert_not_called()
        curl.assert_not_called()
        # The composed alert still reaches the lane's log, title + body, so the
        # readonly-e2e classifier and a human reading the log lose nothing.
        line = next(r.getMessage() for r in caplog.records if "log-only" in r.getMessage())
        assert "[READONLY][V]RENQUANT-104 [full] SHADOW-ACTION: FAILED-EXIT VLO, BUY NVDA x5" in line
        assert "SHADOW — not real" in line
        assert "RENQUANT_SHADOW_NTFY=1" in line

    @pytest.mark.parametrize("value", ["0", "", "false", "yes"])
    def test_only_the_literal_one_enables_the_push(self, monkeypatch, value):
        monkeypatch.setenv("RENQUANT_SHADOW_NTFY", value)
        notify = self._import()
        with patch("urllib.request.urlopen") as m:
            notify("[READONLY][RC]RENQUANT-104", "full", _stub_ctx())
        m.assert_not_called()

    def test_env_override_pushes_the_unchanged_shadow_alert(self, monkeypatch):
        monkeypatch.setenv("RENQUANT_SHADOW_NTFY", "1")
        notify = self._import()
        with patch("urllib.request.urlopen") as m:
            notify("[READONLY][V]RENQUANT-104", "full", self._noise_ctx())
        m.assert_called_once()
        req = m.call_args[0][0]
        assert req.headers.get("Title") == (
            "[READONLY][V]RENQUANT-104 [full] SHADOW-ACTION: FAILED-EXIT VLO, BUY NVDA x5"
        )
        assert req.headers.get("Priority") == "default"
        assert req.data.decode().startswith("SHADOW — not real")

    def test_shadow_gate_is_after_composition_not_before(self):
        """The log-only return sits at the SEND site, below the title/body/
        priority/taxonomy composition — so nothing upstream of the push can
        drift for shadow lanes without the composition tests seeing it."""
        src = (REPO_ROOT / "live" / "runner.py").read_text()
        i_title = src.find('title    = f"{label} [{run_mode}] {tag}"')
        i_gate = src.find("if is_shadow and not _shadow_ntfy_enabled():")
        i_send = src.find("_post_ntfy_with_retries(\n        url,", i_gate)
        assert 0 < i_title < i_gate < i_send

    def test_live_variant_sends_exactly_once_with_the_same_payload_either_way(self, monkeypatch):
        """The live path must be byte-identical with the switch unset AND set:
        RENQUANT_SHADOW_NTFY is consulted only when is_shadow is true."""
        notify = self._import()
        seen = []
        for value in (None, "1", "0"):
            if value is None:
                monkeypatch.delenv("RENQUANT_SHADOW_NTFY", raising=False)
            else:
                monkeypatch.setenv("RENQUANT_SHADOW_NTFY", value)
            with patch("urllib.request.urlopen") as m:
                notify("RENQUANT-104", "full", self._noise_ctx())
            m.assert_called_once()
            req = m.call_args[0][0]
            seen.append((req.headers.get("Title"), req.headers.get("Priority"), req.data))
        assert len(set(seen)) == 1, seen
        title, priority, body = seen[0]
        assert title == "RENQUANT-104 [full] TRADE: FAILED-EXIT VLO, BUY NVDA x5"
        assert priority == "urgent"
        assert "SHADOW" not in body.decode()

    def test_live_pending_buy_still_pushes(self, monkeypatch):
        monkeypatch.delenv("RENQUANT_SHADOW_NTFY", raising=False)
        notify = self._import()
        ctx = _stub_ctx(orders_pending=[{"ticker": "NVDA", "shares": 5,
                                         "status": "accepted", "order_id": "abc"}])
        with patch("urllib.request.urlopen") as m:
            notify("RENQUANT-104", "full", ctx)
        m.assert_called_once()
        req = m.call_args[0][0]
        assert req.headers.get("Title").startswith("RENQUANT-104 [full] PENDING")
        assert req.headers.get("Priority") == "high"


class TestTheCONTRACTWorkflowActuallyRunsTheseTests:
    """The workflow and the marker must name the same set of files.

    2026-08-20: `tests/test_prefilter_is_not_a_rotation.py` was merged (#600)
    with "5/5 checks pass" cited as evidence. Every workflow here runs a
    specific NAMED test file and that one was named by none, so its 16 tests
    never executed in CI — and could not have passed anyway, since `pytest.ini`
    sets `RENQUANT_NO_NOTIFY=1` and the file lacked the opt-out fixture.

    MY FIRST GUARD WAS NOT A CLASSIFIER [codex on #601]. It treated any
    source-text mention of `_notify_decision` / `_no_trade_reason` as a
    contract test, and CI produced the false positives immediately:
    `test_runner_preflight_fail_closed.py` only monkeypatches the helper AWAY,
    and `test_audit_2026_04_24_fixes.py` merely mimics a branch. A substring
    scan cannot encode "asserts on what the operator reads", and it would drag
    an unrelated suite in whenever a helper name appeared in a comment.

    So membership is now an explicit, reviewable **marker** the author applies
    deliberately, and this class asserts the workflow and the marker agree in
    BOTH directions — a marked file the job does not run is uncovered, and a
    file the job runs without the marker means the source of truth has drifted
    from the invocation.

    It lives in a file the workflow itself runs, so it cannot go uncovered.
    """

    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "operator-notification-contract.yml"
    MARKER = "pytest.mark.notification_contract"

    def _marked_files(self) -> set[str]:
        out = set()
        for p in sorted((REPO_ROOT / "tests").glob("test_*.py")):
            try:
                src = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if f"pytestmark = {self.MARKER}" in src:
                out.add(p.name)
        return out

    def _pytest_invocation(self) -> str:
        """ONLY the `python3 -m pytest ...` command, continuations included.

        Searching the whole YAML would be the wrong object: this workflow's own
        comment names test files, so a substring test over the file passes on
        the comment alone. Verified by mutation.
        """
        lines = self.WORKFLOW.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "-m pytest" not in line or line.lstrip().startswith("#"):
                continue
            cmd = [line]
            while cmd[-1].rstrip().endswith("\\") and i + len(cmd) < len(lines):
                cmd.append(lines[i + len(cmd)])
            return "\n".join(cmd)
        return ""

    def _invoked_files(self) -> set[str]:
        return {tok.strip().split("/")[-1]
                for tok in self._pytest_invocation().replace("\\", " ").split()
                if tok.strip().startswith("tests/") and tok.strip().endswith(".py")}

    def test_neither_side_is_empty(self):
        """Either set going empty would make both directions below vacuous."""
        marked, invoked = self._marked_files(), self._invoked_files()
        assert marked, "no file carries the marker — the source of truth is gone"
        assert invoked, "could not parse any test file out of the pytest command"
        assert "test_runner_trade_ntfy.py" in marked & invoked

    def test_every_marked_file_is_run_by_the_workflow(self):
        missing = sorted(self._marked_files() - self._invoked_files())
        assert not missing, (
            f"marked as notification_contract but not run by "
            f"{self.WORKFLOW.name}: {missing}. A file the workflow does not "
            f"name does not run in CI, and a green check then says nothing "
            f"about it."
        )

    def test_every_file_the_workflow_runs_is_marked(self):
        stray = sorted(self._invoked_files() - self._marked_files())
        assert not stray, (
            f"run by {self.WORKFLOW.name} but not marked "
            f"{self.MARKER}: {stray}. The marker is the source of truth; an "
            f"invocation that has drifted from it will drift again."
        )
