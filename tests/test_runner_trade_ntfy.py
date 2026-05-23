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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

    def test_daily_shadow_wrapper_suppresses_inner_preflight_ntfy(self):
        """Shadow failures are non-fatal; daily wrapper sends one alert."""
        src = (REPO_ROOT / "scripts" / "daily_104.sh").read_text()
        idx_shadow = src.find("Step 4: Shadow e2e")
        idx_suppress = src.find("RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1", idx_shadow)
        idx_python = src.find('"$PYTHON" - <<PY', idx_shadow)
        assert idx_shadow > 0
        assert idx_suppress > idx_shadow
        assert idx_python > idx_suppress
        assert idx_python - idx_suppress < 80


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
        assert req.headers.get("Title") == "RENQUANT-104 [full] TRADE"
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
        assert req.headers.get("Title") == "RENQUANT-104 [sell-only] TRADE"

    def test_shadow_exit_is_marked_hypothetical_not_live_trade(self):
        notify = self._import()
        exit_sig = SimpleNamespace(ticker="FTNT", exit_type="qp_sell")
        ctx = _stub_ctx(exits=[exit_sig])
        with patch("urllib.request.urlopen") as m:
            notify("[SHADOW]RENQUANT-104", "full", ctx)
        req = m.call_args[0][0]
        body = req.data.decode()
        assert req.headers.get("Title") == (
            "[SHADOW]RENQUANT-104 [full] SHADOW-ACTION"
        )
        assert req.headers.get("Priority") == "default"
        assert "SHADOW/HYPOTHETICAL (no live orders)" in body
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
        assert m.call_args[0][0].headers.get("Title").endswith("TRADE")

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
