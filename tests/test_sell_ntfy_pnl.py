"""Tests for the 2026-05-18 sell-ntfy $-P/L mandate.

User mandate: "每次卖出的时候 ntfy 里给我算一下具体 pl 是多少". Adapter
stamps realized_pnl_dollar + realized_pnl_pct on the ExitSignal when a
sell fires; live/runner.py builds the ntfy body with these fields when
present. Falls back to the bare `EXIT tkr (reason)` line when missing
(e.g. stop_n_sigma fires with no broker fill yet, sim path, etc.).

Tests are source-substring level (consistent with test_runner_state_fixes.py
style) because the full runtime path needs a broker mock; here we pin
the wire structure.
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADAPTER_SRC = (REPO / "backtesting/renquant_104/adapters/runner.py").read_text()
LIVE_SRC = (REPO / "live/runner.py").read_text()


class TestAdapterStampsPnl:
    def test_realized_pnl_fields_stamped(self):
        assert "sig.realized_pnl_dollar = float(gain_dollar)" in ADAPTER_SRC
        assert "sig.realized_pnl_pct = float(gain_pct)" in ADAPTER_SRC

    def test_cost_basis_falls_back_to_holding_entry_price(self):
        # When positions_cache.avg_entry_price is missing/zero, must
        # fall back to HoldingState.entry_price (running avg-cost).
        assert 'getattr(hs, "entry_price", 0.0)' in ADAPTER_SRC

    def test_pnl_computation_uses_current_price(self):
        # Sell price taken from ctx.prices (close-to-fill snapshot).
        # When bracket/limit orders are later wired, replace with the
        # broker filled_avg_price from the order result.
        assert "price = ctx.prices.get(ticker, 0.0)" in ADAPTER_SRC

    def test_log_line_includes_pl(self):
        assert "P/L=$%" not in ADAPTER_SRC  # f-string, not %-format
        assert 'P/L=${sig.realized_pnl_dollar:+.2f}' in ADAPTER_SRC

    def test_2026_05_18_marker(self):
        assert "2026-05-18: stamp P/L on the ExitSignal" in ADAPTER_SRC


class TestLiveRunnerRendersPnlInNtfy:
    def test_extracts_fields_from_sig(self):
        assert 'getattr(sig, "realized_pnl_dollar", None)' in LIVE_SRC
        assert 'getattr(sig, "realized_pnl_pct", None)' in LIVE_SRC

    def test_pnl_string_format(self):
        # Body line: "EXIT TKR (reason) P/L=$+12.34 (+5.67%)"
        # Tolerance: matches the substring pattern; exact format pinned
        # by the f-string `${pnl_d:+.2f}` / `{pnl_p:+.2f}%`
        assert 'P/L=${pnl_d:+.2f} ({pnl_p:+.2f}%)' in LIVE_SRC

    def test_fallback_when_pnl_missing(self):
        # No P/L attrs → fallback to bare EXIT tkr (reason) line.
        # Pin both forms exist in the source.
        assert 'parts.append(f"EXIT {tkr} ({reason})")' in LIVE_SRC

    def test_2026_05_18_marker(self):
        assert "2026-05-18: include explicit $ realized P/L" in LIVE_SRC
