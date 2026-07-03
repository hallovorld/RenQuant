"""Parking-sleeve daily price coverage — st104 #39 umbrella follow-up.

Pins adapters/sleeve_prices.py and its wiring: when ``sleeve.enabled`` is
true the umbrella must subscribe/fetch daily prices for
``sleeve.spy_symbol`` and ``sleeve.sgov_symbol`` (SPY is already covered
as benchmark/watchlist; SGOV previously was fetched nowhere), mirroring
the ``benchmark_sleeve`` conditional-subscription precedent. With the
flag off/absent (the shipped strategy-104 default) every call site must
be byte-inert.

main.py / adapters wiring is pinned by source inspection — the same
convention as TestAdapterParity in test_universe_alignment.py (LEAN
main.py cannot be imported under plain pytest).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_price import context_price_tickers  # noqa: E402
from adapters.sleeve_prices import (  # noqa: E402
    is_parking_sleeve_enabled,
    parking_sleeve_config,
    parking_sleeve_price_tickers,
)

_ENABLED = {"sleeve": {"enabled": True}}


class TestParkingSleevePriceTickers:
    def test_absent_section_is_empty(self):
        assert parking_sleeve_price_tickers({}) == []
        assert parking_sleeve_price_tickers({"watchlist": ["AAPL"]}) == []

    def test_disabled_is_empty(self):
        cfg = {"sleeve": {"enabled": False, "spy_symbol": "SPY",
                          "sgov_symbol": "SGOV"}}
        assert parking_sleeve_price_tickers(cfg) == []

    def test_malformed_section_is_empty(self):
        assert parking_sleeve_price_tickers({"sleeve": "yes"}) == []
        assert parking_sleeve_price_tickers({"sleeve": None}) == []
        assert parking_sleeve_price_tickers(None) == []

    def test_enabled_defaults_to_spy_and_sgov(self):
        # Defaults mirror renquant-pipeline task_parking_sleeve.py reads
        # and the strategy-104 #39 config pins.
        assert parking_sleeve_price_tickers(_ENABLED) == ["SPY", "SGOV"]

    def test_enabled_custom_symbols_normalized(self):
        cfg = {"sleeve": {"enabled": True, "spy_symbol": " ivv ",
                          "sgov_symbol": "bil"}}
        assert parking_sleeve_price_tickers(cfg) == ["IVV", "BIL"]

    def test_blank_symbols_fall_back_to_defaults(self):
        cfg = {"sleeve": {"enabled": True, "spy_symbol": "  ",
                          "sgov_symbol": ""}}
        assert parking_sleeve_price_tickers(cfg) == ["SPY", "SGOV"]

    def test_identical_legs_deduped(self):
        cfg = {"sleeve": {"enabled": True, "spy_symbol": "SGOV",
                          "sgov_symbol": "SGOV"}}
        assert parking_sleeve_price_tickers(cfg) == ["SGOV"]

    def test_ctx_like_object_with_config_attr(self):
        ctx = NS(config={"sleeve": {"enabled": True}})
        assert parking_sleeve_price_tickers(ctx) == ["SPY", "SGOV"]
        assert is_parking_sleeve_enabled(ctx) is True

    def test_enabled_predicate_and_config_accessor(self):
        assert is_parking_sleeve_enabled({}) is False
        assert is_parking_sleeve_enabled(_ENABLED) is True
        assert parking_sleeve_config(_ENABLED) == {"enabled": True}
        assert parking_sleeve_config({"sleeve": 3}) == {}


class TestSimContextCoverage:
    """context_price_tickers is the sim pricing-universe choke point."""

    _BASE = {"watchlist": ["AAPL", "MSFT"], "benchmark": "SPY"}

    def _tickers(self, config):
        return context_price_tickers(config=config, models={},
                                     sector_etf_map={}, holdings={})

    def test_flag_off_is_byte_inert(self):
        without_section = self._tickers(dict(self._BASE))
        disabled = self._tickers(
            {**self._BASE,
             "sleeve": {"enabled": False, "spy_symbol": "SPY",
                        "sgov_symbol": "SGOV"}})
        assert disabled == without_section
        assert "SGOV" not in disabled

    def test_flag_on_includes_both_legs(self):
        out = self._tickers({**self._BASE, "sleeve": {"enabled": True}})
        assert "SPY" in out and "SGOV" in out
        assert out == list(dict.fromkeys(out))  # still deduped

    def test_flag_on_dedupes_against_benchmark_and_watchlist(self):
        out = self._tickers({**self._BASE, "sleeve": {"enabled": True}})
        assert out.count("SPY") == 1


class TestAdapterCoverageParity:
    """LEAN main.py / lean.py / runner.py wiring — via source inspection.

    Same convention as TestAdapterParity (test_universe_alignment.py):
    LEAN main.py imports AlgorithmImports and cannot be imported here, and
    RunnerAdapter.make_context needs a live broker. Behavioral gating
    (flag-off => [], flag-on => both legs) is covered above; these pins
    prove every price path routes through that single implementation.
    """

    def _read(self, rel: str) -> str:
        return (_REPO_ROOT / rel).read_text()

    def test_lean_main_subscribes_sleeve_legs_conditionally(self):
        src = self._read("backtesting/renquant_104/main.py")
        assert "from adapters.sleeve_prices import parking_sleeve_price_tickers" in src
        assert "for parking_ticker in parking_sleeve_price_tickers(CONFIG):" in src
        # Mirrors the benchmark_sleeve dedup guards.
        assert "parking_ticker != self._benchmark" in src
        assert "parking_ticker not in self.symbols" in src
        assert "parking_ticker not in self._sector_etf_symbols" in src

    def test_lean_adapter_fetches_sleeve_legs(self):
        src = self._read("backtesting/renquant_104/adapters/lean.py")
        assert "all_tickers.extend(parking_sleeve_price_tickers(config))" in src

    def test_live_runner_fetches_sleeve_legs(self):
        src = self._read("backtesting/renquant_104/adapters/runner.py")
        assert "extra_symbols.extend(parking_sleeve_price_tickers(config))" in src

    def test_sim_universe_includes_sleeve_legs(self):
        src = self._read("backtesting/renquant_104/adapters/sim_price.py")
        assert "tickers.extend(parking_sleeve_price_tickers(config))" in src
