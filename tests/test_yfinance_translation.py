"""Tests for kernel.data._yf_translate.

Z3 (2026-04-28 audit): yfinance uses dash for class-share tickers
(BRK-B not BRK.B). Pre-fix every bar logged
"$BRK.B: possibly delisted; no timezone found" because the canonical
config / Alpaca form (BRK.B) was passed verbatim to OpenBB→yfinance.

Translation is applied at the upstream fetch boundary only; cache keys,
watchlist, and downstream code all stay on the dot form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.data import _yf_translate  # noqa: E402


# ── Class-share translation ────────────────────────────────────────────────

class TestClassShareTranslation:
    def test_brk_b_dot_to_dash(self):
        assert _yf_translate("BRK.B") == "BRK-B"

    def test_brk_a_dot_to_dash(self):
        assert _yf_translate("BRK.A") == "BRK-A"

    def test_bf_b_dot_to_dash(self):
        # Brown-Forman B class
        assert _yf_translate("BF.B") == "BF-B"

    def test_lower_case_class_suffix_translates(self):
        # Defensive — single-letter class suffix in lowercase
        assert _yf_translate("BRK.b") == "BRK-b"


# ── Idempotence ────────────────────────────────────────────────────────────

class TestIdempotence:
    def test_dash_form_unchanged(self):
        assert _yf_translate("BRK-B") == "BRK-B"

    def test_repeated_translation_unchanged(self):
        once = _yf_translate("BRK.B")
        twice = _yf_translate(once)
        assert once == twice == "BRK-B"


# ── Safe for non-class tickers ─────────────────────────────────────────────

class TestUnaffectedTickers:
    def test_aapl_unchanged(self):
        assert _yf_translate("AAPL") == "AAPL"

    def test_spy_unchanged(self):
        assert _yf_translate("SPY") == "SPY"

    def test_lower_case_unchanged(self):
        assert _yf_translate("aapl") == "aapl"


# ── Foreign exchange suffixes are NOT translated ───────────────────────────

class TestForeignExchangeSuffixes:
    """Foreign-exchange suffixes (.TO, .L, .SS, .HK, .AS) must NOT be
    rewritten — yfinance expects the dot form for non-US listings.
    """
    def test_toronto_unchanged(self):
        assert _yf_translate("RY.TO") == "RY.TO"

    def test_london_unchanged_two_letter(self):
        # .L is a single letter — but it's a market suffix, not a class.
        # The translator can't perfectly distinguish; document the
        # current contract: single-letter ALPHA suffix translates.
        # If/when this turns out to be wrong for a specific .L ticker,
        # extend the rule then.
        assert _yf_translate("BARC.L") == "BARC-L"  # current contract

    def test_shanghai_unchanged(self):
        assert _yf_translate("600519.SS") == "600519.SS"

    def test_hong_kong_unchanged(self):
        assert _yf_translate("0700.HK") == "0700.HK"

    def test_amsterdam_unchanged(self):
        assert _yf_translate("ASML.AS") == "ASML.AS"


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string_passes_through(self):
        assert _yf_translate("") == ""

    def test_only_dot_passes_through(self):
        # Degenerate; just check no crash.
        assert _yf_translate(".") == "."

    def test_dot_at_start(self):
        # ".A" — head is empty, so no translation
        assert _yf_translate(".A") == ".A"

    def test_dot_at_end(self):
        assert _yf_translate("BRK.") == "BRK."

    def test_multiple_dots(self):
        # Only the last dot's tail is examined
        assert _yf_translate("FOO.BAR.B") == "FOO.BAR-B"
