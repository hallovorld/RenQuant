"""Tests for LEAN data normalization mode (Track C7, 2026-05-10).

Pins the explicit `DataNormalizationMode.Adjusted` choice in
backtesting/renquant_104/main.py:Initialize so that LEAN's price series
matches yfinance (sim) on split + dividend handling.

Per CLAUDE.md §5.13.2, we read main.py source directly and grep for the
production wiring — fixture-only tests would not catch the regression
where the line is silently removed.

Per CLAUDE.md §5.13.3, TestDataNormalizationModeSet is the audit
regression guard pinning the invariant: if anyone removes the line,
this test fires red.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_MAIN_PY = (
    Path(__file__).resolve().parent.parent
    / "backtesting"
    / "renquant_104"
    / "main.py"
)


@pytest.fixture(scope="module")
def main_py_source() -> str:
    assert _MAIN_PY.exists(), f"main.py not found at {_MAIN_PY}"
    return _MAIN_PY.read_text()


# ─────────────────────────────────────────────────────────────────────────────
# Wiring tests — prove the call exists in production source
# ─────────────────────────────────────────────────────────────────────────────


class TestDataNormalizationModeWired:
    """§5.13.2 — verify the call is actually present in main.py source."""

    def test_main_py_sets_data_normalization_mode(self, main_py_source: str) -> None:
        """main.py must explicitly set a data normalization mode.

        Either via per-AddEquity `.SetDataNormalizationMode(...)` or via
        `UniverseSettings.DataNormalizationMode = ...` at the top of
        Initialize. The default LEAN behavior (implicit / data-dependent)
        is forbidden by the 2026-05-10 execution audit.
        """
        has_universe_setting = (
            "UniverseSettings.DataNormalizationMode" in main_py_source
        )
        has_per_security_call = "SetDataNormalizationMode(" in main_py_source
        assert has_universe_setting or has_per_security_call, (
            "main.py must set DataNormalizationMode explicitly via "
            "UniverseSettings.DataNormalizationMode=... or per-security "
            ".SetDataNormalizationMode(...). Neither found."
        )

    def test_normalization_mode_is_adjusted(self, main_py_source: str) -> None:
        """The selected mode must be Adjusted — matches yfinance default.

        Forbids accidental drift to Raw (price unadjusted, sim divergence
        on every split/dividend) or SplitAdjusted (still diverges on
        dividends, AAPL ≈ 0.6%/yr).
        """
        assert "DataNormalizationMode.Adjusted" in main_py_source, (
            "Expected DataNormalizationMode.Adjusted in main.py — "
            "this is the only mode that matches yfinance (sim) defaults."
        )

    def test_normalization_mode_not_raw_or_splitadjusted(
        self, main_py_source: str
    ) -> None:
        """Reject Raw / SplitAdjusted explicitly to catch silent demotions."""
        # Note: these are negative-lookup checks — they pass on absence.
        # If a maintainer ever switches to Raw or SplitAdjusted, this
        # test fires red and forces a deliberate review.
        assert "DataNormalizationMode.Raw" not in main_py_source, (
            "DataNormalizationMode.Raw breaks LEAN/sim parity on "
            "split-adjusted price series."
        )
        assert "DataNormalizationMode.SplitAdjusted" not in main_py_source, (
            "DataNormalizationMode.SplitAdjusted still diverges from "
            "yfinance on dividend-paying stocks (e.g. AAPL ~0.6%/y)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Audit regression guard (§5.13.3)
# ─────────────────────────────────────────────────────────────────────────────


class TestDataNormalizationModeSet:
    """AUDIT REGRESSION GUARD — Track C7, 2026-05-10.

    Class-of-bug invariant: LEAN data normalization must be explicitly
    set to Adjusted in main.py. Without this, LEAN's default depends on
    data source and may diverge from sim on dividend-paying stocks.

    If anyone removes the explicit DataNormalizationMode call, this
    test fires red.
    """

    def test_explicit_normalization_call_present(
        self, main_py_source: str
    ) -> None:
        """Pin: explicit Adjusted setting MUST be present in Initialize."""
        # Combined invariant check — at least one of the two valid forms
        # AND it must be Adjusted (not Raw/SplitAdjusted).
        adjusted_universe = (
            "UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted"
            in main_py_source
        )
        adjusted_per_security = (
            ".SetDataNormalizationMode(DataNormalizationMode.Adjusted)"
            in main_py_source
        )
        assert adjusted_universe or adjusted_per_security, (
            "AUDIT REGRESSION GUARD (Track C7, 2026-05-10): "
            "main.py:Initialize must set DataNormalizationMode.Adjusted "
            "via either UniverseSettings or per-security AddEquity result. "
            "Removing this line breaks LEAN/sim parity — restore it."
        )

    def test_parity_invariant_documented(self, main_py_source: str) -> None:
        """Module docstring must document the LEAN/sim parity invariant.

        The trade-off (no separate dividend events; price continuity
        instead) is load-bearing context for future maintainers.
        """
        assert "LEAN data normalization: Adjusted" in main_py_source, (
            "main.py module docstring must document the Adjusted choice "
            "and its sim-parity rationale."
        )
        assert "yfinance" in main_py_source, (
            "Docstring must reference yfinance as the sim data source "
            "anchoring the parity invariant."
        )
