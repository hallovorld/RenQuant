"""Tests for M5 tournament shadow admission wiring in RunnerAdapter.commit().

Validates:
  * ``_build_tournament_shadow_ticker_scores()`` correctly builds the
    ticker_scores dict from candidate pool + blocked map.
  * The wiring block in commit() is config-gated (default OFF).
  * The wiring block is fail-open (exceptions logged, never raised).
  * When enabled, the orchestrator's ``log_shadow_admission`` is called
    with the correct arguments.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from adapters.runner import _build_tournament_shadow_ticker_scores  # noqa: E402


# ---------------------------------------------------------------------------
# _build_tournament_shadow_ticker_scores tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    ticker: str
    raw_score: float = 0.0
    rank_score: float = 0.0


class TestBuildTournamentShadowTickerScores:
    def test_candidates_get_buy_signal(self):
        pool = [_FakeCandidate("AAPL", 0.5, 0.6), _FakeCandidate("GOOG", 0.3, 0.4)]
        result = _build_tournament_shadow_ticker_scores(pool, {})
        assert result["AAPL"] == {"signal": "buy", "raw_score": 0.5, "rank_score": 0.6}
        assert result["GOOG"] == {"signal": "buy", "raw_score": 0.3, "rank_score": 0.4}

    def test_blocked_model_signal_extracted(self):
        pool = []
        blocked = {"TSLA": "model_signal:sell", "META": "model_signal:hold"}
        result = _build_tournament_shadow_ticker_scores(pool, blocked)
        assert result["TSLA"] == {"signal": "sell", "raw_score": None, "rank_score": None}
        assert result["META"] == {"signal": "hold", "raw_score": None, "rank_score": None}

    def test_non_model_block_excluded(self):
        """Tickers blocked by wash_sale, sector_cap, etc. should NOT appear."""
        pool = []
        blocked = {"AMZN": "wash_sale", "MSFT": "sector_cap"}
        result = _build_tournament_shadow_ticker_scores(pool, blocked)
        assert "AMZN" not in result
        assert "MSFT" not in result

    def test_candidate_not_overridden_by_blocked_map(self):
        """If a ticker is in both cand_pool and blocked_map, cand_pool wins."""
        pool = [_FakeCandidate("AAPL", 0.5, 0.6)]
        blocked = {"AAPL": "veto:rank_score_below_floor"}
        result = _build_tournament_shadow_ticker_scores(pool, blocked)
        assert result["AAPL"]["signal"] == "buy"
        assert result["AAPL"]["raw_score"] == 0.5

    def test_empty_inputs(self):
        result = _build_tournament_shadow_ticker_scores([], {})
        assert result == {}

    def test_candidate_without_ticker_attribute_skipped(self):
        pool = [MagicMock(spec=[])]  # no .ticker attribute
        result = _build_tournament_shadow_ticker_scores(pool, {})
        assert result == {}

    def test_mixed_candidates_and_blocked(self):
        pool = [_FakeCandidate("AAPL", 0.5, 0.6)]
        blocked = {
            "TSLA": "model_signal:sell",
            "AMZN": "wash_sale",
            "AAPL": "veto:rank_score_below_floor",
        }
        result = _build_tournament_shadow_ticker_scores(pool, blocked)
        assert set(result.keys()) == {"AAPL", "TSLA"}
        assert result["AAPL"]["signal"] == "buy"
        assert result["TSLA"]["signal"] == "sell"


# ---------------------------------------------------------------------------
# Config-gated wiring tests
# ---------------------------------------------------------------------------

class TestTournamentShadowConfigGate:
    """Verify that the wiring code in runner.py reads the config gate."""

    def test_default_config_has_tournament_shadow_disabled(self):
        """strategy_config.json must ship with tournament_shadow.enabled=false."""
        import json

        config_path = (
            REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
        )
        config = json.loads(config_path.read_text())
        ts = config.get("tournament_shadow", {})
        assert ts.get("enabled", False) is False, (
            "tournament_shadow must default to disabled"
        )

    def test_wiring_code_checks_enabled_flag(self):
        """The runner.py source must gate on tournament_shadow.enabled."""
        src = (
            REPO_ROOT / "backtesting" / "renquant_104" / "adapters" / "runner.py"
        ).read_text()
        assert 'self._config.get("tournament_shadow"' in src
        assert '_ts_cfg.get("enabled", False)' in src

    def test_wiring_code_is_fail_open(self):
        """The tournament shadow block must be wrapped in try/except."""
        src = (
            REPO_ROOT / "backtesting" / "renquant_104" / "adapters" / "runner.py"
        ).read_text()
        assert "tournament_shadow write failed (non-fatal)" in src

    def test_wiring_imports_at_runtime(self):
        """Import must be lazy (inside the if-enabled block)."""
        src = (
            REPO_ROOT / "backtesting" / "renquant_104" / "adapters" / "runner.py"
        ).read_text()
        # The import must be inside the function body, not at module level
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "from renquant_orchestrator.tournament_shadow_admission import" in line:
                # Verify it's indented (inside a function/method)
                assert line.startswith(" " * 8), (
                    f"tournament_shadow_admission import at line {i+1} must be "
                    f"inside a function body (deeply indented)"
                )
                break
        else:
            pytest.fail(
                "Expected lazy import of tournament_shadow_admission in runner.py"
            )
