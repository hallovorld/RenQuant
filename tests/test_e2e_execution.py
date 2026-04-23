"""End-to-end execution tests — notebook + LEAN.

Per user spec (April 22): verify both the notebook and LEAN backtest run
from beginning to end without raising. These are smoke tests — they do
NOT validate numerical outputs, just that the full execution pipeline
works. Validating numbers is the job of unit tests + sim invariants.

Gated behind environment variables because both are slow:
  RENQUANT_E2E_NOTEBOOK=1  pytest tests/test_e2e_execution.py::TestNotebookE2E
  RENQUANT_E2E_LEAN=1      pytest tests/test_e2e_execution.py::TestLeanE2E

Budget:
  Notebook e2e : ~10 min (training + sim + charts)
  LEAN e2e     : ~20-30 min (full Docker backtest)

These should run in nightly CI, not on every unit-test pass. The
fast-CI layer is still the 974+ unit tests.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"


# ── Notebook e2e ─────────────────────────────────────────────────────────────

_RUN_NOTEBOOK = os.environ.get("RENQUANT_E2E_NOTEBOOK") == "1"
_RUN_LEAN     = os.environ.get("RENQUANT_E2E_LEAN")     == "1"


@pytest.mark.skipif(not _RUN_NOTEBOOK,
                    reason="Set RENQUANT_E2E_NOTEBOOK=1 to run")
class TestNotebookE2E:
    """Headlessly execute renquant_104.ipynb top-to-bottom.

    The assertions are intentionally loose — pass if no cell raises, and
    at least one expected chart artifact lands on disk.
    """

    NOTEBOOK = _STRATEGY_DIR / "renquant_104.ipynb"
    EXPECTED_CHARTS = [
        # Keys in order they appear in the reorganised notebook (Section 2-4)
        _STRATEGY_DIR / "img" / "regime_analysis.png",
        _STRATEGY_DIR / "img" / "model_sharpe_summary.png",
        _STRATEGY_DIR / "img" / "correlation_heatmap.png",
        _STRATEGY_DIR / "img" / "portfolio_simulation.png",
        _STRATEGY_DIR / "img" / "trade_analysis.png",
        _STRATEGY_DIR / "img" / "holdings_over_time.png",
        _STRATEGY_DIR / "img" / "per_symbol_oos.png",
    ]

    def test_notebook_executes_all_cells_without_raising(self, tmp_path):
        out = tmp_path / "executed.ipynb"
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.kernel_name=python3",
                "--output", str(out),
                str(self.NOTEBOOK),
            ],
            capture_output=True, text=True, timeout=1800,
        )
        assert result.returncode == 0, (
            f"nbconvert exited {result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    def test_notebook_produces_expected_charts(self):
        """After a successful notebook run, key chart files should exist
        and have nonzero size."""
        missing = [p for p in self.EXPECTED_CHARTS
                   if not p.exists() or p.stat().st_size == 0]
        assert not missing, f"missing/empty chart outputs: {missing}"


# ── LEAN e2e ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _RUN_LEAN, reason="Set RENQUANT_E2E_LEAN=1 to run")
class TestLeanE2E:
    """Run `lean backtest .` against the renquant_104 strategy.

    Requires Docker + `lean` CLI installed (see doc/setup.md). The test
    reads the new backtest-run directory to confirm LEAN actually
    finished — a non-zero exit code or a missing output directory fails.
    """

    STRATEGY_DIR = _STRATEGY_DIR
    BACKTESTS_DIR = _STRATEGY_DIR / "backtests"

    def _baseline_backtest_count(self) -> int:
        if not self.BACKTESTS_DIR.exists():
            return 0
        return sum(1 for p in self.BACKTESTS_DIR.iterdir() if p.is_dir())

    def test_lean_backtest_succeeds(self):
        before = self._baseline_backtest_count()
        result = subprocess.run(
            ["lean", "backtest", "."],
            cwd=str(self.STRATEGY_DIR),
            capture_output=True, text=True, timeout=3600,
        )
        assert result.returncode == 0, (
            f"lean backtest exited {result.returncode}\n"
            f"stdout: {result.stdout[-3000:]}\n"
            f"stderr: {result.stderr[-3000:]}"
        )
        after = self._baseline_backtest_count()
        assert after > before, (
            "LEAN ran but no new backtest directory appeared — did the "
            "engine actually finish?"
        )

    def test_lean_backtest_produced_result_json(self):
        """The latest backtest directory should have a *.json result."""
        if not self.BACKTESTS_DIR.exists():
            pytest.fail("backtests/ directory is missing — LEAN never ran?")
        dirs = sorted(
            (p for p in self.BACKTESTS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not dirs:
            pytest.fail("no backtest subdirs")
        latest = dirs[0]
        result_json = list(latest.glob("*-summary.json")) + list(latest.glob("*.json"))
        assert result_json, f"no result JSON in {latest}"


# ── Fast-CI smoke test — verify the infrastructure works ────────────────────

class TestE2EFrameworkPresent:
    """These never run the expensive paths — just verify the entry points
    are importable / CLI-reachable, so a broken jupyter / lean install
    surfaces on every run, not only when someone flips the env var."""

    def test_nbconvert_is_installed(self):
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"jupyter nbconvert not installed? {result.stderr}"
        )

    def test_notebook_file_exists(self):
        assert (_STRATEGY_DIR / "renquant_104.ipynb").exists()

    def test_lean_cli_available_or_gracefully_skipped(self):
        """If lean CLI isn't installed, the lean-e2e class should SKIP cleanly
        when invoked, not crash. Here we just surface whether lean is available."""
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                pytest.skip(f"lean CLI returned non-zero: {result.stderr or result.stdout}")
        except FileNotFoundError:
            pytest.skip("lean CLI not installed (brew install or pip install lean)")
