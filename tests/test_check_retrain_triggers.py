"""Unit tests for scripts/check_retrain_triggers.py.

Can't easily test the real yfinance fetch in CI — use monkeypatch on
_pct_change to inject synthetic pct moves and assert the right trigger
tags + exit codes come out.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestTriggerLogic:
    def _run(self, patches: dict, extra_args: list | None = None):
        """Invoke the script's main() with patched _pct_change + captured stdout."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_check_retrain",
            REPO_ROOT / "scripts" / "check_retrain_triggers.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Prepare sys.argv
        argv_backup = sys.argv
        sys.argv = ["check_retrain_triggers.py", *(extra_args or [])]
        try:
            with patch.object(mod, "_pct_change", side_effect=lambda s: patches.get(s)):
                try:
                    rc = mod.main()
                except SystemExit as e:
                    rc = e.code
        finally:
            sys.argv = argv_backup
        return rc

    def test_no_anomaly_exits_zero(self, capsys):
        rc = self._run({"^SPY": 0.005, "SPY": 0.005, "^VIX": 0.01})
        assert rc == 0

    def test_spy_over_threshold_exits_one(self, capsys):
        rc = self._run({"^SPY": 0.025, "SPY": 0.025, "^VIX": 0.01})
        assert rc == 1
        out = capsys.readouterr().out
        assert "anomaly_spy_2pct" in out

    def test_spy_negative_over_threshold_also_fires(self, capsys):
        rc = self._run({"^SPY": -0.03, "SPY": -0.03, "^VIX": 0.01})
        assert rc == 1
        assert "anomaly_spy_2pct" in capsys.readouterr().out

    def test_vix_over_threshold(self, capsys):
        rc = self._run({"^SPY": 0.005, "SPY": 0.005, "^VIX": 0.08})
        assert rc == 1
        assert "anomaly_vix_5pct" in capsys.readouterr().out

    def test_both_fire_both_tags_printed(self, capsys):
        rc = self._run({"^SPY": 0.03, "SPY": 0.03, "^VIX": 0.08})
        assert rc == 1
        out = capsys.readouterr().out
        assert "anomaly_spy_2pct" in out
        assert "anomaly_vix_5pct" in out

    def test_dry_run_always_exits_zero(self, capsys):
        rc = self._run({"^SPY": 0.05, "SPY": 0.05, "^VIX": 0.10},
                        extra_args=["--dry-run"])
        assert rc == 0

    def test_custom_threshold_produces_custom_tag(self, capsys):
        rc = self._run({"^SPY": 0.02, "SPY": 0.02, "^VIX": 0.01},
                        extra_args=["--spy-pct", "0.015"])
        assert rc == 1
        out = capsys.readouterr().out
        # 0.015 → 15 bp × 1000 = 15 → tag suffix
        assert "anomaly_spy_15bp" in out

    def test_no_data_for_ticker_no_fire(self, capsys):
        """If yfinance returns None for a ticker, that check is skipped."""
        rc = self._run({"^SPY": None, "SPY": None, "^VIX": 0.01})
        assert rc == 0
