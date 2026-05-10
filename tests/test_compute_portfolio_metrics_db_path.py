"""Regression test for compute_portfolio_metrics.py DB path fix (2026-05-09).

Bug: scripts/compute_portfolio_metrics.py:122-123 defaulted to data/runs.db
for --source live. After the broker-isolation switch (kernel/state_paths.py),
live data is in data/runs.{broker}.db. Pre-fix the script silently warned
'No rows found' on every cron run since the migration, leaving
portfolio_daily_metrics empty and the dashboard 12+ days stale.

Fix: when --db is omitted and --source live, route through runs_db_path()
helper using --broker (default 'alpaca').

This test pins the invariant: with no --db, the script MUST resolve to a
broker-tagged path that the dashboard reads from — not the legacy untagged path.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


def _load_script_module():
    """Load scripts/compute_portfolio_metrics.py without running main()."""
    spec = importlib.util.spec_from_file_location(
        "compute_portfolio_metrics",
        REPO / "scripts" / "compute_portfolio_metrics.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_portfolio_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestBrokerTaggedDBResolution:

    def test_live_default_broker_resolves_alpaca_db(self):
        """Default --source live + default --broker alpaca → data/runs.alpaca.db."""
        from kernel.state_paths import runs_db_path
        resolved = runs_db_path("data/runs.db", "alpaca")
        assert resolved.name == "runs.alpaca.db", \
            f"Expected runs.alpaca.db, got {resolved.name}"

    def test_live_broker_paper_resolves_paper_db(self):
        from kernel.state_paths import runs_db_path
        resolved = runs_db_path("data/runs.db", "paper")
        assert resolved.name == "runs.paper.db"

    def test_live_default_NOT_legacy_runs_db(self):
        """Critical invariant: pre-fix bug returned data/runs.db (0 live rows
        after broker isolation). Verify we no longer hit that path."""
        from kernel.state_paths import runs_db_path
        # The legacy data/runs.db path is the BUG path. Any broker tag
        # other than 'unknown' (default for None) must NOT produce data/runs.db.
        for broker in ["alpaca", "alpaca_paper", "paper", "ibkr"]:
            resolved = runs_db_path("data/runs.db", broker)
            assert resolved.name != "runs.db", \
                f"broker={broker} resolved to legacy data/runs.db — bug regressed"

    def test_idempotent_against_already_tagged_path(self):
        """If somebody passes data/runs.alpaca.db + --broker alpaca, no double tag."""
        from kernel.state_paths import runs_db_path
        resolved = runs_db_path("data/runs.alpaca.db", "alpaca")
        assert resolved.name == "runs.alpaca.db", \
            "double-tagging regressed — got " + resolved.name


class TestScriptArgumentParsing:
    """Verify the script exposes --broker argument with correct default."""

    def test_broker_arg_present_with_alpaca_default(self):
        mod = _load_script_module()
        # Reconstruct just the arg parser without running main()
        # The script's main() builds the parser inline; verify by reading source.
        src = (REPO / "scripts" / "compute_portfolio_metrics.py").read_text()
        assert '"--broker"' in src, \
            "Fix #A regressed: --broker argument removed from compute_portfolio_metrics.py"
        assert 'default="alpaca"' in src, \
            "Fix #A regressed: --broker default no longer 'alpaca'"

    def test_db_default_routes_through_runs_db_path(self):
        """The script's --db default branch must call runs_db_path(), not
        hardcode 'data/runs.db'."""
        src = (REPO / "scripts" / "compute_portfolio_metrics.py").read_text()
        # Must reference the helper
        assert "runs_db_path" in src, \
            "Fix #A regressed: --db live default no longer uses runs_db_path()"
        # Must NOT have the old hardcoded buggy line
        assert 'else "data/runs.db"' not in src, \
            "Fix #A regressed: legacy hardcoded data/runs.db default returned"


class TestEndToEndDBExists:
    """Smoke test that the DB the script will resolve to actually exists
    on disk (else cron warns 'No rows found' and dashboard stays stale)."""

    def test_alpaca_db_exists_or_xfail(self):
        alpaca_db = REPO / "data" / "runs.alpaca.db"
        if not alpaca_db.exists():
            pytest.xfail("runs.alpaca.db not present — live cron not yet "
                         "established. Test will turn green once daily_104.sh "
                         "writes its first row.")
        assert alpaca_db.stat().st_size > 0
