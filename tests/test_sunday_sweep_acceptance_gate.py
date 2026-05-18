"""Tests for the 2026-05-17 Sunday-sweep acceptance gate.

Pre-fix: sunday_panel_sweep.py wrote any newly-trained artifact straight to
prod, no quality check. Today's incident: xgboost backend trained a 21-feat
stub panel-LTR + ngboost val_IC=-0.0165, and the .bak rollback chain
couldn't recover because there was no pre-sweep backup. prod sat with
val_IC=-0.0165 for ~3h until detected.

Fix invariants:
  • Pre-sweep state is backed up FIRST (before any backend runs)
  • Each backend's post-train artifact passes _gate_check_vs_baseline
    before being marked OK
  • Best gate-passing backend wins (not "first in list")
  • If 0/N backends pass → restore pre-sweep state (never write garbage
    to prod)

Tests are source-substring level (consistent with test_runner_state_fixes.py
style) because the full sweep is heavy (Docker LEAN + 3 trainings).
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "scripts/sunday_panel_sweep.py").read_text()


class TestPreSweepBackup:
    def test_pre_sweep_backup_called(self):
        assert '_backup_artifacts(strategy, "pre-sweep")' in SRC, \
            "must backup pre-sweep state before any backend trains"

    def test_pre_sweep_backup_before_loop(self):
        idx_backup = SRC.index('_backup_artifacts(strategy, "pre-sweep")')
        idx_loop = SRC.index("for backend in backends_to_run")
        assert idx_backup < idx_loop, \
            "pre-sweep backup must precede the backend training loop"


class TestAcceptanceGate:
    def test_gate_function_exists(self):
        assert "def _gate_check_vs_baseline(" in SRC

    def test_gate_called_per_backend(self):
        assert "_gate_check_vs_baseline(\n                    strategy, baseline_metrics\n                )" in SRC \
            or "_gate_check_vs_baseline(strategy, baseline_metrics)" in SRC

    def test_hard_checks_present(self):
        # H1 pool_ic exists; H2 ≥0; H3 drop ≤2pp; H4 scorer ≥0
        assert "H1 pool_ic" in SRC
        assert "H2 pool_ic=" in SRC
        assert "H3 pool_ic dropped" in SRC
        assert "H4 scorer_oos_mean_ic" in SRC

    def test_rejection_marks_backend_not_ok(self):
        assert 'GATE_REJECTED' in SRC
        assert 'ok = False' in SRC


class TestBestBackendSelection:
    def test_picks_best_by_oos_ic(self):
        assert 'max(passing, key=lambda r: r.get("scorer_oos_mean_ic")' in SRC

    def test_restores_pre_sweep_when_all_fail(self):
        assert '_restore_artifacts(strategy, "pre-sweep")' in SRC, \
            "if 0/N backends pass, pre-sweep state must be restored"
        assert "0/%d backends passed acceptance gates" in SRC


class TestRegressionGuard:
    def test_acceptance_gate_uses_pool_ic_baseline_compare(self):
        """The 2026-05-17 incident: pool_ic dropped from +0.094 to negative.
        Gate must catch this baseline regression."""
        assert "base_pool" in SRC
        assert "drop > 0.02" in SRC, "must catch 2pp pool_ic regression"
