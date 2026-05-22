"""Tests for monthly_calibrator_refresh.sh 2026-05-17 acceptance gate.

Pre-fix: fit_panel_calibrator.py wrote a new calibrator straight to prod
with only a smoke-test check after. If the new calibrator's pool_ic
regressed (e.g. dropped from +0.094 to +0.01 or to negative), nothing
caught it — same bug class as today's Sunday-sweep NGB val_IC=-0.0165
incident.

Fix invariants:
  • Pre-refit backup → rollback target always exists
  • H1 smoke-test gate after refit (existing)
  • H2a non-collapse: n_unique_prob_y >= 10 (was display-only)
  • H2b IC regression: pool_ic must not drop > 0.02 vs baseline
  • Gate failure → rollback to backup + smoke-test the rollback +
    appropriate ntfy (REJECT vs CRITICAL if rollback also breaks)
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "scripts/monthly_calibrator_refresh.sh").read_text()
ATOMIC_BACKUP = 'cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"'
ATOMIC_ROLLBACK = 'cp "$ROLLBACK_CAL" "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"'


class TestPreRefitBackup:
    def test_backup_called_before_fit(self):
        assert ATOMIC_BACKUP in SRC
        idx_backup = SRC.index(ATOMIC_BACKUP)
        # Locate the actual python invocation, not the comment mention
        idx_fit = SRC.index('"$PYTHON" scripts/fit_panel_calibrator.py')
        assert idx_backup < idx_fit, "backup must precede the fit invocation"

    def test_baseline_metrics_captured(self):
        assert 'BASELINE_POOL_IC=' in SRC
        assert 'BASELINE_N_UNIQUE=' in SRC


class TestAcceptanceGate:
    def test_smoke_failure_rolls_back(self):
        # Find the smoke-failure branch
        smoke_fail_idx = SRC.index("Post-fit smoke test FAILED")
        nearby = SRC[smoke_fail_idx: smoke_fail_idx + 400]
        assert ATOMIC_ROLLBACK in nearby, \
            "smoke-fail branch must rollback to baseline"

    def test_non_collapse_hard_gate(self):
        assert "n_unique_prob_y={n_uniq} < 10 (collapsed)" in SRC

    def test_ic_regression_2pp_threshold(self):
        # Match the threshold check in the Python heredoc
        assert "if drop > 0.02:" in SRC, \
            "pool_ic > 2pp regression must trigger reject"

    def test_gate_failure_rolls_back(self):
        gate_fail_idx = SRC.index("ACCEPTANCE GATE FAILED")
        nearby = SRC[gate_fail_idx: gate_fail_idx + 800]
        assert ATOMIC_ROLLBACK in nearby
        assert "MONTHLY-REJECT" in nearby

    def test_rollback_smoke_test_catches_double_failure(self):
        nearby = SRC[SRC.index("ACCEPTANCE GATE FAILED"):]
        assert "MONTHLY-CRITICAL" in nearby, \
            "if rollback breaks smoke too, ntfy CRITICAL for operator action"


class TestFixTag:
    def test_2026_05_17_marker(self):
        assert "2026-05-17 ACCEPTANCE GATE" in SRC
        assert "Sunday-sweep corruption" in SRC, \
            "must reference the incident that motivated the fix"
