"""Tests for monthly_calibrator_refresh.sh's acceptance gate.

2026-05-17 acceptance gate: Pre-fix, fit_panel_calibrator.py wrote a new
calibrator straight to prod with only a smoke-test check after. If the new
calibrator's pool_ic regressed (e.g. dropped from +0.094 to +0.01 or to
negative), nothing caught it — same bug class as the Sunday-sweep NGB
val_IC=-0.0165 incident.

2026-07-01 REVIEW FIX ROUND 2 (PR #425 CHANGES_REQUESTED, Codex): the
2026-05-17 fix above still wrote fit_calibrator's OUTPUT directly to
PROD_CAL (the live production path) BEFORE running any of the validation
gates — so the live runtime could read an unvalidated/mismatched
calibrator during the fit-to-validation window, and a first-ever fit (no
prior calibrator) that failed validation left the REJECTED artifact
sitting at PROD_CAL. "Roll back after exposure" is not the same as "never
publish an unvalidated artifact". Fixed by staging: fit_calibrator() now
writes to a unique $STAGING_CAL path; every gate evaluates $STAGING_CAL;
PROD_CAL is only touched once, atomically, via
scripts/monthly_calibrator_atomic_swap.py AFTER every gate passes. Any
gate failure quarantines staging (never touches PROD_CAL) — see
tests/test_monthly_calibrator_atomic_swap.py for the integration-level
(byte-for-byte, concurrent-read) proof of that invariant. This file keeps
the string-pattern-level regression guards over the shell script text.

Fix invariants:
  • Pre-refit archival backup of the (pre-run) PROD_CAL → dated snapshot,
    for operator reference only (NOT used for automated rollback — see
    "no expose-then-rollback" below)
  • fit_calibrator() writes to $STAGING_CAL, never $PROD_CAL directly
  • H0 staged-calibrator smoke check (load + map two distinct scores)
  • H2a non-collapse: n_unique_prob_y >= 10 (was display-only)
  • H2b IC regression: pool_ic must not drop > 0.02 vs baseline
  • Every gate failure → quarantine staging via
    monthly_calibrator_atomic_swap.py quarantine + appropriate ntfy;
    PROD_CAL is NEVER copied/overwritten by any failure branch
  • Atomic publish (Step 3c) only runs after Step 3 AND Step 3b pass
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "scripts/monthly_calibrator_refresh.sh").read_text()
ATOMIC_BACKUP = 'cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"'
# Pre-fix pattern this round's fix eliminates entirely: writing/overwriting
# PROD_CAL as part of a "rollback" after it had already been exposed.
LEGACY_EXPOSE_THEN_ROLLBACK = 'cp "$ROLLBACK_CAL" "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"'


class TestPreRefitBackup:
    def test_backup_called_before_fit(self):
        assert ATOMIC_BACKUP in SRC
        idx_backup = SRC.index(ATOMIC_BACKUP)
        # Locate the actual python invocation, not the comment mention
        idx_fit = SRC.index('"$PYTHON" -m renquant_model_gbdt.fit_calibrator_alpha158_fund')
        assert idx_backup < idx_fit, "backup must precede the fit invocation"

    def test_baseline_metrics_captured(self):
        assert 'BASELINE_POOL_IC=' in SRC
        assert 'BASELINE_N_UNIQUE=' in SRC

    def test_backup_is_archival_only_not_auto_rollback(self):
        """2026-07-01 round 2: the backup is no longer an automated rollback
        target — PROD_CAL is never written until Step 3c, so there is
        nothing to roll back to on a gate failure."""
        assert "archival" in SRC.lower()
        assert "not used for auto-rollback" in SRC or "NOT used for automated rollback" in SRC


class TestStagingNeverExposesProd:
    """2026-07-01 round 2 core invariant: fit writes to a unique staging
    path, PROD_CAL is only touched by the Step 3c atomic publish, and the
    pre-fix expose-then-rollback pattern is gone entirely."""

    def test_run_id_and_staging_path_defined(self):
        assert "RUN_ID=" in SRC
        assert 'STAGING_CAL="${PROD_CAL}.staging-${RUN_ID}.json"' in SRC
        assert "RECEIPT=" in SRC

    def test_fit_writes_to_staging_not_prod(self):
        assert '--out "$STAGING_CAL"' in SRC
        assert '--out "$PROD_CAL"' not in SRC

    def test_legacy_expose_then_rollback_pattern_removed(self):
        assert LEGACY_EXPOSE_THEN_ROLLBACK not in SRC, (
            "regression: PROD_CAL must never be written as part of a "
            "post-hoc rollback — fit must stage first, validate, then "
            "publish atomically"
        )

    def test_no_gate_reads_or_writes_prod_cal_directly(self):
        """Steps 3 / 3b (quality + binding gates) must evaluate the STAGED
        candidate, not PROD_CAL."""
        gate_verdict_idx = SRC.index("GATE_VERDICT=")
        # The actual invocation, not the docstring-reference comment mention.
        binding_call_idx = SRC.index(
            'BINDING_VERDICT=$("$PYTHON" scripts/verify_calibrator_scorer_binding.py'
        )
        gate_block = SRC[gate_verdict_idx: gate_verdict_idx + 60]
        binding_block = SRC[binding_call_idx: binding_call_idx + 200]
        assert '"$STAGING_CAL"' in gate_block
        assert '--calibrator "$STAGING_CAL"' in binding_block

    def test_atomic_publish_step_present_after_both_gates(self):
        gate_idx = SRC.index("ACCEPTANCE GATE FAILED")
        binding_idx = SRC.index("SCORER/CALIBRATOR BINDING GATE FAILED")
        # The actual invocation, not the earlier design-comment mention.
        publish_idx = SRC.index(
            'scripts/monthly_calibrator_atomic_swap.py publish \\\n    --staging'
        )
        assert gate_idx < binding_idx < publish_idx, \
            "atomic publish must come after both the quality gate and the binding gate"

    def test_publish_verifies_digest_before_swap(self):
        assert "CANDIDATE_SHA256=" in SRC
        assert "--expected-sha256" in SRC

    def test_publish_binds_scorer_identity_into_receipt(self):
        assert "--receipt-out" in SRC
        assert "--scorer-fingerprints-json" in SRC
        assert "--scorer-path" in SRC


class TestQuarantineOnAnyFailure:
    """Every failure branch (fit / smoke / quality gate / binding gate /
    publish itself) must quarantine staging, never touch PROD_CAL."""

    def _failure_branches(self):
        markers = [
            "Calibrator fit FAILED",
            "Post-fit smoke test FAILED on staged calibrator",
            "ACCEPTANCE GATE FAILED",
            "SCORER/CALIBRATOR BINDING GATE FAILED",
            "ATOMIC PUBLISH FAILED",
        ]
        for marker in markers:
            idx = SRC.index(marker)
            yield marker, SRC[idx: idx + 500]

    def test_every_failure_branch_quarantines_staging(self):
        for marker, block in self._failure_branches():
            assert "monthly_calibrator_atomic_swap.py quarantine" in block, \
                f"{marker!r} branch must quarantine staging"

    def test_every_failure_branch_asserts_prod_untouched_in_message(self):
        for marker, block in self._failure_branches():
            lowered = block.lower()
            assert "untouched" in lowered or "unchanged" in lowered or "never" in lowered or "not modified" in lowered, \
                f"{marker!r} branch should tell the operator production was not touched"


class TestAcceptanceGate:
    def test_non_collapse_hard_gate(self):
        assert "n_unique_prob_y={n_uniq} < 10 (collapsed)" in SRC

    def test_ic_regression_2pp_threshold(self):
        # Match the threshold check in the Python heredoc
        assert "if drop > 0.02:" in SRC, \
            "pool_ic > 2pp regression must trigger reject"

    def test_gate_failure_quarantines_and_notifies_reject(self):
        gate_fail_idx = SRC.index("ACCEPTANCE GATE FAILED")
        nearby = SRC[gate_fail_idx: gate_fail_idx + 500]
        assert "monthly_calibrator_atomic_swap.py quarantine" in nearby
        assert "MONTHLY-REJECT" in nearby

    def test_atomic_publish_failure_is_critical_not_reject(self):
        """The one failure mode that CAN'T be "gates never touched prod" is
        the publish step itself failing after every gate passed — that's
        the rare/critical case worth a human, distinct from a normal
        REJECT."""
        publish_fail_idx = SRC.index("ATOMIC PUBLISH FAILED")
        nearby = SRC[publish_fail_idx: publish_fail_idx + 500]
        assert "MONTHLY-CRITICAL" in nearby
        assert "Operator action REQUIRED" in nearby


class TestFixTag:
    def test_2026_05_17_marker(self):
        assert "2026-05-17 ACCEPTANCE GATE" in SRC
        assert "Sunday-sweep corruption" in SRC, \
            "must reference the incident that motivated the fix"

    def test_2026_07_01_round_2_marker(self):
        assert "REVIEW FIX ROUND 2" in SRC
        assert "PR #425" in SRC


class TestMonthlyEnvironment:
    def test_uses_project_venv_and_detected_threads(self):
        non_comment = "\n".join(
            line for line in SRC.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "CONDA_PREFIX" not in non_comment
        assert "miniconda" not in non_comment
        assert 'VENV_DIR="$REPO_DIR/.venv"' in SRC
        assert "os.cpu_count()" in SRC
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f'export {var}="$THREADS"' in SRC
