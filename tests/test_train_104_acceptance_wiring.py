"""Tests for train_104.py acceptance-gate wiring (round-7, 2026-04-26).

Per user spec: when retrain fails acceptance, train_104.py must:
1. Preserve the prior panel-ltr.json at active path (live runner sees no change)
2. Archive the rejected staging artifact under _acceptance_log/
3. Exit non-zero so daily_104.sh sees the failure
4. (Optional) fire ntfy alert

These tests use string-level source contracts (same pattern as
test_runner_state_fixes / test_bug20). The full integration is
exercised end-to-end via the next sim run; here we pin the
load-bearing lines so a refactor can't silently lose them.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts/train_104.py"
SCRIPT_SRC = SCRIPT_PATH.read_text()


class TestAcceptanceWiringPresent:
    def test_audit_tag(self):
        assert "Audit fix #152" in SCRIPT_SRC, (
            "Audit tag for acceptance-gate wiring must be in source so "
            "future readers can find the rationale."
        )

    def test_imports_acceptance_module(self):
        assert "from kernel.model_acceptance import" in SCRIPT_SRC
        assert "ModelAcceptanceGate" in SCRIPT_SRC
        assert "promote" in SCRIPT_SRC
        assert "reject" in SCRIPT_SRC

    def test_default_enabled_true(self):
        """`acceptance.enabled` must default to True (safety opt-OUT, not opt-IN)."""
        assert 'acceptance_cfg.get("enabled", True)' in SCRIPT_SRC, (
            "Default ON: forgetting to enable wouldn't ship the safety net."
        )

    def test_skip_acceptance_flag_exists(self):
        """Operator escape hatch for known-broken edge cases."""
        assert "--skip-acceptance" in SCRIPT_SRC

    def test_pre_train_snapshot_taken(self):
        """Active artifact is snapshotted BEFORE training (so we have
        the prior content for gate G4 + recovery)."""
        assert "pre_train_snapshot" in SCRIPT_SRC
        assert ".pre-train.json" in SCRIPT_SRC


class TestPromoteOrReject:
    def test_promote_called_on_pass(self):
        """All-hard-pass path: must call promote(staging, active)."""
        # The promotion line must reference both promote() and the
        # all_hard_passed verdict.
        idx = SCRIPT_SRC.find("verdict.all_hard_passed")
        assert idx >= 0
        # Widened from 800 to 1500 — 2026-05-09 added WF gate override
        # comment between hard-pass branch and promote() call
        block = SCRIPT_SRC[idx:idx + 1500]
        assert "promote(staging_path, active_path)" in block

    def test_reject_called_on_fail(self):
        """Hard-fail path: reject() to archive the bad artifact + log."""
        # Locate the else branch
        idx = SCRIPT_SRC.find("HARD GATE FAILED")
        assert idx >= 0
        block = SCRIPT_SRC[idx:idx + 800]
        assert "reject(staging_path" in block
        assert "archive_dir" in block

    def test_archive_dir_resolved_correctly(self):
        """archive_dir must point under strategy_dir/artifacts/_acceptance_log."""
        assert '"_acceptance_log"' in SCRIPT_SRC
        assert "strategy_dir / \"artifacts\" / \"_acceptance_log\"" in SCRIPT_SRC

    def test_exit_nonzero_on_reject(self):
        """daily_104.sh must SEE the failure (its `if PIPE; then` branch)."""
        idx = SCRIPT_SRC.find("HARD GATE FAILED")
        assert idx >= 0
        block = SCRIPT_SRC[idx:idx + 1500]
        assert "sys.exit(2)" in block, (
            "must exit non-zero so daily_104.sh's `if train_104.py` "
            "sees the rejection"
        )

    def test_ntfy_alert_on_reject(self):
        """Best-effort ntfy alert so operator sees the rejection promptly."""
        idx = SCRIPT_SRC.find("HARD GATE FAILED")
        assert idx >= 0
        block = SCRIPT_SRC[idx:idx + 1500]
        assert "ntfy.sh/renquant" in block
        assert "RETRAIN REJECTED" in block


class TestStagingActiveFlow:
    def test_staging_path_has_correct_suffix(self):
        """Convention: staging artifact at panel-ltr.staging.json."""
        assert '.staging.json' in SCRIPT_SRC

    def test_active_path_resolved_from_config(self):
        """BUG-G7 fix (2026-04-28): active_path comes from
        config.panel_ltr.artifact_path (default `artifacts/panel-ltr.json`),
        NOT a hardcoded literal. This lets side configs (wl178, ablations)
        write to side paths and have the acceptance gate evaluate the
        correct artifact, not the production prior.
        """
        # The default fallback string still appears (inside .get(...))
        assert '"artifacts/panel-ltr.json"' in SCRIPT_SRC
        # Resolution must come from config, not hardcoded composition
        assert 'panel_cfg.get("artifact_path"' in SCRIPT_SRC
        # And the OLD hardcoded line must be gone
        assert 'active_path = strategy_dir / "artifacts" / "panel-ltr.json"' not in SCRIPT_SRC

    def test_pre_train_cleanup_after_promote(self):
        """Audit fix #9 (2026-04-26): cleanup MUST run on both success
        AND rejection. Pre-fix, .pre-train.json files lingered after
        rejected retrains, confusing operators investigating failures.
        Post-fix: cleanup is in a finally: block."""
        # The cleanup must be in a finally: block, not just the success path.
        assert "finally:" in SCRIPT_SRC
        # And the unlink call must reference pre_train_snapshot
        assert "pre_train_snapshot.unlink()" in SCRIPT_SRC
        # Sanity: the finally is in the acceptance flow (after the
        # ALL HARD GATES PASSED branch)
        idx = SCRIPT_SRC.find("ALL HARD GATES PASSED")
        assert idx >= 0
        finally_idx = SCRIPT_SRC.find("finally:", idx)
        unlink_idx  = SCRIPT_SRC.find("pre_train_snapshot.unlink", finally_idx)
        assert finally_idx > 0 and unlink_idx > finally_idx


class TestSkipAcceptanceBypass:
    def test_skip_flag_short_circuits(self):
        """--skip-acceptance disables the gate entirely (acceptance_enabled
        becomes False, no snapshot/promote/reject path runs)."""
        # The conditional must AND the config flag with NOT skip_acceptance
        assert "acceptance_cfg.get(\"enabled\", True)) and not args.skip_acceptance" in SCRIPT_SRC


class TestNoBehaviorChangeWhenDisabled:
    def test_disabled_path_does_not_touch_artifact(self):
        """When acceptance is disabled, the script behaves exactly as
        before this commit — no staging swap, no snapshot, no exit_2."""
        # Verify the snapshot is taken ONLY if acceptance_enabled
        assert "if acceptance_enabled and active_path.exists():" in SCRIPT_SRC
        assert "if acceptance_enabled:\n" in SCRIPT_SRC
