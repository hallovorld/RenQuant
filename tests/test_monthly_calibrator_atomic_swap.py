"""Tests for scripts/monthly_calibrator_atomic_swap.py — the staging/publish
module added 2026-07-01 (review round 2) to close the "fit writes straight
to PROD_CAL" exposure window Codex flagged CHANGES_REQUESTED on PR #425.

Loads the script as a module (importlib, no strategy venv required) — same
convention as tests/test_verify_calibrator_scorer_binding.py.

Covers (per the review's required test list):
  * an INTEGRATION test with a concurrent read hook proving PROD_CAL stays
    byte-identical to its pre-run bytes throughout a FAILED
    fit/binding-check simulation, start-to-finish
  * the first-install (no baseline) failure case leaves NO production
    artifact at all
  * the successful case: staging -> PROD_CAL only after "gates pass"
  * TOCTOU protection: atomic_publish refuses to swap (and touches
    NEITHER file) if the staging bytes changed after the digest was
    captured
  * the acceptance receipt binds the checked scorer identity/fingerprints
    + the exact candidate digest
  * CLI wiring (sha256 / publish / quarantine subcommands), since the
    shell script depends on this argv contract exactly
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "monthly_calibrator_atomic_swap.py"
_spec = importlib.util.spec_from_file_location("atomicswap", _SCRIPT)
atomicswap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atomicswap)  # type: ignore[union-attr]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Unit: sha256 / receipt building ────────────────────────────────────────

class TestSha256AndReceipt:
    def test_sha256_file_matches_hashlib(self, tmp_path):
        p = tmp_path / "cal.json"
        p.write_bytes(b'{"metadata": {"pool_ic": 0.05}}')
        assert atomicswap.sha256_file(p) == _sha256(p.read_bytes())

    def test_receipt_binds_scorer_identity_and_candidate_digest(self, tmp_path):
        staging = tmp_path / "cal.staging-run1.json"
        staging.write_bytes(b'{"metadata": {}}')
        digest = atomicswap.sha256_file(staging)
        receipt = atomicswap.build_receipt(
            status="published",
            candidate_path=staging,
            candidate_sha256=digest,
            scorer_path=tmp_path / "panel-ltr.json",
            scorer_fingerprints=["sha256:abc123"],
            calibrator_fingerprints=["sha256:abc123"],
            pool_ic=0.05,
            n_unique_prob_y=42,
            reason="all gates passed",
        )
        assert receipt["candidate_sha256"] == digest
        assert receipt["scorer_fingerprints"] == ["sha256:abc123"]
        assert receipt["scorer_path"].endswith("panel-ltr.json")
        assert receipt["status"] == "published"
        assert "timestamp" in receipt

    def test_write_receipt_round_trips_json(self, tmp_path):
        receipt = atomicswap.build_receipt(
            status="rejected", candidate_path=tmp_path / "x.json",
            candidate_sha256="deadbeef", reason="quality gate failed",
        )
        out = tmp_path / "nested" / "receipt.json"
        atomicswap.write_receipt(out, receipt)
        loaded = json.loads(out.read_text())
        assert loaded["reason"] == "quality gate failed"
        assert loaded["candidate_sha256"] == "deadbeef"


# ── atomic_publish: success, missing staging, TOCTOU ───────────────────────

class TestAtomicPublishSuccess:
    def test_publish_only_after_gates_pass_moves_staging_to_prod(self, tmp_path):
        """The exact 'staging -> PROD_CAL only after gates pass' case."""
        prod = tmp_path / "panel-rank-calibration.json"
        staging = tmp_path / "panel-rank-calibration.json.staging-run1.json"
        candidate_bytes = b'{"metadata": {"pool_ic": 0.09, "n_unique_prob_y": 40}}'
        staging.write_bytes(candidate_bytes)
        digest = atomicswap.sha256_file(staging)

        # Simulate: gates already passed elsewhere, THIS is the only call
        # that is allowed to touch prod.
        published_sha = atomicswap.atomic_publish(staging, prod, expected_sha256=digest)

        assert published_sha == digest
        assert prod.exists()
        assert prod.read_bytes() == candidate_bytes
        assert not staging.exists(), "staging must be consumed by the swap"

    def test_publish_overwrites_prior_prod_bytes(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        prod.write_bytes(b'{"metadata": {"pool_ic": 0.02}}')
        staging = tmp_path / "panel-rank-calibration.json.staging-run2.json"
        new_bytes = b'{"metadata": {"pool_ic": 0.10}}'
        staging.write_bytes(new_bytes)
        digest = atomicswap.sha256_file(staging)

        atomicswap.atomic_publish(staging, prod, expected_sha256=digest)
        assert prod.read_bytes() == new_bytes


class TestAtomicPublishFailureLeavesProdUntouched:
    def test_missing_staging_raises_and_prod_untouched(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        prod.write_bytes(b"OLD")
        staging = tmp_path / "panel-rank-calibration.json.staging-run3.json"
        with pytest.raises(FileNotFoundError):
            atomicswap.atomic_publish(staging, prod, expected_sha256="whatever")
        assert prod.read_bytes() == b"OLD"

    def test_toctou_digest_mismatch_refuses_swap_and_touches_neither_file(self, tmp_path):
        """The staging file's bytes changed AFTER the digest used for the
        gate checks was captured (simulated race). Both files must be left
        exactly as they were — no partial swap, no silent publish of the
        wrong bytes."""
        prod = tmp_path / "panel-rank-calibration.json"
        prod.write_bytes(b"OLD-PROD-BYTES")
        staging = tmp_path / "panel-rank-calibration.json.staging-run4.json"
        original_candidate = b'{"metadata": {"pool_ic": 0.09}}'
        staging.write_bytes(original_candidate)
        checked_digest = atomicswap.sha256_file(staging)  # captured at gate-check time

        # Something mutates staging AFTER the digest was captured (TOCTOU).
        tampered = b'{"metadata": {"pool_ic": -0.5, "malicious": true}}'
        staging.write_bytes(tampered)

        with pytest.raises(atomicswap.DigestMismatchError):
            atomicswap.atomic_publish(staging, prod, expected_sha256=checked_digest)

        assert prod.read_bytes() == b"OLD-PROD-BYTES", \
            "prod must be untouched when the candidate digest no longer matches"
        assert staging.read_bytes() == tampered, \
            "staging must be left as-is for the caller to quarantine"


# ── quarantine_staging: never touches prod, no-op when nothing to quarantine ─

class TestQuarantine:
    def test_quarantine_moves_staging_out_of_the_way(self, tmp_path):
        staging = tmp_path / "panel-rank-calibration.json.staging-run5.json"
        staging.write_bytes(b'{"metadata": {}}')
        dest = atomicswap.quarantine_staging(staging, reason="pool_ic regressed")
        assert dest is not None
        assert not staging.exists()
        assert dest.exists()
        assert dest.parent.name == "_rejected_calibrators"
        reason_file = dest.with_name(dest.name + ".reason.txt")
        assert reason_file.exists()
        assert "pool_ic regressed" in reason_file.read_text()

    def test_quarantine_missing_staging_is_a_safe_no_op(self, tmp_path):
        """fit_calibrator crashed before writing anything — quarantine must
        not raise and must not fabricate a file."""
        staging = tmp_path / "does-not-exist.json"
        dest = atomicswap.quarantine_staging(staging, reason="fit crashed")
        assert dest is None

    def test_quarantine_never_touches_a_sibling_prod_path(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        prod.write_bytes(b"OLD-PROD-BYTES")
        staging = tmp_path / "panel-rank-calibration.json.staging-run6.json"
        staging.write_bytes(b"bad-candidate")
        atomicswap.quarantine_staging(staging, reason="binding mismatch")
        assert prod.read_bytes() == b"OLD-PROD-BYTES"


# ── Integration: concurrent-read hook proving PROD_CAL is untouched ────────

class TestConcurrentReadDuringFailedRun:
    """Required by the review: an integration test with a concurrent/read
    hook proving PROD_CAL remains the OLD bytes throughout a FAILED
    fit/binding check — start to finish, not just checked once at the end.
    """

    def test_prod_bytes_never_observed_to_change_during_a_failing_run(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        old_bytes = b'{"metadata": {"pool_ic": 0.07, "n_unique_prob_y": 30}}'
        prod.write_bytes(old_bytes)

        staging = tmp_path / "panel-rank-calibration.json.staging-runX.json"

        observed_mismatch: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            # Simulates the live runtime reading PROD_CAL while the monthly
            # job is mid-flight. Polls aggressively for the duration of the
            # simulated failing run.
            while not stop.is_set():
                if prod.exists():
                    current = prod.read_bytes()
                    if current != old_bytes:
                        observed_mismatch.append(current.decode(errors="replace"))
                else:
                    observed_mismatch.append("<<MISSING>>")
                time.sleep(0.001)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        try:
            # ── Simulated failing run ──
            # Step 2: fit writes ONLY to staging.
            staging.write_bytes(b'{"metadata": {"pool_ic": -0.30, "n_unique_prob_y": 3}}')
            time.sleep(0.02)  # give the reader thread several polls
            # Step 3: quality gate fails (n_unique_prob_y=3 < 10, pool_ic
            # regressed hard) -> quarantine, PROD_CAL must never be touched.
            dest = atomicswap.quarantine_staging(
                staging, reason="n_unique_prob_y=3 < 10 (collapsed)",
            )
            time.sleep(0.02)  # more polls after the "failure"
        finally:
            stop.set()
            reader_thread.join(timeout=2)

        assert not observed_mismatch, (
            "concurrent reader observed PROD_CAL bytes change/disappear "
            f"during a failing run: {observed_mismatch}"
        )
        assert prod.read_bytes() == old_bytes
        assert dest is not None and dest.exists()
        assert not staging.exists()

    def test_prod_bytes_unchanged_through_a_toctou_publish_failure(self, tmp_path):
        """Same concurrent-read hook, but the failure is a TOCTOU digest
        mismatch discovered at the publish step itself (not an earlier
        gate) — PROD_CAL must still never be observed to change."""
        prod = tmp_path / "panel-rank-calibration.json"
        old_bytes = b"OLD-CALIBRATOR-BYTES"
        prod.write_bytes(old_bytes)
        staging = tmp_path / "panel-rank-calibration.json.staging-runY.json"
        staging.write_bytes(b"candidate-v1")
        checked_digest = atomicswap.sha256_file(staging)

        observed_mismatch: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                current = prod.read_bytes() if prod.exists() else b"<<MISSING>>"
                if current != old_bytes:
                    observed_mismatch.append(repr(current))
                time.sleep(0.001)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            # Race: staging mutates after the digest that gate-checks used.
            staging.write_bytes(b"candidate-v1-TAMPERED")
            time.sleep(0.02)
            with pytest.raises(atomicswap.DigestMismatchError):
                atomicswap.atomic_publish(staging, prod, expected_sha256=checked_digest)
            time.sleep(0.02)
        finally:
            stop.set()
            t.join(timeout=2)

        assert not observed_mismatch
        assert prod.read_bytes() == old_bytes


# ── First-install (no baseline) failure leaves NO production artifact ──────

class TestFirstInstallFailureLeavesNoProductionArtifact:
    def test_no_prior_calibrator_and_failed_fit_leaves_prod_absent(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        assert not prod.exists()  # first-ever fit: no baseline
        staging = tmp_path / "panel-rank-calibration.json.staging-runZ.json"
        # fit_calibrator crashed before writing staging at all.
        dest = atomicswap.quarantine_staging(staging, reason="fit crashed, no baseline")
        assert dest is None
        assert not prod.exists(), \
            "a failed first-ever fit must leave NO production artifact, not a rejected one"

    def test_no_prior_calibrator_and_failed_gate_leaves_prod_absent(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        assert not prod.exists()
        staging = tmp_path / "panel-rank-calibration.json.staging-runZ2.json"
        staging.write_bytes(b'{"metadata": {"pool_ic": -0.9, "n_unique_prob_y": 1}}')
        dest = atomicswap.quarantine_staging(staging, reason="collapsed, no baseline")
        assert dest is not None  # the bad candidate is archived...
        assert not prod.exists(), \
            "...but PROD_CAL must still not exist — no baseline means nothing to expose"


# ── CLI wiring (argv contract monthly_calibrator_refresh.sh depends on) ────

class TestCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *args],
            capture_output=True, text=True, timeout=15,
        )

    def test_sha256_subcommand_prints_digest(self, tmp_path):
        p = tmp_path / "cal.json"
        p.write_bytes(b"hello")
        result = self._run("sha256", "--path", str(p))
        assert result.returncode == 0
        assert result.stdout.strip() == _sha256(b"hello")

    def test_publish_subcommand_end_to_end(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        staging = tmp_path / "panel-rank-calibration.json.staging-cli1.json"
        staging.write_bytes(b'{"metadata": {"pool_ic": 0.08, "n_unique_prob_y": 25}}')
        digest = _sha256(staging.read_bytes())
        receipt_path = tmp_path / "receipt.json"

        result = self._run(
            "publish",
            "--staging", str(staging), "--prod", str(prod),
            "--expected-sha256", digest,
            "--receipt-out", str(receipt_path),
            "--scorer-path", str(tmp_path / "panel-ltr.json"),
            "--scorer-fingerprints-json", json.dumps(["sha256:abc"]),
            "--calibrator-fingerprints-json", json.dumps(["sha256:abc"]),
            "--pool-ic", "0.08", "--n-unique", "25",
        )
        assert result.returncode == 0, result.stderr
        assert prod.exists()
        assert not staging.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "published"
        assert receipt["candidate_sha256"] == digest
        assert receipt["scorer_fingerprints"] == ["sha256:abc"]

    def test_publish_subcommand_digest_mismatch_exits_nonzero_prod_untouched(self, tmp_path):
        prod = tmp_path / "panel-rank-calibration.json"
        prod.write_bytes(b"OLD")
        staging = tmp_path / "panel-rank-calibration.json.staging-cli2.json"
        staging.write_bytes(b"candidate")

        result = self._run(
            "publish",
            "--staging", str(staging), "--prod", str(prod),
            "--expected-sha256", "0" * 64,
        )
        assert result.returncode == 1
        assert prod.read_bytes() == b"OLD"
        assert staging.exists()

    def test_quarantine_subcommand_end_to_end(self, tmp_path):
        staging = tmp_path / "panel-rank-calibration.json.staging-cli3.json"
        staging.write_bytes(b'{"metadata": {}}')
        receipt_path = tmp_path / "receipt.json"
        result = self._run(
            "quarantine",
            "--staging", str(staging),
            "--reason", "binding mismatch",
            "--receipt-out", str(receipt_path),
        )
        assert result.returncode == 0, result.stderr
        assert not staging.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "binding mismatch"

    def test_quarantine_subcommand_missing_staging_is_no_op_success(self, tmp_path):
        staging = tmp_path / "never-existed.json"
        result = self._run("quarantine", "--staging", str(staging), "--reason", "fit crashed")
        assert result.returncode == 0
        assert "no-op" in result.stdout.lower()
