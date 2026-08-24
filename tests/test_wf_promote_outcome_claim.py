"""A refusal must never be reported as a success (2026-08-21 / 2026-08-23).

WHAT THE RUN-HEALTH SCAN CAUGHT, and then what happened four days later:

  * `conditional-retrain104`: "2 of them CLAIMED SUCCESS while
    weekly-wf-promote's own log for that date shows no promotion (the wrapper
    reports its child's exit code, and a CALM_FRESH refusal exits 0)". On
    2026-08-19 and 2026-08-20 it printed "chain complete" and PUSHED
    "RenQuant 104 WF promote OK" while production was untouched.
  * `retrain_panel`: on 2026-08-23 it logged "delegated weekly_wf_promote PASS"
    for a chain whose own verdict was `VERDICT: FAIL` (genuine_ic=+0.0000,
    aligned_real_ic == placebo_ic to four decimals) and which promoted nothing.
    That log line is also what the run-health scan reads to decide whether the
    job "acted", so the false PASS corrupted the scan as well as the reader.

Both wrappers branched on the child's EXIT CODE alone, and `weekly_wf_promote.sh`
exits 0 on a refusal DELIBERATELY — "Reject disposition: prod FRESH ...
governance nominal, calm notify, exit 0" — because a gate declining is the gate
working.

WHY THESE TESTS LOOK LIKE THIS [codex on RenQuant#603]. The first version read
`logs/weekly_wf_promote/*.log` and re-applied the wrapper's regex in Python. In
a clean checkout that was `8 passed, 3 skipped` — the three incident cases,
which are the entire point, silently did not run, because those logs are
workstation state. And re-applying the regex verifies neither the shell branch,
nor the notification title and body, nor the exit code, nor the seams.

So: a hermetic fake repo (stub `python`, stub `subrepo_env.sh`, stub child), the
REAL wrappers and the REAL shared classifier, driven end to end, asserting on
what the operator would actually receive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest_harness import build_repo, run  # noqa: E402

pytestmark = pytest.mark.integration

REFUSAL_OUTPUT = """\
#!/bin/bash
cat <<'EOF'
2026-08-23 10:12:41 [INFO] VERDICT: FAIL
WF gate REJECTED staged model — consulting the RFC#210 freshness fallback.
RFC#210 fallback verdict: REFUSE — production unchanged.
Reject disposition: prod FRESH (trained 2026-08-02, 21d <= 28d SLA) — governance nominal, calm notify, exit 0.
EOF
exit 0
"""

PROMOTED_OUTPUT = """\
#!/bin/bash
echo "=== weekly_wf_promote PASSED at Sun Aug 23 10:12:43 PDT 2026 — gate summary ==="
exit 0
"""

FALLBACK_PROMOTED_OUTPUT = """\
#!/bin/bash
echo "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at Sun Aug 23 — summary ==="
exit 0
"""

CRASH_OUTPUT = """\
#!/bin/bash
echo "boom" >&2
exit 3
"""

SILENT_ZERO_OUTPUT = """\
#!/bin/bash
echo "started"
echo "some unrelated chatter"
exit 0
"""


# ── conditional_retrain_104.sh ────────────────────────────────────────────────

def _run_conditional(tmp_path, child_body):
    repo = build_repo(tmp_path, child_body, scripts=["conditional_retrain_104.sh"])
    notes = tmp_path / "notify.log"
    return run(repo, "conditional_retrain_104.sh", {
        "RQ_CONDITIONAL_REPO_DIR": str(repo),
        "RQ_CONDITIONAL_NOTIFY_LOG": str(notes),
    })


class TestConditionalRetrainReportsWhatActuallyHappened:
    def test_a_refusal_is_NOT_reported_as_OK(self, tmp_path):
        """The 2026-08-19/20 incident, reproduced hermetically."""
        rc, log, notes = _run_conditional(tmp_path, REFUSAL_OUTPUT)
        assert rc == 0, "a gate declining is not a failure"
        assert "RAN, NOTHING PROMOTED" in log, log[-800:]
        assert "chain complete" not in log
        assert "WF promote OK" not in notes, notes
        assert "no change" in notes, notes
        assert "VERDICT: FAIL" in notes or "Reject disposition" in notes, notes

    @pytest.mark.parametrize("body", [PROMOTED_OUTPUT, FALLBACK_PROMOTED_OUTPUT])
    def test_a_real_promotion_is_reported_as_one(self, tmp_path, body):
        rc, log, notes = _run_conditional(tmp_path, body)
        assert rc == 0
        assert "PROMOTED" in log, log[-800:]
        assert "PROMOTED" in notes, notes
        assert "no change" not in notes

    def test_a_nonzero_child_still_alarms_and_exits_1(self, tmp_path):
        rc, log, notes = _run_conditional(tmp_path, CRASH_OUTPUT)
        assert rc == 1, "a crashed chain must fail the job"
        assert "FAILED" in log
        assert "ERROR" in notes, notes

    def test_zero_exit_with_no_recognisable_outcome_is_UNVERIFIED(self, tmp_path):
        """The polarity that matters: if the child's markers are ever renamed,
        this must surface instead of inheriting a permanent false OK.

        And it must surface in the EXIT STATUS, not only in the text. An
        earlier revision exited 0 here, which handed launchd a successful job
        for an outcome nobody could establish -- the same false OK in a new
        place. 2, not 1, so automation can separate "the child failed" from
        "the child's contract drifted" (codex review, 2026-08-24).
        """
        rc, log, notes = _run_conditional(tmp_path, SILENT_ZERO_OUTPUT)
        assert rc == 2, "an unestablished outcome must not present as success"
        assert "UNVERIFIED" in log, log[-800:]
        assert "UNVERIFIED" in notes, notes
        assert "OK" not in notes.replace("UNVERIFIED", ""), notes


# ── retrain_panel.sh ──────────────────────────────────────────────────────────

def _run_panel(tmp_path, child_body):
    repo = build_repo(tmp_path, child_body, scripts=["retrain_panel.sh"])
    return run(repo, "retrain_panel.sh", {
        "RQ_RETRAIN_PANEL_REPO_DIR": str(repo),
        "RQ_RETRAIN_PANEL_LOCK_FILE": str(tmp_path / "panel.lock"),
    })


class TestRetrainPanelReportsWhatActuallyHappened:
    """Same bug, second wrapper. It emits no ntfy by design, so only the log
    line is asserted — but that line is what the run-health scan reads."""

    def test_the_2026_08_23_false_PASS(self, tmp_path):
        rc, log, _ = _run_panel(tmp_path, REFUSAL_OUTPUT)
        assert rc == 0
        assert "RAN, NOTHING PROMOTED" in log, log[-800:]
        assert "weekly_wf_promote PASS at" not in log, (
            "this is the exact string logged on 2026-08-23 for a run that "
            "promoted nothing"
        )

    def test_a_real_promotion_is_reported_as_one(self, tmp_path):
        rc, log, _ = _run_panel(tmp_path, PROMOTED_OUTPUT)
        assert rc == 0
        assert "PROMOTED" in log, log[-800:]

    def test_a_nonzero_child_fails_the_job(self, tmp_path):
        rc, log, _ = _run_panel(tmp_path, CRASH_OUTPUT)
        assert rc == 1
        assert "FAILED" in log

    def test_silent_zero_is_UNVERIFIED(self, tmp_path):
        """This wrapper emits NO notification, so the exit status is the only
        signal that leaves the process. Exiting 0 on an unestablished outcome
        would make a renamed marker produce a log line nobody reads and a green
        job -- strictly more silent than the bug this file removes."""
        rc, log, _ = _run_panel(tmp_path, SILENT_ZERO_OUTPUT)
        assert rc == 2, "an unestablished outcome must not present as success"
        assert "UNVERIFIED" in log, log[-800:]


class TestTheTwoWrappersShareOneDefinition:
    """Two copies of this rule would drift; that is how the second wrapper kept
    the bug for two days after the first was fixed."""

    def test_both_source_the_shared_classifier(self):
        real = Path(__file__).resolve().parent.parent / "scripts"
        for name in ("conditional_retrain_104.sh", "retrain_panel.sh"):
            src = (real / name).read_text(encoding="utf-8")
            assert "lib/wf_promote_outcome.sh" in src, name
            assert "classify_wf_promote_outcome" in src, name

    def test_neither_reimplements_the_markers(self):
        real = Path(__file__).resolve().parent.parent / "scripts"
        for name in ("conditional_retrain_104.sh", "retrain_panel.sh"):
            src = (real / name).read_text(encoding="utf-8")
            assert "weekly_wf_promote (PASSED|FALLBACK-PROMOTED)" not in src, (
                f"{name} re-states the marker pattern instead of using the helper"
            )
