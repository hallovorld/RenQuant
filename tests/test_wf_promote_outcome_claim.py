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


# ── stubs that behave like the REAL child: `exec >>` its dated log FIRST ──────
# weekly_wf_promote.sh redirects its own stdout/stderr into
# logs/weekly_wf_promote/<date>.log before printing any terminal marker, so a
# wrapper reading only the tee'd stdout sees nothing. These stubs reproduce
# that; the stubs above (markers on stdout) are the shape the 08-21 fix was
# tested against and the shape production never had.

def _redirected(body_after_redirect: str) -> str:
    return (
        "#!/bin/bash\n"
        "REPO_DIR=\"${RQ_WEEKLY_PROMOTE_REPO_DIR:-$PWD}\"\n"
        "mkdir -p \"$REPO_DIR/logs/weekly_wf_promote\"\n"
        "exec >> \"$REPO_DIR/logs/weekly_wf_promote/$(date +%Y-%m-%d).log\" 2>&1\n"
        "echo \"=== weekly_wf_promote started at $(date) ===\"\n"
        + body_after_redirect
    )


REDIRECTED_REFUSAL_OUTPUT = _redirected(
    "echo \"RFC#210 fallback verdict: REFUSE — production unchanged.\"\n"
    "echo \"Reject disposition: prod FRESH (trained 2026-08-31, 3d <= 28d SLA) — governance nominal, calm notify, exit 0.\"\n"
    "exit 0\n")
REDIRECTED_FALLBACK_PROMOTED_OUTPUT = _redirected(
    "echo \"=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at $(date) — summary ===\"\n"
    "exit 0\n")
REDIRECTED_ALARM_OUTPUT = _redirected(
    "echo \"Reject disposition: ALARM|refused on 'quality_floor', not prod_stale — alarm notify, exit 1.\"\n"
    "exit 1\n")
PRIOR_RUN_PROMOTED_LINE = "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at 09:09 — an earlier run today ===\n"


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


class TestConditionalRetrainReadsTheChildsOwnLog:
    """2026-09-03: the 13:10 VIX chain refused correctly ('prod FRESH … exit 0')
    and the wrapper still reported OUTCOME UNVERIFIED (rc 2) — the classifier
    was reading the tee'd stdout, which the child had already redirected away
    from. The evidence is the segment of the child's dated log written by
    THIS run."""

    def test_a_redirected_refusal_is_NOTHING_PROMOTED(self, tmp_path):
        rc, log, notes = _run_conditional(tmp_path, REDIRECTED_REFUSAL_OUTPUT)
        assert rc == 0, log[-800:]
        assert "RAN, NOTHING PROMOTED" in log, log[-800:]
        assert "UNVERIFIED" not in log
        assert "no change" in notes, notes
        # describe_wf_promote_outcome quotes the FIRST recognisable phrase of the
        # run — the verdict line precedes the disposition line in the real log.
        assert "fallback verdict: REFUSE" in notes or "Reject disposition" in notes, notes

    def test_a_redirected_promotion_is_PROMOTED(self, tmp_path):
        rc, log, notes = _run_conditional(tmp_path, REDIRECTED_FALLBACK_PROMOTED_OUTPUT)
        assert rc == 0
        assert "UNVERIFIED" not in log
        assert "PROMOTED" in notes and "no change" not in notes, notes

    def test_an_earlier_promotion_in_todays_log_does_not_leak_into_this_run(self, tmp_path):
        """The dated log accumulates every run of the day; an operator
        --promote-staged in the morning must not make the afternoon's refusal
        read as PROMOTED."""
        repo = build_repo(tmp_path, REDIRECTED_REFUSAL_OUTPUT, scripts=["conditional_retrain_104.sh"])
        child_log = repo / "logs" / "weekly_wf_promote" / f"{_today()}.log"
        child_log.parent.mkdir(parents=True, exist_ok=True)
        child_log.write_text(PRIOR_RUN_PROMOTED_LINE)
        notes = tmp_path / "notify.log"
        rc, log, notes_text = run(repo, "conditional_retrain_104.sh", {
            "RQ_CONDITIONAL_REPO_DIR": str(repo),
            "RQ_CONDITIONAL_NOTIFY_LOG": str(notes),
        })
        assert rc == 0
        assert "RAN, NOTHING PROMOTED" in log, log[-800:]
        assert "no change" in notes_text, notes_text
        assert "WF promote: PROMOTED" not in notes_text and "chain complete" not in log

    def test_a_redirected_alarm_exit_still_FAILS(self, tmp_path):
        rc, log, notes = _run_conditional(tmp_path, REDIRECTED_ALARM_OUTPUT)
        assert rc == 1
        assert "FAILED" in log and "ERROR" in notes, notes

    def test_no_child_log_at_all_stays_UNVERIFIED(self, tmp_path):
        """A child that never wrote its dated log and printed nothing useful:
        the polarity the 08-21 fix established must survive the new evidence
        path — never success."""
        rc, log, notes = _run_conditional(tmp_path, SILENT_ZERO_OUTPUT)
        assert rc == 2 and "UNVERIFIED" in notes, notes


def _today() -> str:
    import datetime as _dt
    return _dt.date.today().isoformat()


class TestRetrainPanelReadsTheChildsOwnLog:
    def test_a_redirected_refusal_is_NOTHING_PROMOTED(self, tmp_path):
        rc, log, _ = _run_panel(tmp_path, REDIRECTED_REFUSAL_OUTPUT)
        assert rc == 0, log[-800:]
        assert "NOTHING PROMOTED" in log and "UNVERIFIED" not in log, log[-800:]
    # NOTE: no "earlier promotion in today's log" case here — retrain_panel.sh
    # already no-ops when the child's dated log exists ("weekly_wf_promote already
    # ran today"), so the segmentation can only be exercised through the
    # conditional wrapper (TestConditionalRetrainReadsTheChildsOwnLog).
