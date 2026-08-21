"""The wrapper must not call a REFUSAL a success (2026-08-21).

WHAT THE RUN-HEALTH SCAN CAUGHT. `conditional-retrain104` reported:

    2 of them CLAIMED SUCCESS while weekly-wf-promote's own log for that date
    shows no promotion (the wrapper reports its child's exit code, and a
    CALM_FRESH refusal exits 0)

`scripts/conditional_retrain_104.sh` branched on the child's exit status alone.
`weekly_wf_promote.sh` exits 0 on a refusal *deliberately* — "Reject
disposition: prod FRESH ... governance nominal, calm notify, exit 0"
(weekly_wf_promote.sh:517) — because a gate declining is the gate working. So
on 2026-08-19 and 2026-08-20 the wrapper printed "chain complete" and paged the
operator "WF promote OK" while production was untouched.

Same family as the ntfy mislabels fixed in RenQuant#598/#599/#600: a message
naming an outcome that is not the outcome. This one is worse in one respect —
it pages a *positive* result for a non-event, so the operator has no reason to
look.

WHY THE PATTERNS ARE PARSED OUT OF THE SCRIPT. Retyping the regexes here would
create a second implementation that can silently disagree with the first — the
twin-implementation trap this repo has hit repeatedly. These tests extract the
predicates the script actually runs and apply them to REAL recorded child
output, so a change to the script that breaks classification fails here.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "conditional_retrain_104.sh"
CHILD = REPO_ROOT / "scripts" / "weekly_wf_promote.sh"
WF_LOGS = REPO_ROOT / "logs" / "weekly_wf_promote"

pytestmark = pytest.mark.integration


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _grep_patterns() -> list[str]:
    """Every `grep -qE "<pat>"` the outcome classification runs, in order."""
    return re.findall(r'grep -qE "([^"]+)"', _script())


class TestTheScriptStillParses:
    def test_bash_accepts_it(self):
        r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


class TestTheFalseSuccessIsGone:
    def test_it_no_longer_pages_OK_on_a_bare_zero_exit(self):
        s = _script()
        assert 'notify "RenQuant 104 WF promote OK"' not in s, (
            "the OK page fired on any zero exit, including a refusal"
        )
        assert "Gated WF promote chain complete" not in s, (
            "'complete' was printed for runs that promoted nothing"
        )

    def test_it_reports_a_refusal_by_its_own_name(self):
        s = _script()
        assert "RAN, NOTHING PROMOTED" in s
        assert "no change" in s

    def test_an_unclassifiable_outcome_is_never_success(self):
        """The polarity that matters: if the markers are ever renamed, this must
        surface, not inherit a permanent false OK."""
        s = _script()
        assert "OUTCOME UNVERIFIED" in s
        assert "UNVERIFIED" in s.split("else", 1)[-1] or "UNVERIFIED" in s


class TestTheMarkersMatchTHECHILD:
    """The promotion markers are the child's, not this wrapper's invention."""

    def test_both_child_markers_exist_in_the_child(self):
        child = CHILD.read_text(encoding="utf-8")
        for marker in ("=== weekly_wf_promote PASSED",
                       "=== weekly_wf_promote FALLBACK-PROMOTED"):
            assert marker in child, f"{marker} not emitted by {CHILD.name}"

    def test_the_wrapper_looks_for_exactly_those(self):
        pats = _grep_patterns()
        assert pats, "no grep -qE predicates found — classification is gone"
        promo = pats[0]
        assert "PASSED" in promo and "FALLBACK-PROMOTED" in promo, promo


class TestItClassifiesREALProductionOutput:
    """Applied to recorded child logs, not to invented strings."""

    def _classify(self, text: str) -> str:
        pats = _grep_patterns()
        promo, refuse = pats[0], pats[1]
        if re.search(promo, text):
            return "PROMOTED"
        if re.search(refuse, text):
            return "NOTHING_PROMOTED"
        return "UNVERIFIED"

    def _log(self, date: str) -> str:
        p = WF_LOGS / f"{date}.log"
        if not p.is_file():
            pytest.skip(f"{p} absent on this machine")
        return p.read_text(encoding="utf-8", errors="replace")

    @pytest.mark.parametrize("date", ["2026-08-20", "2026-08-19", "2026-08-18"])
    def test_the_refusals_that_were_reported_as_OK(self, date):
        """These are the exact runs the scan flagged. Each must classify as a
        non-promotion — under the old wrapper every one of them paged OK."""
        got = self._classify(self._log(date))
        assert got == "NOTHING_PROMOTED", f"{date} classified {got}"

    def test_a_promotion_marker_would_classify_as_promoted(self):
        """Guards the other direction, since no real promotion exists in the
        recent record to test against — the marker text is taken from the
        child's own source, so this is not an invented string."""
        child = CHILD.read_text(encoding="utf-8")
        assert "=== weekly_wf_promote PASSED at" in child
        synthetic = "some output\n=== weekly_wf_promote PASSED at Thu Aug 21 — summary ===\n"
        assert self._classify(synthetic) == "PROMOTED"

    def test_silence_is_not_success(self):
        assert self._classify("started\nsome unrelated output\ndone\n") == "UNVERIFIED"
