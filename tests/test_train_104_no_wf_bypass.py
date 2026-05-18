"""Tests for the 2026-05-17 train_104.py WF-bypass removal.

Pre-fix (§5.13.15 violation): every daily retrain set RQ_ALLOW_NO_WF=1
and called promote(), turning the WF gate into theater. Today's
Sunday-sweep corruption (NGB val_IC=-0.0165 to prod) was an instance
of the broader problem — light gates alone can't catch silent quality
regressions.

Fix invariants:
  • train_104.py does NOT set RQ_ALLOW_NO_WF=1 by default.
  • Daily retrain STAGES the new artifact, does not promote unless
    operator explicitly sets RQ_ALLOW_NO_WF=1 in the shell env.
  • Promotion happens via weekly_wf_promote.sh → run_wf_gate.py.
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_104 = (REPO / "scripts/train_104.py").read_text()
WEEKLY_SH = (REPO / "scripts/weekly_wf_promote.sh").read_text()


class TestNoDailyWfBypass:
    def test_no_setdefault_wf_bypass(self):
        """The script must NOT setdefault RQ_ALLOW_NO_WF=1."""
        assert 'setdefault("RQ_ALLOW_NO_WF", "1")' not in TRAIN_104, \
            "daily retrain must NOT set RQ_ALLOW_NO_WF=1 as a default"

    def test_external_override_still_honored(self):
        """If operator sets RQ_ALLOW_NO_WF=1 externally (e.g. emergency),
        promote() is still called. This is the emergency override path."""
        assert 'RQ_ALLOW_NO_WF") == "1"' in TRAIN_104
        # Followed by promote(...)
        idx = TRAIN_104.index('RQ_ALLOW_NO_WF") == "1"')
        nearby = TRAIN_104[idx:idx + 600]
        assert "promote(staging_path, active_path)" in nearby, \
            "override branch must call promote"

    def test_default_branch_stages_only(self):
        """Default behavior: stage only, no promote."""
        # The default branch log message should reflect STAGE-ONLY
        assert "STAGED at" in TRAIN_104
        assert "Production NOT updated" in TRAIN_104

    def test_fix_tag_present(self):
        assert "2026-05-17 §5.13.15 fix" in TRAIN_104

    def test_weekly_wf_promote_still_uses_strict_gate(self):
        """The weekly cron MUST NOT bypass WF — that's the only path
        to promote post-fix. Comments mentioning RQ_ALLOW_NO_WF as
        explanation are OK; only actual exports/sets are forbidden."""
        # Strip comment lines for the bypass-presence check
        non_comment = "\n".join(
            line for line in WEEKLY_SH.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "RQ_ALLOW_NO_WF=1" not in non_comment, \
            "weekly_wf_promote.sh should NEVER set RQ_ALLOW_NO_WF=1 in code"
        assert "export RQ_ALLOW_NO_WF" not in non_comment
        assert "run_wf_gate.py" in WEEKLY_SH, \
            "weekly promote must invoke run_wf_gate.py"
