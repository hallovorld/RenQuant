"""Regression test: scripts/run_ablation_followups.sh must NOT swallow
sanity-script errors with `|| true`.

2026-05-03 incident: the wrapper used `python ... > LOG 2>&1 || true`,
silently masking an ImportError in run_sanity_checks.py. The chain
proceeded to B2 sim claiming sanity had passed when it had never run.

This test inspects the source of the wrapper to enforce: any sanity
invocation that uses `|| true` is a bug. The wrapper must propagate
exit codes and bail on FAIL grep.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestFollowupsNoSilentSanity(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO / "scripts" / "run_ablation_followups.sh"
        self.src = self.path.read_text()

    def test_sanity_invocation_does_not_use_or_true(self) -> None:
        """Sanity script call must not have `|| true` — propagate failures."""
        sanity_block = re.search(
            r"run_sanity_checks\.py.*?(?:\n.{0,200})*?(?=#)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(
            sanity_block,
            "Could not locate run_sanity_checks invocation in followups script",
        )
        block_src = sanity_block.group(0)
        self.assertNotIn(
            "|| true", block_src,
            f"`|| true` found near run_sanity_checks invocation. "
            f"This silently swallows failures. Block: {block_src[:200]}",
        )

    def test_followups_exits_nonzero_on_sanity_fail_grep(self) -> None:
        """The wrapper must `exit` (non-zero) when sanity reports FAIL."""
        # Search for a pattern: grep "FAIL" and exit-non-zero on hit
        pattern = re.compile(
            r'grep.*FAIL.*\n[^#]*exit\s+\d+',
            re.DOTALL,
        )
        self.assertTrue(
            pattern.search(self.src),
            "Followups must exit non-zero when sanity log contains FAIL — "
            "got source without grep-FAIL→exit guard",
        )

    def test_skip_sanity_escape_hatch_documented(self) -> None:
        """An explicit `SKIP_SANITY=1` env override must exist for emergency
        bypass — but never silent."""
        self.assertIn("SKIP_SANITY", self.src,
                      "Need SKIP_SANITY=1 escape hatch (explicit, loud)")
        self.assertIn("NOT for ship decisions", self.src,
                      "Bypass must warn user it's not for ship decisions")


if __name__ == "__main__":
    unittest.main()
