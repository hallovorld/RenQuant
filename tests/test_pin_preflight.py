"""Contract test: trading entrypoints must align subrepo checkouts to the
audited pins before trading, fail-closed.

Context (2026-06-11): a daily full silently traded STALE code because nothing
verified the runtime subrepo checkouts matched subrepos.lock.json (the umbrella
was a day behind origin/main). The fix is a shared preflight that auto-aligns
each isolated runtime checkout to its PINNED commit and ABORTS if a repo is
dirty or unreachable. These source-level checks pin that wiring so a future
refactor can't silently drop the guard from a trading path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PREFLIGHT = (_SCRIPTS / "preflight_pin_align.sh").read_text()
DAILY = (_SCRIPTS / "daily_104.sh").read_text()
INTRADAY = (_SCRIPTS / "intraday_sell_104.sh").read_text()


def _pos(src: str, needle: str) -> int:
    idx = src.find(needle)
    assert idx >= 0, needle
    return idx


class TestPreflightHelper:
    def test_aligns_to_pins_via_assemble_sync(self):
        # Aligns clean-but-drifted checkouts to the lock commit (--sync); the
        # tool refuses on a dirty repo, which is our fail-closed signal.
        assert "subrepo_assemble.py" in PREFLIGHT
        assert "--sync" in PREFLIGHT

    def test_aligns_isolated_runtime_root_by_default(self):
        # Production should validate the isolated runtime root, not dirty
        # sibling developer worktrees.
        assert "--runtime-root" in PREFLIGHT
        assert "$REPO_DIR/.subrepo_runtime/repos" in PREFLIGHT
        assert "RENQUANT_PIN_SYNC_RUNTIME_ROOT" in PREFLIGHT

    def test_fail_closed_aborts_on_misalignment(self):
        # A non-zero assemble result must abort the caller (never trade).
        assert "exit 1" in PREFLIGHT
        assert "Refusing to trade unaudited code" in PREFLIGHT

    def test_umbrella_main_is_not_auto_pulled(self):
        # Deliberate-deploy policy: warn on lag, never `git pull`/`git merge`
        # the umbrella main into a live-trading run.
        assert "git pull" not in PREFLIGHT
        assert "git merge" not in PREFLIGHT
        assert "git -C \"$REPO_DIR\" fetch" in PREFLIGHT  # warn-only freshness probe
        assert "PREFLIGHT_CHECK_UMBRELLA" in PREFLIGHT

    def test_has_emergency_escape_hatch(self):
        assert "RENQUANT_SKIP_PIN_SYNC" in PREFLIGHT


class TestTradingEntrypointsWireThePreflight:
    @pytest.mark.parametrize("name,src", [("daily_104.sh", DAILY),
                                          ("intraday_sell_104.sh", INTRADAY)])
    def test_entrypoint_sources_preflight(self, name: str, src: str):
        assert "preflight_pin_align.sh" in src, (
            f"{name} must source the pin-alignment preflight before trading"
        )

    def test_daily_checks_umbrella_lag(self):
        assert "PREFLIGHT_CHECK_UMBRELLA=1" in DAILY

    def test_intraday_does_not_check_umbrella_lag(self):
        # The 12-minute loop must not fetch the umbrella every run.
        assert "PREFLIGHT_CHECK_UMBRELLA=1" not in INTRADAY

    def test_daily_preflight_runs_after_lock_and_market_guard(self):
        preflight = _pos(DAILY, 'source "$REPO_DIR/scripts/preflight_pin_align.sh"')
        assert _pos(DAILY, "trap \"rm -f '$LOCK_FILE'\" EXIT") < preflight
        assert _pos(DAILY, 'echo "NYSE open today') < preflight
        assert preflight < _pos(DAILY, "runtime_qp_sanity_check.py")

    def test_intraday_preflight_runs_after_lock_and_market_guard(self):
        preflight = _pos(INTRADAY, 'source "$REPO_DIR/scripts/preflight_pin_align.sh"')
        assert _pos(INTRADAY, "sys.exit(0 if len(sched) > 0 else 1)") < preflight
        assert _pos(INTRADAY, "trap \"rm -f '$LOCK_FILE'\" EXIT") < preflight
        assert _pos(INTRADAY, 'cd "$REPO_DIR"') < preflight
        assert preflight < _pos(INTRADAY, "runtime_qp_sanity_check.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
