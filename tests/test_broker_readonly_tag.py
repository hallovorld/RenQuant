"""shadow_blend rail (2026-07-27): parameterized readonly shadow-lane tag.

Contract under test:
1. Default identity — no env, no ctor arg → broker_name == "alpaca_shadow",
   byte-identical to the legacy PatchTST shadow lane (daily_104 Step 4).
2. Tag threading — RENQUANT_READONLY_TAG=alpaca_shadow_blend (daily_104
   Step 5) routes the wrapper's broker_name, and kernel/state_paths derives
   live_state.alpaca_shadow_blend.json + runs.alpaca_shadow_blend.db.
3. Fail-closed validation — a set-but-invalid tag raises ValueError instead
   of silently writing into the legacy lane's state files.
4. ntfy label — legacy tag keeps the literal "[READONLY]" prefix; the blend
   tag gets "[READONLY][RC]" (fleet callsign, 2026-08-04); BOTH still start with
   "[READONLY]" so _notify_decision's is_shadow classification holds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from live.broker_readonly import (  # noqa: E402
    DEFAULT_READONLY_TAG,
    READONLY_TAG_ENV,
    ReadOnlyBrokerWrapper,
    resolve_readonly_tag,
    validate_readonly_tag,
)
from live.runner import LANE_CALLSIGNS, _readonly_label_prefix  # noqa: E402
from kernel.state_paths import (  # noqa: E402
    ALLOWED_BROKERS,
    live_state_path,
    runs_db_path,
)

BLEND_TAG = "alpaca_shadow_blend"


class TestTagResolution:
    def test_default_is_legacy_alpaca_shadow(self, monkeypatch):
        """Byte-identical legacy: env unset → the historical tag."""
        monkeypatch.delenv(READONLY_TAG_ENV, raising=False)
        assert resolve_readonly_tag() == "alpaca_shadow"
        assert DEFAULT_READONLY_TAG == "alpaca_shadow"

    def test_empty_env_is_legacy(self, monkeypatch):
        monkeypatch.setenv(READONLY_TAG_ENV, "")
        assert resolve_readonly_tag() == "alpaca_shadow"

    def test_blend_env_threads_through(self, monkeypatch):
        monkeypatch.setenv(READONLY_TAG_ENV, BLEND_TAG)
        assert resolve_readonly_tag() == BLEND_TAG

    @pytest.mark.parametrize("bad", [
        "alpaca",                 # would collide with PROD state files
        "alpaca-shadow-blend",    # hyphens: not filename-token safe here
        "shadow_blend",           # missing the alpaca_shadow prefix
        "alpaca_shadow/../evil",  # path traversal
        "alpaca_shadow blend",    # whitespace
    ])
    def test_invalid_env_fails_closed(self, monkeypatch, bad):
        """Invalid tag must ABORT, never fall back to the legacy lane."""
        monkeypatch.setenv(READONLY_TAG_ENV, bad)
        with pytest.raises(ValueError):
            resolve_readonly_tag()

    def test_validate_accepts_legacy_and_blend(self):
        assert validate_readonly_tag("alpaca_shadow") == "alpaca_shadow"
        assert validate_readonly_tag(BLEND_TAG) == BLEND_TAG


class TestWrapperTag:
    def _wrapper(self, **kwargs) -> ReadOnlyBrokerWrapper:
        return ReadOnlyBrokerWrapper(SimpleNamespace(), **kwargs)

    def test_default_identity(self, monkeypatch):
        """No env, no ctor arg → legacy broker_name (Step 4 unchanged)."""
        monkeypatch.delenv(READONLY_TAG_ENV, raising=False)
        assert self._wrapper().broker_name == "alpaca_shadow"

    def test_env_tag_selects_blend_lane(self, monkeypatch):
        monkeypatch.setenv(READONLY_TAG_ENV, BLEND_TAG)
        assert self._wrapper().broker_name == BLEND_TAG

    def test_ctor_tag_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(READONLY_TAG_ENV, BLEND_TAG)
        assert self._wrapper(tag="alpaca_shadow").broker_name == "alpaca_shadow"

    def test_ctor_tag_validated(self, monkeypatch):
        monkeypatch.delenv(READONLY_TAG_ENV, raising=False)
        with pytest.raises(ValueError):
            self._wrapper(tag="alpaca")


class TestStatePathRouting:
    """kernel/state_paths must route the blend tag to its own lane."""

    def test_blend_tag_in_allowlist(self):
        assert BLEND_TAG in ALLOWED_BROKERS

    def test_live_state_path_blend(self, tmp_path):
        assert live_state_path(tmp_path, BLEND_TAG) == (
            tmp_path / "live_state.alpaca_shadow_blend.json"
        )

    def test_runs_db_path_blend(self):
        assert runs_db_path(Path("data/runs.db"), BLEND_TAG) == (
            Path("data/runs.alpaca_shadow_blend.db")
        )

    def test_lanes_are_disjoint(self, tmp_path):
        """prod / legacy shadow / blend shadow: three distinct state files."""
        paths = {
            str(live_state_path(tmp_path, b))
            for b in ("alpaca", "alpaca_shadow", BLEND_TAG)
        }
        assert len(paths) == 3
        dbs = {
            str(runs_db_path(Path("data/runs.db"), b))
            for b in ("alpaca", "alpaca_shadow", BLEND_TAG)
        }
        assert len(dbs) == 3


class TestNtfyLabelPrefix:
    def test_legacy_tag_is_byte_identical(self):
        assert _readonly_label_prefix("alpaca_shadow") == "[READONLY]"

    def test_blend_tag_carries_full_tag(self):
        assert _readonly_label_prefix(BLEND_TAG) == (
            "[READONLY][RC]"
        )

    def test_both_prefixes_keep_is_shadow_contract(self):
        """_notify_decision keys is_shadow off label.startswith('[READONLY]')."""
        for tag in ("alpaca_shadow", BLEND_TAG):
            assert _readonly_label_prefix(tag).startswith("[READONLY]")

    def test_non_shadow_brokers_get_no_prefix(self):
        for name in ("alpaca", "paper", "alpaca-paper", "ibkr", ""):
            assert _readonly_label_prefix(name) == ""


class TestCallsignCoverage:
    """Every lane the daily script LAUNCHES must have a callsign.

    `_readonly_label_prefix` ends in `LANE_CALLSIGNS.get(tag, tag.upper())`.
    A permissive default means adding a lane cannot fail — it degrades, and
    the degraded form is the worst one for the only surface that matters:
    "[READONLY][ALPACA_SHADOW_VOL_WINDOW]" is 36 characters of title, so on a
    phone the reader sees the lane marker and nothing else, or the marker gets
    truncated away and the body reads like a live fill. That is the same harm
    as orch#1014 (the dropped SHADOW disclaimer), reached by a different route.

    So this asserts coverage against the AUTHORITY on what runs — the
    RENQUANT_READONLY_TAG assignments in scripts/daily_104.sh — not against a
    second hand-maintained list, which would rot in exactly the same way.
    """

    LAUNCH_TAG_RE = re.compile(r"RENQUANT_READONLY_TAG=([a-z0-9_]+)")

    def _launched_tags(self) -> set[str]:
        script = Path(__file__).resolve().parents[1] / "scripts" / "daily_104.sh"
        assert script.is_file(), f"daily_104.sh not found at {script}"
        tags = set(self.LAUNCH_TAG_RE.findall(script.read_text()))
        # Guard the guard: if the launch idiom is ever refactored away, this
        # test must fail loudly rather than pass over an empty set.
        assert tags, (
            "no RENQUANT_READONLY_TAG=... assignments found in daily_104.sh — "
            "the launch idiom changed and this coverage check is now vacuous"
        )
        return tags

    def test_every_running_shadow_lane_has_a_callsign(self):
        missing = sorted(
            t for t in self._launched_tags()
            if t != "alpaca_shadow" and t not in LANE_CALLSIGNS
        )
        assert not missing, (
            f"shadow lanes launched by daily_104.sh with no entry in "
            f"LANE_CALLSIGNS: {missing}. Each falls back to its tag in caps, "
            f"e.g. [READONLY][{missing[0].upper()}] — add a callsign in "
            f"live/runner.py."
        )

    def test_the_vol_window_lane_specifically(self):
        """The lane that was missing for six days. Regression pin."""
        assert _readonly_label_prefix("alpaca_shadow_vol_window") == "[READONLY][V]"

    def test_callsigns_stay_short_enough_to_leave_room_for_a_decision(self):
        """Terseness is the point (operator directive 2026-08-04 简练).

        A prefix is pure overhead on a notification title; the budget below is
        deliberately loose — it exists to catch a tag-shaped value landing in
        the map, not to police the operator's naming.
        """
        for tag, sign in LANE_CALLSIGNS.items():
            assert 0 < len(sign) <= 4, f"{tag} → {sign!r} is not a terse callsign"
            assert sign.isalnum(), f"{tag} → {sign!r} should be alphanumeric"

    def test_callsigns_are_unique(self):
        """Two lanes sharing a marker is worse than no marker: the reader is
        confidently told the wrong lane."""
        signs = list(LANE_CALLSIGNS.values())
        assert len(signs) == len(set(signs)), f"duplicate callsigns: {signs}"

    def test_coverage_holds_for_the_prefix_function_not_just_the_map(self):
        """The map is the mechanism; the prefix is the contract. Assert on the
        rendered title, so a future refactor that stops consulting the map is
        caught here too."""
        for tag in self._launched_tags():
            prefix = _readonly_label_prefix(tag)
            assert prefix.startswith("[READONLY]"), tag
            assert tag.upper() not in prefix, (
                f"{tag} renders as {prefix} — the shouted-tag fallback, "
                f"meaning it has no callsign"
            )


class TestNtfyTitleBlendLane:
    """End-to-end title check through _notify_decision for the blend label."""

    @pytest.fixture(autouse=True)
    def _allow_mocked_ntfy(self, monkeypatch):
        monkeypatch.delenv("RENQUANT_NO_NOTIFY", raising=False)

    def test_blend_no_trade_cycle_is_shadow_decision(self):
        from live.runner import _notify_decision
        ctx = SimpleNamespace(
            orders=[], orders_placed=[], orders_skipped=[], exits=[],
            regime="BULL_CALM", confidence=0.50, portfolio_value=10071.0,
            holdings={}, bear_only=False,
            regime_state=SimpleNamespace(in_transition=False),
            skip_buys=False, buy_blocked=False, counters={}, ranked=[],
        )
        with patch("urllib.request.urlopen") as m:
            _notify_decision(
                "[READONLY][RC]RENQUANT-104", "full", ctx,
            )
        m.assert_called_once()
        req = m.call_args[0][0]
        title = req.headers.get("Title")
        assert title == (
            "[READONLY][RC]RENQUANT-104 [full] SHADOW-DECISION"
        )
        assert req.headers.get("Priority") == "default"
        # 2026-08-04: boilerplate body sentence removed (title carries the
        # shadow identity twice); the safety property that remains load-
        # bearing is the [READONLY] title classification, asserted above.
        assert "SHADOW/HYPOTHETICAL" not in req.data.decode()


def test_validate_accepts_the_s1_blend_mom_tag():
    """GOAL-8 S1 lane tag (Step 5b): alpaca_shadow* prefix rule admits it
    with no validator change; state files stay disjoint by construction."""
    tag = "alpaca_shadow_blend_mom"
    assert validate_readonly_tag(tag) == tag
    assert str(live_state_path(Path("s"), tag)).endswith(
        "live_state.alpaca_shadow_blend_mom.json")
    assert str(runs_db_path(Path("data/runs.db"), tag)).endswith(
        "runs.alpaca_shadow_blend_mom.db")


# ── fleet callsign contract (2026-08-04, codex on RQ#578) ────────────────────

import pytest as _pytest
from live.runner import LANE_CALLSIGNS, _readonly_label_prefix as _prefix

#: This file IS the operator-notification contract: it calls the composition
#: helpers and asserts on what the operator would read. Membership is this
#: MARKER, deliberately applied, not a substring scan over the source — a scan
#: cannot tell a contract from a file that merely monkeypatches a helper away
#: [codex on RenQuant#601]. The workflow runs exactly the marked files, and
#: TestTheCONTRACTWorkflowActuallyRunsTheseTests asserts both directions.
pytestmark = pytest.mark.notification_contract


@_pytest.mark.parametrize("tag,callsign", [
    ("alpaca_shadow_blend", "RC"),
    ("alpaca_shadow_blend_mom", "RSs"),
    ("alpaca_shadow_blend_mom_fast", "Rf"),
    ("alpaca_shadow_blend_rb_mom", "RCS"),
    ("alpaca_shadow_blend_rb_fast", "RCf"),
])
def test_every_fleet_tag_maps_to_its_callsign(tag, callsign):
    """Operator-facing lane identity: each configured mapping is a direct
    regression guard, and every prefix still STARTS with [READONLY] (the
    is_shadow classification a shadow message must never lose)."""
    assert LANE_CALLSIGNS[tag] == callsign
    p = _prefix(tag)
    assert p == f"[READONLY][{callsign}]"
    assert p.startswith("[READONLY]")


def test_unknown_shadow_tag_falls_back_to_full_upper_tag():
    """Forward-compatible fallback: an UNKNOWN alpaca_shadow_* tag gets its
    uppercased full tag — never a bare [READONLY] (that literal belongs to
    the legacy lane's byte-identical contract), never a wrong callsign."""
    p = _prefix("alpaca_shadow_blend_newlane_x")
    assert p == "[READONLY][ALPACA_SHADOW_BLEND_NEWLANE_X]"
    assert p != "[READONLY]"


def test_legacy_tag_stays_byte_identical():
    assert _prefix("alpaca_shadow") == "[READONLY]"
    assert _prefix("alpaca") == ""
