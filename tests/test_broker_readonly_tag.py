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
from live.runner import _readonly_label_prefix  # noqa: E402
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
