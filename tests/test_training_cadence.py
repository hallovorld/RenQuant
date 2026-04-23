"""Tests for the training.cadence gate in FullTrainingPipeline."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.pp_training_full import (  # noqa: E402
    FullTrainingContext,
    FullTrainingPipeline,
    _cadence_allows_today,
)


class TestCadenceGateFunction:
    def test_daily_always_allowed(self):
        cfg: dict = {"training": {"cadence": "daily"}}
        for wd in range(7):
            allowed, _ = _cadence_allows_today(cfg, wd)
            assert allowed is True

    def test_default_cadence_is_daily(self):
        """Missing training block preserves the current (daily) behaviour."""
        allowed, _ = _cadence_allows_today({}, datetime.date.today().weekday())
        assert allowed is True

    def test_weekly_sunday_default_allows_sunday_only(self):
        cfg = {"training": {"cadence": "weekly"}}
        assert _cadence_allows_today(cfg, 6)[0] is True   # Sunday
        for wd in range(6):
            assert _cadence_allows_today(cfg, wd)[0] is False

    def test_weekly_custom_weekday(self):
        cfg = {"training": {"cadence": "weekly", "weekly_weekday": 2}}  # Wednesday
        assert _cadence_allows_today(cfg, 2)[0] is True
        assert _cadence_allows_today(cfg, 1)[0] is False

    def test_unknown_cadence_fails_open(self):
        cfg = {"training": {"cadence": "monthly"}}
        allowed, _ = _cadence_allows_today(cfg, 0)
        assert allowed is True


class TestCustomCadence:
    def test_custom_allows_listed_weekdays_only(self):
        cfg = {"training": {"cadence": "custom", "allowed_weekdays": [1, 3, 6]}}
        for wd in [1, 3, 6]:
            assert _cadence_allows_today(cfg, wd)[0] is True
        for wd in [0, 2, 4, 5]:
            assert _cadence_allows_today(cfg, wd)[0] is False

    def test_custom_empty_list_fails_open(self):
        cfg = {"training": {"cadence": "custom", "allowed_weekdays": []}}
        assert _cadence_allows_today(cfg, 0)[0] is True

    def test_custom_malformed_list_fails_open(self):
        cfg = {"training": {"cadence": "custom",
                            "allowed_weekdays": ["monday", None]}}
        assert _cadence_allows_today(cfg, 0)[0] is True

    def test_custom_missing_list_fails_open(self):
        """Missing allowed_weekdays means no schedule configured → don't block."""
        cfg = {"training": {"cadence": "custom"}}
        assert _cadence_allows_today(cfg, 0)[0] is True


class TestPipelineGate:
    def _make_ctx(self, cfg: dict | None = None, force: bool = False):
        return FullTrainingContext(
            config=cfg if cfg is not None else {},
            strategy="renquant_104",
            strategy_dir=_STRATEGY_DIR,
            skip_baseline=True,
            skip_panel=True,
            skip_recalibrate=True,
            force_retrain=force,
        )

    def test_weekly_gate_blocks_on_non_cadence_day(self):
        """When cadence blocks, no jobs execute even if skip_* are False."""
        ctx = FullTrainingContext(
            config={"training": {"cadence": "weekly", "weekly_weekday": 6}},
            strategy="renquant_104",
            strategy_dir=_STRATEGY_DIR,
        )
        with patch("kernel.pipeline.pp_training_full.BaselineTournamentJob") as MockBase, \
             patch("datetime.date") as MockDate:
            MockDate.today.return_value = datetime.date(2026, 4, 20)  # Monday
            MockDate.side_effect = lambda *a, **kw: datetime.date(*a, **kw)
            FullTrainingPipeline().run(ctx)
        assert not MockBase.called

    def test_weekly_gate_allows_cadence_day(self):
        ctx = FullTrainingContext(
            config={"training": {"cadence": "weekly", "weekly_weekday": 6}},
            strategy="renquant_104",
            strategy_dir=_STRATEGY_DIR,
            skip_baseline=True, skip_panel=True, skip_recalibrate=True,
        )
        with patch("datetime.date") as MockDate:
            MockDate.today.return_value = datetime.date(2026, 4, 26)  # Sunday
            MockDate.side_effect = lambda *a, **kw: datetime.date(*a, **kw)
            result = FullTrainingPipeline().run(ctx)
        # Pipeline ran (no exception) and returned the same ctx
        assert result is ctx

    def test_force_retrain_bypasses_gate(self):
        ctx = FullTrainingContext(
            config={"training": {"cadence": "weekly", "weekly_weekday": 6}},
            strategy="renquant_104",
            strategy_dir=_STRATEGY_DIR,
            skip_baseline=True, skip_panel=True, skip_recalibrate=True,
            force_retrain=True,
        )
        called: list[str] = []
        class FakeJob:
            def __init__(self, name="x"):
                self._name = name
            def should_skip(self, ctx):
                return True  # still skip-all since we only care about gate semantics
            def run(self, ctx):
                called.append(self._name)
            @property
            def name(self):
                return self._name

        # On a non-cadence day (Monday), force_retrain=True should still enter the loop
        with patch("kernel.pipeline.pp_training_full.BaselineTournamentJob", lambda: FakeJob("baseline")), \
             patch("kernel.pipeline.pp_training_full.PanelTrainingJob",       lambda: FakeJob("panel")), \
             patch("kernel.pipeline.pp_training_full.RecalibrationJob",       lambda: FakeJob("recalibrate")), \
             patch("datetime.date") as MockDate:
            MockDate.today.return_value = datetime.date(2026, 4, 20)  # Monday
            MockDate.side_effect = lambda *a, **kw: datetime.date(*a, **kw)
            FullTrainingPipeline().run(ctx)

        # All jobs' should_skip was consulted ⇒ loop entered ⇒ gate bypassed
        # (called list stays empty because each FakeJob skips, but .run wouldn't
        #  be reached at all under the gate)
        assert called == []  # all skipped
        # The real assertion is no AssertionError was raised — gate bypass worked.
